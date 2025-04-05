import argparse
import json
import logging
import os
import pathlib
import subprocess
import time
from typing import Dict, Optional

import datasets
from anthropic_client import AnthropicClient
from data_types import GitRepoData, RequirementsData
from docker_utils import write_docker_files
from git_utils import clone_and_get_tree, load_file_contents
from pipeline.collect import collect_requirements
from pipeline.localize import localize_requirements
from pipeline.retry import retry_installation
from version_finder import check_and_replace_version, get_version_at_time

import docker  # type: ignore
from logger import setup_child_logger, setup_logger
from model_utils import anthropic_generate_json


def process_localization(
    result: GitRepoData, logger: Optional[logging.Logger] = None
) -> Dict[str, str]:
    """
    Process the localization of requirements for a given commit.

    Args:
        result (GitRepoData): Dictionary containing tree_output, commit_hash, and commit_date
        client (AnthropicClient): Client for making requests to Anthropic API
        logger (logging.Logger, optional): Parent logger for tracking operations

    Returns:
        Dict[str, str]: Dictionary mapping file paths to their contents

    Raises:
        AnthropicResponseError: If there's an issue with the Anthropic API response
        RequirementsError: If there's an issue parsing the requirements
    """
    # Create specific logger for this function
    loc_logger = setup_child_logger(
        logger_name="localization", parent_logger=logger, subfolder="localization"
    )

    # Localize requirements
    prompt = localize_requirements(
        tree_result=result["tree_output"],
        commit_hash=result["commit_hash"],
        commit_date=result["commit_date"],
    )
    loc_logger.info(
        f"Localizing requirements for {result['commit_hash']} from {result['commit_date']}"
    )

    # Use the abstracted function to generate JSON from Anthropic API
    file_paths = anthropic_generate_json(
        prompt=prompt, logger=loc_logger, output_filename="localization_output.json"
    )

    return file_paths


def time_travel_requirements(
    requirements_data: RequirementsData, commit_date: str, logger: Optional[logging.Logger] = None
) -> RequirementsData:
    """
    Update package versions in requirements data based on the commit date.

    Args:
        requirements_data (RequirementsData): Dictionary containing the requirements data
        commit_date (str): The date of the commit in YYYY-MM-DD format
        logger (logging.Logger, optional): Logger for tracking operations

    Returns:
        RequirementsData: Updated requirements data with appropriate versions
    """
    if logger:
        logger.info(f"Time traveling requirements to {commit_date}")

    pip_packages: Dict[str, str] = {}
    updated_requirements: RequirementsData = {
        # TODO: this should also use time travel
        "python_version": requirements_data.get("python_version", "3.8"),
        # TODO: we should have some automatic validation of the apt packages
        "apt_packages": requirements_data.get("apt_packages", []),
        "pip_packages": pip_packages,
        "install_commands": requirements_data.get("install_commands", ""),
    }

    # Update pip package versions based on commit date
    for package, version in requirements_data.get("pip_packages", {}).items():
        if version.startswith("=="):
            # Use specific version as is (remove == prefix)
            updated_version = version[2:]
        elif version.startswith(">=") or version == "":
            # Find the appropriate version at the commit date
            updated_version = get_version_at_time(package_name=package, timestamp=commit_date)
            if logger:
                logger.info(
                    f"Updated {package} version to {updated_version} for date {commit_date}"
                )
        else:
            # Keep the version as is
            updated_version = version

        pip_packages[package] = updated_version

    return updated_requirements


def process_requirements_collection(
    file_contents: Dict[str, str],
    result: GitRepoData,
    logger: Optional[logging.Logger] = None,
) -> RequirementsData:
    """
    Process the collection of requirements and build the Docker environment.

    Args:
        file_contents (dict): Dictionary mapping file paths to their contents
        result (GitRepoData): Dictionary containing commit_hash and commit_date
        client (AnthropicClient): Client for making requests to Anthropic API
        logger (logging.Logger, optional): Parent logger for tracking operations
        repo_url (str, optional): URL of the git repository

    Raises:
        AnthropicResponseError: If there's an issue with the Anthropic API response
        RequirementsError: If there's an issue parsing the requirements
    """
    # Create specific logger for this function
    req_logger = setup_child_logger(
        logger_name="requirements_collection", parent_logger=logger, subfolder="requirements"
    )

    # Collect requirements
    prompt = collect_requirements(
        file_contents=file_contents,
        commit_hash=result["commit_hash"],
        commit_date=result["commit_date"],
    )
    req_logger.info(prompt)

    # Use the abstracted function to generate JSON from Anthropic API
    requirements_data_raw = anthropic_generate_json(
        prompt=prompt, logger=req_logger, output_filename="requirements_output.json"
    )

    # Ensure required fields are present
    requirements_data: RequirementsData = {
        "python_version": requirements_data_raw.get("python_version", "3.8"),
        "apt_packages": requirements_data_raw.get("apt_packages", []),
        "pip_packages": requirements_data_raw.get("pip_packages", {}),
        "install_commands": requirements_data_raw.get("install_commands", ""),
    }

    req_logger.info(f"Parsed requirements data: {requirements_data}")

    return requirements_data


def process_trial_and_error(
    file_contents: Dict[str, str],
    requirements_data: RequirementsData,
    git_data: GitRepoData,
    error_message: str,
    logger: logging.Logger,
) -> RequirementsData:
    """
    Process the trial and error of the Docker environment.

    Args:
        file_contents (dict): Dictionary mapping file paths to their contents
        requirements_json: The requirements data that failed
        result (GitRepoData): Dictionary containing commit_hash and commit_date
        error_message (str): The error message from the failed Docker build
        client (AnthropicClient): Client for making requests to Anthropic API
        logger (logging.Logger, optional): Parent logger for tracking operations
    """
    # Create specific logger for this function
    print("trial_and_error parent logger: ", logger)
    trial_logger = setup_child_logger(
        logger_name="trial_and_error", parent_logger=logger, subfolder="trial_and_error"
    )

    trial_logger.info("Starting trial and error process for failed Docker build")
    trial_logger.info(f"Error message: {error_message}")

    prompt = retry_installation(
        file_contents=file_contents,
        requirements_json=json.dumps(requirements_data, indent=2),
        commit_hash=git_data["commit_hash"],
        commit_date=git_data["commit_date"],
        error_message=error_message,
    )
    trial_logger.info(prompt)

    # Use the abstracted function to generate JSON from Anthropic API
    requirements_data_raw = anthropic_generate_json(
        prompt=prompt,
        logger=trial_logger,
        output_filename="trial_and_error_output.json",
    )

    # Ensure required fields are present
    casted_requirements_data: RequirementsData = {
        "python_version": requirements_data_raw.get("python_version", "3.8"),
        "apt_packages": requirements_data_raw.get("apt_packages", []),
        "pip_packages": requirements_data_raw.get("pip_packages", {}),
        "install_commands": requirements_data_raw.get("install_commands", ""),
    }

    trial_logger.info(f"Parsed requirements data: {casted_requirements_data}")

    return casted_requirements_data


def build_docker_images(logger: logging.Logger):
    """
    Build Docker images for the testbed environment using the Docker Python API.

    Args:
        logger (logging.Logger): Logger for tracking operations

    Returns:
        bool: True if Docker build was successful, False otherwise
    """
    # Initialize Docker client
    client = docker.from_env()  # type: ignore[attr-defined]

    # Build base image
    logger.info("Building docker image (base)")
    start_time = time.time()
    try:
        base_image, base_logs = client.images.build(
            path="./docker",
            dockerfile="Dockerfile.base",
            tag="testbed-base:latest",
        )
        base_build_time = time.time() - start_time
        logger.info(f"Base docker image build completed in {base_build_time:.2f} seconds")
        logger.debug(f"Base image ID: {base_image.id}")
    except docker.errors.BuildError as e:  # type: ignore[attr-defined]
        logger.error(f"Error building base docker image: {e}")
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=str(e))
    except Exception as e:
        logger.error(f"Unexpected error building base docker image: {e}")
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=str(e))

    # Build testbed image
    logger.info("Building docker image (testbed)")
    start_time = time.time()
    testbed_logs: list[dict] = []  # Initialize testbed_logs to prevent UnboundLocalError
    try:
        testbed_image, logs = client.images.build(
            path="./docker",
            dockerfile="Dockerfile",
            tag="testbed:latest",
        )
        for log in logs:
            testbed_logs.append(log)  # type: ignore[arg-type]
        testbed_build_time = time.time() - start_time
        logger.info(f"Testbed docker image build completed in {testbed_build_time:.2f} seconds")
        logger.debug(f"Testbed image ID: {testbed_image.id}")

        # Save stdout to log directory if build successful
        build_log = f"Build successful\nBase build time: {base_build_time:.2f} seconds\nTestbed build time: {testbed_build_time:.2f} seconds\nBase image ID: {base_image.id}\nTestbed image ID: {testbed_image.id}"

        # Return success
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=build_log, stderr="")
    except docker.errors.BuildError as e:  # type: ignore[attr-defined]
        logger.error(f"Error building testbed docker image: {e}")
        error_message = "\n".join(
            [
                log.get("stream", "")
                for log in testbed_logs
                if "error" in log.get("stream", "").lower()
            ]
        )
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=error_message or str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error building testbed docker image: {e}")
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=str(e))


def get_pip_freeze_requirements(
    result: GitRepoData, logger: Optional[logging.Logger] = None
) -> Dict[str, str]:
    """
    Run pip freeze in Docker container and process the requirements.

    Args:
        result (GitRepoData): Dictionary containing commit date and other info
        logger (logging.Logger, optional): Logger for tracking operations

    Returns:
        Dict[str, str]: Dictionary mapping package names to versions
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Running pip freeze in Docker container")
    # Create and start a container from the testbed image
    container = docker.from_env().containers.run(
        "testbed:latest", command="pip freeze", remove=True, detach=False
    )
    # container output is in bytes, convert to string
    output_str = container.decode("utf-8")
    logger.info(output_str)

    # process the requirements being spit out by pip freeze
    pip_freeze_requirements: Dict[str, str] = {}
    for line in output_str.split("\n"):
        if "==" not in line:
            continue
        package, version = line.split("==")
        pip_freeze_requirements[package] = check_and_replace_version(
            package, version, result["commit_date"]
        )

    logger.info(f"Processed {len(pip_freeze_requirements)} packages from pip freeze")
    return pip_freeze_requirements


# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SWE-Bench environment builder")
    parser.add_argument(
        "--debug",
        type=str,
        help="Load results from a specific experiment timestamp (format: YYYYMMDD_HHMMSS)",
    )
    args = parser.parse_args()

    # Set up main logger
    logger = setup_logger(debug=args.debug is not None)

    swe_bench = datasets.load_dataset("princeton-nlp/SWE-Bench", split="test")
    swe_bench = swe_bench.filter(lambda x: x["repo"] == "scikit-learn/scikit-learn").sort(
        "created_at"
    )

    example = swe_bench[10]
    git_data = clone_and_get_tree(
        repo_name=example["repo"],
        commit=example["base_commit"],
        logger=logger,
    )

    # Initialize Anthropic client
    client = AnthropicClient()

    if args.debug:
        logger.info(f"Debug mode enabled, loading results from specified experiment: {args.debug}")
        exp_dir = os.path.join("exps", args.debug)
        if not os.path.exists(exp_dir) or not os.path.isdir(exp_dir):
            logger.error(f"Specified experiment directory does not exist: {exp_dir}")
            exit(1)

        logger.info(f"Loading from experiment directory: {exp_dir}")

        # Load localization results
        localization_file = os.path.join(exp_dir, "localization", "localization_output.json")
        if not os.path.exists(localization_file):
            logger.error(f"Localization output file not found: {localization_file}")
            exit(1)

        with open(localization_file, "r") as f:
            file_paths = json.load(f)
            logger.info(f"Loaded localization data: {file_paths}")

        # Load file contents
        file_contents = load_file_contents(
            file_paths=file_paths, base_dir=pathlib.Path("playground/scikit-learn"), logger=logger
        )

        # Load requirements collection results
        requirements_file = os.path.join(exp_dir, "requirements", "requirements_output.json")
        if not os.path.exists(requirements_file):
            logger.error(f"Requirements output file not found: {requirements_file}")
            exit(1)

        with open(requirements_file, "r") as f:
            requirements_data = json.load(f)
            logger.info(f"Loaded requirements data: {requirements_data}")
    else:
        # Process localization - now creates its own logger
        file_paths = process_localization(result=git_data, logger=logger)

        # Load file contents
        file_contents = load_file_contents(
            file_paths=file_paths, base_dir=pathlib.Path("playground/scikit-learn"), logger=logger
        )

        # Process requirements collection - now creates its own logger
        requirements_data = process_requirements_collection(
            file_contents=file_contents,
            result=git_data,
            logger=logger,
        )

    # Write Docker configuration files
    time_traveled_requirements = time_travel_requirements(
        requirements_data=requirements_data, commit_date=git_data["commit_date"], logger=logger
    )
    write_docker_files(
        requirements_data=time_traveled_requirements,
        git_data=git_data,
        logger=logger,
    )

    # Build Docker images
    output = build_docker_images(logger=logger)

    # Check if Docker build was successful
    if output.returncode != 0:
        logger.error(f"Docker build failed with exit code {output.returncode}")
        logger.error(f"Error output: {output.stderr}")
        error_message = output.stderr
        time_traveled_requirements = process_trial_and_error(
            file_contents=file_contents,
            requirements_data=time_traveled_requirements,
            git_data=git_data,
            error_message=error_message,
            logger=logger,
        )
        write_docker_files(
            requirements_data=time_traveled_requirements,
            git_data=git_data,
            logger=logger,
        )
        output = build_docker_images(logger=logger)
        if output.returncode != 0:
            logger.error(
                f"After trial and error, Docker build failed with exit code {output.returncode}"
            )
            logger.error(f"Error output: {output.stderr}")
            logger.error("Failed to build Docker images")
            exit(1)
        else:
            logger.info("After trial and error, Docker build completed successfully")
            logger.info(output.stdout)

    # # Get pip freeze requirements from the built Docker container
    # pip_freeze_requirements = get_pip_freeze_requirements(result=result, logger=logger)

    # # Update the requirements data
    # time_traveled_requirements["pip_packages"] = pip_freeze_requirements

    # # Write the docker files again
    # write_docker_files(
    #     requirements_data=time_traveled_requirements,
    #     result=result,
    #     repo_url=REPO_URL,
    #     logger=logger,
    # )

    # # Build the docker image again
    # logger.info("Building docker image (testbed) with time traveled pip freeze requirements")
    # output = build_docker_images(logger=logger)
    # if output.returncode != 0:
    #     logger.error(f"Docker build failed with exit code {output.returncode}")
    #     logger.error(f"Error output: {output.stderr}")
    #     exit(1)
    # else:
    #     logger.info(output.stdout)
    #     logger.info("Docker image built successfully")
