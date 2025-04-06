import argparse
import json
import logging
import os
import pathlib
from typing import Dict

import datasets
from anthropic_client import AnthropicClient
from data_types import GitRepoData, RequirementsData
from docker_utils import build_docker_images, run_test_in_container, write_docker_files
from exceptions import RequirementsError
from git_utils import clone_and_get_tree, load_file_contents
from pipeline.build_retry import retry_installation
from pipeline.collect import collect_requirements
from pipeline.localize import localize_requirements
from pipeline.test_retry import retry_test
from version_finder import time_travel_requirements

from logger import CustomLogger, setup_logger
from model_utils import anthropic_generate_json


def process_localization(result: GitRepoData, logger: CustomLogger) -> Dict[str, str]:
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
    loc_logger = setup_logger(
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


def process_requirements_collection(
    file_contents: Dict[str, str],
    result: GitRepoData,
    logger: CustomLogger,
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
    req_logger = setup_logger(
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
    trial_logger = setup_logger(
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
    build_logger = setup_logger(
        logger_name="first_build", parent_logger=logger, subfolder="first_build"
    )
    write_docker_files(
        requirements_data=time_traveled_requirements,
        git_data=git_data,
        logger=build_logger,
    )

    # Build Docker images
    build_output = build_docker_images(logger=logger, build_name="first_build")

    # Check if Docker build was successful
    num_trial = 0
    while build_output.returncode != 0 and num_trial < 3:
        trial_logger = setup_logger(
            logger_name=f"trial_and_error_{num_trial}",
            parent_logger=logger,
            subfolder="trial_and_error",
        )
        trial_logger.error(f"Docker build failed with exit code {build_output.returncode}")
        trial_logger.error(f"Error output: {build_output.stderr}")
        trial_logger.info(f"Trial {num_trial} failed")
        error_message = build_output.stderr
        time_traveled_requirements = process_trial_and_error(
            file_contents=file_contents,
            requirements_data=time_traveled_requirements,
            git_data=git_data,
            error_message=error_message,
            logger=trial_logger,
        )
        write_docker_files(
            requirements_data=time_traveled_requirements,
            git_data=git_data,
            logger=trial_logger,
        )
        build_output = build_docker_images(
            logger=trial_logger, build_name=f"trial_and_error_{num_trial}"
        )
        num_trial += 1
        if build_output.returncode == 0:
            logger.info("After trial and error, Docker build completed successfully")

    # Run the test in a Docker container
    test_command = (
        "pytest sklearn/cluster/tests/test_affinity_propagation.py::test_affinity_propagation"
    )
    exit_code, logs = run_test_in_container(test_command, logger)

    num_test_retry = 0
    while exit_code != 0 and num_test_retry < 3:
        test_retry_logger = setup_logger(
            logger_name=f"test_retry_{num_test_retry}", parent_logger=logger, subfolder="test_retry"
        )
        error_message = logs
        prompt = retry_test(
            file_contents=file_contents,
            requirements_json=json.dumps(time_traveled_requirements, indent=2),
            commit_hash=git_data["commit_hash"],
            commit_date=git_data["commit_date"],
            error_message=error_message,
            test_command=test_command,
        )
        try:
            requirements_data_raw = anthropic_generate_json(
                prompt=prompt,
                logger=test_retry_logger,
                output_filename="test_retry_output.json",
            )
            # NOTE: I hate python typing ... this is so immature
            requirements_data_raw_cast: RequirementsData = {
                "python_version": requirements_data_raw.get("python_version", "3.8"),
                "apt_packages": requirements_data_raw.get("apt_packages", []),
                "pip_packages": requirements_data_raw.get("pip_packages", {}),
                "install_commands": requirements_data_raw.get("install_commands", ""),
            }
            time_traveled_requirements = time_travel_requirements(
                requirements_data=requirements_data_raw_cast,
                commit_date=git_data["commit_date"],
                logger=test_retry_logger,
            )
            write_docker_files(
                requirements_data=time_traveled_requirements,
                git_data=git_data,
                logger=test_retry_logger,
            )
            build_output = build_docker_images(
                logger=test_retry_logger, build_name=f"test_retry_{num_test_retry}"
            )
            if build_output.returncode != 0:
                raise ValueError(
                    f"During test retry, Docker build failed with exit code {build_output.returncode}"
                )

            exit_code, logs = run_test_in_container(test_command, test_retry_logger)
            num_test_retry += 1
        except RequirementsError:
            logger.error("Claude thinks the error cannot be solved by updating the requirements.")
