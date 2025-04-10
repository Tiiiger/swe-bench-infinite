import argparse
import json
import os
import pathlib
import subprocess
from pathlib import Path
from typing import Dict

import datasets
from anthropic_client import AnthropicClient
from data_types import GitRepoData, RequirementsData
from docker_utils import build_docker_images, copy_to_container, exec_run_with_timeout
from exceptions import RequirementsError
from git_utils import clone_and_get_tree, load_file_contents
from pipeline.build_retry import retry_installation
from pipeline.collect import collect_requirements
from pipeline.localize import localize_requirements
from swebench.harness.test_spec.test_spec import make_test_spec
from version_finder import time_travel_requirements

import docker
from logger import CustomLogger, setup_logger
from model_utils import anthropic_generate_json


def process_localization(result: GitRepoData, logger: CustomLogger) -> list[str]:
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
    loc_logger = setup_logger(logger_name="localization", parent_logger=logger)

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
    req_logger = setup_logger(logger_name="requirements_collection", parent_logger=logger)

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


def process_retry_build(
    file_contents: Dict[str, str],
    requirements_data: RequirementsData,
    git_data: GitRepoData,
    error_message: str,
    trial_num: int,
    parent_logger: CustomLogger,
) -> tuple[RequirementsData, subprocess.CompletedProcess]:
    """
    Process the retry of a failed build by attempting to fix requirements and rebuild.

    Args:
        file_contents (dict): Dictionary mapping file paths to their contents
        requirements_data (RequirementsData): Current requirements data
        git_data (GitRepoData): Dictionary containing commit_hash and commit_date
        error_message (str): The error message from the failed build
        trial_num (int): The current trial number
        parent_logger (CustomLogger): Parent logger for tracking operations

    Returns:
        tuple[RequirementsData, subprocess.CompletedProcess]: Updated requirements data and build output
    """
    build_retry_logger = setup_logger(
        logger_name=f"retry_build_{trial_num}",
        parent_logger=parent_logger,
    )

    prompt = retry_installation(
        file_contents=file_contents,
        requirements_json=json.dumps(requirements_data, indent=2),
        commit_hash=git_data["commit_hash"],
        commit_date=git_data["commit_date"],
        error_message=error_message,
    )

    try:
        requirements_data_raw = anthropic_generate_json(
            prompt=prompt,
            logger=build_retry_logger,
            output_filename="retry_build_output.json",
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
            logger=build_retry_logger,
        )
        build_output = build_docker_images(
            requirements_data=time_traveled_requirements,
            git_data=git_data,
            logger=build_retry_logger,
            build_name=f"retry_build_{trial_num}",
        )

        return time_traveled_requirements, build_output
    except RequirementsError:
        parent_logger.error(
            "Claude thinks the error cannot be solved by updating the requirements."
        )
        raise


# def process_retry_test(
#     file_contents: Dict[str, str],
#     requirements_data: RequirementsData,
#     git_data: GitRepoData,
#     error_message: str,
#     test_command: str,
#     trial_num: int,
#     parent_logger: CustomLogger,
# ) -> tuple[RequirementsData, subprocess.CompletedProcess]:
#     """
#     Process the retry of a failed test by attempting to fix requirements and rebuild.

#     Args:
#         file_contents (dict): Dictionary mapping file paths to their contents
#         requirements_data (RequirementsData): Current requirements data
#         git_data (GitRepoData): Dictionary containing commit_hash and commit_date
#         error_message (str): The error message from the failed test
#         test_command (str): The test command to run
#         trial_num (int): The current trial number
#         parent_logger (CustomLogger): Parent logger for tracking operations

#     Returns:
#         tuple[RequirementsData, subprocess.CompletedProcess]: Updated requirements data and test output
#     """
#     test_retry_logger = setup_logger(
#         logger_name=f"retry_test_{trial_num}",
#         parent_logger=parent_logger,
#     )

#     prompt = retry_test(
#         file_contents=file_contents,
#         requirements_json=json.dumps(requirements_data, indent=2),
#         commit_hash=git_data["commit_hash"],
#         commit_date=git_data["commit_date"],
#         error_message=error_message,
#         test_command=test_command,
#     )

#     try:
#         requirements_data_raw = anthropic_generate_json(
#             prompt=prompt,
#             logger=test_retry_logger,
#             output_filename="retry_test_output.json",
#         )
#         # NOTE: I hate python typing ... this is so immature
#         requirements_data_raw_cast: RequirementsData = {
#             "python_version": requirements_data_raw.get("python_version", "3.8"),
#             "apt_packages": requirements_data_raw.get("apt_packages", []),
#             "pip_packages": requirements_data_raw.get("pip_packages", {}),
#             "install_commands": requirements_data_raw.get("install_commands", ""),
#         }
#         time_traveled_requirements = time_travel_requirements(
#             requirements_data=requirements_data_raw_cast,
#             commit_date=git_data["commit_date"],
#             logger=test_retry_logger,
#         )
#         build_output = build_docker_images(
#             requirements_data=time_traveled_requirements,
#             git_data=git_data,
#             logger=test_retry_logger,
#             build_name=f"retry_test_{trial_num}",
#         )
#         if build_output.returncode != 0:
#             raise ValueError(
#                 f"During test retry, Docker build failed with exit code {build_output.returncode}"
#             )

#         test_output = run_test_in_container(test_command, test_retry_logger)
#         return time_traveled_requirements, test_output
#     except RequirementsError:
#         parent_logger.error(
#             "Claude thinks the error cannot be solved by updating the requirements."
#         )
#         raise


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

    swe_bench = datasets.load_dataset("princeton-nlp/SWE-Bench", split="test")
    swe_bench = swe_bench.filter(lambda x: x["repo"] == "scikit-learn/scikit-learn").sort(
        "created_at"
    )  # type: ignore

    example = swe_bench[100]
    test_spec = make_test_spec(example)  # type: ignore

    # Set up main logger
    logger = setup_logger(debug=args.debug is not None, instance_id=example["instance_id"])
    git_data = clone_and_get_tree(
        repo_name=example["repo"],  # type: ignore
        commit=example["base_commit"],  # type: ignore
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
        requirements_file = os.path.join(
            exp_dir, "requirements_collection", "requirements_output.json"
        )
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
    # Build Docker images
    build_output = build_docker_images(
        requirements_data=time_traveled_requirements,
        git_data=git_data,
        logger=logger,
        build_name="first_build",
    )

    # Check if Docker build was successful
    num_trial = 0
    while build_output.returncode != 0 and num_trial < 3:
        if num_trial > 0:
            logger.info(f"Retry build trial {num_trial} failed")
        else:
            logger.info("First build failed")
        try:
            time_traveled_requirements, build_output = process_retry_build(
                file_contents=file_contents,
                requirements_data=time_traveled_requirements,
                git_data=git_data,
                error_message=build_output.stderr,
                trial_num=num_trial,
                parent_logger=logger,
            )
            if build_output.returncode == 0:
                logger.info(
                    f"After retry build trial {num_trial}, Docker build completed successfully"
                )
        except (RequirementsError, ValueError) as e:
            logger.error(f"Build retry {num_trial} failed: {str(e)}")
            break
        num_trial += 1

    if build_output.returncode != 0:
        logger.error("Docker build failed after 3 retries")
        exit(1)
    else:
        logger.info("Instance Docker build completed successfully")

    # Run the test in a Docker container
    # Create and start container
    container = docker.from_env().containers.run(
        "testbed:latest",
        command="tail -f /dev/null",  # Keep container running
        remove=True,  # Auto-remove when stopped
        detach=True,  # Run in background
    )

    # write to logdir
    eval_script_path = Path(logger.get_logdir()) / "eval.sh"
    with open(eval_script_path, "w") as f:
        f.write(test_spec.eval_script)

    copy_to_container(container, eval_script_path, Path("/eval.sh"))
    exit_code, test_output_log, timed_out, total_runtime = exec_run_with_timeout(
        container, "conda run -n testbed /bin/bash /eval.sh", timeout=600
    )
    logger.info(f"Test output: {test_output_log}")
    with open(Path(logger.get_logdir()) / "test_output.log", "w") as f:
        f.write(test_output_log)

    logger.info(f"Test exit code: {exit_code}")

    # num_test_retry = 0
    # while exit_code != 0 and num_test_retry < 3:
    #     try:
    #         time_traveled_requirements, test_output = process_retry_test(
    #             file_contents=file_contents,
    #             requirements_data=time_traveled_requirements,
    #             git_data=git_data,
    #             error_message=test_output_log,
    #             test_command=test_spec.eval_script,
    #             trial_num=num_test_retry,
    #             parent_logger=logger,
    #         )
    #         if test_output.returncode == 0:
    #             logger.info(f"After retry test trial {num_test_retry}, test completed successfully")
    #     except (RequirementsError, ValueError) as e:
    #         logger.error(f"Test retry {num_test_retry} failed: {str(e)}")
    #         break
    #     num_test_retry += 1
