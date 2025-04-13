import argparse
import json
import os
import pathlib
import shutil
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import datasets
from data_types import GitRepoData, RequirementsData
from docker_utils import build_docker_images, copy_to_container, exec_run_with_timeout
from exceptions import RequirementsError
from experiment_utils import (
    find_docker_image,
    find_latest_build_folder,
    load_example_data,
    save_example_data,
)
from git_utils import clone_and_get_tree, load_file_contents
from pipeline.build_retry import retry_installation
from pipeline.collect import collect_requirements
from pipeline.localize import localize_requirements
from pipeline.test_retry import retry_test
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore
from version_finder import time_travel_requirements

import docker
from logger import CustomLogger, setup_logger
from model_utils import anthropic_generate_json

# Create a global lock for git operations
git_lock = threading.Lock()


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
        logger_name=f"retry_build_{trial_num}", parent_logger=parent_logger
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


def process_retry_test(
    file_contents: Dict[str, str],
    requirements_data: RequirementsData,
    git_data: GitRepoData,
    error_message: str,
    test_command: str,
    trial_num: int,
    parent_logger: CustomLogger,
) -> subprocess.CompletedProcess:
    """
    Process the retry of a failed test by attempting to fix requirements and rebuild.

    Args:
        file_contents (dict): Dictionary mapping file paths to their contents
        requirements_data (RequirementsData): Current requirements data
        git_data (GitRepoData): Dictionary containing commit_hash and commit_date
        error_message (str): The error message from the failed test
        test_command (str): The test command to run
        trial_num (int): The current trial number
        parent_logger (CustomLogger): Parent logger for tracking operations

    Returns:
        tuple[RequirementsData, subprocess.CompletedProcess]: Updated requirements data and test output
    """
    test_retry_logger = setup_logger(
        logger_name=f"retry_test_{trial_num}", parent_logger=parent_logger
    )

    prompt = retry_test(
        file_contents=file_contents,
        requirements_json=json.dumps(requirements_data, indent=2),
        commit_hash=git_data["commit_hash"],
        commit_date=git_data["commit_date"],
        error_message=error_message,
        test_command=test_command,
    )

    try:
        requirements_data_raw = anthropic_generate_json(
            prompt=prompt,
            logger=test_retry_logger,
            output_filename="retry_test_output.json",
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
        build_output = build_docker_images(
            requirements_data=time_traveled_requirements,
            git_data=git_data,
            logger=test_retry_logger,
            build_name=f"retry_test_{trial_num}",
        )
        if build_output.returncode != 0:
            raise ValueError(
                f"During test retry, Docker build failed with exit code {build_output.returncode}"
            )

        return build_output
    except RequirementsError:
        parent_logger.error(
            "Claude thinks the error cannot be solved by updating the requirements."
        )
        raise


def process_eval_report(
    test_spec: Any,
    example: Any,
    parent_logger: CustomLogger,
    trial_num: int,
) -> subprocess.CompletedProcess:
    """
    Process the evaluation report by running tests and generating a report.

    Args:
        test_spec: The test specification object
        example: Dictionary containing instance data including patch and instance_id
        logger: Logger instance for logging operations
        container: Docker container instance

    Returns:
        Dict containing the evaluation report
    """
    # setup a separate logger for eval report
    eval_report_logger = setup_logger(
        logger_name=f"eval_report_{trial_num}", parent_logger=parent_logger
    )
    instance_id = eval_report_logger.get_instance_id()

    # Run the test in a Docker container
    # Create and start container
    container = docker.from_env().containers.run(
        f"{instance_id}.test:latest",
        command="tail -f /dev/null",  # Keep container running
        remove=True,  # Auto-remove when stopped
        detach=True,  # Run in background
    )

    # get patch diff
    patch_diff_path = Path(eval_report_logger.get_logdir()) / "patch.diff"
    with open(patch_diff_path, "w") as f:
        f.write(example["patch"])  # type: ignore

    copy_to_container(container, patch_diff_path, Path("/patch.diff"))
    exit_code, test_output_log, timed_out, total_runtime = exec_run_with_timeout(
        container, "conda run -n testbed git apply --verbose /patch.diff", timeout=600
    )
    eval_report_logger.info(f"Patch output: {test_output_log}")
    if exit_code != 0:
        raise RuntimeError(f"Patch failed with exit code {exit_code}. This should not be retried.")

    # write to logdir
    eval_script_path = Path(eval_report_logger.get_logdir()) / "eval.sh"
    with open(eval_script_path, "w") as f:
        f.write(test_spec.eval_script)

    copy_to_container(container, eval_script_path, Path("/eval.sh"))
    exit_code, test_output_log, timed_out, total_runtime = exec_run_with_timeout(
        container, "conda run -n testbed /bin/bash /eval.sh", timeout=600
    )
    eval_report_logger.info(f"Test output: {test_output_log}")
    with open(Path(eval_report_logger.get_logdir()) / "test_output.log", "w") as f:
        f.write(test_output_log)

    if exit_code != 0:
        return subprocess.CompletedProcess(
            args=[], returncode=exit_code, stdout=test_output_log, stderr=test_output_log
        )

    report = get_eval_report(
        test_spec=test_spec,
        prediction={
            "instance_id": example["instance_id"],
            "model_patch": example["patch"],  # type: ignore
            "model_name_or_path": "gold",
        },
        test_log_path=f"{eval_report_logger.get_logdir()}/test_output.log",
        include_tests_status=True,
    )
    with open(Path(eval_report_logger.get_logdir()) / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    return subprocess.CompletedProcess(args=[], returncode=0, stdout=None, stderr=None)


def process_single_example(
    example: Dict[str, Any],
    exp_name: str,
    debug: Optional[str] = None,
) -> bool:
    """
    Process a single example from SWE-Bench.

    Args:
        example: The example to process
        exp_name: Name of the experiment
        debug: Optional debug experiment timestamp (deprecated, kept for backward compatibility)

    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        test_spec = make_test_spec(example)  # type: ignore

        # Set up main logger with the example's instance_id directly
        logger = setup_logger(
            debug=False,  # Set debug to False, since we now use debug_path for debugging
            instance_id=example["instance_id"],
            root_dir=exp_name,
        )

        # Save example data to JSON file
        example_path = save_example_data(example, logger.get_logdir())
        logger.info(f"Saved example data to {example_path}")

        # Protect git cloning operation with a lock
        print("Trying to get git lock")
        with git_lock:
            print("Got git lock")
            git_data = clone_and_get_tree(
                repo_name=example["repo"],  # type: ignore
                commit=example["base_commit"],  # type: ignore
                logger=logger,
                instance_id=example["instance_id"],
            )

        if debug:
            # This branch is deprecated and only kept for backward compatibility
            logger.info(f"Debug mode enabled, loading results from specified experiment: {debug}")
            exp_dir = os.path.join(exp_name, debug)
            if not os.path.exists(exp_dir) or not os.path.isdir(exp_dir):
                logger.error(f"Specified experiment directory does not exist: {exp_dir}")
                return False

            logger.info(f"Loading from experiment directory: {exp_dir}")

            # Load localization results
            localization_file = os.path.join(exp_dir, "localization", "localization_output.json")
            if not os.path.exists(localization_file):
                logger.error(f"Localization output file not found: {localization_file}")
                return False

            with open(localization_file, "r") as f:
                file_paths = json.load(f)
                logger.info(f"Loaded localization data: {file_paths}")

            # Load file contents
            with git_lock:
                file_contents = load_file_contents(
                    file_paths=file_paths,
                    base_dir=pathlib.Path("playground") / example["repo"].split("/")[1],
                    logger=logger,
                    git_data=git_data,
                )

            # Load requirements collection results
            requirements_file = os.path.join(
                exp_dir, "requirements_collection", "requirements_output.json"
            )
            if not os.path.exists(requirements_file):
                logger.error(f"Requirements output file not found: {requirements_file}")
                return False

            with open(requirements_file, "r") as f:
                requirements_data = json.load(f)
                logger.info(f"Loaded requirements data: {requirements_data}")
        else:
            # Process localization - now creates its own logger
            file_paths = process_localization(
                result=git_data,
                logger=logger,
            )

            # Load file contents
            with git_lock:
                file_contents = load_file_contents(
                    file_paths=file_paths,
                    base_dir=pathlib.Path("playground") / example["repo"].split("/")[1],
                    logger=logger,
                    git_data=git_data,
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
            return False
        else:
            logger.info("Instance Docker build completed successfully")

        success = False
        for num_eval_trial in range(3):
            eval_report_result = process_eval_report(
                test_spec=test_spec,
                example=example,
                parent_logger=logger,
                trial_num=num_eval_trial,
            )

            if eval_report_result.returncode == 0:
                logger.info(f"After retry eval trial {num_eval_trial}, eval completed successfully")
                # copy the report.json from the eval_report_logger to the main logger
                shutil.copy(
                    Path(f"{logger.get_logdir()}/eval_report_{num_eval_trial}") / "report.json",
                    Path(logger.get_logdir()) / "final_report.json",
                )
                success = True
                break
            else:
                logger.error(f"Eval retry {num_eval_trial} failed: {eval_report_result.stderr}")
                # retry test
                process_retry_test(
                    file_contents=file_contents,
                    requirements_data=time_traveled_requirements,
                    git_data=git_data,
                    error_message=eval_report_result.stderr,
                    test_command=test_spec.eval_script,
                    trial_num=num_eval_trial,
                    parent_logger=logger,
                )

        if not success:
            logger.error("Eval failed after 3 retries")
            return False
        else:
            logger.info("Eval completed successfully")
            return True

    except Exception as e:
        logger.error(f"Error processing example {example['instance_id']}: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        return False


def debug_example(
    debug_path: str,
):
    """
    Debug a single example from SWE-Bench using previously generated artifacts.

    Args:
        debug_path: Path to the debug directory containing previous runs

    Returns:
        bool: True if debugging was successful, False otherwise
    """
    # Load example data from the debug path
    example = load_example_data(debug_path)
    if example is None:
        return False

    print(f"Loaded example data from {os.path.join(debug_path, 'example.json')}")
    instance_id = example["instance_id"]

    # Make test spec from the loaded example
    test_spec = make_test_spec(example)  # type: ignore

    # Set up main logger with the example's instance_id directly
    logger = setup_logger(
        debug=True,
        instance_id=instance_id,
        root_dir=os.path.dirname(debug_path),
    )

    logger.info(f"Debug mode enabled, loading results from: {debug_path}")

    if not os.path.exists(debug_path) or not os.path.isdir(debug_path):
        logger.error(f"Specified debug directory does not exist: {debug_path}")
        return False

    # Find the latest retry build or test folder
    latest_folder = find_latest_build_folder(debug_path)
    if latest_folder is None:
        logger.error(f"No build or test folders found in {debug_path}")
        return False

    latest_folder_name, latest_folder_path = latest_folder
    logger.info(f"Using latest folder: {latest_folder_name} at {latest_folder_path}")

    # Check if Docker image exists
    if not find_docker_image(instance_id, logger):
        return False

    # Run eval report using the latest folder and existing Docker image
    logger.info(f"Running evaluation report using image from {latest_folder_name}")
    process_eval_report(
        test_spec=test_spec,
        example=example,
        parent_logger=logger,
        trial_num=0,  # Use 0 as we're in debug mode
    )


# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SWE-Bench environment builder")
    parser.add_argument(
        "--debug_path",
        type=str,
        help="Path to a specific experiment directory to debug (format: exp_name/YYYYMMDD_HHMMSS/instance_id)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of processes to use for parallel processing",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index in the dataset",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Ending index in the dataset (exclusive)",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="batchA",
        help="Name of the experiment",
    )
    args = parser.parse_args()

    # Force single process mode when debug_path is enabled
    if args.debug_path is not None and args.num_processes > 1:
        print(
            f"Warning: Debug mode enabled. Setting num_processes to 1 (was {args.num_processes})."
        )
        args.num_processes = 1

    if args.debug_path:
        # In debug mode, just run the debug_example function
        success = debug_example(args.debug_path)
        print(f"Debug {'succeeded' if success else 'failed'}")
    else:
        # Normal processing mode - load dataset and process examples
        # Load SWE-Bench dataset
        swe_bench = datasets.load_dataset("princeton-nlp/SWE-Bench", split="test")
        swe_bench = swe_bench.filter(lambda x: x["repo"] == "sympy/sympy").sort(  # type: ignore
            "created_at"
        )

        # Determine the range of examples to process
        end_idx = args.end_index if args.end_index is not None else len(swe_bench)
        examples = [swe_bench[i] for i in range(args.start_index, end_idx)]

        # Create a pool of workers
        with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
            # Process examples in parallel
            results = list(
                executor.map(
                    lambda x: process_single_example(*x),
                    [(example, args.exp_name, None) for example in examples],
                )
            )

        # Print summary of results
        total = len(results)
        successful = sum(1 for r in results if r)
        print("\nProcessing complete!")
        print(f"Total examples processed: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success rate: {(successful/total)*100:.2f}%")
