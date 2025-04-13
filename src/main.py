import argparse
import json
import os
import pathlib
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

from data import get_even_sample_dataset
from docker_utils import build_docker_images
from exceptions import RequirementsError
from experiment_utils import (
    find_docker_image,
    load_example_data,
    save_example_data,
)
from git_utils import clone_and_get_tree, load_file_contents
from pipeline import (
    process_eval_report,
    process_localization,
    process_requirements_collection,
    process_retry_build,
    process_retry_test,
)
from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore
from version_finder import time_travel_requirements

from logger import setup_logger

# Create a global lock for git operations
git_lock = threading.Lock()


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
        with git_lock:
            git_data = clone_and_get_tree(
                repo_name=example["repo"],  # type: ignore
                commit=example["base_commit"],  # type: ignore
                logger=logger,
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
            # Process localization - now loaded from pipeline module
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

            # Process requirements collection - now loaded from pipeline module
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
            eval_run_result, report = process_eval_report(
                test_spec=test_spec,
                example=example,
                parent_logger=logger,
                trial_num=num_eval_trial,
            )

            # Note(tianyi)
            # return code != 0 means
            # 1. pytest has failed -> there are environment issues
            # 2. pytest has failed -> there are tests that are not passing
            #
            # resolved == True means
            # 1. all tests we expect to pass, have passed
            #
            # we only need to retry if there are environment issues
            if eval_run_result.returncode == 0 or list(report.values())[0]["resolved"]:
                logger.info(f"After retry eval trial {num_eval_trial}, eval completed successfully")
                # copy the report.json from the eval_report_logger to the main logger
                shutil.copy(
                    Path(f"{logger.get_logdir()}/eval_report_{num_eval_trial}") / "report.json",
                    Path(logger.get_logdir()) / "final_report.json",
                )
                success = True
                break
            else:
                logger.error(f"Eval retry {num_eval_trial} failed: {eval_run_result.stderr}")
                # retry test
                process_retry_test(
                    file_contents=file_contents,
                    requirements_data=time_traveled_requirements,
                    git_data=git_data,
                    error_message=eval_run_result.stderr,
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

    # Check if Docker image exists
    if not find_docker_image(instance_id, logger):
        raise ValueError(f"Docker image for {instance_id} not found")

    # Run eval report using the latest folder and existing Docker image
    output = process_eval_report(
        test_spec=test_spec,
        example=example,
        parent_logger=logger,
        trial_num=0,  # Use 0 as we're in debug mode
    )

    return output


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
        swe_bench = get_even_sample_dataset(verbose=False)
        swe_bench = list(filter(lambda x: x["repo"] == "pylint-dev/pylint", swe_bench))

        # Determine the range of examples to process
        end_idx = (
            min(args.end_index, len(swe_bench)) if args.end_index is not None else len(swe_bench)
        )
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
