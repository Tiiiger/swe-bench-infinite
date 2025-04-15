import json
import subprocess
from typing import Dict, Tuple

from data_types import GitRepoData, RequirementsData
from docker_utils import build_docker_images
from exceptions import RequirementsError
from version_finder import time_travel_requirements

from logger import CustomLogger, setup_logger
from model_utils import anthropic_generate_json
from pipeline.utils import make_file_contents_str


def retry_installation(
    file_contents: dict[str, str],
    requirements_json: str,
    commit_hash: str,
    commit_date: str,
    repo_name: str,
    error_message: str,
):
    # load prompt from prompts/trial_and_error.txt
    with open("src/prompts/build_trial_and_error.txt", "r") as f:
        prompt = f.read()
    with open("src/prompts/common.txt", "r") as f:
        common_prompt = f.read()
    prompt = prompt.replace("{common_instructions}", common_prompt)
    prompt = prompt.replace("{file_contents_str}", make_file_contents_str(file_contents))
    prompt = prompt.replace("{requirements_json}", requirements_json)
    prompt = prompt.replace("{commit_hash}", commit_hash)
    prompt = prompt.replace("{commit_date}", commit_date)
    prompt = prompt.replace("{error_message}", error_message)
    prompt = prompt.replace("{repo_name}", repo_name)
    return prompt


def process_retry_build(
    file_contents: Dict[str, str],
    requirements_data: RequirementsData,
    git_data: GitRepoData,
    error_message: str,
    trial_num: int,
    parent_logger: CustomLogger,
) -> Tuple[RequirementsData, subprocess.CompletedProcess]:
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
        repo_name=git_data["repo_name"],
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


if __name__ == "__main__":
    pass
