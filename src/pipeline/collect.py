from typing import Dict

from data_types import GitRepoData, RequirementsData

from logger import CustomLogger, setup_logger
from model_utils import anthropic_generate_json
from pipeline.utils import make_file_contents_str


def collect_requirements(
    file_contents: dict[str, str], commit_hash: str, commit_date: str, repo_name: str
):
    # load prompt from prompts/requirements_collection.txt
    with open("src/prompts/requirements_collection.txt", "r") as f:
        prompt = f.read()
    prompt = prompt.replace("{file_contents_str}", make_file_contents_str(file_contents))
    prompt = prompt.replace("{commit_hash}", commit_hash)
    prompt = prompt.replace("{commit_date}", commit_date)
    prompt = prompt.replace("{repo_name}", repo_name)
    return prompt


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
        logger (CustomLogger): Parent logger for tracking operations

    Returns:
        RequirementsData: Structured requirements data for building the environment

    Raises:
        RequirementsError: If there's an issue parsing the requirements
    """
    # Create specific logger for this function
    req_logger = setup_logger(logger_name="requirements_collection", parent_logger=logger)

    # Collect requirements
    prompt = collect_requirements(
        file_contents=file_contents,
        commit_hash=result["commit_hash"],
        commit_date=result["commit_date"],
        repo_name=result["repo_name"],
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
