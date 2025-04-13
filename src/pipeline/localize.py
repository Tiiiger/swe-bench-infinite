from data_types import GitRepoData

from logger import CustomLogger, setup_logger
from model_utils import anthropic_generate_json


def localize_requirements(tree_result: str, commit_hash: str, commit_date: str, repo_name: str):
    # load prompt from prompts/requirements_localization.txt
    with open("src/prompts/requirements_localization.txt", "r") as f:
        prompt = f.read()
    prompt = prompt.replace("{tree_result}", tree_result)
    prompt = prompt.replace("{commit_hash}", commit_hash)
    prompt = prompt.replace("{commit_date}", commit_date)
    prompt = prompt.replace("{repo_name}", repo_name)
    prompt = prompt.replace("{top_k}", str(10))
    return prompt


def process_localization(result: GitRepoData, logger: CustomLogger) -> list[str]:
    """
    Process the localization of requirements for a given commit.

    Args:
        result (GitRepoData): Dictionary containing tree_output, commit_hash, and commit_date
        logger (CustomLogger): Parent logger for tracking operations

    Returns:
        List[str]: List of file paths to analyze

    Raises:
        RequirementsError: If there's an issue parsing the requirements
    """
    # Create specific logger for this function
    loc_logger = setup_logger(logger_name="localization", parent_logger=logger)

    # Localize requirements
    prompt = localize_requirements(
        tree_result=result["tree_output"],
        commit_hash=result["commit_hash"],
        commit_date=result["commit_date"],
        repo_name=result["repo_name"],
    )
    loc_logger.info(
        f"Localizing requirements for {result['commit_hash']} from {result['commit_date']}"
    )

    # Use the abstracted function to generate JSON from Anthropic API
    file_paths = anthropic_generate_json(
        prompt=prompt, logger=loc_logger, output_filename="localization_output.json"
    )

    return file_paths


if __name__ == "__main__":
    pass
