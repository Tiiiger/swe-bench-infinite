import logging
import os
import pathlib
import subprocess
import unittest
from typing import Dict, List, Optional

from data_types import GitRepoData
from exceptions import GitError
from github import Github

from logger import setup_logger


def get_repo_url(repo_name: str) -> str:
    """
    Get the URL of a GitHub repository given its name in the format 'owner/repo'.

    Args:
        repo_name (str): Repository name in the format 'owner/repo' (e.g., 'scikit-learn/scikit-learn')

    Returns:
        str: The repository URL

    Raises:
        Exception: If the repository cannot be found
    """
    # Initialize the GitHub client with authentication token from environment variable
    github_token = os.environ.get("GITHUB_TOKEN")
    g = Github(github_token)
    # Get the repository
    repo = g.get_repo(repo_name)

    # Return the repository URL
    return repo.html_url


# Example usage
if __name__ == "__main__":
    repo_name = "scikit-learn/scikit-learn"
    try:
        repo_url = get_repo_url(repo_name)
        print(f"Repository URL: {repo_url}")
    except Exception as e:
        print(e)


def load_file_contents(
    file_paths: List[str], base_dir: str | pathlib.Path, logger: Optional[logging.Logger] = None
) -> Dict[str, str]:
    """
    Load the contents of files from a list of file paths, changing to a base directory first.

    Args:
        file_paths (list): List of file paths to load
        base_dir (str): Base directory to change to before loading files
        logger (logging.Logger, optional): Logger for tracking operations

    Returns:
        dict: Dictionary mapping file paths to their contents
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Save current working directory
    current_dir = os.getcwd()

    file_contents = {}
    try:
        # Change to base directory
        os.chdir(base_dir)
        logger.info(f"Changed directory to {base_dir}")

        # Load file contents
        for file_path in file_paths:
            try:
                with open(file_path, "r") as f:
                    file_contents[file_path] = f.read()
                logger.info(f"Loaded content for {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    finally:
        # Restore original directory
        os.chdir(current_dir)
        logger.info(f"Restored directory to {current_dir}")

    return file_contents


def clone_and_get_tree(
    repo_name: str,
    commit: str,
    logger: logging.Logger,
    tree_depth: int = 2,
) -> GitRepoData:
    """
    Clone a git repository if it doesn't exist, checkout a specific commit,
    and return the directory tree.

    Args:
        repo_name (str): Name of the git repository to clone
        commit (str): The commit hash to checkout
        logger (logging.Logger): Logger for tracking operations
        tree_depth (int): Depth level for the tree command

    Returns:
        GitRepoData: A dictionary containing the tree output and the commit information

    Raises:
        GitError: If any git operation fails
    """
    # Extract timestamp from parent logger
    clone_logger = setup_logger(logger_name="clone_and_get_tree", parent_logger=logger)

    # save current working directory
    current_dir = os.getcwd()

    # Move to playground
    os.makedirs("playground", exist_ok=True)
    clone_logger.info("Changing to playground directory")
    os.chdir("playground")

    # get the repo url
    repo_url = get_repo_url(repo_name)
    target_dir = repo_name.split("/")[-1]

    # Check if repo already exists
    if not os.path.exists(target_dir):
        # Clone the repo
        clone_logger.info(f"Cloning repository {repo_url} to {target_dir}...")
        subprocess.run(args=["git", "clone", repo_url, target_dir])
        clone_logger.info(f"Repository cloned successfully to {target_dir}")
    else:
        clone_logger.info(f"Repository already exists at {target_dir}, skipping clone")

    # Move into the repo directory
    clone_logger.debug(f"Changing directory to {target_dir}")
    os.chdir(target_dir)

    # Verify the commit exists
    clone_logger.debug(f"Verifying commit {commit} exists...")
    verify_result = subprocess.run(
        args=["git", "cat-file", "-e", commit], capture_output=True, text=True
    )
    if verify_result.returncode != 0:
        clone_logger.error(f"Error: Commit {commit} does not exist!")
        clone_logger.error(f"Error output: {verify_result.stderr}")
        raise GitError(f"Commit {commit} does not exist")

    # Check out the commit
    clone_logger.info(f"Checking out commit {commit}...")
    checkout_result = subprocess.run(
        args=["git", "checkout", commit], capture_output=True, text=True
    )
    if checkout_result.returncode != 0:
        clone_logger.error("Error checking out commit:")
        clone_logger.error(checkout_result.stderr)
        raise GitError(f"Failed to checkout commit {commit}: {checkout_result.stderr}")

    clone_logger.info(f"Successfully checked out commit {commit}")

    # Get the commit timestamp
    clone_logger.info("Getting commit timestamp...")
    timestamp_result = subprocess.run(
        args=["git", "show", "-s", "--format=%ad", "--date=short", commit],
        capture_output=True,
        text=True,
        check=True,
    )
    commit_date = timestamp_result.stdout.strip()
    clone_logger.info(f"Commit date: {commit_date}")

    # Get the repo tree
    clone_logger.info(f"Generating tree with depth {tree_depth}...")
    tree_result = subprocess.run(
        args=f"tree -L {tree_depth}", shell=True, capture_output=True, text=True
    )

    clone_logger.debug("Returning results")

    # Move back to the current directory
    os.chdir(current_dir)

    return {
        "commit_hash": commit,
        "commit_date": commit_date,
        "tree_output": tree_result.stdout,
        "repo_url": repo_url,
    }


def get_head_commit_timestamp(
    repo_path: str, logger: Optional[logging.Logger] = None
) -> Optional[str]:
    """
    Get the timestamp of the HEAD commit in a git repository.

    Args:
        repo_path (str): Path to the git repository
        logger (logging.Logger, optional): Logger for tracking operations

    Returns:
        str: Timestamp of the HEAD commit in ISO format
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Save current working directory
    current_dir = os.getcwd()

    try:
        # Change to repository directory
        os.chdir(repo_path)
        logger.info(f"Changed directory to {repo_path}")

        # Get timestamp of HEAD commit
        logger.info("Getting timestamp of HEAD commit...")
        result = subprocess.run(
            args=["git", "show", "-s", "--format=%ci", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse the timestamp
        timestamp = result.stdout.strip()
        logger.info(f"HEAD commit timestamp: {timestamp}")

        return timestamp
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting HEAD commit timestamp: {e}")
        logger.error(f"Error output: {e.stderr}")
        return None
    finally:
        # Restore original directory
        os.chdir(current_dir)
        logger.debug(f"Restored directory to {current_dir}")


class TestGitUtils(unittest.TestCase):
    def test_get_repo_url(self):
        # Call the function being tested
        result = get_repo_url("scikit-learn/scikit-learn")

        # Verify the result
        self.assertEqual(result, "https://github.com/scikit-learn/scikit-learn")

    def test_clone_and_get_tree(self):
        # Call the function being tested
        result = clone_and_get_tree("scikit-learn/scikit-learn", "scikit-learn", "HEAD")

        # Verify the result
        self.assertEqual(result, "https://github.com/scikit-learn/scikit-learn")


if __name__ == "__main__":
    unittest.main()
