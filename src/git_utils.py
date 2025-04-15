import logging
import os
import pathlib
import subprocess
import sys
import unittest
from typing import Dict, List, Optional

from data_types import GitRepoData
from exceptions import GitError
from github import Github

from logger import setup_logger

# These are cached here to avoid github api rate limiting
GITHUB_URLS = {
    "scikit-learn/scikit-learn": "https://github.com/scikit-learn/scikit-learn",
    "sympy/sympy": "https://github.com/sympy/sympy",
    "pytest-dev/pytest": "https://github.com/pytest-dev/pytest",
    "pydata/xarray": "https://github.com/pydata/xarray",
    "psf/requests": "https://github.com/psf/requests",
}


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
    if repo_name in GITHUB_URLS:
        return GITHUB_URLS[repo_name]
    # Initialize the GitHub client with authentication token from environment variable
    github_token = os.environ.get("GITHUB_TOKEN")
    g = Github(github_token)
    # Get the repository
    repo = g.get_repo(repo_name)

    # Return the repository URL
    return repo.html_url


def load_file_contents(
    git_data: GitRepoData,
    file_paths: List[str],
    base_dir: str | pathlib.Path,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """
    Load the contents of files from a list of file paths, using absolute paths.

    Args:
        file_paths (list): List of file paths to load
        base_dir (str): Base directory to load files from
        logger (logging.Logger, optional): Logger for tracking operations

    Returns:
        dict: Dictionary mapping file paths to their contents
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    file_contents = {}
    try:
        # Convert base_dir to absolute path
        base_dir = pathlib.Path(base_dir).resolve()
        logger.info(f"Using base directory: {base_dir}")

        # checkout the commit
        checkout_result = subprocess.run(
            args=["git", "checkout", git_data["commit_hash"]],
            cwd=base_dir,
            capture_output=True,
            text=True,
        )
        if checkout_result.stdout:
            logger.debug(f"Git checkout stdout: {checkout_result.stdout}")
        if checkout_result.stderr:
            logger.debug(f"Git checkout stderr: {checkout_result.stderr}")

        # Load file contents
        for file_path in file_paths:
            # Convert to absolute path
            abs_file_path = base_dir / file_path
            # if file_path is a directory, skip
            if abs_file_path.is_dir():
                logger.info(f"Skipping directory {file_path}")
                continue
            try:
                with open(abs_file_path, "r") as f:
                    file_contents[file_path] = f.read()
                logger.info(f"Loaded content for {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error in load_file_contents: {e}")
        raise

    return file_contents


def generate_tree(directory, max_depth=2, current_depth=0, prefix=""):
    """
    Generate a directory tree representation similar to the Unix 'tree' command.

    Args:
        directory (str or Path): Directory to generate tree for
        max_depth (int): Maximum depth of directories to display
        current_depth (int): Current depth in the recursion
        prefix (str): Prefix for the current line

    Returns:
        str or list: String representation of the tree at depth 0,
                    or list of lines for recursive calls
    """
    directory_path = pathlib.Path(directory)
    contents = list(directory_path.iterdir())
    files = [item for item in contents if item.is_file()]
    subdirs = [item for item in contents if item.is_dir()]

    tree_output = []

    for i, file in enumerate(sorted(files)):
        tree_output.append(
            f"{prefix}├── {file.name}"
            if i < len(files) - 1 or subdirs
            else f"{prefix}└── {file.name}"
        )

    if current_depth < max_depth:
        for i, subdir in enumerate(sorted(subdirs)):
            is_last = i == len(subdirs) - 1
            tree_output.append(f"{prefix}{'└── ' if is_last else '├── '}{subdir.name}")
            if current_depth < max_depth - 1:
                extension = "    " if is_last else "│   "
                tree_output.extend(
                    generate_tree(subdir, max_depth, current_depth + 1, prefix + extension)
                )

    if current_depth == 0:
        return "\n".join([directory_path.name] + tree_output)
    return tree_output


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

    # Create playground directory with absolute path
    playground_dir = pathlib.Path("playground").resolve()
    playground_dir.mkdir(exist_ok=True)
    clone_logger.info(f"Using playground directory: {playground_dir}")

    # get the repo url
    repo_url = get_repo_url(repo_name)
    target_dir = playground_dir / repo_name.split("/")[-1]

    # Check if repo already exists
    if not target_dir.exists():
        # Clone the repo
        clone_logger.info(f"Cloning repository {repo_url} to {target_dir}...")
        clone_result = subprocess.run(
            args=["git", "clone", repo_url, str(target_dir)], capture_output=True, text=True
        )
        if clone_result.stdout:
            clone_logger.debug(f"Git clone stdout: {clone_result.stdout}")
        if clone_result.stderr:
            clone_logger.debug(f"Git clone stderr: {clone_result.stderr}")
        clone_logger.info(f"Repository cloned successfully to {target_dir}")
    else:
        clone_logger.info(f"Repository already exists at {target_dir}, skipping clone")

    # Verify the commit exists
    clone_logger.debug(f"Verifying commit {commit} exists...")
    verify_result = subprocess.run(
        args=["git", "cat-file", "-e", commit], capture_output=True, text=True, cwd=target_dir
    )
    if verify_result.stdout:
        clone_logger.debug(f"Git cat-file stdout: {verify_result.stdout}")
    if verify_result.stderr:
        clone_logger.debug(f"Git cat-file stderr: {verify_result.stderr}")
    if verify_result.returncode != 0:
        clone_logger.error(f"Error: Commit {commit} does not exist!")
        clone_logger.error(f"Error output: {verify_result.stderr}")
        raise GitError(f"Commit {commit} does not exist")

    # Check out the commit
    clone_logger.info(f"Checking out commit {commit}...")
    checkout_result = subprocess.run(
        args=["git", "checkout", commit], capture_output=True, text=True, cwd=target_dir
    )
    if checkout_result.stdout:
        clone_logger.debug(f"Git checkout stdout: {checkout_result.stdout}")
    if checkout_result.stderr:
        clone_logger.debug(f"Git checkout stderr: {checkout_result.stderr}")
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
        cwd=target_dir,
    )
    if timestamp_result.stdout:
        clone_logger.debug(f"Git show stdout: {timestamp_result.stdout}")
    if timestamp_result.stderr:
        clone_logger.debug(f"Git show stderr: {timestamp_result.stderr}")
    commit_date = timestamp_result.stdout.strip()
    clone_logger.info(f"Commit date: {commit_date}")

    # Get the repo tree using the Python implementation
    clone_logger.info(f"Generating tree with depth {tree_depth}...")
    # Note(tianyi): I have experimented with setting the depth but there doesn't seem to be a good default
    # So now I am setting this to infinite depth
    tree_output = generate_tree(target_dir, max_depth=sys.maxsize)
    # Ensure tree_output is always a string to match GitRepoData type
    if isinstance(tree_output, list):
        tree_output = "\n".join(tree_output)
    clone_logger.debug(f"Tree output: {tree_output}")

    clone_logger.debug("Returning results")

    return {
        "commit_hash": commit,
        "commit_date": commit_date,
        "tree_output": tree_output,
        "repo_url": repo_url,
        "repo_name": repo_name,
    }


def get_head_commit_timestamp(
    repo_path: str | pathlib.Path, logger: Optional[logging.Logger] = None
) -> Optional[str]:
    """
    Get the timestamp of the HEAD commit in a git repository.

    Args:
        repo_path (str | pathlib.Path): Path to the git repository
        logger (logging.Logger, optional): Logger for tracking operations

    Returns:
        str: Timestamp of the HEAD commit in ISO format
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # Convert to absolute path
        repo_path = pathlib.Path(repo_path).resolve()
        logger.info(f"Using repository path: {repo_path}")

        # Get timestamp of HEAD commit
        logger.info("Getting timestamp of HEAD commit...")
        result = subprocess.run(
            args=["git", "show", "-s", "--format=%ci", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_path),  # Convert Path to string for cwd
        )
        if result.stdout:
            logger.debug(f"Git show stdout: {result.stdout}")
        if result.stderr:
            logger.debug(f"Git show stderr: {result.stderr}")

        # Parse the timestamp
        timestamp = result.stdout.strip()
        logger.info(f"HEAD commit timestamp: {timestamp}")

        return timestamp
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting HEAD commit timestamp: {e}")
        logger.error(f"Error output: {e.stderr}")
        return None


class TestGitUtils(unittest.TestCase):
    def test_get_repo_url(self):
        # Call the function being tested
        result = get_repo_url("scikit-learn/scikit-learn")

        # Verify the result
        self.assertEqual(result, "https://github.com/scikit-learn/scikit-learn")

    def test_clone_and_get_tree(self):
        # Call the function being tested
        result = clone_and_get_tree(
            "scikit-learn/scikit-learn",
            "HEAD",
            logger=logging.getLogger(__name__),
        )

        # Verify the result
        self.assertEqual(result, "https://github.com/scikit-learn/scikit-learn")


if __name__ == "__main__":
    unittest.main()
