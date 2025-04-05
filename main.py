import argparse
import json
import logging
import os
import pathlib
import subprocess
import time
from typing import Dict, List, Optional, TypedDict

from anthropic_client import AnthropicClient
from exceptions import (
    GitError,
)
from logger import setup_child_logger, setup_logger
from model_utils import anthropic_generate_json
from pipeline.collect import collect_requirements
from pipeline.localize import localize_requirements
from pipeline.retry import retry_installation
from version_finder import check_and_replace_version, get_version_at_time


class RequirementsData(TypedDict):
    """
    Type definition for requirements data structure.

    Attributes:
        python_version (str): The required Python version (e.g., "3.8")
        apt_packages (List[str]): List of packages to install via apt-get
        pip_packages (Dict[str, str]): Dictionary of pip package names to versions
        install_commands (str): Command to install dev environment after packages
    """

    python_version: str
    apt_packages: List[str]
    pip_packages: Dict[str, str]
    install_commands: str


class GitRepoData(TypedDict):
    """
    Type definition for git repository data structure.

    Attributes:
        commit_hash (str): The commit hash that was checked out
        commit_date (str): The date of the commit in YYYY-MM-DD format
        tree_output (str): The output of the tree command showing the repo structure
    """

    commit_hash: str
    commit_date: str
    tree_output: str


def clone_and_get_tree(
    repo_url: str,
    target_dir: str,
    date_from: str,
    date_to: str,
    tree_depth: int = 2,
    logger: Optional[logging.Logger] = None,
) -> GitRepoData:
    """
    Clone a git repository if it doesn't exist, checkout a commit from a specific date range,
    and return the directory tree.

    Args:
        repo_url (str): URL of the git repository to clone
        target_dir (str): Directory name for the cloned repository
        date_from (str): Start date in YYYY-MM-DD format
        date_to (str): End date in YYYY-MM-DD format
        tree_depth (int): Depth level for the tree command
        logger (logging.Logger): Logger for tracking operations

    Returns:
        GitRepoData: A dictionary containing the tree output and the commit information

    Raises:
        GitError: If any git operation fails
    """
    # Create a specific logger for clone_and_get_tree if a parent logger is provided
    if logger:
        # Extract timestamp from parent logger
        clone_logger = setup_child_logger(logger_name="clone_and_get_tree", parent_logger=logger)
    else:
        # Create a standalone logger
        clone_logger = logging.getLogger("clone_and_get_tree")
        clone_logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        clone_logger.addHandler(console_handler)

    # save current working directory
    current_dir = os.getcwd()

    # Move to playground
    clone_logger.info("Changing to playground directory")
    os.chdir("playground")

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

    # Check out back to main branch
    clone_logger.info("Checking out back to main branch")
    subprocess.run(args=["git", "checkout", "main"], capture_output=True, text=True)

    # Get commit history for the date range and capture output
    clone_logger.info(f"Getting commit history from {date_from} to {date_to}...")
    result = subprocess.run(
        args=[
            "git",
            "log",
            f"--since='{date_from}'",
            f"--until='{date_to}'",
            "--pretty=format:%H %ad",
            "--date=short",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    # Parse the output into a dictionary
    clone_logger.debug("Parsing commit history...")
    commits: Dict[str, List[str]] = {}
    for line in result.stdout.strip().split("\n"):
        if line:  # skip empty lines
            commit_hash, date = line.split()
            if date not in commits:
                commits[date] = []
            commits[date].append(commit_hash)
            clone_logger.debug(f"Found commit {commit_hash} from {date}")

    # Get the earliest commit in the date range
    sorted_dates = sorted(commits.keys())

    if not sorted_dates:
        clone_logger.error(f"No commits found in date range {date_from} to {date_to}")
        raise GitError(f"No commits found in date range {date_from} to {date_to}")

    earliest_date = sorted_dates[0]
    earliest_commit = commits[earliest_date][-1]  # get the last commit from the earliest date

    clone_logger.info(f"Earliest commit from {earliest_date}: {earliest_commit}")

    # Verify the commit exists
    clone_logger.debug(f"Verifying commit {earliest_commit} exists...")
    verify_result = subprocess.run(
        args=["git", "cat-file", "-e", earliest_commit], capture_output=True, text=True
    )
    if verify_result.returncode != 0:
        clone_logger.error(f"Error: Commit {earliest_commit} does not exist!")
        clone_logger.error(f"Error output: {verify_result.stderr}")
        raise GitError(f"Commit {earliest_commit} does not exist")

    # Check out the commit
    clone_logger.info(f"Checking out commit {earliest_commit}...")
    checkout_result = subprocess.run(
        args=["git", "checkout", earliest_commit], capture_output=True, text=True
    )
    if checkout_result.returncode != 0:
        clone_logger.error("Error checking out commit:")
        clone_logger.error(checkout_result.stderr)
        raise GitError(f"Failed to checkout commit {earliest_commit}: {checkout_result.stderr}")

    clone_logger.info(f"Successfully checked out commit {earliest_commit}")

    # Get the repo tree
    clone_logger.info(f"Generating tree with depth {tree_depth}...")
    tree_result = subprocess.run(
        args=f"tree -L {tree_depth}", shell=True, capture_output=True, text=True
    )

    clone_logger.debug("Returning results")

    # Move back to the current directory
    os.chdir(current_dir)

    return {
        "commit_hash": earliest_commit,
        "commit_date": earliest_date,
        "tree_output": tree_result.stdout,
    }


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


def process_localization(
    result: GitRepoData, client: AnthropicClient, logger: Optional[logging.Logger] = None
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
        prompt=prompt, client=client, logger=loc_logger, output_filename="localization_output.json"
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


def write_docker_files(
    requirements_data: RequirementsData, result: GitRepoData, repo_url: str, logger: logging.Logger
) -> None:
    """
    Write Docker configuration files based on requirements data.

    Args:
        requirements_data (RequirementsData): Dictionary containing requirements data
        result (GitRepoData): Dictionary containing commit_hash and other info
        repo_url (str): URL of the git repository
        logger (logging.Logger): Logger for tracking operations
    """
    # write to requirements.txt
    with open("docker/conda_setup.sh", "w") as f:
        python_version = requirements_data["python_version"]
        f.write(f"mamba create -n testbed python={python_version}\n")

    # write to github_setup.sh
    with open("docker/github_setup.sh", "w") as f:
        f.write(f"git clone {repo_url} testbed\n")
        f.write("cd testbed\n")
        f.write(f"git checkout {result['commit_hash']}\n")

    with open("docker/apt_install.sh", "w") as f:
        f.write(
            f"apt-get update && apt-get install -y {' '.join(requirements_data['apt_packages'])}\n"
        )

    with open("docker/pip_install.sh", "w") as f:
        for package, version in requirements_data["pip_packages"].items():
            f.write(f"pip install {package}=={version}\n")

    # If install_commands is provided, write them to a file
    if "install_commands" in requirements_data and requirements_data["install_commands"]:
        with open("docker/install_commands.sh", "w") as f:
            f.write(requirements_data["install_commands"])

    logger.info("Docker configuration files written successfully")


def process_requirements_collection(
    file_contents: Dict[str, str],
    result: GitRepoData,
    client: AnthropicClient,
    logger: Optional[logging.Logger] = None,
    repo_url: Optional[str] = None,
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
        prompt=prompt, client=client, logger=req_logger, output_filename="requirements_output.json"
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
    result: GitRepoData,
    error_message: str,
    client: AnthropicClient,
    logger: Optional[logging.Logger] = None,
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
        commit_hash=result["commit_hash"],
        commit_date=result["commit_date"],
        error_message=error_message,
    )
    trial_logger.info(prompt)

    # Use the abstracted function to generate JSON from Anthropic API
    requirements_data_raw = anthropic_generate_json(
        prompt=prompt,
        client=client,
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
    Build Docker images for the testbed environment.

    Args:
        time_traveled_requirements (RequirementsData): Requirements with appropriate versions
        file_contents (Dict[str, str]): Dictionary mapping file paths to their contents
        result (GitRepoData): Dictionary containing commit_hash and other info
        client (AnthropicClient): Client for making requests to Anthropic API
        logger (logging.Logger, optional): Logger for tracking operations

    Returns:
        bool: True if Docker build was successful, False otherwise
    """
    # Run subprocess to build docker image
    logger.info("Building docker image (base)")
    start_time = time.time()
    subprocess.run(
        args=[
            "docker",
            "build",
            "-t",
            "testbed-base:latest",
            "-f",
            "./docker/Dockerfile.base",
            "./docker",
        ],
        capture_output=True,
        text=True,
    )
    base_build_time = time.time() - start_time
    logger.info(f"Base docker image build completed in {base_build_time:.2f} seconds")

    logger.info("Building docker image (testbed)")
    start_time = time.time()
    output = subprocess.run(
        args=["docker", "build", "-t", "testbed:latest", "-f", "./docker/Dockerfile", "./docker"],
        capture_output=True,
        text=True,
    )
    testbed_build_time = time.time() - start_time
    logger.info(f"Testbed docker image build completed in {testbed_build_time:.2f} seconds")

    return output


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

    REPO_URL = "https://github.com/scikit-learn/scikit-learn.git"

    # Clone and get tree for scikit-learn from January 2018
    logger.info("Starting clone_and_get_tree operation")
    result = clone_and_get_tree(
        repo_url=REPO_URL,
        target_dir="scikit-learn",
        date_from="2019-01-01",
        date_to="2019-01-31",
        tree_depth=3,
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
        file_paths = process_localization(result=result, client=client, logger=logger)

        # Load file contents
        file_contents = load_file_contents(
            file_paths=file_paths, base_dir=pathlib.Path("playground/scikit-learn"), logger=logger
        )

        # Process requirements collection - now creates its own logger
        requirements_data = process_requirements_collection(
            file_contents=file_contents,
            result=result,
            client=client,
            logger=logger,
            repo_url=REPO_URL,
        )

    # Write Docker configuration files
    time_traveled_requirements = time_travel_requirements(
        requirements_data=requirements_data, commit_date=result["commit_date"], logger=logger
    )
    write_docker_files(
        requirements_data=time_traveled_requirements,
        result=result,
        repo_url=REPO_URL,
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
            result=result,
            error_message=error_message,
            client=client,
            logger=logger,
        )
        write_docker_files(
            requirements_data=time_traveled_requirements,
            result=result,
            repo_url=REPO_URL,
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

    # after the docker image is built, we run pip freeze and get all the installed packages
    # we need to first activate the testbed environment
    output = subprocess.run(
        args=["docker", "exec", "testbed", "pip", "freeze"],
        capture_output=True,
        text=True,
    )
    logger.info(output.stdout)
    # process the requirements being spit out by pip freeze
    pip_freeze_requirements: dict[str, str] = dict()
    for line in output.stdout.split("\n"):
        if "==" not in line:
            continue
        package, version = line.split("==")
        pip_freeze_requirements[package] = check_and_replace_version(
            package, version, result["commit_date"]
        )

    # update the requirements data
    time_traveled_requirements["pip_packages"] = pip_freeze_requirements
    breakpoint()

    # write the docker files again
    write_docker_files(
        requirements_data=time_traveled_requirements,
        result=result,
        repo_url=REPO_URL,
        logger=logger,
    )

    # build the docker image again
    logger.info("Building docker image (testbed) with time traveled pip freeze requirements")
    output = build_docker_images(logger=logger)
    if output.returncode != 0:
        logger.error(f"Docker build failed with exit code {output.returncode}")
        logger.error(f"Error output: {output.stderr}")
        logger.error("Failed to build Docker images")
        exit(1)
    else:
        logger.info(output.stdout)
        logger.info("Docker image built successfully")
