import os
import re
import json
import subprocess
import logging
import datetime
import pathlib
import docker
from pathlib import Path
from typing import Dict, List, TypedDict
from pipeline.localize import localize_requirements
from anthropic_client import AnthropicClient
from pipeline.collect import collect_requirements
from version_finder import get_version_at_time

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

def setup_logger():
    """Set up and return a logger that writes to both console and a file."""
    # Create timestamp for the log directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create log directory
    log_dir = f"exps/{timestamp}"
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    log_file = f"{log_dir}/clone_and_get_tree.log"
    
    # Create logger
    logger = logging.getLogger("clone_and_get_tree")
    logger.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to {log_file}")
    return logger


def clone_and_get_tree(repo_url, target_dir, date_from, date_to, tree_depth=2, logger=None):
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
        dict: A dictionary containing the tree output and the commit information
    """
    # save current working directory
    current_dir = os.getcwd()
        
    # Move to playground
    logger.info("Changing to playground directory")
    os.chdir("playground")

    if logger is None:
        logger = logging.getLogger("clone_and_get_tree")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    
    # Check if repo already exists
    if not os.path.exists(target_dir):
        # Clone the repo
        logger.info(f"Cloning repository {repo_url} to {target_dir}...")
        subprocess.run(["git", "clone", repo_url, target_dir])
        logger.info(f"Repository cloned successfully to {target_dir}")
    else:
        logger.info(f"Repository already exists at {target_dir}, skipping clone")

    # Move into the repo directory
    logger.debug(f"Changing directory to {target_dir}")
    os.chdir(target_dir)

    # Check out back to main branch
    logger.info(f"Checking out back to main branch")
    subprocess.run(["git", "checkout", "main"], capture_output=True, text=True)

    # Get commit history for the date range and capture output
    logger.info(f"Getting commit history from {date_from} to {date_to}...")
    result = subprocess.run(["git", "log", f"--since='{date_from}'", f"--until='{date_to}'", 
                            "--pretty=format:%H %ad", "--date=short"], 
                           capture_output=True, text=True, check=True)

    # Parse the output into a dictionary
    logger.debug("Parsing commit history...")
    commits = {}
    for line in result.stdout.strip().split('\n'):
        if line:  # skip empty lines
            commit_hash, date = line.split()
            if date not in commits:
                commits[date] = []
            commits[date].append(commit_hash)
            logger.debug(f"Found commit {commit_hash} from {date}")

    # Get the earliest commit in the date range
    sorted_dates = sorted(commits.keys())
    
    if not sorted_dates:
        logger.error(f"No commits found in date range {date_from} to {date_to}")
        return {"error": "No commits found in specified date range"}
        
    earliest_date = sorted_dates[0]
    earliest_commit = commits[earliest_date][-1]  # get the last commit from the earliest date

    logger.info(f"Earliest commit from {earliest_date}: {earliest_commit}")

    # Verify the commit exists
    logger.debug(f"Verifying commit {earliest_commit} exists...")
    verify_result = subprocess.run(["git", "cat-file", "-e", earliest_commit], 
                                 capture_output=True, text=True)
    if verify_result.returncode != 0:
        logger.error(f"Error: Commit {earliest_commit} does not exist!")
        logger.error(f"Error output: {verify_result.stderr}")
        return {"error": f"Commit {earliest_commit} does not exist"}

    # Check out the commit
    logger.info(f"Checking out commit {earliest_commit}...")
    checkout_result = subprocess.run(["git", "checkout", earliest_commit], 
                                   capture_output=True, text=True)
    if checkout_result.returncode != 0:
        logger.error("Error checking out commit:")
        logger.error(checkout_result.stderr)
        return {"error": f"Failed to checkout commit {earliest_commit}"}

    logger.info(f"Successfully checked out commit {earliest_commit}")

    # Get the repo tree
    logger.info(f"Generating tree with depth {tree_depth}...")
    tree_result = subprocess.run(f"tree -L {tree_depth}", shell=True, 
                               capture_output=True, text=True)
    
    logger.debug("Returning results")

    # Move back to the current directory
    os.chdir(current_dir)

    return {
        "commit_hash": earliest_commit,
        "commit_date": earliest_date,
        "tree_output": tree_result.stdout,
        "all_commits": commits
    }

def dump_anthropic_response(response, logger):
    all_blocks = []
    for content_block in response.content:
        if content_block.type == "text":
            all_blocks.append({
                "type": "text",
                "text": content_block.text,
            })
        elif content_block.type == "thinking":
            all_blocks.append({
                "type": "thinking",
                "thinking": content_block.thinking
            })
    return json.dumps(all_blocks, indent=4)

def load_file_contents(file_paths, base_dir, logger=None):
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

def get_head_commit_timestamp(repo_path, logger=None):
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
            ["git", "show", "-s", "--format=%ci", "HEAD"],
            capture_output=True, text=True, check=True
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

def process_localization(result, client, logger):
    """
    Process the localization of requirements for a given commit.
    
    Args:
        result (dict): Dictionary containing tree_output, commit_hash, and commit_date
        client (AnthropicClient): Client for making requests to Anthropic API
        logger (logging.Logger): Logger for tracking operations
        
    Returns:
        List[str]: List of file paths from the response
    """
    # Localize requirements
    prompt = localize_requirements(result["tree_output"], result["commit_hash"], result["commit_date"])
    logger.info(f"Localizing requirements for {result['commit_hash']} from {result['commit_date']}")
    response = client.create_message_with_retry(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    logger.info("Response content:")
    text_block = None
    for content_block in response.content:
        if content_block.type == "text" and content_block.text is not None:
            text_block = content_block.text
    response_dump = dump_anthropic_response(response, logger)
    if text_block is None:
        logger.error("Expected text content block, got %s", content_block.type)
        exit(1)
    logger.info(response_dump)

    # regex grab the ```json block
    logger.info("Extracting JSON block...")
    json_block = re.search(r"```json(.*)```", text_block, re.DOTALL).group(1)
    logger.info(json_block)

    # parse the json block
    logger.info("Parsing JSON block...")
    try:
        json_data: List[str] = json.loads(json_block)
    except json.JSONDecodeError as e:
        logger.error("Error parsing JSON block: %s", e)
        exit(1)
    logger.info(json_data)
    
    return json_data

def write_docker_files(requirements_data: RequirementsData, result, repo_url, logger):
    """
    Write Docker configuration files based on requirements data.
    
    Args:
        requirements_data (RequirementsData): Dictionary containing requirements data
        result (dict): Dictionary containing commit_hash and other info
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
        f.write(f"cd testbed\n")
        f.write(f"git checkout {result['commit_hash']}\n")

    with open("docker/apt_install.sh", "w") as f:
        f.write(f"apt-get update && apt-get install -y {' '.join(requirements_data['apt_packages'])}\n")

    with open("docker/requirements_collection.txt", "w") as f:
        for package, version in requirements_data['pip_packages'].items():
            if version.startswith(">=") or version.startswith("=="):
                version = version[2:]
            elif version == "":
                version = get_version_at_time(package, result["commit_date"])
            f.write(f"{package}=={version}\n")
    
    # If install_commands is provided, write them to a file
    if "install_commands" in requirements_data and requirements_data["install_commands"]:
        with open("docker/install_commands.sh", "w") as f:
            f.write(requirements_data["install_commands"])
    
    logger.info("Docker configuration files written successfully")

def process_requirements_collection(file_contents, result, client, logger, repo_url):
    """
    Process the collection of requirements and build the Docker environment.
    
    Args:
        file_contents (dict): Dictionary mapping file paths to their contents
        result (dict): Dictionary containing commit_hash and commit_date
        client (AnthropicClient): Client for making requests to Anthropic API
        logger (logging.Logger): Logger for tracking operations
        repo_url (str): URL of the git repository
        
    Returns:
        int: Exit code from the Docker build process
    """
    # Collect requirements
    prompt = collect_requirements(file_contents, result["commit_hash"], result["commit_date"])
    logger.info(prompt)

    response = client.create_message_with_retry(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    logger.info("Response content:")
    text_block = None
    for content_block in response.content:
        if content_block.type == "text" and content_block.text is not None:
            text_block = content_block.text
    response_dump = dump_anthropic_response(response, logger)
    if text_block is None:
        logger.error("Expected text content block, got %s", content_block.type)
        exit(1)
    logger.info(response_dump)

    # regex grab the ```json block
    logger.info("Extracting JSON block...")
    json_block = re.search(r"```json(.*)```", text_block, re.DOTALL).group(1)

    # parse the json block
    logger.info("Parsing JSON block...")
    try:
        requirements_data_raw = json.loads(json_block)
        
        # Ensure required fields are present
        requirements_data: RequirementsData = {
            "python_version": requirements_data_raw.get("python_version", "3.8"),
            "apt_packages": requirements_data_raw.get("apt_packages", []),
            "pip_packages": requirements_data_raw.get("pip_packages", {}),
            "install_commands": requirements_data_raw.get("install_commands", "")
        }
        
        logger.info(f"Parsed requirements data: {requirements_data}")
    except json.JSONDecodeError as e:
        logger.error("Error parsing JSON block: %s", e)
        exit(1)

    # Write Docker configuration files
    write_docker_files(requirements_data, result, repo_url, logger)
    return exit_code

# Main execution
if __name__ == "__main__":
    # Set up logger
    logger = setup_logger()

    REPO_URL = "https://github.com/scikit-learn/scikit-learn.git"
    
    # Clone and get tree for scikit-learn from January 2018
    logger.info("Starting clone_and_get_tree operation")
    result = clone_and_get_tree(
        repo_url=REPO_URL,
        target_dir="scikit-learn",
        date_from="2018-01-01",
        date_to="2018-01-31",
        tree_depth=3,
        logger=logger
    )
    
    # Print the tree output
    if "error" in result:
        logger.error(f"Operation failed: {result['error']}")
        exit(1)

    # Initialize Anthropic client
    client = AnthropicClient()

    # Process localization
    json_data = process_localization(result, client, logger)
    
    # Load file contents
    file_contents = load_file_contents(
        file_paths=json_data,
        base_dir=pathlib.Path("playground/scikit-learn"),
        logger=logger
    )

    # Process requirements collection and build Docker
    process_requirements_collection(file_contents, result, client, logger, REPO_URL)

    
    # run subprocess to build docker image
    output = subprocess.run(["docker", "build", "-t", "sweb.infinite.py", "./docker"], capture_output=True, text=True)
    print(output.stdout)
    
    # Get and check the exit code
    exit_code = output.returncode
    if exit_code == 0:
        logger.info("Docker build completed successfully")
        return exit_code

    logger.error(f"Docker build failed with exit code {exit_code}")
    logger.error(f"Error output: {output.stderr}")