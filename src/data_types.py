from typing import Dict, List, Optional, TypedDict


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
    repo_url: str


class SWEBenchExample(TypedDict):
    """
    Type definition for SWE-Bench example data structure.

    Attributes:
        instance_id (str): Unique identifier for the example
        repo (str): Repository name in the format 'owner/repo'
        base_commit (str): Base commit hash
        patch (str): Git patch to apply for the fix
        created_at (str, optional): Creation timestamp
        issue_url (str, optional): URL to the GitHub issue
        issue_title (str, optional): Title of the GitHub issue
        issue_body (str, optional): Body content of the GitHub issue
    """

    instance_id: str
    repo: str
    base_commit: str
    patch: str
    created_at: Optional[str]
    issue_url: Optional[str]
    issue_title: Optional[str]
    issue_body: Optional[str]
