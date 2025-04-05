import logging

from data_types import GitRepoData, RequirementsData


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
        f.write("#!/bin/bash\n")
        f.write("mamba activate testbed\n")
        for package, version in requirements_data["pip_packages"].items():
            f.write(f"pip install {package}=={version}\n")

    # If install_commands is provided, write them to a file
    if "install_commands" in requirements_data and requirements_data["install_commands"]:
        with open("docker/install_commands.sh", "w") as f:
            f.write(requirements_data["install_commands"])

    logger.info("Docker configuration files written successfully")
