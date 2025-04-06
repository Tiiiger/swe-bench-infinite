import logging
import os
import shutil
import subprocess
import time

from data_types import GitRepoData, RequirementsData

import docker


def write_docker_files(
    requirements_data: RequirementsData,
    git_data: GitRepoData,
    logger: logging.Logger,
    build_name: str,
) -> None:
    """
    Write Docker configuration files based on requirements data.

    Args:
        requirements_data (RequirementsData): Dictionary containing requirements data
        git_data (GitRepoData): Dictionary containing commit_hash and other info
        logger (logging.Logger): Logger for tracking operations
        build_name (str): Name used to create a subfolder in the logdir
    """
    # Create directories
    os.makedirs("docker", exist_ok=True)
    log_subdir = os.path.join("logdir", build_name)
    os.makedirs(log_subdir, exist_ok=True)

    # File paths in docker directory
    docker_files = [
        "conda_setup.sh",
        "github_setup.sh",
        "apt_install.sh",
        "pip_install.sh",
        "install_repo.sh",
    ]

    # write to requirements.txt
    with open("docker/conda_setup.sh", "w") as f:
        python_version = requirements_data["python_version"]
        f.write(f"mamba create -n testbed python={python_version}\n")

    # write to github_setup.sh
    with open("docker/github_setup.sh", "w") as f:
        f.write(f"git clone {git_data['repo_url']} testbed\n")
        f.write("cd testbed\n")
        f.write(f"git checkout {git_data['commit_hash']}\n")

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
    with open("docker/install_repo.sh", "w") as f:
        f.write(requirements_data["install_commands"])

    # Copy all docker files to the log directory
    for filename in docker_files:
        src_path = os.path.join("docker", filename)
        dst_path = os.path.join(log_subdir, filename)
        shutil.copy2(src_path, dst_path)

    logger.info(f"Docker configuration files written successfully and copied to {log_subdir}")


def build_docker_images(logger: logging.Logger, build_name: str):
    """
    Build Docker images for the testbed environment using the Docker Python API.

    Args:
        logger (logging.Logger): Logger for tracking operations

    Returns:
        bool: True if Docker build was successful, False otherwise
    """
    # Initialize Docker client
    client = docker.from_env()  # type: ignore[attr-defined]

    # Build base image
    logger.info("Building docker image (base)")
    start_time = time.time()
    try:
        base_image, base_logs = client.images.build(
            path="./docker",
            dockerfile="Dockerfile.base",
            tag="testbed-base:latest",
        )
        base_build_time = time.time() - start_time
        logger.info(f"Base docker image build completed in {base_build_time:.2f} seconds")
        logger.debug(f"Base image ID: {base_image.id}")
    except docker.errors.BuildError as e:  # type: ignore[attr-defined]
        logger.error(f"Error building base docker image: {e}")
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=str(e))
    except Exception as e:
        logger.error(f"Unexpected error building base docker image: {e}")
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=str(e))

    # Build testbed image
    logger.info("Building docker image (testbed)")
    start_time = time.time()
    testbed_logs: list[dict] = []  # Initialize testbed_logs to prevent UnboundLocalError
    try:
        testbed_image, logs = client.images.build(
            path="./docker",
            dockerfile="Dockerfile.env",
            tag="testbed:latest",
        )
        for log in logs:
            testbed_logs.append(log)  # type: ignore[arg-type]
        testbed_build_time = time.time() - start_time
        logger.info(f"Testbed docker image build completed in {testbed_build_time:.2f} seconds")
        logger.debug(f"Testbed image ID: {testbed_image.id}")

        # Save stdout to log directory if build successful
        build_log = f"Build successful\nBase build time: {base_build_time:.2f} seconds\nTestbed build time: {testbed_build_time:.2f} seconds\nBase image ID: {base_image.id}\nTestbed image ID: {testbed_image.id}"

        # Return success
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=build_log, stderr="")
    except docker.errors.BuildError as e:  # type: ignore[attr-defined]
        logger.error(f"Error building testbed docker image: {e}")
        error_message = "\n".join(
            [
                log.get("stream", "")
                for log in testbed_logs
                if "error" in log.get("stream", "").lower()
            ]
        )
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=error_message or str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error building testbed docker image: {e}")
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=str(e))
