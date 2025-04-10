import hashlib
import logging
import os
import shutil
import subprocess
import tarfile
import threading
import time
from pathlib import Path

from data_types import GitRepoData, RequirementsData

import docker
from docker.models.containers import Container
from logger import setup_logger


def write_docker_files(
    requirements_data: RequirementsData,
    git_data: GitRepoData,
    logger,
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
    log_subdir = logger.get_logdir()

    # File paths in docker directory
    docker_files = [
        "conda_setup.sh",
        "github_setup.sh",
        "apt_install.sh",
        "pip_install.sh",
        "install_repo.sh",
    ]

    # write to log_subdir first
    log_paths = {}

    # write conda_setup.sh to logdir
    log_conda_path = os.path.join(log_subdir, "conda_setup.sh")
    with open(log_conda_path, "w") as f:
        python_version = requirements_data["python_version"]
        version_str = "".join(python_version.split(".")).replace(".", "")
        f.write(f"conda rename -n py{version_str} testbed\n")
    log_paths["conda_setup.sh"] = log_conda_path

    # write github_setup.sh to logdir
    log_github_path = os.path.join(log_subdir, "github_setup.sh")
    with open(log_github_path, "w") as f:
        f.write(f"git clone {git_data['repo_url']} testbed\n")
        f.write("cd testbed\n")
        f.write(f"git checkout {git_data['commit_hash']}\n")
    log_paths["github_setup.sh"] = log_github_path

    # write apt_install.sh to logdir
    log_apt_path = os.path.join(log_subdir, "apt_install.sh")
    with open(log_apt_path, "w") as f:
        f.write(
            f"apt-get update && apt-get install -y {' '.join(requirements_data['apt_packages'])}\n"
        )
    log_paths["apt_install.sh"] = log_apt_path

    # write pip_install.sh to logdir
    log_pip_path = os.path.join(log_subdir, "pip_install.sh")
    with open(log_pip_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("mamba activate testbed\n")
        for package, version in requirements_data["pip_packages"].items():
            f.write(f"pip install {package}=={version}\n")
    log_paths["pip_install.sh"] = log_pip_path

    # write install_repo.sh to logdir
    log_install_path = os.path.join(log_subdir, "install_repo.sh")
    with open(log_install_path, "w") as f:
        f.write(requirements_data["install_commands"])
    log_paths["install_repo.sh"] = log_install_path

    # Copy from logdir to docker folder only if content changed
    # NOTE: this is because Docker also checks the timestamp of the file
    # so if the content hasn't changed, we don't update these files to retrigger build
    for filename in docker_files:
        log_path = log_paths[filename]
        docker_path = os.path.join("docker", filename)

        # Compute hash of the new file in logdir
        with open(log_path, "rb") as f:
            new_content_hash = hashlib.md5(f.read()).hexdigest()

        # Check if docker file exists and compute its hash
        copy_needed = True
        if os.path.exists(docker_path):
            with open(docker_path, "rb") as f:
                existing_content_hash = hashlib.md5(f.read()).hexdigest()
            if existing_content_hash == new_content_hash:
                copy_needed = False

        # Copy only if content has changed
        if copy_needed:
            shutil.copy2(log_path, docker_path)
            logger.info(f"Updated {filename} in docker folder with new content")
        else:
            logger.info(f"Skipped copying {filename} to docker folder (content unchanged)")

    logger.info(f"Docker configuration files written successfully to {log_subdir}")


def custom_build_docker_images(path: str, dockerfile: str, tag: str) -> subprocess.CompletedProcess:
    """
    Build Docker images for the testbed environment using the Docker Python API.

    Args:
        logger (logging.Logger): Logger for tracking operations
    """
    import re

    client = docker.from_env()  # type: ignore[attr-defined]
    # copy from docker.models.images.ImageCollection.build
    resp = client.api.build(path=path, dockerfile=dockerfile, tag=tag, decode=True)
    if isinstance(resp, str):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=resp, stderr="")
    image_id = None
    full_log = ""
    has_error = False
    for chunk in resp:
        if "error" in chunk:
            full_log += "[ERROR] " + chunk["error"].strip() + "\n"
            has_error = True
        elif "stream" in chunk:
            full_log += chunk["stream"].strip() + "\n"
            match = re.search(r"(^Successfully built |sha256:)([0-9a-f]+)$", chunk["stream"])
            if match:
                image_id = match.group(2)
        elif "status" in chunk:
            full_log += "[STATUS] " + chunk["status"].strip() + "\n"
        elif "progress" in chunk:
            full_log += "[PROGRESS] " + chunk["progress"].strip() + "\n"
        elif "progressDetail" in chunk:
            full_log += "[PROGRESS DETAIL] " + chunk["progressDetail"].strip() + "\n"
        elif "aux" in chunk:
            full_log += "[AUX] " + str(chunk["aux"]) + "\n"
        else:
            full_log += "[UNKNOWN] " + str(chunk).strip() + "\n"
    if image_id and not has_error:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=full_log, stderr="")
    else:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=full_log)


def build_docker_images(
    requirements_data: RequirementsData,
    git_data: GitRepoData,
    logger: logging.Logger,
    build_name: str,
):
    """
    Build Docker images for the testbed environment using the Docker Python API.

    Args:
        logger (logging.Logger): Logger for tracking operations

    Returns:
        bool: True if Docker build was successful, False otherwise
    """
    # Initialize Docker client
    docker.from_env()  # type: ignore[attr-defined]

    # add a child logger for docker build logs
    docker_build_logger = setup_logger(f"docker_{build_name}", parent_logger=logger)

    # write docker files
    write_docker_files(requirements_data, git_data, docker_build_logger)

    # Build base image
    docker_build_logger.info("Building docker image (base)")
    start_time = time.time()
    base_result = custom_build_docker_images(
        path="./docker",
        dockerfile="Dockerfile.base",
        tag="testbed-base:latest",
    )
    if base_result.returncode != 0:
        docker_build_logger.error("Failed to retrieve or build base docker image")
        docker_build_logger.error(base_result.stderr)
        raise RuntimeError("Failed to retrieve or build base docker image")
    else:
        docker_build_logger.info(base_result.stdout)

    base_build_time = time.time() - start_time
    docker_build_logger.info(f"Base docker image build completed in {base_build_time:.2f} seconds")

    # Build testbed image
    docker_build_logger.info("Building docker image (testbed)")
    start_time = time.time()
    testbed_result = custom_build_docker_images(
        path="./docker",
        dockerfile="Dockerfile.env",
        tag="testbed:latest",
    )
    testbed_build_time = time.time() - start_time
    docker_build_logger.info(
        f"Testbed docker image build completed in {testbed_build_time:.2f} seconds"
    )
    if testbed_result.returncode != 0:
        docker_build_logger.error("Failed to retrieve or build testbed docker image")
        docker_build_logger.error(testbed_result.stderr)
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=testbed_result.stderr
        )
    else:
        docker_build_logger.info(testbed_result.stdout)

    # Save stdout to log directory if build successful
    build_log = f"Build successful\nBase build time: {base_build_time:.2f} seconds\nTestbed build time: {testbed_build_time:.2f} seconds"

    # Return success
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=build_log, stderr="")


def exec_run_with_timeout(
    container: Container, cmd: str, timeout: int | None = 60
) -> tuple[int, str, bool, float]:
    """
    Run a command in a container with a timeout.

    Args:
        container (docker.Container): Container to run the command in.
        cmd (str): Command to run.
        timeout (int): Timeout in seconds.
    """
    # Local variables to store the result of executing the command
    exec_result = b""
    exec_id = None
    exit_code = 1
    exception = None
    timed_out = False

    # Wrapper function to run the command
    def run_command():
        nonlocal exec_result, exec_id, exception, exit_code
        try:
            exec_id = container.client.api.exec_create(container.id, cmd)["Id"]  # type: ignore
            exec_stream = container.client.api.exec_start(exec_id, stream=True)  # type: ignore
            for chunk in exec_stream:
                exec_result += chunk
            # finish the command
            exit_code = 0
        except Exception as e:
            exception = e

    # Start the command in a separate thread
    thread = threading.Thread(target=run_command)
    start_time = time.time()
    thread.start()
    thread.join(timeout)

    if exception:
        raise exception  # type: ignore

    # If the thread is still alive, the command timed out
    if thread.is_alive():
        if exec_id is not None:
            exec_pid = container.client.api.exec_inspect(exec_id)["Pid"]  # type: ignore
            container.exec_run(f"kill -TERM {exec_pid}", detach=True)
        timed_out = True
    end_time = time.time()
    return exit_code, exec_result.decode(), timed_out, end_time - start_time


def copy_to_container(container: Container, src: Path, dst: Path):
    """
    Copy a file from local to a docker container

    Args:
        container (Container): Docker container to copy to
        src (Path): Source file path
        dst (Path): Destination file path in the container
    """
    # Check if destination path is valid
    if os.path.dirname(dst) == "":
        raise ValueError(f"Destination path parent directory cannot be empty!, dst: {dst}")

    # temporary tar file
    tar_path = src.with_suffix(".tar")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(
            src, arcname=dst.name
        )  # use destination name, so after `put_archive`, name is correct

    # get bytes for put_archive cmd
    with open(tar_path, "rb") as tar_file:
        data = tar_file.read()

    # Make directory if necessary
    container.exec_run(f"mkdir -p {dst.parent}")

    # Send tar file to container and extract
    container.put_archive(os.path.dirname(dst), data)

    # clean up in locally and in container
    tar_path.unlink()
