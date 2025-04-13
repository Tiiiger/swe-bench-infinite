import json
import os
from typing import Any, Dict, Optional, cast

from data_types import SWEBenchExample

import docker
from logger import CustomLogger


def save_example_data(example: Dict[str, Any], log_dir: str) -> str:
    """
    Save SWE-Bench example data to a JSON file.

    Args:
        example: The example to save
        log_dir: Directory to save the example data in

    Returns:
        Path to the saved example file
    """
    example_path = os.path.join(log_dir, "example.json")

    # Extract only the needed fields
    example_data: SWEBenchExample = {
        "instance_id": example["instance_id"],
        "repo": example["repo"],
        "base_commit": example["base_commit"],
        "patch": example.get("patch", ""),
        "created_at": example.get("created_at", None),
        "issue_url": example.get("issue_url", None),
        "issue_title": example.get("issue_title", None),
        "issue_body": example.get("issue_body", None),
    }

    with open(example_path, "w") as f:
        json.dump(example_data, f, indent=2)

    return example_path


def load_example_data(debug_path: str) -> Optional[SWEBenchExample]:
    """
    Load SWE-Bench example data from a JSON file.

    Args:
        debug_path: Path to the directory containing the example.json file

    Returns:
        The loaded example data or None if the file doesn't exist
    """
    example_path = os.path.join(debug_path, "example.json")
    if not os.path.exists(example_path):
        print(f"Example data not found in debug path: {example_path}")
        return None

    with open(example_path, "r") as f:
        example = json.load(f)

    return cast(SWEBenchExample, example)


def find_latest_build_folder(debug_path: str) -> Optional[tuple[str, str]]:
    """
    Find the latest build or test folder in the debug path.

    Args:
        debug_path: Path to the debug directory

    Returns:
        Tuple of (folder_name, folder_path) for the latest folder, or None if no folders found
    """
    build_folders = []

    # Check for first build folder
    first_build_path = os.path.join(debug_path, "first_build")
    if os.path.exists(first_build_path) and os.path.isdir(first_build_path):
        build_folders.append(("first_build", first_build_path))

    # Check for retry build folders
    for i in range(3):  # We try up to 3 retries
        retry_build_path = os.path.join(debug_path, f"retry_build_{i}")
        if os.path.exists(retry_build_path) and os.path.isdir(retry_build_path):
            build_folders.append((f"retry_build_{i}", retry_build_path))

    # Check for retry test folders
    for i in range(3):  # We try up to 3 retries
        retry_test_path = os.path.join(debug_path, f"retry_test_{i}")
        if os.path.exists(retry_test_path) and os.path.isdir(retry_test_path):
            build_folders.append((f"retry_test_{i}", retry_test_path))

    if not build_folders:
        return None

    return build_folders[-1]


def find_docker_image(instance_id: str, logger: CustomLogger) -> bool:
    """
    Find a valid Docker image for the given instance.

    Args:
        instance_id: The instance ID to look for
        logger: Logger instance for logging operations

    Returns:
        True if a valid Docker image was found, False otherwise
    """
    client = docker.from_env()

    # Try different possible image names
    image_names = [f"{instance_id}.test", f"{instance_id}.test:latest", f"{instance_id}"]

    for image_name in image_names:
        try:
            client.images.get(image_name)  # type: ignore
            logger.info(f"Found existing Docker image: {image_name}")
            return True
        except Exception as e:  # type: ignore
            logger.warning(f"Docker image {image_name} not found: {str(e)}")

    logger.error("No valid Docker image found. Cannot proceed with debugging.")
    return False
