# https://pypi.org/project/numpy/#history
# https://pypi.org/project/pandas/#history
# https://pypi.org/project/scikit-learn/#history
import time
import unittest
from collections import OrderedDict
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from data_types import RequirementsData

# Import Selenium components
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from logger import CustomLogger


def get_versions_with_requests(package_name):
    """Get package version history using the requests library."""
    url = f"https://pypi.org/project/{package_name}/#history"
    response = requests.get(url)

    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    version_to_date = OrderedDict()

    release_cards = soup.find_all("a", class_="release__card")

    for card in release_cards:
        try:
            version_element = card.find("p", class_="release__version")
            version_number = version_element.text.strip().split("\n")[0]

            version_date = card.find("p", class_="release__version-date").find("time")
            datetime_attr = version_date.get("datetime")

            if datetime_attr:
                timestamp = datetime.fromisoformat(datetime_attr.replace("Z", "+00:00")).timestamp()
                version_to_date[version_number] = timestamp
            else:
                print(f"No date found for {version_number}")
        except Exception as e:
            print(f"Error processing version: {e}")

    # Make sure the dates are sorted
    version_to_date = OrderedDict(sorted(version_to_date.items(), key=lambda x: x[1]))

    return version_to_date


def get_versions_with_selenium(package_name):
    """Get package version history using Selenium."""
    # Set up headless Firefox browser
    firefox_options = Options()
    firefox_options.add_argument("--headless")

    # Initialize the Firefox driver
    driver = webdriver.Firefox(options=firefox_options)

    try:
        # Navigate to the package history page
        url = f"https://pypi.org/project/{package_name}/#history"
        driver.get(url)

        # Wait for the history section to load
        WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.CLASS_NAME, "release__card"))
        )

        # Allow time for JavaScript to fully render the page
        time.sleep(0.1)

        # Parse the page content
        version_to_date = OrderedDict()
        release_cards = driver.find_elements(By.CLASS_NAME, "release__card")

        for card in release_cards:
            try:
                version_element = card.find_element(By.CLASS_NAME, "release__version")
                version_number = version_element.text.strip().split("\n")[0]

                version_date = card.find_element(
                    By.CLASS_NAME, "release__version-date"
                ).find_element(By.TAG_NAME, "time")
                datetime_attr = version_date.get_attribute("datetime")

                if datetime_attr:
                    # Convert datetime string to timestamp when saving
                    timestamp = datetime.fromisoformat(
                        datetime_attr.replace("Z", "+00:00")
                    ).timestamp()
                    version_to_date[version_number] = timestamp
                else:
                    print(f"No date found for {version_number}")
            except Exception as e:
                print(f"Error processing version: {e}")

        # make sure the dates are sorted
        version_to_date = OrderedDict(sorted(version_to_date.items(), key=lambda x: x[1]))

        return version_to_date

    finally:
        # Always close the browser
        driver.quit()


def get_versions(package_name):
    """
    Get package version history using the specified backend.

    Args:
        package_name (str): Name of the package
        backend (str): Backend to use, either "requests" or "selenium"

    Returns:
        OrderedDict: Mapping of version numbers to release timestamps
    """
    # First try with requests backend
    versions = get_versions_with_requests(package_name)

    # If requests failed, try with selenium
    if versions is None or len(versions) == 0:
        versions = get_versions_with_selenium(package_name)

    # If both failed, raise error
    if versions is None or len(versions) == 0:
        raise ValueError(f"Failed to get versions for {package_name}")

    return versions


def check_and_replace_version(package_name, version, timestamp):
    timestamp = datetime.strptime(timestamp, "%Y-%m-%d").timestamp()
    versions = get_versions(package_name)

    if version in versions:
        version_timestamp = versions[version]
    else:
        return version

    if timestamp < version_timestamp:
        return get_version_at_time(package_name, timestamp)
    else:
        return version


def get_version_at_time(package_name, timestamp):
    """
    Find the version of a package at a given timestamp.

    Args:
        package_name (str): Name of the package
        timestamp (float or str): Unix timestamp or YYYY-MM-DD

    Returns:
        str: Version number at the given timestamp
    """
    # Convert string timestamp to float if necessary
    if isinstance(timestamp, str):
        timestamp = datetime.strptime(timestamp, "%Y-%m-%d").timestamp()

    versions = get_versions(package_name)
    # Since versions is sorted by timestamp, we can just find the first version
    # that was released after our target timestamp and return the previous one
    prev_version = None
    for version, ts in versions.items():
        if ts > timestamp:
            return prev_version
        prev_version = version

    if prev_version is None:
        raise ValueError(f"No version found for {package_name} at {timestamp}")

    return prev_version


def time_travel_requirements(
    requirements_data: RequirementsData, commit_date: str, logger: CustomLogger
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

    pip_packages: dict[str, str] = {}
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


class TestVersionFinder(unittest.TestCase):
    def test_numpy(self):
        # Test with a reasonable date that should have a known version
        version = get_version_at_time("numpy", "2023-01-01")
        self.assertIsNotNone(version)

    def test_pandas(self):
        # Test another package
        version = get_version_at_time("pandas", "2022-01-01")
        self.assertIsNotNone(version)

    def test_alabaster(self):
        version = check_and_replace_version("alabaster", "1.0.0", "2019-01-01")
        self.assertEqual(version, "0.7.12")

    def test_attrs(self):
        version = check_and_replace_version("attrs", "25.3.0", "2019-01-01")
        self.assertEqual(version, "18.2.0")


if __name__ == "__main__":
    unittest.main()
