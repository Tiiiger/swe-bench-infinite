# https://pypi.org/project/numpy/#history
# https://pypi.org/project/pandas/#history
# https://pypi.org/project/scikit-learn/#history
import requests
from bs4 import BeautifulSoup
import unittest
from collections import OrderedDict
from datetime import datetime

def get_versions(package_name):
    url = f"https://pypi.org/project/{package_name}/#history"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    version_to_date = OrderedDict()
    for version in soup.find_all("a", class_="release__card"):
        version_element = version.find(class_="release__version")
        version_number = version_element.find(string=True, recursive=False).strip()
        version_date = version.find(class_="release__version-date").find("time")
        if version_date and version_date.get('datetime'):
            # Convert datetime string to timestamp when saving
            timestamp = datetime.fromisoformat(version_date['datetime'].replace('Z', '+00:00')).timestamp()
            version_to_date[version_number] = timestamp
        else:
            print(f"No date found for {version_number}")

    # make sure the dates are sorted
    version_to_date = OrderedDict(sorted(version_to_date.items(), key=lambda x: x[1]))

    return version_to_date

def get_version_at_time(package_name, timestamp):
    """
    Find the version of a package at a given timestamp.
    
    Args:
        package_name (str): Name of the package
        timestamp (float): YYYY-MM-DD
        
    Returns:
        str: Version number at the given timestamp
    """
    versions = get_versions(package_name)
    timestamp = datetime.strptime(timestamp, "%Y-%m-%d").timestamp()
    
    # Since versions is sorted by timestamp, we can just find the first version
    # that was released after our target timestamp and return the previous one
    prev_version = None
    for version, ts in versions.items():
        if ts > timestamp:
            return prev_version
        prev_version = version
    
    # If we get here, all versions are before our timestamp
    return prev_version

class TestVersionFinder(unittest.TestCase):
    def test_numpy(self):
        # Test with a known timestamp (e.g., January 1, 2024)
        version = get_version_at_time("numpy", "2025-04-02")
        self.assertEqual(version, "2.2.4")

if __name__ == "__main__":
    unittest.main()