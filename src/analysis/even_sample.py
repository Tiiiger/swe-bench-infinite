import datetime
import random
from collections import Counter
from typing import Any, Dict, List, cast

import datasets

swe_bench = datasets.load_dataset("princeton-nlp/SWE-Bench", split="test")


def get_quarter_bin(timestamp_str: str) -> str:
    """Convert ISO timestamp to quarterly bin (e.g., '2022 Q1')"""
    date = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
    quarter = (date.month - 1) // 3 + 1
    return f"{date.year} Q{quarter}"


# Count repository distribution
repo_counter: Counter[str] = Counter()
# Count by quarter
quarter_counter: Counter[str] = Counter()
# Count by repo and quarter
repo_quarter_counter: Dict[str, Dict[str, int]] = {}
# Store items by repo and quarter for sampling
repo_quarter_items: Dict[str, Dict[str, List[Any]]] = {}

for item in swe_bench:
    repo = item["repo"]  # type: ignore
    created_at = item["created_at"]  # type: ignore
    quarter = get_quarter_bin(created_at)

    # Update repo counter
    repo_counter[repo] += 1

    # Update quarter counter
    quarter_counter[quarter] += 1

    # Update repo-quarter counter
    if repo not in repo_quarter_counter:
        repo_quarter_counter[repo] = {}
        repo_quarter_items[repo] = {}
    if quarter not in repo_quarter_counter[repo]:
        repo_quarter_counter[repo][quarter] = 0
        repo_quarter_items[repo][quarter] = []
    repo_quarter_counter[repo][quarter] += 1
    repo_quarter_items[repo][quarter].append(cast(dict, item))

# Print the repository distribution
print("\nRepository Distribution:")
for repo, count in sorted(repo_counter.items(), key=lambda x: x[1], reverse=True):
    print(f"{repo}: {count}")

# Print the quarterly distribution
print("\nQuarterly Distribution:")
for quarter, count in sorted(quarter_counter.items()):
    print(f"{quarter}: {count}")

# Print the stratified report
print("\nStratified Report (Repo × Quarter):")
for repo, quarters in sorted(
    repo_quarter_counter.items(), key=lambda x: repo_counter[x[0]], reverse=True
):
    print(f"\n{repo} ({repo_counter[repo]} total):")
    for quarter, count in sorted(quarters.items()):
        print(f"  {quarter}: {count}")

# Total count
print(f"\nTotal repositories: {len(repo_counter)}")
print(f"Total issues: {len(swe_bench)}")  # type: ignore


def create_evenly_sampled_dataset(target_total: int = 300, min_repo_samples: int = 10):
    """
    Create an evenly sampled dataset with approximately target_total samples,
    ensuring minimum representation for each repository while maintaining quarter stratification.

    Args:
        target_total: Target total number of samples for the entire dataset.
        min_repo_samples: Minimum number of samples to include for each repository.

    Returns:
        List of sampled items
    """
    # For each repository, determine how many samples we should take
    # Initial allocation ensures minimum representation for small repositories
    repo_allocation = {}
    remaining_target = target_total

    # First pass: Ensure minimum samples per repository
    # Make sure we don't exceed actual available samples
    for repo, count in repo_counter.items():
        repo_allocation[repo] = min(min_repo_samples, count)
        remaining_target -= repo_allocation[repo]

    # If we have samples left to allocate, distribute proportionally among repos
    # with more than min_repo_samples items
    if remaining_target > 0:
        total_remaining_items = sum(
            max(0, count - min_repo_samples) for count in repo_counter.values()
        )
        if total_remaining_items > 0:
            for repo, count in repo_counter.items():
                if count > min_repo_samples:
                    # Calculate extra samples proportionally
                    extra_proportion = (count - min_repo_samples) / total_remaining_items
                    extra_samples = int(extra_proportion * remaining_target)
                    repo_allocation[repo] += extra_samples

    # Adjust to ensure we hit target_total as closely as possible
    total_allocated = sum(repo_allocation.values())
    if total_allocated < target_total:
        # Distribute remaining samples to larger repositories
        remaining = target_total - total_allocated
        for repo in sorted(repo_counter.keys(), key=lambda r: repo_counter[r], reverse=True):
            if repo_allocation[repo] < repo_counter[repo] and remaining > 0:
                repo_allocation[repo] += 1
                remaining -= 1
            if remaining == 0:
                break

    print("\nSampling Strategy:")
    print(f"- Target total samples: {target_total}")
    print(f"- Minimum samples per repository: {min_repo_samples}")
    print("- Repository allocations:")
    for repo, allocation in sorted(repo_allocation.items(), key=lambda x: x[1], reverse=True):
        original = repo_counter[repo]
        percentage = (allocation / original * 100) if original > 0 else 0
        print(f"  {repo}: {allocation} (from {original}, {percentage:.1f}%)")

    # Sample from each repository according to its allocation
    sampled_items = []

    for repo, allocation in repo_allocation.items():
        if allocation <= 0:
            continue

        # Get all available quarters for this repo
        available_quarters = sorted(repo_quarter_counter[repo].keys())
        if not available_quarters:
            continue

        # Calculate how many samples to take from each quarter
        # Try to distribute evenly across quarters
        quarter_allocations = {}
        samples_per_quarter = max(1, allocation // len(available_quarters))
        remainder = allocation - (samples_per_quarter * len(available_quarters))

        for quarter in available_quarters:
            quarter_allocations[quarter] = min(
                samples_per_quarter, repo_quarter_counter[repo][quarter]
            )

        # Distribute any remainder to quarters with more samples
        if remainder > 0:
            for quarter in sorted(
                available_quarters, key=lambda q: repo_quarter_counter[repo][q], reverse=True
            ):
                extra = min(
                    remainder, repo_quarter_counter[repo][quarter] - quarter_allocations[quarter]
                )
                if extra > 0:
                    quarter_allocations[quarter] += extra
                    remainder -= extra
                if remainder == 0:
                    break

        # Sample from each quarter
        repo_samples = []
        for quarter, quarter_allocation in quarter_allocations.items():
            if quarter_allocation <= 0:
                continue

            available_items = repo_quarter_items[repo][quarter]
            sample_size = min(quarter_allocation, len(available_items))
            if sample_size > 0:
                sample = random.sample(available_items, sample_size)
                repo_samples.extend(sample)

        sampled_items.extend(repo_samples)

    print(f"Total sampled items: {len(sampled_items)}")

    # Create detailed distribution reports
    generate_distribution_report(sampled_items)

    return sampled_items


def generate_distribution_report(sampled_items: List[Any]):
    """Generate detailed distribution reports for the sampled dataset."""
    # Report on the distribution in the sampled dataset
    sampled_repo_counter = Counter([item["repo"] for item in sampled_items])

    # Calculate quarter for each sampled item
    sampled_quarters = [get_quarter_bin(item["created_at"]) for item in sampled_items]
    sampled_quarter_counter = Counter(sampled_quarters)

    # Calculate repo-quarter distribution
    sampled_repo_quarter: Dict[str, Dict[str, int]] = {}
    for item in sampled_items:
        repo = item["repo"]
        quarter = get_quarter_bin(item["created_at"])
        if repo not in sampled_repo_quarter:
            sampled_repo_quarter[repo] = {}
        if quarter not in sampled_repo_quarter[repo]:
            sampled_repo_quarter[repo][quarter] = 0
        sampled_repo_quarter[repo][quarter] += 1

    # Print sampled repository distribution
    print("\nSampled Repository Distribution:")
    for repo, count in sorted(sampled_repo_counter.items(), key=lambda x: x[1], reverse=True):
        original_count = repo_counter[repo]
        percentage = (count / original_count * 100) if original_count > 0 else 0
        print(f"{repo}: {count} (from {original_count}, {percentage:.1f}%)")

    # Print sampled quarterly distribution
    print("\nSampled Quarterly Distribution:")
    for quarter, count in sorted(sampled_quarter_counter.items()):
        original_count = quarter_counter[quarter]
        percentage = (count / original_count * 100) if original_count > 0 else 0
        print(f"{quarter}: {count} (from {original_count}, {percentage:.1f}%)")

    # Print sampled repo-quarter distribution
    print("\nSampled Repo × Quarter Distribution:")
    for repo, quarters in sorted(
        sampled_repo_quarter.items(), key=lambda x: sum(x[1].values()), reverse=True
    ):
        print(f"\n{repo} ({sum(quarters.values())} sampled, {repo_counter[repo]} original):")
        for quarter, count in sorted(quarters.items()):
            original_count = repo_quarter_counter[repo].get(quarter, 0)
            percentage = (count / original_count * 100) if original_count > 0 else 0
            print(f"  {quarter}: {count} (from {original_count}, {percentage:.1f}%)")


# Create an evenly sampled dataset with target 300 total samples
# and ensuring minimum representation for smaller repositories
print("\n--- Creating Evenly Sampled Dataset (Target: 300 samples) ---")
sampled_dataset = create_evenly_sampled_dataset(target_total=300, min_repo_samples=10)

# You can use the sampled dataset for further analysis or save it
print(f"\nFinal sample size: {len(sampled_dataset)} items")
