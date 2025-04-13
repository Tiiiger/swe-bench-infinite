import datetime
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, cast

import datasets


def get_quarter_bin(timestamp_str: str) -> str:
    """Convert ISO timestamp to quarterly bin (e.g., '2022 Q1')"""
    date = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
    quarter = (date.month - 1) // 3 + 1
    return f"{date.year} Q{quarter}"


def generate_distribution_report(
    sampled_items: List[Any],
    repo_counter: Counter,
    quarter_counter: Dict,
    repo_quarter_counter: Dict,
):
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


def get_even_sample_dataset(
    target_total: int = 300,
    min_repo_samples: int = 25,
    verbose: bool = True,
    cache: bool = True,
    cache_path: Optional[str] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    Get an evenly sampled dataset from SWE-Bench.

    Args:
        target_total: Target total number of samples for the entire dataset.
        min_repo_samples: Minimum number of samples to include for each repository.
        verbose: Whether to print distribution reports.
        cache: Whether to cache the results.
        cache_path: Path to cache file. If None, uses 'sampled_dataset.json' in same directory.
        seed: Random seed for reproducibility.

    Returns:
        List of sampled items
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Set default cache path if not provided
    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(__file__), "sampled_dataset.json")

    # Check if cached dataset exists
    if cache and not verbose and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    # Load the dataset
    swe_bench = datasets.load_dataset("princeton-nlp/SWE-Bench", split="test")

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

    if verbose:
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

    # For each repository, determine how many samples we should take
    # Initial allocation ensures minimum representation for small repositories
    repo_allocation = {}
    remaining_target = target_total

    # First pass: For repos with fewer samples than min_repo_samples, take all available
    for repo, count in repo_counter.items():
        if count <= min_repo_samples:
            # For small repos, take all available samples
            repo_allocation[repo] = count
            remaining_target -= count
        else:
            # Initialize larger repos with zero for now
            repo_allocation[repo] = 0

    # Second pass: Ensure minimum samples or available samples for larger repositories
    # and distribute remaining proportionally
    total_larger_repos = sum(1 for count in repo_counter.values() if count > min_repo_samples)

    if total_larger_repos > 0:
        # Calculate baseline allocation for larger repos
        baseline_larger = (
            min(min_repo_samples, remaining_target // total_larger_repos)
            if total_larger_repos > 0
            else 0
        )

        # Allocate baseline to each larger repo
        for repo, count in repo_counter.items():
            if count > min_repo_samples:
                repo_allocation[repo] = min(baseline_larger, count)
                remaining_target -= repo_allocation[repo]

        # Third pass: Distribute any remaining samples proportionally among larger repos
        if remaining_target > 0:
            # Calculate total remaining capacity in larger repos
            total_remaining_capacity = sum(
                max(0, count - repo_allocation[repo])
                for repo, count in repo_counter.items()
                if count > min_repo_samples
            )

            if total_remaining_capacity > 0:
                for repo, count in repo_counter.items():
                    if count > min_repo_samples and count > repo_allocation[repo]:
                        # Calculate proportional extra allocation
                        extra_capacity = count - repo_allocation[repo]
                        proportion = extra_capacity / total_remaining_capacity
                        extra_allocation = min(int(proportion * remaining_target), extra_capacity)

                        repo_allocation[repo] += extra_allocation
                        remaining_target -= extra_allocation

    # Final pass: If we still have remaining samples, add them one by one to repos that can take more
    if remaining_target > 0:
        for repo in sorted(
            repo_counter.keys(), key=lambda r: repo_counter[r] - repo_allocation[r], reverse=True
        ):
            available_capacity = repo_counter[repo] - repo_allocation[repo]
            if available_capacity > 0 and remaining_target > 0:
                extra = min(available_capacity, remaining_target)
                repo_allocation[repo] += extra
                remaining_target -= extra
            if remaining_target == 0:
                break

    if verbose:
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
    actual_repo_samples = {}  # Track how many samples we actually take per repo

    for repo, allocation in repo_allocation.items():
        if allocation <= 0:
            continue

        # Special case: For repos with fewer samples than min_repo_samples, take ALL samples
        if repo_counter[repo] <= min_repo_samples:
            repo_samples = []
            for quarter in repo_quarter_counter[repo].keys():
                available_items = repo_quarter_items[repo][quarter]
                repo_samples.extend(available_items)  # Take all samples from this quarter

            sampled_items.extend(repo_samples)
            actual_repo_samples[repo] = len(repo_samples)
            continue  # Skip the normal quarter-based allocation for these small repos

        # Regular case: For larger repos, distribute across quarters
        # Get all available quarters for this repo
        available_quarters = sorted(repo_quarter_counter[repo].keys())
        if not available_quarters:
            continue

        # Calculate how many samples to take from each quarter
        # Try to distribute evenly across quarters without using ceil to avoid oversampling
        quarter_allocations = {}
        samples_per_quarter = allocation / len(
            available_quarters
        )  # Use float division instead of ceiling

        # First pass: Allocate integer portion to each quarter
        base_per_quarter = int(samples_per_quarter)
        remainder = allocation - (base_per_quarter * len(available_quarters))

        # Allocate base amount to each quarter
        for quarter in available_quarters:
            quarter_allocations[quarter] = min(
                base_per_quarter, repo_quarter_counter[repo][quarter]
            )

        # Distribute remainder to quarters with more samples, one at a time
        if remainder > 0:
            # Sort quarters by available samples (more samples first)
            for quarter_i in sorted(
                available_quarters,
                key=lambda q: repo_quarter_counter[repo][q] - quarter_allocations[q],
                reverse=True,
            ):
                if (
                    quarter_allocations[quarter_i] < repo_quarter_counter[repo][quarter_i]
                    and remainder > 0
                ):
                    quarter_allocations[quarter_i] += 1
                    remainder -= 1
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
        actual_repo_samples[repo] = len(repo_samples)

    # Final adjustment to ensure we don't exceed target_total
    if len(sampled_items) > target_total:
        # Calculate how many samples to remove
        excess = len(sampled_items) - target_total

        # Create a priority list for removing samples, focusing on repos with most samples first
        # But don't reduce any repo below min_repo_samples unless absolutely necessary
        removal_candidates = []
        for repo, count in sorted(actual_repo_samples.items(), key=lambda x: x[1], reverse=True):
            # Only consider repos with more than minimum samples
            if count > min_repo_samples:
                eligible_to_remove = count - min_repo_samples
                removal_candidates.append((repo, eligible_to_remove))

        # If we still need to remove more after considering eligible repos
        if sum(eligible for _, eligible in removal_candidates) < excess:
            # Also consider repos at minimum samples if absolutely necessary
            for repo, count in sorted(
                actual_repo_samples.items(), key=lambda x: x[1], reverse=True
            ):
                if count <= min_repo_samples and count > 1:  # Ensure at least 1 sample remains
                    removal_candidates.append((repo, count - 1))

        # Remove samples based on priority list
        to_remove = excess
        indices_to_remove = []

        # Create a map of samples by repo for easier removal
        samples_by_repo: Dict[str, List[int]] = {repo: [] for repo in actual_repo_samples}
        for i, item in enumerate(sampled_items):
            samples_by_repo[item["repo"]].append(i)

        # Remove samples from each repo according to our priority
        for repo, eligible in removal_candidates:
            if to_remove <= 0:
                break

            # Calculate how many to remove from this repo
            remove_from_repo = min(eligible, to_remove)
            if remove_from_repo <= 0:
                continue

            # Select indices to remove
            repo_indices = samples_by_repo[repo]
            selected_indices = random.sample(repo_indices, remove_from_repo)
            indices_to_remove.extend(selected_indices)

            to_remove -= remove_from_repo

        # Sort indices in descending order to avoid shifting problems
        indices_to_remove.sort(reverse=True)

        # Remove the selected samples
        for idx in indices_to_remove:
            sampled_items.pop(idx)

    if verbose:
        print(f"Total sampled items: {len(sampled_items)}")
        # Create detailed distribution reports
        generate_distribution_report(
            sampled_items, repo_counter, quarter_counter, repo_quarter_counter
        )

    # Cache the results if requested
    if cache:
        if verbose:
            print(f"\nCaching results to: {cache_path}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)

        with open(cache_path, "w") as f:
            # Convert dataset items to serializable format
            serializable_data = []
            for item in sampled_items:
                serializable_item = {k: v for k, v in item.items()}
                serializable_data.append(serializable_item)
            json.dump(serializable_data, f, indent=2)

        if verbose:
            print(f"Cached {len(serializable_data)} items successfully.")

        return serializable_data

    return sampled_items


# Run the function if this file is executed directly
if __name__ == "__main__":
    print("\n--- Creating Evenly Sampled Dataset (Target: 300 samples) ---")
    sampled_dataset = get_even_sample_dataset(
        target_total=300, min_repo_samples=25, verbose=True, cache=True
    )
    print(f"\nFinal sample size: {len(sampled_dataset)} items")
