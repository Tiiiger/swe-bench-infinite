import json
import subprocess
from pathlib import Path

from docker_utils import copy_to_container, exec_run_with_timeout
from swebench.harness.grading import get_eval_report

import docker
from logger import CustomLogger, setup_logger


def process_eval_report(
    test_spec,
    example,
    parent_logger: CustomLogger,
    trial_num: int,
) -> tuple[subprocess.CompletedProcess, dict]:
    """
    Process the evaluation report by running tests and generating a report.

    Args:
        test_spec: The test specification object
        example: Dictionary containing instance data including patch and instance_id
        parent_logger: Logger instance for logging operations
        trial_num: The current trial number

    Returns:
        tuple[subprocess.CompletedProcess, dict]: Test output and evaluation report
    """
    # setup a separate logger for eval report
    eval_report_logger = setup_logger(
        logger_name=f"eval_report_{trial_num}", parent_logger=parent_logger
    )
    instance_id = eval_report_logger.get_instance_id()

    # Run the test in a Docker container
    # Create and start container
    container = docker.from_env().containers.run(
        f"{instance_id}.test:latest",
        command="tail -f /dev/null",  # Keep container running
        remove=True,  # Auto-remove when stopped
        detach=True,  # Run in background
    )

    # get patch diff
    patch_diff_path = Path(eval_report_logger.get_logdir()) / "patch.diff"
    with open(patch_diff_path, "w") as f:
        f.write(example["patch"])  # type: ignore

    copy_to_container(container, patch_diff_path, Path("/patch.diff"))
    exit_code, test_output_log, timed_out, total_runtime = exec_run_with_timeout(
        container, "conda run -n testbed git apply --verbose /patch.diff", timeout=600
    )
    eval_report_logger.info(f"Patch output: {test_output_log}")
    if exit_code != 0:
        raise RuntimeError(f"Patch failed with exit code {exit_code}. This should not be retried.")

    # write to logdir
    eval_script_path = Path(eval_report_logger.get_logdir()) / "eval.sh"
    with open(eval_script_path, "w") as f:
        f.write(test_spec.eval_script)

    copy_to_container(container, eval_script_path, Path("/eval.sh"))
    exit_code, test_output_log, timed_out, total_runtime = exec_run_with_timeout(
        container, "conda run -n testbed /bin/bash /eval.sh", timeout=600
    )
    eval_report_logger.info(f"Test output: {test_output_log}")
    with open(Path(eval_report_logger.get_logdir()) / "test_output.log", "w") as f:
        f.write(test_output_log)

    report = get_eval_report(
        test_spec=test_spec,
        prediction={
            "instance_id": example["instance_id"],
            "model_patch": example["patch"],  # type: ignore
            "model_name_or_path": "gold",
        },
        test_log_path=f"{eval_report_logger.get_logdir()}/test_output.log",
        include_tests_status=True,
    )
    with open(Path(eval_report_logger.get_logdir()) / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    # stop the container
    container.stop()

    return (
        subprocess.CompletedProcess(
            args=[], returncode=exit_code, stdout=test_output_log, stderr=test_output_log
        ),
        report,
    )
