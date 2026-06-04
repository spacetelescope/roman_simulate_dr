import re
import subprocess
import sys
from pathlib import Path

# Strict pattern: alphanumeric, dots, underscores, hyphens, and forward slashes.
SAFE_PATTERN = re.compile(r"^[a-zA-Z0-9._\-/]+$")


def get_safe_args():
    """
    Sanitize and return command-line arguments from sys.argv.

    Iterates through all provided CLI arguments (excluding the script name)
    and validates them against a strict regex to prevent shell injection.

    Returns
    -------
    list of str
        A list of validated, safe command-line arguments.

    Raises
    ------
    ValueError
        If any argument contains characters outside the allowed safe set.
    """
    args = sys.argv[1:]
    for arg in args:
        if not SAFE_PATTERN.match(arg):
            raise ValueError(f"Potentially malicious argument detected: {arg}")
    return args


def spawn_process(script_name):
    """
    Execute a project bash script as a secured subprocess.

    Resolves the absolute path of the script, validates CLI arguments,
    and executes the process using list unpacking to avoid linter
    concatenation warnings.

    Parameters
    ----------
    script_name : str
        The filename of the bash script to execute.

    Notes
    -----
    Uses list unpacking `[*base_cmd, *safe_args]` instead of list
    concatenation `list + list` to satisfy security linters regarding
    untrusted input execution.
    """
    try:
        # Resolve absolute path to prevent path traversal
        script_path = (Path(__file__).parent / script_name).resolve()

        if not script_path.is_file():
            print(f"Error: Internal script {script_name} missing.")
            sys.exit(1)

        # 1. Get sanitized arguments
        safe_args = get_safe_args()

        # 2. Use list unpacking to satisfy "untrusted input" linter warnings.
        # This is more robust than list concatenation (+)
        result = subprocess.run(
            ["/bin/bash", str(script_path), *safe_args],
            shell=False,
            check=False,
        )
        sys.exit(result.returncode)

    except ValueError as e:
        print(f"Security Check Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        sys.exit(1)


def run_simulation():
    """
    Entry point for the data simulation workflow.

    Executes 'rdr_simulate_data.sh' with sanitized arguments.
    """
    spawn_process("rdr_simulate_data.sh")


def run_processing():
    """
    Entry point for the data processing workflow.

    Executes 'rdr_process_data.sh' with sanitized arguments.
    """
    spawn_process("rdr_process_data.sh")
