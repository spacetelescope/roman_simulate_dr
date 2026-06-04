import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from astropy.table import Table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_obs_plan(filename: str) -> Table:
    """
    Reads an observation plan from an ECSV file.

    Parameters
    ----------
    filename : str
        Path to the ECSV file.

    Returns
    -------
    astropy.table.Table
        The observation plan as an Astropy Table.
    """
    return Table.read(filename, format="ascii.ecsv")


def parallelize_jobs(method, jobs, max_workers: int | None = None):
    """
    Run jobs in parallel using ThreadPoolExecutor.

    Parameters
    ----------
    method : callable
        The function or method to execute for each job.
    jobs : list of dict
        Each dict contains the keyword arguments for one call to `method`.
    max_workers : int or None, optional
        Number of parallel workers. If None or <= 1, jobs are run sequentially.

    Returns
    -------
    None
    """
    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(method, **job) for job in jobs]
            for future, _ in zip(as_completed(futures), jobs, strict=False):
                future.result()
    else:
        for job in jobs:
            method(**job)


def generate_roman_filename(
    program: int,
    plan: int,
    passno: int,
    segment: int,
    observation: int,
    visit: int,
    exposure: int,
    sca: int,
    bandpass: str,
    suffix: str,
) -> str:
    """
    Generate a standardized Roman filename based on observation parameters.

    The filename encodes key metadata about the observation, including program,
    plan, pass number, segment, observation, visit, exposure, SCA, bandpass, and
    a custom suffix.

    Parameters
    ----------
    program : int
        Program identifier.
    plan : int
        Plan identifier.
    passno : int
        Pass number.
    segment : int
        Segment number.
    observation : int
        Observation number.
    visit : int
        Visit number.
    exposure : int
        Exposure number.
    sca : int
        Sensor Chip Assembly (SCA) number.
    bandpass : str
        Bandpass filter name (will be converted to lowercase).
    suffix : str
        Custom suffix to append to the filename.

    Returns
    -------
    str
        The generated Roman filename encoding all provided parameters.
    """
    filename = (
        f"r{program}{plan:02d}{passno:03d}{segment:03d}"
        f"{observation:03d}{visit:03d}_{exposure:04d}"
        f"_wfi{sca:02d}_{bandpass.lower()}_{suffix}.asdf"
    )
    return filename


def generate_catalog_name(obs_plan_filename: str) -> str:
    """
    Generate a catalog filename by appending '_cat' before the file extension.

    Parameters
    ----------
    obs_plan_filename : str
        The observation plan filename.

    Returns
    -------
    str
        The derived catalog filename.
    """
    path = Path(obs_plan_filename)
    return str(path.with_name(path.stem + "_cat" + path.suffix))
