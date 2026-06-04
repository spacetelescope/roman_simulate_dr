import argparse
import os
import time
from pathlib import Path

from roman_simulate_dr.scripts.logger import logger
from roman_simulate_dr.scripts.utils import (
    generate_roman_filename,
    parallelize_jobs,
    read_obs_plan,
)


class RomanisimImages:
    """
    Handles generation of simulated Roman L1 images based on an observation plan and input catalog.
    """

    def __init__(
        self,
        obs_plan_filename: str,
        input_filename: str,
        max_workers: int = 1,
        sca_ids: list[int] | None = None,
    ):
        """
        Initialize the RomanisimImages object.

        Parameters
        ----------
        obs_plan_filename : str
            Path to the observation plan file.
        input_filename : str
            Path to the input catalog file.
        max_workers : int
            Number of parallel workers to use for processing (default 1).
        sca_ids : list of int or None, optional
            List of SCA IDs to use. If None, uses SCA 1.

        Raises
        ------
        FileNotFoundError
            If obs_plan_filename or input_filename is not provided.
        """
        self.input_root = Path(os.getenv("RDR_INPUT_PATH", "."))
        self.output_root = Path(os.getenv("RDR_OUTPUT_PATH", "."))

        # Helper to resolve paths without doubling up directories
        def resolve_input(fname):
            p = Path(fname)
            if p.exists():
                return p
            return self.input_root / fname

        plan_path = resolve_input(obs_plan_filename)
        # Crucial: Step 2 should check if the catalog exists in Input OR Output root
        input_path = Path(input_filename)
        if not input_path.exists():
            input_path = (
                self.output_root / input_filename
            )  # Try output root (intermediate file)

        if not plan_path.exists():
            raise FileNotFoundError(f"Plan not found: {plan_path}")
        if not input_filename:
            raise FileNotFoundError("An input catalog filename must be provided.")

        self.plan = read_obs_plan(str(plan_path))
        self.input_filename = str(input_path)
        self.max_workers = max_workers
        self.sca_ids = self._create_sca_id_list(sca_ids)

    def _create_sca_id_list(self, sca_ids: list[int] | None = None) -> list[int]:
        """
        Generate a list of SCA IDs for catalog creation.

        Returns a list of SCA IDs based on the input:
        - If sca_ids is None, returns [1] (default SCA 1).
        - If sca_ids is a single negative value, returns all SCA IDs from 1 to 17.
        - Otherwise, returns the provided sca_ids list.

        Parameters
        ----------
        sca_ids : list of int or None, optional
            List of SCA IDs to use, or None for default behavior.

        Returns
        -------
        list of int
            The list of SCA IDs to use for catalog generation.
        """
        if sca_ids is None:
            return [1]
        if len(sca_ids) == 1 and sca_ids[0] < 0:
            return list(range(1, 18))
        return sca_ids

    def _generate_simulated_images(
        self,
        radec: tuple = (270.0, 66.0),
        level: int = 1,
        sca: int = 1,
        bandpass: str = "F062",
        roll: float = 0.0,
        catalog: str = "",
        stpsf: bool = True,
        ma_table_number: int = 109,
        date: str = "2027-06-01T00:00:00",
        drop_extra_dq: bool = True,
        output_filename: str = "romanisim_simulated_image.asdf",
    ):
        """
        Run the romanisim simulation with the given parameters.
        """
        import subprocess

        start_time = time.time()

        cmd = [
            "romanisim-make-image",
            "--radec",
            str(radec[0]),
            str(radec[1]),
            "--level",
            str(level),
            "--sca",
            str(sca),
            "--bandpass",
            str(bandpass),
            "--usecrds",
            "--roll",
            str(roll),
            "--catalog",
            str(catalog),
            *(["--psftype", "stpsf"] if stpsf else []),
            "--ma_table_number",
            str(ma_table_number),
            "--date",
            date,
            "--rng_seed",
            "1",
            *(["--drop-extra-dq"] if drop_extra_dq else []),
            output_filename,
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, shell=False, check=False
        )
        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"[{output_filename}] Finished in {duration}.")

        if result.returncode != 0:
            logger.error(
                f"[{output_filename}] FAILED after {duration}. STDERR:\n{result.stderr}"
            )

        return output_filename, result.returncode

    def run(self):
        """
        Run the romanisim simulation workflow for all passes and visits in the plan.
        """
        jobs = []
        for (
            ra_ref,
            dec_ref,
            pa,
            bandpass,
            ma_table_number,
            _,  # duration, not used here
            plan,
            pidx,
            segment,
            observation,
            vidx,
            eidx,
        ) in self.plan:
            for sca in self.sca_ids:
                bandpass = bandpass.upper()
                base_filename = generate_roman_filename(
                    program=1,
                    plan=plan,
                    passno=int(pidx),
                    segment=segment,
                    observation=observation,
                    visit=int(vidx),
                    exposure=int(eidx),
                    sca=int(sca),
                    bandpass=bandpass,
                    suffix="uncal",
                )
                full_output_path = self.output_root / base_filename
                jobs.append(
                    dict(
                        radec=(ra_ref, dec_ref),
                        sca=sca,
                        bandpass=bandpass,
                        roll=pa,
                        ma_table_number=ma_table_number,
                        catalog=self.input_filename,
                        output_filename=str(full_output_path),
                    )
                )
        parallelize_jobs(
            self._generate_simulated_images,
            jobs,
            max_workers=self.max_workers,
        )


def _cli():
    """
    Command-line interface for generating Romanisim simulated images.
    """
    parser = argparse.ArgumentParser(
        description="Generate Romanisim input catalogs based on an observation plan."
    )
    parser.add_argument(
        "--obs-plan",
        type=str,
        default="obs_plan.ecsv",
        help="Observation plan filename (default: obs_plan.ecsv)",
    )
    parser.add_argument(
        "--input-filename",
        type=str,
        default="romanisim_input_catalog.ecsv",
        required=False,
        help="Input catalog filename (default: romanisim_input_catalog.ecsv)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel workers for exposure generation (default: 1, disables parallelization)",
    )
    parser.add_argument(
        "--sca-ids",
        type=int,
        nargs="+",
        default=[1],
        help="List of SCA IDs to simulate (default: [1])",
    )
    args = parser.parse_args()
    input_catalog = RomanisimImages(
        obs_plan_filename=args.obs_plan,
        input_filename=args.input_filename,
        max_workers=args.max_workers,
        sca_ids=args.sca_ids,
    )
    input_catalog.run()
    logger.info("Done.")


if __name__ == "__main__":
    """
    Entry point for the script when run as a standalone program.
    """
    _cli()
