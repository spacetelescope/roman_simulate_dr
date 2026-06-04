import argparse
import gc
import os
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from romanisim.catalog import (
    make_cosmos_galaxies,
    make_gaia_stars,
    make_stars,
    read_catalog,
)

from roman_simulate_dr.scripts.logger import logger
from roman_simulate_dr.scripts.utils import generate_catalog_name, read_obs_plan


class InputCatalog:
    """
    Class to generate Romanisim input catalogs based on an observation plan.

    """

    def __init__(
        self,
        obs_plan_filename: str,
        output_catalog_filename: str | None = None,
        ra: float | None = None,
        dec: float | None = None,
        radius: float | None = None,
        flux_catalog_filename: str | None = None,
    ):
        """
        Initialize the InputCatalog object.

        Parameters
        ----------
        obs_plan_filename : str
            Path to the observation plan file.
        output_catalog_filename : str or None, optional
            Filename for the final output catalog.
        ra : float or None, optional
            Override for the RA (deg) of the whole catalog.
        dec : float or None, optional
            Override for the Dec (deg) of the whole catalog.
        radius : float or None, optional
            Override for the radius (deg) of the whole catalog.
        flux_catalog_filename : str or None, optional
            Path to a flux_catalog file previously produced by roman_photoz (e.g. parquet).
            If provided, the generated catalog will be updated using this flux catalog.
        """
        self.input_root = Path(os.getenv("RDR_INPUT_PATH", "."))
        self.output_root = Path(os.getenv("RDR_OUTPUT_PATH", "."))

        def resolve_path(fname, root, must_exist=False):
            if not fname:
                return None
            p = Path(fname)
            # If path is already absolute or exists relative to CWD, use it
            if p.exists():
                return p
            # Otherwise, check the designated root
            resolved = root / fname
            if must_exist and not resolved.exists():
                raise FileNotFoundError(
                    f"Required file not found: {resolved.absolute()}"
                )
            return resolved

        # Resolve Plan (Input)
        plan_path = resolve_path(obs_plan_filename, self.input_root, must_exist=True)
        self.plan = read_obs_plan(str(plan_path))

        # Resolve Flux Catalog (Input; externally-provided flux catalog file produced by roman_photoz)
        self.flux_catalog_filename = resolve_path(
            flux_catalog_filename, self.input_root
        )

        # Resolve Output Catalog
        if output_catalog_filename:
            # If the user provided a full path (like DATASET/OUTPUT/file.parquet), use it.
            # Otherwise, join the filename with our output root.
            out_p = Path(output_catalog_filename)
            self.catalog_filename = (
                out_p if out_p.parent != Path(".") else self.output_root / out_p
            )
        else:
            self.catalog_filename = self.output_root / generate_catalog_name(
                str(plan_path)
            )

        # set reference coordinates and radius to simulate
        self.ra = ra if ra is not None else float(np.mean(np.array(self.plan["RA"])))
        self.dec = (
            dec if dec is not None else float(np.mean(np.array(self.plan["DEC"])))
        )
        self.radius = radius if radius is not None else 0.3

    def _generate_catalog(self, filter_list=None):
        """
        Generate a single catalog covering the full area and keep components in memory.

        Parameters
        ----------
        filter_list : list of str or None, optional
            List of filter names to use for bandpasses. If None, uses default filters.
        """
        logger.info(
            f"Generating catalog at RA={self.ra} Dec={self.dec} radius={self.radius} deg"
        )
        if filter_list is None:
            filter_list = [
                "f062",
                "f087",
                "f106",
                "f129",
                "f146",
                "f158",
                "f184",
                "f213",
            ]
        bandpasses = [bp.upper() for bp in filter_list]

        coords = SkyCoord(ra=self.ra, dec=self.dec, unit="deg", frame="icrs")

        # 1. Generate Galaxies
        catalog = make_cosmos_galaxies(
            coord=coords, bandpasses=bandpasses, seed=42, radius=self.radius
        )
        logger.info(f"Galaxies generated. Rows: {len(catalog)}")

        # 2. Add Gaia Stars (and release temp memory)
        try:
            temp_stars = make_gaia_stars(
                coord=coords, bandpasses=bandpasses, seed=42, radius=self.radius
            )
        except Exception as e:
            logger.warning(f"Gaia stars failed: {e}. Falling back to read_catalog.")
            temp_stars = read_catalog(
                "/grp/roman/gaia/healpix128",
                coord=coords,
                bandpasses=bandpasses,
                radius=self.radius,
            )

        catalog = vstack([catalog, temp_stars])
        del temp_stars  # Explicitly delete the temporary star table
        gc.collect()  # Force Python to reclaim that RAM immediately

        # 3. Add General Stars
        temp_stars = make_stars(
            coord=coords,
            n=1000,
            bandpasses=bandpasses,
            seed=42,
            radius=self.radius,
        )
        catalog = vstack([catalog, temp_stars])
        del temp_stars
        gc.collect()

        # 4. Save to Disk
        catalog.write(self.catalog_filename, format="parquet", overwrite=True)

        logger.info(
            f"""
              Final concatenated catalog saved to '{self.catalog_filename}'.
              Total sources: {len(catalog)}.
              """
        )

        # If the user supplied a flux_catalog filename, update the generated catalog
        # using that file (user must run roman_photoz separately and provide the flux_catalog output).
        if self.flux_catalog_filename is not None:
            logger.info(
                f"Updating generated catalog using flux catalog: {self.flux_catalog_filename}"
            )
            updated = self.update_catalog_fluxes(catalog)
            # If update_catalog_fluxes succeeds it already writes the updated catalog.
            logger.info(
                f"Updated catalog with roman_photoz fluxes saved to '{self.catalog_filename}'. Total sources: {len(updated)}"
            )

    def update_catalog_fluxes(self, catalog: Table) -> Table:
        """
        Update the provided catalog's fluxes using an externally provided flux catalog file.

        This method isolates the logic of reading the flux catalog file, importing the
        update helper from roman_photoz, performing the update, and writing the result.

        Parameters
        ----------
        catalog : astropy.table.Table
            The generated romanisim catalog to be updated.

        Returns
        -------
        updated : astropy.table.Table
            The updated catalog (also written to self.catalog_filename).
        """
        if self.flux_catalog_filename is None:
            raise RuntimeError(
                "No flux_catalog_filename provided to update_catalog_fluxes."
            )

        try:
            flux_catalog = Table.read(self.flux_catalog_filename, format="parquet")
        except Exception as exc:
            logger.error(
                f"Could not read flux catalog '{self.flux_catalog_filename}': {exc}"
            )
            raise

        # import the helpers that perform the update (keep import local to avoid
        # forcing roman_photoz to be installed when users just want to generate catalogs).
        try:
            from roman_photoz.update_romanisim_catalog_fluxes import (
                create_random_catalog,
                update_fluxes,
            )
        except Exception:
            logger.error(
                "Failed to import modules from roman_photoz. "
                "If you want to update fluxes using a roman_photoz output file, "
                "please ensure the 'roman_photoz' package is installed and importable."
            )
            raise

        try:
            # create same number of entries in both catalogs for updating
            flux_catalog = create_random_catalog(flux_catalog, n=len(catalog), seed=42)
            gc.collect()  # Clean up any shards left by create_random_catalog

            # additional columns (e.g., label, redshift_true) are added here
            updated = update_fluxes(target_catalog=catalog, flux_catalog=flux_catalog)
            # Clear the old references immediately
            del catalog
            del flux_catalog
            gc.collect()
        except Exception as exc:
            logger.error(f"Failed to update catalog fluxes: {exc}")
            raise

        # Overwrite the previously written catalog with the updated version
        updated.write(self.catalog_filename, format="parquet", overwrite=True)
        return updated

    def run(self, filter_list=None) -> None:
        """
        Run the Romanisim input catalog generation workflow.

        This method creates a single catalog for all exposures.
        """
        self._generate_catalog(filter_list=filter_list)


def _cli():
    """
    Command-line interface for generating Romanisim input catalogs.

    Parses arguments and runs the catalog generation workflow.
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
        "--output-filename",
        type=str,
        default=None,
        required=False,
        help="Output catalog filename (default: append '_cat' to the observation plan filename)",
    )
    parser.add_argument(
        "--ra",
        type=float,
        default=None,
        help="Override: RA center of catalog (deg)",
    )
    parser.add_argument(
        "--dec",
        type=float,
        default=None,
        help="Override: Dec center of catalog (deg)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.3,
        help="Override: Radius of catalog (deg; default 0.3)",
    )
    parser.add_argument(
        "--flux-catalog",
        type=str,
        default=None,
        required=False,
        help="Path to a flux_catalog file produced by roman_photoz (parquet). If provided, the generated catalog will be updated using this file.",
    )
    parser.add_argument(
        "--filter-list",
        type=str,
        nargs="+",
        default=None,
        help="List of filter names to use for bandpasses (default: all Roman WFI filters)",
    )
    args = parser.parse_args()

    input_catalog = InputCatalog(
        obs_plan_filename=args.obs_plan,
        output_catalog_filename=args.output_filename,
        ra=args.ra,
        dec=args.dec,
        radius=args.radius,
        flux_catalog_filename=args.flux_catalog,
    )
    input_catalog.run(args.filter_list)

    logger.info("Done.")


if __name__ == "__main__":
    """
    Entry point for the script when run as a standalone program.
    """
    _cli()
