"""
plot_zphot_vs_ztrue.py

A command-line tool for matching photometric catalogs to a truth catalog,
selecting outliers, and generating diagnostic plots for roman-photoz.

Usage:
    # With shell expansion (recommended)
    python -m \
      roman_simulate_dr.scripts.plot_utils.plot_zphot_vs_ztrue \
      r00001_r0_full_*y[0-9][0-9]_cat.parquet \
      --ref-cat romanisim_input_catalog.parquet 

    # Or with explicit file listing
    python -m \
      roman_simulate_dr.scripts.plot_utils.plot_zphot_vs_ztrue \
      cat_filename_1.parquet cat_filename_2.parquet ... \
      --ref-cat romanisim_input_catalog.parquet 

Arguments:
    cat_files: List of catalog files to process (use shell expansion for multiple files).
    --ref-cat: Reference catalog file (truth catalog) for matching (this is the input
                 file used in the romanisim simulation step).

Outputs:
    - matched_results.txt: Table of matched sources.
    - soi.txt: Table of sources of interest (outlier samples).
    - photoz_vs_truez.png: Plot of photometric vs. true redshift.
    - magnitude_histograms.png: Histograms of AB magnitudes for each filter.
    - outlier_seds: Directory containing SED plots for selected outlier sources.

Requires:
    - astropy
    - numpy
    - matplotlib

"""

import argparse
import sys
from pathlib import Path

import astropy.units as u
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack

# Roman filter effective wavelengths (microns)
ROMAN_FILTERS = {
    "F062": 0.620,
    "F087": 0.869,
    "F106": 1.060,
    "F129": 1.292,
    "F146": 1.464,
    "F158": 1.577,
    "F184": 1.842,
    "F213": 2.125,
}


def save_current_plot(filename, dpi=150):
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()


def compute_statistics(z_true, z_phot):
    if len(z_true) == 0:
        return {
            "delta_z": np.array([]),
            "nmad": np.nan,
            "mad": np.nan,
            "bias": np.nan,
            "scatter": np.nan,
            "outlier_frac": np.nan,
            "conf68": np.nan,
            "conf95": np.nan,
        }
    delta_z = z_phot - z_true
    # Robust scatter calculation normalized by (1 + z)
    nmad = 1.48 * np.median(np.abs(delta_z - np.median(delta_z)) / (1 + z_true))

    # Original calculations
    mad = np.median(np.abs(delta_z - np.median(delta_z)))
    bias = np.mean(delta_z)
    scatter = np.std(delta_z)

    # The official Roman outlier threshold
    outlier_frac = np.mean(np.abs(delta_z) > 0.15 * (1 + z_true))

    # Percentiles
    conf68 = np.percentile(np.abs(delta_z), 68)
    conf95 = np.percentile(np.abs(delta_z), 95)

    return {
        "delta_z": delta_z,
        "nmad": nmad,
        "mad": mad,
        "bias": bias,
        "scatter": scatter,
        "outlier_frac": outlier_frac,
        "conf68": conf68,
        "conf95": conf95,
    }


def plot_individual_seds(
    catalog_table, results_table, selected_info, output_dir="outlier_seds"
):
    """
    Finalized SED plotter with color-coded titles and specific line styles.
    True Redshift elements: Royal Blue
    Photometric Redshift elements: Crimson
    Lyman Breaks: Dashed (--) | Balmer Breaks: Dotted (:)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for info in selected_info:
        label = info["label"]
        region_name = info["region"]

        cat_row = catalog_table[catalog_table["label"] == label][0]
        res_row = results_table[results_table["matched_label"] == label][0]

        # Metadata extraction
        z_true, z_phot = cat_row["redshift_true"], res_row["photoz"]
        chi2 = res_row["photoz_gof"]

        waves, fluxes, errors, snr_list = [], [], [], []
        for f_name, f_wave in ROMAN_FILTERS.items():
            if f_name in cat_row.colnames:
                waves.append(f_wave)
                f_val = res_row.get(f"segment_{f_name.lower()}_flux")
                fluxes.append(f_val)
                err_val = res_row.get(f"segment_{f_name.lower()}_flux_err", f_val * 0.1)
                errors.append(err_val)
                if err_val > 0:
                    snr_list.append(f_val / err_val)
                else:
                    breakpoint()

        avg_snr = np.mean(snr_list) if snr_list else 0.0

        fig, ax = plt.subplots(figsize=(8, 6))

        # 1. Photometry and Spectral Breaks
        ax.errorbar(
            waves,
            fluxes,
            yerr=errors,
            fmt="o",
            color="black",
            ecolor="gray",
            capsize=3,
            zorder=10,
        )
        ax.plot(waves, fluxes, color="black", alpha=0.1, ls="-", zorder=9)

        # z_true: Royal Blue | Lyman=dashed, Balmer=dotted
        ax.axvline(
            0.1216 * (1 + z_true),
            color="royalblue",
            ls="--",
            alpha=0.8,
            label=r"Lyman break ($z_{true}$)",
        )
        ax.axvline(
            0.3646 * (1 + z_true),
            color="royalblue",
            ls=":",
            alpha=0.8,
            label=r"Balmer break ($z_{true}$)",
        )

        # z_phot: Crimson | Lyman=dashed, Balmer=dotted
        ax.axvline(
            0.1216 * (1 + z_phot),
            color="crimson",
            ls="--",
            alpha=0.8,
            label=r"Lyman break ($z_{phot}$)",
        )
        ax.axvline(
            0.3646 * (1 + z_phot),
            color="crimson",
            ls=":",
            alpha=0.8,
            label=r"Balmer break ($z_{phot}$)",
        )

        # 2. THE TITLE SOLUTION
        # We use suptitle for the top line and ax.set_title for the subtitle to manage spacing
        fig.suptitle(f"Source {label}", y=0.95, fontsize=12, ha="center")

        # boxstyle padding=0.6 provides the wider margins seen in your target image
        bbox_props = dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor="lightgray",
            alpha=0.8,
        )

        # Line 1: z_true creates the box background
        # We add two newlines to ensure the box height accommodates the second line comfortably
        ax.text(
            0.04,
            0.94,
            f"$z_{{true}}$: {z_true:.2f}\n\n ",
            color="royalblue",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            bbox=bbox_props,
        )

        # Line 2: z_phot positioned precisely within the existing box
        ax.text(
            0.04,
            0.87,
            f"$z_{{phot}}$: {z_phot:.2f}",
            color="crimson",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )

        # Subtitle
        subtitle = f"(Region: {region_name}, $\chi^2$={chi2:.2f}, <SNR>={avg_snr:.1f})"
        ax.set_title(subtitle, fontsize=10, pad=10, style="italic", loc="center")

        # 3. Scale and Style
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 20)  # Headroom for legend and title

        ax.legend(fontsize="small", loc="upper right", framealpha=0.8)
        ax.set_xlabel(r"Wavelength ($\mu$m)")
        ax.set_ylabel("Flux (mJy)")
        ax.grid(True, which="both", ls="-", alpha=0.05)

        plt.savefig(
            out_path / f"sed_{region_name.lower()}_{label}.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()


def plot_results(save_path=None, ref_cat_filename="romanisim_input_catalog.parquet"):
    rng = np.random.default_rng(42)  # For reproducibility of random selections

    results = Table.read("matched_results.txt", format="ascii.fixed_width_two_line")
    cat_full = Table.read(ref_cat_filename, format="parquet")

    # 1. Initial Processing
    min_sep = 0.1
    sep_mask = np.array(results["separation_arcsec"]) <= min_sep
    filtered = results[sep_mask]
    z_true = np.array(filtered["matched_redshift_true"])
    z_phot = np.array(filtered["photoz"])
    labels = np.array(filtered["matched_label"])

    stats = compute_statistics(z_true, z_phot)
    outlier_mask = np.abs(z_phot - z_true) > 0.15 * (1 + z_true)

    # 2. Precise Region Definitions (zt_min, zp_min, width, height)
    # These match the orange boxes in the provided screenshot
    box_defs = {
        "R1": (0.4, 5.1, 2.0, 2.8),  # Top Left: High z_phot / Low z_true
        "R2": (0.1, 3.9, 3.0, 0.9),  # Mid Left: Balmer/Lyman confusion
        "R3": (0.1, 2.0, 1.15, 0.2),  # Lower Left
        "R4": (5.8, 4.3, 2.0, 0.3),  # Mid Right: High-z undershooting
        "R5": (3.2, 2.0, 4.5, 0.2),  # Lower Right: Low-z overshooting
        "R6": (3.1, 0.2, 4.7, 1.5),  # Bottom Right: Massive high-to-low z failure
    }

    plt.figure(figsize=(10, 10))

    # 3. Scatter Plot Layers
    plt.scatter(
        z_true[~outlier_mask],
        z_phot[~outlier_mask],
        s=5,
        alpha=0.3,
        color="blue",
        edgecolors="none",
    )
    plt.scatter(
        z_true[outlier_mask],
        z_phot[outlier_mask],
        s=8,
        alpha=0.7,
        color="lightgray",
        edgecolors="none",
    )

    selected_info = []
    ax = plt.gca()

    # 4. Draw Regions and Sample Points
    for r_id, (zt_min, zp_min, w, h) in box_defs.items():
        # Draw the orange rounded rectangles
        rect = patches.FancyBboxPatch(
            (zt_min, zp_min),
            w,
            h,
            boxstyle="round,pad=0.1",
            linewidth=2,
            edgecolor="orange",
            facecolor="none",
            alpha=0.8,
        )
        ax.add_patch(rect)

        # Region Labeling
        plt.text(
            zt_min + w / 2,
            zp_min + h / 2,
            r_id,
            fontsize=24,
            fontweight="bold",
            ha="center",
            va="center",
            alpha=0.8,
        )

        # Logic for selecting 10 random sources for SED plotting
        mask = (
            (z_true >= zt_min)
            & (z_true <= zt_min + w)
            & (z_phot >= zp_min)
            & (z_phot <= zp_min + h)
        )
        indices = np.where(mask)[0]
        if len(indices) > 0:
            sample_count = min(3, len(indices))
            selected_idx = rng.choice(indices, sample_count, replace=False)

            plt.scatter(
                z_true[selected_idx],
                z_phot[selected_idx],
                color="red",
                s=40,
                edgecolors="white",
                zorder=10,
            )
            for idx in selected_idx:
                plt.text(
                    z_true[idx] + 0.05,
                    z_phot[idx],
                    str(labels[idx]),
                    color="red",
                    fontsize=7,
                    fontweight="bold",
                )
                selected_info.append({"label": labels[idx], "region": r_id})

    # 5. Original Reference Lines & Branding
    z_range = np.linspace(0, 8, 100)
    plt.plot([0, 8], [0, 8], "k--", alpha=0.8, label="1:1 line")
    plt.plot(
        z_range,
        z_range + 0.15 * (1 + z_range),
        "r:",
        linewidth=1.5,
        label="15% Boundary",
    )
    plt.plot(z_range, z_range - 0.15 * (1 + z_range), "r:", linewidth=1.5)

    plt.xlabel("True Redshift ($z_{true}$)")
    plt.ylabel("Photometric Redshift ($z_{phot}$)")
    plt.title(
        'L4 Data Release Validation: Photo-z vs True-z\n(Combined Filters, Sep $\leq$ 0.1")'
    )

    # Original Stats Box
    stats_text = (
        f"Total N: {len(z_true)}\n$\sigma_{{NMAD}}$: {stats.get('nmad', 0):.4f}\n"
        f"Bias: {stats['bias']:.4f}\nOutlier Frac: {stats['outlier_frac']:.2%}"
    )
    plt.gca().text(
        0.05,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.legend(loc="lower right")
    plt.xlim(0, 8)
    plt.ylim(0, 8)
    plt.grid(alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(Path(save_path), dpi=150)
        plt.close()

    # Run diagnostic SED generator (defined in previous steps)
    plot_individual_seds(cat_full, filtered, selected_info)


def plot_magnitude_histograms(
    ref_cat_filename="romanisim_input_catalog.parquet",
    save_path=None,
):
    cat = Table.read(ref_cat_filename, format="parquet")
    flux_cols = [col for col in cat.colnames if col.startswith("F")]
    ncols = 2
    nrows = (len(flux_cols) + 1) // ncols
    _, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for i, flux_col in enumerate(flux_cols):
        ax = axes[i // ncols][i % ncols]
        flux = cat[flux_col]
        positive = flux > 0
        flux = flux[positive]
        abmag = flux.to(u.ABmag, u.zero_point_flux(3631.1 * u.Jy)).value
        ax.hist(abmag, bins=40, alpha=0.7, color="C0")
        ax.set_xlim(10, 40)
        ax.set_title(f"{flux_col}")
        ax.set_xlabel("AB Magnitude")
        ax.set_ylabel("Count")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def run_matching(catalog_list, ref_cat_filename="romanisim_input_catalog.parquet"):
    # 1. Load Target Data (Photometric Catalog)
    target_tables = [Table.read(f, format="parquet") for f in catalog_list]
    all_targets = vstack(target_tables)

    # 2. Load Input Catalog (Truth Catalog)
    catalog_table = Table.read(ref_cat_filename, format="parquet")

    # Pre-filter catalog for valid entries
    valid = (
        ~np.isnan(catalog_table["ra"])
        & ~np.isnan(catalog_table["dec"])
        & (catalog_table["type"] != "PSF")
    )
    filtered_cat = catalog_table[valid]

    # 3. Perform Coordinate Matching
    targets_coords = SkyCoord(ra=all_targets["ra"], dec=all_targets["dec"])
    catalog_coords = SkyCoord(
        ra=filtered_cat["ra"] * u.deg, dec=filtered_cat["dec"] * u.deg
    )
    idx, sep2d, _ = targets_coords.match_to_catalog_sky(catalog_coords)

    # 4. Build Output Table
    output = Table()
    output["matched_label"] = filtered_cat["label"][idx]
    output["target_ra"] = all_targets["ra"]
    output["target_dec"] = all_targets["dec"]
    output["photoz"] = all_targets["photoz"]
    output["photoz_gof"] = all_targets["photoz_gof"]
    output["photoz_sed"] = all_targets["photoz_sed"]
    output["matched_redshift_true"] = filtered_cat["redshift_true"][idx]
    output["separation_arcsec"] = sep2d.arcsec

    # 5. Dynamically Add Fluxes and Errors
    roman_filters = ["f062", "f087", "f106", "f129", "f146", "f158", "f184", "f213"]

    for f in roman_filters:
        flux_col = f"segment_{f}_flux"
        err_col = f"segment_{f}_flux_err"

        if flux_col in all_targets.colnames:
            output[flux_col] = all_targets[flux_col]
        if err_col in all_targets.colnames:
            output[err_col] = all_targets[err_col]

    # 6. Save Results
    output.write(
        "matched_results.txt", format="ascii.fixed_width_two_line", overwrite=True
    )


def select_soi_by_standard_threshold(
    results_filename="matched_results.txt",
    soi_filename="soi.txt",
    min_sep=0.1,
):
    results = Table.read(results_filename, format="ascii.fixed_width_two_line")
    sep_mask = np.array(results["separation_arcsec"]) <= min_sep
    filtered = results[sep_mask]

    z_true = np.array(filtered["matched_redshift_true"])
    z_phot = np.array(filtered["photoz"])

    # Official Level 4 outlier definition
    outlier_mask = np.abs(z_phot - z_true) > 0.15 * (1 + z_true)

    soi = filtered[outlier_mask]
    soi.write(soi_filename, format="ascii.fixed_width_two_line", overwrite=True)


def main():
    parser = argparse.ArgumentParser(
        description="Assess results: run matching, select SOI, plot results and histograms."
    )
    parser.add_argument(
        "cat_files",
        nargs="+",
        help="List of catalog files (use shell expansion, e.g. r00001_r0_full_*y[0-9][0-9]_cat.parquet)",
    )
    parser.add_argument(
        "--ref-cat",
        required=True,
        help="Reference catalog file (truth catalog) for matching (e.g. romanisim_input_catalog.parquet)",
        default="romanisim_input_catalog.parquet",
    )
    args = parser.parse_args()

    soi_filename = "soi.txt"

    # get list of all catalog files matching the pattern
    cat_files = args.cat_files

    if not cat_files:
        print(f"No files matched the pattern: {args.pattern}", file=sys.stderr)
        sys.exit(1)

    run_matching(cat_files, ref_cat_filename=args.ref_cat)
    select_soi_by_standard_threshold(min_sep=0.1, soi_filename=soi_filename)
    plot_results(save_path="photoz_vs_truez.png", ref_cat_filename=args.ref_cat)
    plot_magnitude_histograms(
        save_path="magnitude_histograms.png", ref_cat_filename=args.ref_cat
    )


if __name__ == "__main__":
    main()
