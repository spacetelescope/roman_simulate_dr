#!/usr/bin/env python3
"""Plot catalog sources by dust_ebv as scatter or binned-mean maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Display sources from a parquet catalog using scatter points "
            "or a binned-mean dust_ebv image."
        )
    )
    parser.add_argument(
        "catalog",
        nargs="?",
        help="Full or relative path to a parquet catalog file.",
    )
    parser.add_argument(
        "--mode",
        choices=("scatter", "binned-mean"),
        default="scatter",
        help="Plot mode (default: scatter).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=150,
        help="Number of bins per axis for --mode binned-mean (default: 150).",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.0,
        help=(
            "Gaussian smoothing sigma in pixels for --mode binned-mean "
            "(default: 0, disabled)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, image is only shown.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (useful with --output).",
    )
    parser.add_argument(
        "--how-to-run",
        action="store_true",
        help="Show command examples and exit.",
    )

    args = parser.parse_args()
    if not args.catalog and not args.how_to_run:
        parser.exit(
            1,
            "Warning: missing required catalog path. "
            "Provide a full/relative path or use --how-to-run.\n",
        )
    return args


def print_run_helper() -> None:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    try:
        display_script = script_path.relative_to(project_root)
    except ValueError:
        display_script = script_path

    print("Run examples:")
    print(f"  uv run python {display_script} DATASET/OUTPUT/catalog.parquet")
    print(
        f"  uv run python {display_script} "
        "../DATASET/OUTPUT/catalog.parquet --mode binned-mean"
    )
    print(
        f"  uv run python {display_script} "
        "/full/path/to/catalog.parquet --output dust_map.png"
    )


def resolve_catalog_path(catalog_input: str) -> Path:
    candidate = Path(catalog_input).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    if not candidate.is_file():
        raise FileNotFoundError(
            f"Catalog file not found: {catalog_input}. "
            "Provide a valid full or relative path to an existing file."
        )

    return candidate.resolve()


def load_catalog(catalog_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(catalog_path, columns=["ra", "dec", "dust_ebv"])
    df = df.dropna(subset=["ra", "dec", "dust_ebv"])
    if df.empty:
        raise ValueError("No valid rows remain after dropping missing ra/dec/dust_ebv.")
    return df


def plot_scatter(ax: plt.Axes, df: pd.DataFrame) -> plt.Collection:
    return ax.scatter(
        df["ra"],
        df["dec"],
        c=df["dust_ebv"],
        s=10,
        cmap="viridis",
        linewidths=0,
    )


def smooth_grid_nan_aware(grid: np.ndarray, sigma: float) -> np.ndarray:
    valid = np.isfinite(grid)
    if not np.any(valid):
        return grid

    values = np.where(valid, grid, 0.0)
    smooth_values = gaussian_filter(values, sigma=sigma, mode="nearest")
    smooth_weights = gaussian_filter(valid.astype(float), sigma=sigma, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.divide(
            smooth_values,
            smooth_weights,
            out=np.full_like(grid, np.nan, dtype=float),
            where=smooth_weights > 0,
        )


def plot_binned_mean(
    ax: plt.Axes,
    df: pd.DataFrame,
    bins: int,
    gaussian_sigma: float = 0.0,
) -> plt.AxesImage:
    if bins < 2:
        raise ValueError("--bins must be >= 2 for binned-mean mode.")
    if gaussian_sigma < 0:
        raise ValueError("--gaussian-sigma must be >= 0.")

    ra = df["ra"].to_numpy()
    dec = df["dec"].to_numpy()
    dust = df["dust_ebv"].to_numpy()

    weighted_sum, ra_edges, dec_edges = np.histogram2d(
        ra,
        dec,
        bins=[bins, bins],
        weights=dust,
    )
    counts, _, _ = np.histogram2d(
        ra,
        dec,
        bins=[ra_edges, dec_edges],
    )
    mean_grid = np.divide(
        weighted_sum,
        counts,
        out=np.full_like(weighted_sum, np.nan, dtype=float),
        where=counts > 0,
    )
    if gaussian_sigma > 0:
        mean_grid = smooth_grid_nan_aware(mean_grid, sigma=gaussian_sigma)

    return ax.imshow(
        mean_grid.T,
        origin="lower",
        extent=[ra_edges[0], ra_edges[-1], dec_edges[0], dec_edges[-1]],
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
    )


def main() -> None:
    args = parse_args()

    if args.how_to_run:
        print_run_helper()
        return

    catalog_path = resolve_catalog_path(args.catalog)
    df = load_catalog(catalog_path)

    fig, ax = plt.subplots(figsize=(10, 8))

    if args.mode == "scatter":
        mappable = plot_scatter(ax, df)
        title = f"Sources color-coded by dust_ebv ({catalog_path.name})"
    else:
        mappable = plot_binned_mean(
            ax,
            df,
            args.bins,
            gaussian_sigma=args.gaussian_sigma,
        )
        title = f"Binned mean dust_ebv ({args.bins}x{args.bins}) from {catalog_path.name}"
        if args.gaussian_sigma > 0:
            title += f", Gaussian sigma={args.gaussian_sigma}"

    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label("dust_ebv")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if args.output is not None:
        fig.savefig(args.output, dpi=200)
        print(f"Saved plot: {args.output}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
