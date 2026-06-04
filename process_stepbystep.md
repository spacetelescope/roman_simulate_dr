# Process Step-by-Step: Produce the Final Dataset

## Goal

Generate the final processed Roman dataset products (L2/L3 + multiband outputs)
starting from an observation plan and optional `roman_photoz` flux catalog.

## Prerequisites

1. Install project dependencies:
   - `uv sync --all-extras`
2. Ensure these runtime commands are available:
   - `romanisim-make-image`
   - `strun`
   - `skycell_asn`
   - `multiband_asn`
3. Ensure input files exist:
   - `obs_plan.ecsv`
   - optional `roman_photoz_simulated_catalog.parquet`

## Step 1 — Set runtime paths

From repo root:

1. `export RDR_INPUT_PATH="$PWD/DATASET/INPUT"`
2. `export RDR_OUTPUT_PATH="$PWD/DATASET/OUTPUT"`
3. `mkdir -p "$RDR_OUTPUT_PATH"`

## Step 2 — Generate the Romanisim input catalog

Run:
`uv run rdr-generate-input-catalog --obs-plan obs_plan.ecsv --output-filename romanisim_input_catalog.parquet --flux-catalog roman_photoz_simulated_catalog.parquet --radius 0.3 --filter-list f062 f087 f106 f129 f146 f158 f184 f213`

Expected output:

- `$RDR_OUTPUT_PATH/romanisim_input_catalog.parquet`

## Step 3 — Simulate L1 detector data

Run: `uv run rdr-simulate-data obs_plan.ecsv`

What this does:

1. Regenerates/validates the input catalog.
2. Simulates `*_uncal.asdf` detector files.
3. Writes simulation logs and summary markdown.

Expected outputs:

- `$RDR_OUTPUT_PATH/*_uncal.asdf`
- `$RDR_OUTPUT_PATH/dr_logs_generate_input_catalog.log`
- `$RDR_OUTPUT_PATH/dr_logs_generate_simulated_l1_images.log`
- `$RDR_OUTPUT_PATH/rdr_simulation_summary_latest.md`

## Step 4 — Process L1 to final products (L2/L3 + multiband)

Run: `uv run rdr-process-data "$RDR_OUTPUT_PATH"`

This executes:

1. `roman_elp` on all `*_uncal.asdf`
2. skycell association creation
3. `roman_mos` mosaicking
4. `multiband_asn` generation
5. `romancal.step.MultibandCatalogStep`

Expected outputs in `$RDR_OUTPUT_PATH`:

- `*_cal.asdf` (L2)
- `*_coadd.asdf` (L3)
- `*_asn.json` and multiband association JSON files
- multiband catalog products

## Step 5 — Verify final dataset completeness

Check that these exist:

1. At least one `*_cal.asdf`
2. At least one `*_coadd.asdf`
3. Association JSON files (`*_asn.json` and `*r0_full*.json`)
4. Processing logs:
   - `dr_logs_elp.log`
   - `dr_logs_create_skycells_asn.log`
   - `dr_logs_mos.log`
   - `dr_logs_multiband_asn.log`
   - `dr_logs_multiband_catalog_step.log`

If all are present and logs contain no fatal errors, final dataset production is
complete.

## Step 6 — Generate visualization and diagnostic artifacts (recommended)

Run from repo root:

1. L3 mosaic image:
   - `uv run rdr-generate-mosaic \
       $RDR_OUTPUT_PATH/r00001_*_f062_coadd.asdf \
       --output $RDR_OUTPUT_PATH/l3_mosaic_f062`
2. Per-file comparison overlays:
   - `uv run python -m \
       roman_simulate_dr.scripts.visualization_utils.visualize_generic_coadd \
       $RDR_OUTPUT_PATH/r00001_*_f062_coadd.asdf \
       --show-sources`
3. Photo-z vs truth diagnostics:
   - `uv run python -m \
       roman_simulate_dr.scripts.plot_utils.plot_zphot_vs_ztrue \
       $RDR_OUTPUT_PATH/r00001_r0_full_*y[0-9][0-9]_cat.parquet \
       --ref-cat $RDR_OUTPUT_PATH/romanisim_input_catalog.parquet`

Expected visualization/diagnostic outputs:

- `$RDR_OUTPUT_PATH/l3_mosaic_f062.png`
- `$RDR_OUTPUT_PATH/*_comparison.png`
- `matched_results.txt`
- `soi.txt`
- `photoz_vs_truez.png`
- `magnitude_histograms.png`
- `outlier_seds/`
