#!/bin/bash
# Script to simulate Roman L1 data products for a predefined filter list.
#
# Usage: ./rdr_simulate_data.sh <obs_plan.ecsv>
#
# Environment Variables:
#   RDR_INPUT_PATH:  Directory containing obs plans and flux catalogs (default: .)
#   RDR_OUTPUT_PATH: Directory where catalogs and logs will be saved (default: .)

# Ensure pipeline command failures propagate.
set -o pipefail

# --- Argument Check ---
# Usage: ./rdr_simulate_data.sh <obs_plan.ecsv>
if [ -z "$1" ]; then
  echo "Usage: $0 <obs_plan.ecsv>"
  exit 1
fi

OBS_PLAN="$1"
OUTPUT_DIR="${RDR_OUTPUT_PATH:-.}"
mkdir -p "$OUTPUT_DIR"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="$OUTPUT_DIR/dr_logs_rdr_simulate_data_${RUN_TIMESTAMP}.log"
REPORT_FILE="$OUTPUT_DIR/rdr_simulation_summary_${RUN_TIMESTAMP}.md"
LATEST_REPORT="$OUTPUT_DIR/rdr_simulation_summary_latest.md"

# Capture all terminal output to a consolidated run log as well.
exec > >(tee -a "$RUN_LOG") 2>&1

# Filenames only (python handles the paths)
REF_FLUX_CAT="roman_photoz_simulated_catalog.parquet"
ROMANISIM_INPUT_CAT="romanisim_input_catalog.parquet"

echo "Running Simulation for $OBS_PLAN..."
echo "Detailed run log: $RUN_LOG"

status=0

# 1. Generate Input Catalog
python -m roman_simulate_dr.scripts.generate_input_catalog \
  --obs-plan "$OBS_PLAN" \
  --output-filename "$ROMANISIM_INPUT_CAT" \
  --flux-catalog "$REF_FLUX_CAT" \
  --radius 0.3 \
  2>&1 | tee "$OUTPUT_DIR/dr_logs_generate_input_catalog.log"
status=${PIPESTATUS[0]}

# 2. Generate L1 Images
if [ "$status" -eq 0 ]; then
  python -m roman_simulate_dr.scripts.generate_simulated_l1_images \
    --obs-plan "$OBS_PLAN" \
    --input-filename "$ROMANISIM_INPUT_CAT" \
    --sca-ids 1 2 10 11 \
    --max-workers 3 \
    2>&1 | tee "$OUTPUT_DIR/dr_logs_generate_simulated_l1_images.log"
  status=${PIPESTATUS[0]}
else
  echo "Input catalog generation failed; skipping L1 image simulation."
fi

# 3. Generate Simulation Report
python -m roman_simulate_dr.scripts.generate_simulation_report \
  --log-file "$RUN_LOG" \
  --output-file "$REPORT_FILE" \
  --command "rdr-simulate-data $OBS_PLAN" \
  --obs-plan "$OBS_PLAN" \
  --exit-code "$status"
report_status=$?

if [ "$report_status" -eq 0 ]; then
  cp "$REPORT_FILE" "$LATEST_REPORT"
  echo "Summary report: $REPORT_FILE"
  echo "Latest summary: $LATEST_REPORT"
else
  echo "WARNING: Failed to generate summary report at $REPORT_FILE"
fi

if [ "$status" -eq 0 ]; then
  echo "DONE. Results in $OUTPUT_DIR"
else
  echo "FAILED. See logs in $OUTPUT_DIR"
fi

exit "$status"
