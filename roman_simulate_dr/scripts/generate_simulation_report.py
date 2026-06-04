import argparse
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

TIMESTAMP_PREFIX_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3,6})?)"
)
FINISHED_RE = re.compile(
    r"\[(?P<filename>[^\]]+)\]\s+Finished in (?P<seconds>\d+(?:\.\d+)?)\."
)
FAILED_RE = re.compile(
    r"\[(?P<filename>[^\]]+)\]\s+FAILED after (?P<seconds>\d+(?:\.\d+)?)\."
)
FILTER_RE = re.compile(r"_f(?P<filter>\d{3})_", re.IGNORECASE)
CATALOG_PATH_RE = re.compile(r"Final concatenated catalog saved to '(?P<path>[^']+)'\.")
FLUX_CATALOG_RE = re.compile(r"using flux catalog:\s*(?P<path>\S+)")
TOTAL_SOURCES_RE = re.compile(r"Total sources:\s*(?P<count>\d+)")
DONE_RE = re.compile(r"^DONE\. Results in (?P<path>.+)$")


@dataclass
class TimingRecord:
    filename: str
    seconds: float
    timestamp: datetime | None
    filter_name: str | None


def parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_PREFIX_RE.match(line)
    if not match:
        return None
    ts_text = match.group("ts")
    ts_formats = ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S")
    for fmt in ts_formats:
        try:
            return datetime.strptime(ts_text, fmt)
        except ValueError:
            continue
    return None


def format_hms(seconds: float) -> str:
    whole_seconds = int(round(seconds))
    hours, rem = divmod(whole_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_seconds(seconds: float) -> str:
    return f"{seconds:,.2f} s ({format_hms(seconds)})"


def safe_stats(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    median = statistics.median(values)
    minimum = min(values)
    maximum = max(values)
    return mean, median, minimum, maximum


def parse_filter_name(filename: str) -> str | None:
    match = FILTER_RE.search(filename)
    if not match:
        return None
    return f"F{match.group('filter')}"


def build_report(
    *,
    log_text: str,
    log_path: Path,
    output_path: Path,
    command: str,
    obs_plan: str,
    exit_code: int,
) -> str:
    records: list[TimingRecord] = []
    failed_records: list[TimingRecord] = []
    all_timestamps: list[datetime] = []

    catalog_path = None
    flux_catalog_path = None
    total_sources = None
    output_dir = None
    catalog_start = None
    catalog_end = None
    notes: list[str] = []

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        timestamp = parse_timestamp(line)
        if timestamp is not None:
            all_timestamps.append(timestamp)

        if "Generating catalog at RA=" in line and timestamp is not None:
            catalog_start = timestamp
        if (
            "Updated catalog with roman_photoz fluxes saved to" in line
            and timestamp is not None
        ):
            catalog_end = timestamp
        elif "Final concatenated catalog saved to" in line and timestamp is not None:
            catalog_end = timestamp

        catalog_match = CATALOG_PATH_RE.search(line)
        if catalog_match:
            catalog_path = catalog_match.group("path")

        flux_match = FLUX_CATALOG_RE.search(line)
        if flux_match:
            flux_catalog_path = flux_match.group("path")

        total_match = TOTAL_SOURCES_RE.search(line)
        if total_match:
            total_sources = int(total_match.group("count"))

        done_match = DONE_RE.search(line)
        if done_match:
            output_dir = done_match.group("path").strip()

        finished_match = FINISHED_RE.search(line)
        if finished_match:
            filename = finished_match.group("filename")
            records.append(
                TimingRecord(
                    filename=filename,
                    seconds=float(finished_match.group("seconds")),
                    timestamp=timestamp,
                    filter_name=parse_filter_name(filename),
                )
            )

        failed_match = FAILED_RE.search(line)
        if failed_match:
            filename = failed_match.group("filename")
            failed_records.append(
                TimingRecord(
                    filename=filename,
                    seconds=float(failed_match.group("seconds")),
                    timestamp=timestamp,
                    filter_name=parse_filter_name(filename),
                )
            )

        if "archive is unstable" in line.lower():
            notes.append(line)
        if line.startswith("User defined LEPHAREDIR"):
            notes.append(line)
        if line.startswith("User defined LEPHAREWORK"):
            notes.append(line)

    status = "success" if exit_code == 0 else "failed"
    generated_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")

    report_lines = [
        "# Simulation Run Report",
        "",
        f"- Generated (UTC): {generated_utc}",
        "",
        "## Run summary",
        f"- Command: `{command}`",
        f"- Observation plan: `{obs_plan}`" if obs_plan else "- Observation plan: n/a",
        f"- Exit code: `{exit_code}` ({status})",
        f"- Log file: `{log_path}`",
        f"- Report file: `{output_path}`",
    ]

    if output_dir:
        report_lines.append(f"- Output directory: `{output_dir}`")

    if all_timestamps:
        wall_seconds = (all_timestamps[-1] - all_timestamps[0]).total_seconds()
        report_lines.append(
            f"- Wall time observed in logs: **{format_seconds(wall_seconds)}**"
        )

    report_lines.extend(
        [
            "",
            "## Catalog summary",
            f"- Final catalog: `{catalog_path}`"
            if catalog_path
            else "- Final catalog: n/a",
            (
                f"- Total sources: **{total_sources:,}**"
                if total_sources is not None
                else "- Total sources: n/a"
            ),
            (
                f"- Flux catalog used: `{flux_catalog_path}`"
                if flux_catalog_path
                else "- Flux catalog used: not detected"
            ),
        ]
    )

    if catalog_start and catalog_end:
        catalog_elapsed = (catalog_end - catalog_start).total_seconds()
        report_lines.append(
            f"- Catalog generation+update elapsed: **{format_seconds(catalog_elapsed)}**"
        )

    report_lines.extend(
        [
            "",
            "## Detector timing summary",
            f"- Completed detector files: **{len(records)}**",
            f"- Failed detector files: **{len(failed_records)}**",
        ]
    )

    if records:
        durations = [record.seconds for record in records]
        mean, median, minimum, maximum = safe_stats(durations)
        total = sum(durations)
        report_lines.extend(
            [
                f"- Aggregate detector compute time: **{format_seconds(total)}**",
                f"- Mean detector runtime: **{format_seconds(mean)}**",
                f"- Median detector runtime: **{format_seconds(median)}**",
                (
                    f"- Fastest / slowest detector runtime: "
                    f"**{format_seconds(minimum)} / {format_seconds(maximum)}**"
                ),
            ]
        )

        completed_with_timestamp = [r for r in records if r.timestamp is not None]
        if completed_with_timestamp:
            completed_with_timestamp.sort(key=lambda rec: rec.timestamp)
            first_done = completed_with_timestamp[0].timestamp
            last_done = completed_with_timestamp[-1].timestamp
            report_lines.extend(
                [
                    f"- First detector completion: `{first_done}`",
                    f"- Last detector completion: `{last_done}`",
                ]
            )
    else:
        report_lines.append("- No detector completion timings were found in the log.")

    report_lines.extend(["", "## Per-filter timing breakdown"])
    if records:
        grouped: dict[str, list[float]] = defaultdict(list)
        for record in records:
            if record.filter_name:
                grouped[record.filter_name].append(record.seconds)

        for filter_name in sorted(grouped, key=lambda name: int(name[1:])):
            values = grouped[filter_name]
            mean, median, minimum, maximum = safe_stats(values)
            report_lines.append(
                f"- `{filter_name}`: n={len(values)}, "
                f"mean={format_seconds(mean)}, "
                f"median={format_seconds(median)}, "
                f"min/max={format_seconds(minimum)} / {format_seconds(maximum)}"
            )
    else:
        report_lines.append("- n/a")

    report_lines.extend(["", "## Notes"])
    if notes:
        unique_notes = []
        seen = set()
        for note in notes:
            if note not in seen:
                seen.add(note)
                unique_notes.append(note)
        report_lines.extend(f"- {note}" for note in unique_notes)
    else:
        report_lines.append("- None detected.")

    return "\n".join(report_lines) + "\n"


def _cli():
    parser = argparse.ArgumentParser(
        description="Generate a markdown summary report from rdr-simulate-data logs."
    )
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--command", default="rdr-simulate-data")
    parser.add_argument("--obs-plan", default="")
    parser.add_argument("--exit-code", default=0, type=int)
    args = parser.parse_args()

    log_text = args.log_file.read_text(encoding="utf-8", errors="replace")
    report = build_report(
        log_text=log_text,
        log_path=args.log_file,
        output_path=args.output_file,
        command=args.command,
        obs_plan=args.obs_plan,
        exit_code=args.exit_code,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    _cli()
