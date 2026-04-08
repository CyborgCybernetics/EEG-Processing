#!/usr/bin/env python3
"""
Quality-control pipeline: one row per recording file ("trial") with all QC metrics
and pass/fail flags based on configurable thresholds.

Does NOT compute band-ratio event stats — use band_ratio_pipeline.py for that.

Depends on: eeg_qc_pipeline.py (run_eeg_qc, per-file metrics).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eeg_qc_pipeline import run_eeg_qc  # noqa: E402


def metrics_wide_to_trials_long(metrics_wide: "pd.DataFrame") -> "pd.DataFrame":
    """Wide (metrics x files) -> long (one row per filename)."""
    import pandas as pd

    # columns = filenames, index = metric names
    long_df = metrics_wide.T.sort_index().reset_index()
    long_df = long_df.rename(columns={"index": "filename"})
    return long_df


def evaluate_pass_row(row, *, min_percent_clean, min_duration_sec, max_line_noise_ratio):
    reasons: list[str] = []
    pc = row.get("Percent_Clean")
    try:
        pc = float(pc)
    except (TypeError, ValueError):
        pc = float("nan")
    if not (pc >= min_percent_clean):
        reasons.append(f"Percent_Clean={pc:.4f} < {min_percent_clean}")

    try:
        dur = float(row.get("Duration_Sec", float("nan")))
    except (TypeError, ValueError):
        dur = float("nan")
    if not (dur >= min_duration_sec):
        reasons.append(f"Duration_Sec={dur:.2f} < {min_duration_sec}")

    ln = row.get("Line_Noise_Ratio")
    try:
        ln = float(ln)
        if ln == ln and ln > max_line_noise_ratio:
            reasons.append(f"Line_Noise_Ratio={ln:.4f} > {max_line_noise_ratio}")
    except (TypeError, ValueError):
        pass

    if reasons:
        return False, "; ".join(reasons)
    return True, ""


def run_qc_pipeline(args: argparse.Namespace) -> dict:
    """Run QC folder scan, build trials table, apply pass rules, write CSV + JSON summary."""
    folder = os.path.abspath(args.folder)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== QC: scanning {folder} (fs={args.fs} Hz) ===")
    metrics_wide = run_eeg_qc(folder, fs=float(args.fs), n_channels=args.n_channels)
    trials = metrics_wide_to_trials_long(metrics_wide)

    passes: list[bool] = []
    reasons: list[str] = []
    for _, row in trials.iterrows():
        ok, why = evaluate_pass_row(
            row,
            min_percent_clean=args.min_percent_clean,
            min_duration_sec=args.min_duration_sec,
            max_line_noise_ratio=args.max_line_noise_ratio,
        )
        passes.append(ok)
        reasons.append(why)

    trials["qc_pass"] = passes
    trials["qc_fail_reason"] = reasons

    stem = args.prefix
    csv_path = os.path.join(out_dir, f"{stem}_qc_trials.csv")
    trials.to_csv(csv_path, index=False)

    n_pass = int(sum(passes))
    n_fail = len(passes) - n_pass
    summary = {
        "folder": folder,
        "n_files": len(trials),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "rules": {
            "min_percent_clean": args.min_percent_clean,
            "min_duration_sec": args.min_duration_sec,
            "max_line_noise_ratio": args.max_line_noise_ratio,
        },
        "output_csv": csv_path,
    }
    json_path = os.path.join(out_dir, f"{stem}_qc_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Pass: {n_pass}  |  Fail: {n_fail}  (rules: min clean %, min duration s, max line noise)")

    return {"trials_csv": csv_path, "summary_json": json_path, "summary": summary}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=str, required=True, help="Folder with BrainFlow/OpenBCI files.")
    p.add_argument("--output-dir", type=str, required=True, help="Where to write CSV + JSON.")
    p.add_argument("--prefix", type=str, default="qc_run", help="Output filename stem.")
    p.add_argument("--fs", type=float, default=250.0, help="Sampling rate (Hz).")
    p.add_argument("--n-channels", type=int, default=8, help="EEG channel count.")
    p.add_argument(
        "--min-percent-clean",
        type=float,
        default=0.40,
        metavar="P",
        help="Pass if Percent_Clean >= P (0–1). Default 0.40.",
    )
    p.add_argument(
        "--min-duration-sec",
        type=float,
        default=10.0,
        help="Pass if recording duration (s) >= this. Default 10.",
    )
    p.add_argument(
        "--max-line-noise-ratio",
        type=float,
        default=0.30,
        metavar="R",
        help="Pass if Line_Noise_Ratio <= R. Default 0.30.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_qc_pipeline(args)


if __name__ == "__main__":
    main()
