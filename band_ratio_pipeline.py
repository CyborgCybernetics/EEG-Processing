#!/usr/bin/env python3
"""
Single entry point: band-power ratios (3 windows), per-channel/band stats, plots, cross-channel tests.

Mode A — raw EEG folder (BrainFlow/OpenBCI): runs eeg_band_stats.py logic (Welch bands, CAR+1–45 Hz).
Mode B — existing trial-level CSV: skips extraction, only analyzes/plots.

Required Python files (same directory): eeg_band_stats.py, eeg_qc_pipeline.py,
plot_band_trial_triplets.py, analyze_band_ratio_crosschannel.py

Optional: scikit-learn (for GMM/BIC in cross-channel step).

Example (raw data):
  python band_ratio_pipeline.py --raw-folder "D:\\data\\eeg" --output-dir "D:\\out\\run1" --prefix myrun

Example (trial CSV only):
  python band_ratio_pipeline.py --trial-csv "D:\\out\\run1\\myrun_trial_level.csv" --output-dir "D:\\out\\run1"
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

from analyze_band_ratio_crosschannel import run_crosschannel_analysis
from eeg_band_stats import DEFAULT_INTERVAL_MAP_STR, run_band_stats
from plot_band_trial_triplets import run_all_channels_plots

TRIAL_REQUIRED_COLUMNS = frozenset(
    {
        "filename",
        "trial_id",
        "event_time_sec",
        "channel",
        "band",
        "power_rest",
        "power_pre_movement",
        "power_post_movement",
        "log_ratio_pre_vs_rest",
        "log_ratio_post_vs_rest",
        "log_ratio_post_vs_pre",
    }
)


def validate_trial_csv(path: str) -> None:
    import pandas as pd

    df = pd.read_csv(path, nrows=0)
    miss = sorted(TRIAL_REQUIRED_COLUMNS - set(df.columns))
    if miss:
        raise SystemExit(
            "Trial CSV is missing required columns:\n  "
            + ", ".join(miss)
            + "\nExpected a *_trial_level.csv from eeg_band_stats.py."
        )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--raw-folder",
        type=str,
        help="Folder with BrainFlow-RAW_*.csv / OpenBCI-RAW-*.txt (runs band extraction first).",
    )
    src.add_argument(
        "--trial-csv",
        type=str,
        help="Existing *_trial_level.csv (skip raw processing).",
    )

    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="All outputs (CSVs, PNGs, report) go here.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="band_ratio_run",
        help="Filename stem for outputs (default: band_ratio_run).",
    )
    p.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only statistics / cross-channel CSVs, no PNGs.",
    )
    p.add_argument(
        "--skip-crosschannel",
        action="store_true",
        help="Skip GMM/Friedman/pairwise (faster if you only want summary_stats).",
    )
    p.add_argument(
        "--pairwise-nominal-p",
        type=float,
        default=0.05,
        metavar="P",
        help="Exploratory pairwise Holm when Friedman raw p < P (default 0.05). Use 0 to disable.",
    )

    # Raw-only / eeg_band_stats passthrough
    raw = p.add_argument_group("Raw EEG (only with --raw-folder)")
    raw.add_argument("--fs", type=float, default=250.0)
    raw.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=list(range(1, 9)),
        help="1-based channels (default: 1 2 3 4 5 6 7 8).",
    )
    raw.add_argument("--name-contains", type=str, default=None)
    raw.add_argument("--events-csv", type=str, default=None)
    raw.add_argument("--events-from-end-sec", type=float, default=10.0)
    raw.add_argument(
        "--interval-map",
        type=str,
        default=DEFAULT_INTERVAL_MAP_STR,
        help="Block:sec map for auto events, or empty string for uniform spacing.",
    )
    raw.add_argument("--include-t0-event", action="store_true")
    raw.add_argument(
        "--rest-baseline-win",
        type=float,
        nargs=2,
        default=[-2.0, -0.5],
        metavar=("START", "END"),
    )
    raw.add_argument(
        "--pre-movement-win",
        type=float,
        nargs=2,
        default=[-0.5, 0.0],
        metavar=("START", "END"),
    )
    raw.add_argument(
        "--post-movement-win",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        metavar=("START", "END"),
    )

    return p.parse_args()


def build_eeg_namespace(ns: argparse.Namespace):
    """Namespace compatible with eeg_band_stats.run_band_stats."""
    import argparse as ap

    return ap.Namespace(
        folder=ns.raw_folder,
        name_contains=ns.name_contains,
        events_csv=ns.events_csv,
        events_from_end_sec=ns.events_from_end_sec,
        interval_map=ns.interval_map,
        include_t0_event=ns.include_t0_event,
        fs=ns.fs,
        channels=ns.channels,
        rest_baseline_win=ns.rest_baseline_win,
        pre_movement_win=ns.pre_movement_win,
        post_movement_win=ns.post_movement_win,
        output_prefix=ns.prefix,
        out_dir=ns.output_dir,
    )


def run_pipeline(args: argparse.Namespace, *, verbose: bool = True) -> dict:
    """
    Run the full pipeline from an argparse.Namespace (same fields as CLI).
    Use this from testing_script.py or other drivers.
    Returns the report dict (also writes JSON to disk).
    """
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    report: dict = {"output_dir": out_dir, "prefix": args.prefix}

    if getattr(args, "raw_folder", None):
        eeg_ns = build_eeg_namespace(args)
        if verbose:
            print("=== Step 1: Extract band powers & trial-level ratios (eeg_band_stats) ===")
        paths = run_band_stats(eeg_ns)
        trial_csv = paths["trial_level"]
        report["eeg_band_stats"] = paths
    else:
        trial_csv = os.path.abspath(args.trial_csv)
        validate_trial_csv(trial_csv)
        report["trial_csv"] = trial_csv

    if not args.skip_crosschannel:
        if verbose:
            print("=== Step 2: Cross-channel analysis (GMM/BIC, Friedman, pairwise) ===")
        p_nom = args.pairwise_nominal_p
        if p_nom is not None and p_nom <= 0:
            p_nom = None
        cc = run_crosschannel_analysis(
            trial_csv,
            output_dir=out_dir,
            prefix=args.prefix,
            pairwise_nominal_p=p_nom,
            verbose=verbose,
        )
        report["crosschannel"] = cc
    else:
        report["crosschannel"] = None

    if not args.skip_plots:
        if verbose:
            print("=== Step 3: Plots (8 channels x 5 bands, 3 ln-ratios per event) ===")
        plot_paths = run_all_channels_plots(
            trial_csv,
            out_dir,
            file_stem=f"{args.prefix}_trial_level",
        )
        report["plots"] = plot_paths
        if verbose:
            for path in plot_paths:
                print(path)
    else:
        report["plots"] = []

    report_path = os.path.join(out_dir, f"{args.prefix}_pipeline_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["report_json"] = report_path
    if verbose:
        print("=== Done ===")
        print(report_path)
    return report


def main():
    args = parse_args()
    run_pipeline(args, verbose=True)


if __name__ == "__main__":
    main()
