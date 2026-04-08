"""
Band-ratio pipeline only (Welch bands, event windows, ln-ratios, plots, cross-channel stats).
Does NOT run QC — use qc_testing_script.py first if you need pass/fail per file.

    python testing_script.py

Same as: python band_ratio_pipeline.py ...  |  See README.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIG — edit only this section (use raw strings r"..." on Windows)
# -----------------------------------------------------------------------------

# "raw"  = read BrainFlow/OpenBCI files from RAW_FOLDER and compute band stats first
# "trial" = skip extraction; use an existing *_trial_level.csv from a previous run
MODE = "raw"

# Where outputs go (CSVs, PNGs, pipeline_report.json). Folder is created if needed.
OUTPUT_DIR = r"C:\Users\Lab Member\Desktop\CC_DATA\pipeline_output"

# Start of every output filename, e.g. mystudy_trial_level.csv, mystudy_summary_stats.csv
PREFIX = "mystudy"

# --- Used when MODE == "raw" -------------------------------------------------
RAW_FOLDER = r"C:\Users\Lab Member\Desktop\CC_DATA"

# Sampling rate (Hz), channels to analyze (1-based, default all 8)
FS = 250.0
CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8]

# Optional: only files whose basename contains this string (case-insensitive)
NAME_CONTAINS = None  # e.g. "A35" or None for all files

# Auto-events from each file’s end: block spacing (A28:30, …) or empty string + EVENTS_FROM_END_SEC
# Leave empty string to use a fixed spacing for every file (seconds).
INTERVAL_MAP = "A28:30,A29:15,A30:10,A31:5,A32:30,A33:15,A34:10,A35:5"
EVENTS_FROM_END_SEC = 10.0

# Time windows (seconds) relative to each event (t = 0)
REST_BASELINE_WIN = (-2.0, -0.5)   # quiet baseline
PRE_MOVEMENT_WIN = (-0.5, 0.0)    # lead-up
POST_MOVEMENT_WIN = (0.0, 1.0)    # response after event

# --- Used when MODE == "trial" -----------------------------------------------
# Full path to a trial-level CSV (same schema as *_trial_level.csv from eeg_band_stats)
TRIAL_CSV = r"C:\Users\Lab Member\Desktop\CC_DATA\pipeline_output\mystudy_trial_level.csv"

# --- Pipeline toggles ---------------------------------------------------------
SKIP_PLOTS = False           # True = CSV only, no PNGs
SKIP_CROSSCHANNEL = False    # True = skip GMM/Friedman/pairwise (faster)
# Exploratory pairwise table when Friedman raw p < this (0 = disabled)
PAIRWISE_NOMINAL_P = 0.05

# -----------------------------------------------------------------------------
# End of config
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from band_ratio_pipeline import run_pipeline  # noqa: E402


def _build_args() -> argparse.Namespace:
    if MODE not in ("raw", "trial"):
        raise SystemExit('MODE must be "raw" or "trial".')

    ns = argparse.Namespace(
        output_dir=OUTPUT_DIR,
        prefix=PREFIX,
        skip_plots=SKIP_PLOTS,
        skip_crosschannel=SKIP_CROSSCHANNEL,
        pairwise_nominal_p=PAIRWISE_NOMINAL_P,
        fs=FS,
        channels=CHANNELS,
        name_contains=NAME_CONTAINS,
        events_csv=None,
        events_from_end_sec=EVENTS_FROM_END_SEC,
        interval_map=INTERVAL_MAP,
        include_t0_event=False,
        rest_baseline_win=list(REST_BASELINE_WIN),
        pre_movement_win=list(PRE_MOVEMENT_WIN),
        post_movement_win=list(POST_MOVEMENT_WIN),
    )

    if MODE == "raw":
        ns.raw_folder = os.path.abspath(RAW_FOLDER)
        ns.trial_csv = None
    else:
        ns.raw_folder = None
        ns.trial_csv = os.path.abspath(TRIAL_CSV)

    return ns


def main():
    args = _build_args()

    print(f"Mode: {MODE}  |  Output: {os.path.abspath(OUTPUT_DIR)}  |  Prefix: {PREFIX}\n")
    run_pipeline(args, verbose=True)


if __name__ == "__main__":
    main()
