"""
QC-only pipeline (pass/fail per recording file). Edit CONFIG, then:

    python qc_testing_script.py

For band-power / event ratios use testing_script.py (separate pipeline).
See README.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

RAW_FOLDER = r"C:\Users\Lab Member\Desktop\CC_DATA"
OUTPUT_DIR = r"C:\Users\Lab Member\Desktop\CC_DATA\qc_output"
PREFIX = "qc_run"

FS = 250.0
N_CHANNELS = 8

# Pass = ALL of: Percent_Clean >= MIN_PERCENT_CLEAN, Duration_Sec >= MIN_DURATION_SEC,
#                  Line_Noise_Ratio <= MAX_LINE_NOISE_RATIO (if not NaN)
MIN_PERCENT_CLEAN = 0.40
MIN_DURATION_SEC = 10.0
MAX_LINE_NOISE_RATIO = 0.30

# -----------------------------------------------------------------------------
# End of config
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qc_pipeline import run_qc_pipeline  # noqa: E402
import argparse  # noqa: E402


def main():
    args = argparse.Namespace(
        folder=os.path.abspath(RAW_FOLDER),
        output_dir=os.path.abspath(OUTPUT_DIR),
        prefix=PREFIX,
        fs=FS,
        n_channels=N_CHANNELS,
        min_percent_clean=MIN_PERCENT_CLEAN,
        min_duration_sec=MIN_DURATION_SEC,
        max_line_noise_ratio=MAX_LINE_NOISE_RATIO,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    run_qc_pipeline(args)


if __name__ == "__main__":
    main()
