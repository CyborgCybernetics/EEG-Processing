# QC pipeline (per file: metrics + pass / fail)

## What you need

- A folder containing **`BrainFlow-RAW_*.csv`** and/or **`OpenBCI-RAW_*.txt`** files.
- `pip install -r requirements.txt` (same `requirements.txt` as the rest of the repo).

## Easiest run

1. Open **`qc_testing_script.py`**.
2. Set **`RAW_FOLDER`** to your data folder, **`OUTPUT_DIR`** where CSVs should go, and adjust **`MIN_PERCENT_CLEAN`**, **`MIN_DURATION_SEC`**, **`MAX_LINE_NOISE_RATIO`** if you want stricter or looser pass rules.
3. Run:

```bash
python qc_testing_script.py
```

## Outputs

- **`{PREFIX}_qc_trials.csv`** — one row per recording file; all QC metrics plus **`qc_pass`** and **`qc_fail_reason`**.
- **`{PREFIX}_qc_summary.json`** — counts of pass/fail and the rule values used.

Processed `.npy` files are still written under **`RAW_FOLDER/processed/`** by `eeg_qc_pipeline` (same as before).

## Pass / fail rules

A file **passes** only if **all** of these hold:

- **`Percent_Clean`** ≥ `MIN_PERCENT_CLEAN` (fraction of samples not flagged as artifact).
- **`Duration_Sec`** ≥ `MIN_DURATION_SEC`.
- **`Line_Noise_Ratio`** ≤ `MAX_LINE_NOISE_RATIO` (skipped if NaN).

## Command-line (same logic)

```bash
python qc_pipeline.py --folder "path/to/eeg" --output-dir "path/to/out" --prefix qc_run ^
  --min-percent-clean 0.40 --min-duration-sec 10.0 --max-line-noise-ratio 0.30
```

Run `python qc_pipeline.py --help` for all options.

## Modules

- **`qc_pipeline.py`** — CLI and `run_qc_pipeline()`.
- **`eeg_qc_pipeline.py`** — loads files, CAR, filters, metrics.

This pipeline does **not** compute event-locked band ratios. For that, see **README_BAND.md** and **`testing_script.py`**.
