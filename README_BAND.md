# Band-ratio pipeline (events, Welch bands, plots, cross-channel stats)

## What you need

- **Raw mode:** a folder with the same **`BrainFlow-RAW_*.csv`** / **`OpenBCI-RAW_*.txt`** files, **or**
- **Trial mode:** an existing **`*_trial_level.csv`** produced earlier by this pipeline (see schema below).
- `pip install -r requirements.txt`.

This pipeline does **not** run QC. Use **`qc_testing_script.py`** first if you want per-file pass/fail (see **README_QC.md**).

## Easiest run

1. Open **`testing_script.py`**.
2. Set **`MODE`**: `"raw"` to read from **`RAW_FOLDER`**, or `"trial"` to point **`TRIAL_CSV`** at a saved trial-level CSV.
3. Set **`OUTPUT_DIR`**, **`PREFIX`**, windows, **`INTERVAL_MAP`** (or spacing), and **`CHANNELS`** as needed.
4. Run:

```bash
python testing_script.py
```

## Outputs (under `OUTPUT_DIR`, names start with `PREFIX`)

- `{prefix}_trial_level.csv` — one row per event × channel × band  
- `{prefix}_summary_stats.csv` — tests on log-ratios; FDR by contrast  
- `{prefix}_events_used.csv`  
- Cross-channel CSVs (GMM/BIC, Friedman, pairwise) unless `--skip-crosschannel`  
- Eight PNGs (one per channel, all bands, three ln-ratios per event) unless `--skip-plots`  
- `{prefix}_pipeline_report.json`  

## Command-line

```bash
python band_ratio_pipeline.py --raw-folder "path/to/eeg" --output-dir "path/to/out" --prefix myrun
python band_ratio_pipeline.py --trial-csv "path/to/myrun_trial_level.csv" --output-dir "path/to/out" --prefix myrun
```

`python band_ratio_pipeline.py --help` lists flags (`--skip-plots`, `--skip-crosschannel`, `--fs`, `--channels`, event windows, `--interval-map`, etc.).

## Trial CSV schema (`MODE == "trial"`)

Required columns include:

`filename`, `trial_id`, `event_time_sec`, `channel`, `band`,  
`power_rest`, `power_pre_movement`, `power_post_movement`,  
`log_ratio_pre_vs_rest`, `log_ratio_post_vs_rest`, `log_ratio_post_vs_pre`

## Modules

- **`band_ratio_pipeline.py`** — CLI and `run_pipeline()`.
- **`eeg_band_stats.py`** — raw EEG → trial / summary CSVs.
- **`eeg_qc_pipeline.py`** — file loading (used by `eeg_band_stats`).
- **`plot_band_trial_triplets.py`** — figures.
- **`analyze_band_ratio_crosschannel.py`** — cross-channel statistics.
