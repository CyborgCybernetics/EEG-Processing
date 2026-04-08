import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis, skew


# Parent function: run on a folder of files
# ============================================================

def run_eeg_qc(folder_path, fs=250, n_channels=8):
    """
    Run EEG QC on all BrainFlow/OpenBCI files in a folder.

    Parameters:
    ----------
    folder_path : str
        Path to folder containing BrainFlow-RAW_*.csv and/or OpenBCI-RAW-*.txt files.
    fs : int
        Sampling frequency (Hz).
    n_channels : int
        Number of EEG channels (8 for your headset).

    Returns
    -------
    metrics_df : pandas.DataFrame
        DataFrame where:
          - columns = filenames
          - rows    = human-readable metrics aggregated across channels.
    """
    # Find supported files
    brainflow_files = glob.glob(os.path.join(folder_path, "BrainFlow-RAW_*.csv"))
    openbci_files   = glob.glob(os.path.join(folder_path, "OpenBCI-RAW-*.txt"))
    all_files = sorted(brainflow_files + openbci_files)

    if not all_files:
        raise FileNotFoundError("No BrainFlow-RAW_*.csv or OpenBCI-RAW-*.txt files found in folder.")

    # Where to save processed signals
    processed_dir = os.path.join(folder_path, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    metrics_by_file = {}  # filename -> pandas.Series

    for idx, fpath in enumerate(all_files, 1):
        fname = os.path.basename(fpath)
        print(f"[{idx}/{len(all_files)}] Processing {fname}")

        # -------- LOAD EEG --------
        eeg_raw = load_eeg_file(fpath, n_channels=n_channels)

        # -------- PROCESS TRIAL --------
        metrics = process_single_trial(
            eeg_raw,
            fs=fs,
            filename=fname,
            save_dir=processed_dir
        )

        # Store as Series; columns will be filenames
        metrics_by_file[fname] = pd.Series(metrics)

    metrics_df = pd.DataFrame(metrics_by_file)
    return metrics_df


# Loading functions
# ============================================================

def load_eeg_file(fpath, n_channels=8):
    """
    Load EEG from:
      - BrainFlow-RAW_*.csv  (headerless, tab-separated)
      - OpenBCI-RAW-*.txt    (commented header, comma-separated)

    Returns
    -------
    eeg : np.ndarray, shape (n_channels, n_samples)
        EEG data (channels x time) for the 8 EXG channels.
    """
    fname_lower = os.path.basename(fpath).lower()
    ext = os.path.splitext(fpath)[1].lower()

    # ---- BrainFlow CSV: headerless, tab-separated ----
    if "brainflow-raw" in fname_lower and ext == ".csv":
        df = pd.read_csv(fpath, sep="\t", header=None)
        # column 0 = sample index, columns 1–8 = EEG
        if df.shape[1] < n_channels + 1:
            raise ValueError(
                f"BrainFlow file {fpath} has only {df.shape[1]} columns; "
                f"expected at least {n_channels + 1}."
            )
        eeg = df.iloc[:, 1:1 + n_channels].to_numpy().T  # (channels, samples)
        return eeg

    # ---- OpenBCI TXT: comment header, comma-separated ----
    elif "openbci-raw" in fname_lower and ext == ".txt":
        df = read_openbci_txt_to_df(fpath)
        exg_cols = [f"EXG Channel {i}" for i in range(n_channels)]
        missing = [c for c in exg_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing expected EEG columns {missing} in {fpath}. "
                f"Found columns: {list(df.columns)}"
            )
        eeg = df[exg_cols].to_numpy().T
        return eeg

    else:
        raise ValueError(f"Unsupported file type or naming: {fpath}")


def read_openbci_txt_to_df(fpath):
    """
    Read OpenBCI-RAW-*.txt into a DataFrame.
    Skips header lines starting with %, then uses the next line as header.
    Strips whitespace from column names.
    """
    header_line = None
    skip_rows = 0
    with open(fpath, "r") as f:
        for line in f:
            if line.startswith('%') or line.startswith('#') or not line.strip():
                skip_rows += 1
                continue
            header_line = line.strip() # you strip the whitespace (spaces, tabs, etc) so that you get to extract the true string.
            break #otherwise pandas can't read csv!

    if header_line is None:
        df = pd.read_csv(fpath, comment='%', header=None)
    else:
        df = pd.read_csv(fpath, skiprows=skip_rows, header=0)

    df.columns = [c.strip() for c in df.columns]
    return df


# Single-trial processing
# ============================================================

def process_single_trial(eeg_raw, fs, filename, save_dir):
    """
    Full pipeline for one EEG trial:
      - CAR re-reference
      - 1–45 Hz bandpass (broadband)         -> eeg_bb
      - 1–80 Hz bandpass (PSD / line noise) -> eeg_psd
      - 0.1–4 Hz bandpass (slow MRCP/BP)    -> eeg_slow
      - Amplitude-based artifact detection on eeg_bb
      - Winsorization (1–99 percentile) on eeg_bb -> eeg_wins
      - Save eeg_bb, eeg_wins, eeg_slow as .npy
      - Compute metrics aggregated across channels
    """
    n_channels, n_samples = eeg_raw.shape

    # ----- Re-reference: common average reference -----
    car = np.mean(eeg_raw, axis=0, keepdims=True)
    eeg_reref = eeg_raw - car

    # ----- Filtering -----
    nyq = fs / 2.0

    # 1–45 Hz broadband (main QC + features)
    b_bp, a_bp = butter(4, [1.0 / nyq, 45.0 / nyq], btype="band")
    eeg_bb = filtfilt(b_bp, a_bp, eeg_reref, axis=1)

    # 1–80 Hz for PSD/line noise
    b_ln, a_ln = butter(4, [1.0 / nyq, 80.0 / nyq], btype="band")
    eeg_psd = filtfilt(b_ln, a_ln, eeg_reref, axis=1)

    # 0.1–4 Hz slow band for MRCP/BP
    b_slow, a_slow = butter(4, [0.1 / nyq, 4.0 / nyq], btype="band")
    eeg_slow = filtfilt(b_slow, a_slow, eeg_reref, axis=1)

    # ----- Artifact detection on broadband -----
    std_above = 7.5
    buffer_sec = 0.9
    channels_involved = 1
    min_amp = 200.0  # uV

    art_mask = detect_amplitude_artifacts(
        eeg_bb,
        fs=fs,
        std_above=std_above,
        buffer_sec=buffer_sec,
        channels_involved=channels_involved,
        min_amp=min_amp,
    )
    good_mask = ~art_mask 
    n_good = np.sum(good_mask)
    n_total = n_samples

    if n_good < fs:
        print(f"  WARNING {filename}: < 1 second of clean data; metrics may be unreliable.")

    # Clean segments used for metrics/PSD (use broadband & PSD-filtered)
    eeg_clean_bb = eeg_bb[:, good_mask]
    eeg_clean_psd = eeg_psd[:, good_mask]

    # ----- Winsorize broadband (for robust stats / ML) -----
    eeg_wins = winsorize_eeg(eeg_bb, lower_pct=1, upper_pct=99)

    # ----- Save processed signals -----
    base = os.path.splitext(filename)[0]
    np.save(os.path.join(save_dir, f"{base}_bb.npy"),   eeg_bb)
    np.save(os.path.join(save_dir, f"{base}_wins.npy"), eeg_wins)
    np.save(os.path.join(save_dir, f"{base}_slow.npy"), eeg_slow)

    # ----- Compute metrics (aggregated across channels) -----
    metrics = compute_metrics_for_trial(
        eeg_clean_bb,
        eeg_clean_psd,
        fs=fs,
        filename=filename,
    )

    # ----- Trial-level metadata -----
    metrics["Duration_Sec"] = n_samples / fs
    metrics["Percent_Clean"] = float(n_good) / float(n_total)
    metrics["Num_Samples"] = int(n_total)
    metrics["Num_Clean_Samples"] = int(n_good)
    metrics["Num_Channels"] = int(n_channels)

    return metrics

# Artifact detection, winsorizing
# ============================================================

def detect_amplitude_artifacts(eeg, fs, std_above, buffer_sec, channels_involved, min_amp):
    """
    Amplitude-based artifact detection across channels with temporal buffering.

    eeg : (n_channels, n_samples)
    Returns
    -------
    art_mask : (n_samples,) boolean
        True = artifact sample, False = clean.
    """
    n_channels, n_samples = eeg.shape

    mean_ch = np.mean(eeg, axis=1, keepdims=True)
    std_ch  = np.std(eeg, axis=1, ddof=0, keepdims=True)

    # Enforce minimum std so very flat channels don't produce tiny thresholds.
    min_std = min_amp / std_above
    std_ch = np.maximum(std_ch, min_std)

    # Possible artifacts per channel
    poss_arts = np.abs(eeg - mean_ch) > (std_above * std_ch)

    # Combine across channels
    sum_arts = np.sum(poss_arts, axis=0)
    base_mask = sum_arts >= channels_involved

    # Add temporal buffer around each artifact
    art_mask = base_mask.copy()
    idx = np.where(base_mask)[0]
    if idx.size > 0:
        buff = int(round(buffer_sec * fs))
        for i in idx:
            start = max(0, i - buff)
            end   = min(n_samples - 1, i + buff)
            art_mask[start:end+1] = True

    return art_mask


def winsorize_eeg(eeg, lower_pct=1, upper_pct=99):
    """
    Winsorize each channel independently between given percentiles.

    eeg : (n_channels, n_samples)
    """
    eeg_wins = eeg.copy()
    for ch in range(eeg.shape[0]):
        x = eeg[ch, :]
        lo, hi = np.percentile(x, [lower_pct, upper_pct])
        eeg_wins[ch, :] = np.clip(x, lo, hi)
    return eeg_wins


# Metrics: human-readable, averaged across channels
# ============================================================

def compute_metrics_for_trial(eeg_clean_bb, eeg_clean_psd, fs, filename):
    """
    Compute per-channel metrics, then aggregate them across channels
    into human-readable metrics.

    eeg_clean_bb : (n_channels, n_clean_samples), 1–45 Hz (broadband)
    eeg_clean_psd: (n_channels, n_clean_samples), 1–80 Hz (for PSD / line noise)

    Returns
    -------
    metrics : dict
        Keys are human-readable metric names, values are trial-level scalars.
    """
    n_channels, _ = eeg_clean_bb.shape
    metrics = {}

    # PSD parameters
    win  = int(2 * fs)       # 2-second window
    nover = int(win / 2)     # 50% overlap
    nfft = max(256, 2 ** int(np.ceil(np.log2(win))))

    # Frequency bands (Hz)
    bands = {
        "delta": (1, 4),
        "theta": (4, 7),
        "alpha": (8, 12),
        "beta":  (13, 30),
        "gamma": (30, 45),
    }
    band_names = list(bands.keys())

    # Per-channel containers
    ch_min       = np.full(n_channels, np.nan)
    ch_max       = np.full(n_channels, np.nan)
    ch_var       = np.full(n_channels, np.nan)
    ch_std       = np.full(n_channels, np.nan)
    ch_kurt      = np.full(n_channels, np.nan)
    ch_skew      = np.full(n_channels, np.nan)
    ch_flat_frac = np.full(n_channels, np.nan)
    ch_rms       = np.full(n_channels, np.nan)
    ch_zcr       = np.full(n_channels, np.nan)
    ch_spec_ent  = np.full(n_channels, np.nan)
    ch_hj_act    = np.full(n_channels, np.nan)
    ch_hj_mob    = np.full(n_channels, np.nan)
    ch_hj_comp   = np.full(n_channels, np.nan)
    ch_line_noise = np.full(n_channels, np.nan)

    band_abs  = {b: np.full(n_channels, np.nan) for b in band_names}
    band_rel  = {b: np.full(n_channels, np.nan) for b in band_names}
    band_peak = {b: np.full(n_channels, np.nan) for b in band_names}

    # ------------ Loop over channels ------------
    for ch in range(n_channels):
        x_bb  = eeg_clean_bb[ch, :]
        x_psd = eeg_clean_psd[ch, :]

        if x_bb.size == 0:
            # leave NaNs for this channel
            continue

        # ---------- Time-domain stats (1–45 Hz) ----------
        ch_min[ch] = np.min(x_bb)
        ch_max[ch] = np.max(x_bb)
        ch_var[ch] = np.var(x_bb)
        ch_std[ch] = np.std(x_bb)
        ch_kurt[ch] = kurtosis(x_bb, fisher=False, bias=False)
        ch_skew[ch] = skew(x_bb, bias=False)

        dx = np.diff(x_bb)
        ch_flat_frac[ch] = np.sum(dx == 0) / float(dx.size) if dx.size > 0 else np.nan

        ch_rms[ch] = np.sqrt(np.mean(x_bb ** 2))

        # Zero-crossing rate (per second)
        if x_bb.size > 1:
            s = np.sign(x_bb)
            s[s == 0] = 1
            zc = np.sum(s[:-1] * s[1:] < 0)
            duration_clean_sec = x_bb.size / float(fs)
            ch_zcr[ch] = zc / max(duration_clean_sec, 1e-6)
        else:
            ch_zcr[ch] = np.nan

        # ---------- Frequency-domain metrics (1–80 Hz PSD) ----------
        f, pxx = welch(
            x_psd,
            fs=fs,
            window="hann",
            nperseg=win,
            noverlap=nover,
            nfft=nfft,
        )

        idx_total = (f >= 1) & (f <= 40)
        idx_line  = (f >= 58) & (f <= 62)
        total_pow = np.trapezoid(pxx[idx_total], f[idx_total]) if np.any(idx_total) else np.nan
        line_pow  = np.trapezoid(pxx[idx_line],  f[idx_line])  if np.any(idx_line) else np.nan

        if total_pow and total_pow > 0:
            line_ratio = line_pow / total_pow
        else:
            line_ratio = np.nan
        ch_line_noise[ch] = line_ratio

        # spectral entropy
        pnorm = pxx / np.sum(pxx)
        ch_spec_ent[ch] = -np.sum(pnorm * np.log2(pnorm + 1e-16))

        # Hjorth parameters
        dx2 = np.diff(x_bb)
        ddx = np.diff(dx2)
        var0 = np.var(x_bb)
        var1 = np.var(dx2) if dx2.size > 0 else np.nan
        var2 = np.var(ddx) if ddx.size > 0 else np.nan

        ch_hj_act[ch] = var0
        ch_hj_mob[ch] = np.sqrt(var1 / var0) if var0 > 0 and not np.isnan(var1) else np.nan
        if var1 > 0 and not np.isnan(var2) and not np.isnan(ch_hj_mob[ch]) and ch_hj_mob[ch] > 0:
            ch_hj_comp[ch] = np.sqrt(var2 / var1) / ch_hj_mob[ch]
        else:
            ch_hj_comp[ch] = np.nan

        # Band powers & peaks
        if np.any(idx_total):
            total_pow_all = np.trapezoid(pxx[idx_total], f[idx_total])
        else:
            total_pow_all = np.nan

        for bname, (f1, f2) in bands.items():
            idx_band = (f >= f1) & (f <= f2)
            if not np.any(idx_band):
                continue

            abs_pow = np.trapezoid(pxx[idx_band], f[idx_band])
            if total_pow_all and total_pow_all > 0:
                rel_pow = abs_pow / total_pow_all
            else:
                rel_pow = np.nan

            local_pxx = pxx[idx_band]
            local_f   = f[idx_band]
            peak_idx  = np.argmax(local_pxx)
            peak_f    = local_f[peak_idx]

            band_abs[bname][ch]  = abs_pow
            band_rel[bname][ch]  = rel_pow
            band_peak[bname][ch] = peak_f
            
    metrics["Amp_Min"]           = np.nanmean(ch_min)
    metrics["Amp_Max"]           = np.nanmean(ch_max)
    metrics["Amp_Variance"]      = np.nanmean(ch_var)
    metrics["Amp_Std"]           = np.nanmean(ch_std)
    metrics["Amp_Kurtosis"]      = np.nanmean(ch_kurt)
    metrics["Amp_Skewness"]      = np.nanmean(ch_skew)
    metrics["Flatline_Percent"] = np.nanmean(ch_flat_frac)
    metrics["RMS_Amplitude"]     = np.nanmean(ch_rms)
    metrics["Zero_Cross_Rate"]   = np.nanmean(ch_zcr)
    metrics["Spectral_Entropy"]  = np.nanmean(ch_spec_ent)
    metrics["Hjorth_Activity"]   = np.nanmean(ch_hj_act)
    metrics["Hjorth_Mobility"]   = np.nanmean(ch_hj_mob)
    metrics["Hjorth_Complexity"] = np.nanmean(ch_hj_comp)
    metrics["Line_Noise_Ratio"]  = np.nanmean(ch_line_noise)
    for bname in band_names:
        metrics[f"{bname}_Abs_Power"]   = np.nanmean(band_abs[bname])
        metrics[f"{bname}_Rel_Power"]   = np.nanmean(band_rel[bname])
        metrics[f"{bname}_Peak_Freq"]   = np.nanmean(band_peak[bname])

    return metrics