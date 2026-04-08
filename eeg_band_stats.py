import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import ttest_1samp, wilcoxon

from eeg_qc_pipeline import load_eeg_file


# When auto-building events from file end, use this spacing for known block IDs in filenames (A28…A35).
DEFAULT_INTERVAL_MAP_STR = "A28:30,A29:15,A30:10,A31:5,A32:30,A33:15,A34:10,A35:5"

EEG_BANDS = {
    "delta": (1, 4),
    "theta": (4, 7),
    "alpha": (8, 12),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def discover_files(folder, name_contains=None):
    bf = glob.glob(os.path.join(folder, "BrainFlow-RAW_*.csv"))
    ob = glob.glob(os.path.join(folder, "OpenBCI-RAW-*.txt"))
    out = sorted(bf + ob)
    if name_contains:
        token = name_contains.lower()
        out = [f for f in out if token in os.path.basename(f).lower()]
    if not out:
        raise FileNotFoundError(
            f"No supported EEG files found in folder: {folder}"
        )
    return out


def parse_channels(channels, total_channels=8):
    if channels is None:
        return list(range(1, total_channels + 1))
    ch = sorted(set(channels))
    bad = [c for c in ch if c < 1 or c > total_channels]
    if bad:
        raise ValueError(f"Invalid channels {bad}; allowed range is 1..{total_channels}.")
    return ch


def car_reref(eeg):
    car = np.mean(eeg, axis=0, keepdims=True)
    return eeg - car


def bandpass_1_45(eeg, fs):
    nyq = fs / 2.0
    b, a = butter(4, [1.0 / nyq, 45.0 / nyq], btype="band")
    return filtfilt(b, a, eeg, axis=1)


def window_to_samples(event_t, win, fs, n_samples):
    i0 = int(round((event_t + win[0]) * fs))
    i1 = int(round((event_t + win[1]) * fs))
    if i0 < 0 or i1 <= i0 or i1 > n_samples:
        return None
    return i0, i1


def band_power_welch(x, fs, band, nperseg_sec=1.0):
    n = int(x.shape[0]) if hasattr(x, "shape") else len(x)
    if n < 8:
        return np.nan
    nperseg = max(64, int(round(nperseg_sec * fs)))
    nperseg = min(nperseg, n)
    noverlap = int(nperseg // 2)
    if noverlap >= nperseg:
        noverlap = max(0, nperseg - 1)
    f, pxx = welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    lo, hi = band
    m = (f >= lo) & (f <= hi)
    if not np.any(m):
        return np.nan
    return np.trapezoid(pxx[m], f[m])


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    valid = np.isfinite(pvals)
    if not np.any(valid):
        return qvals
    pv = pvals[valid]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * m / (np.arange(1, m + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q_temp = np.empty_like(q_ranked)
    q_temp[order] = q_ranked
    qvals[valid] = q_temp
    return qvals


def parse_interval_map(text):
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        key, val = part.split(":")
        out[key.strip().upper()] = float(val.strip())
    return out


def infer_block_from_filename(name):
    m = re.search(r"(A\d+)_", name.upper())
    return m.group(1) if m else None


def make_events_from_end(duration_sec, step_sec):
    t = duration_sec
    ev = []
    while t > 0:
        ev.append(round(t, 6))
        t -= step_sec
    return sorted(set(x for x in ev if x >= 0))


def load_events(events_csv):
    df = pd.read_csv(events_csv)
    required = {"filename", "event_time_sec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Events CSV missing required columns: {sorted(missing)}. "
            "Required: filename, event_time_sec"
        )
    if "trial_id" not in df.columns:
        df["trial_id"] = np.arange(1, len(df) + 1)
    return df


def _log_pct_pair(p_num, p_den):
    if np.isfinite(p_num) and p_num > 0 and np.isfinite(p_den) and p_den > 0:
        r = p_num / p_den
        return np.log(r), (r - 1.0) * 100.0
    return np.nan, np.nan


def compute_trial_rows(
    eeg_by_file,
    events_df,
    fs,
    channels_1based,
    rest_baseline_win,
    pre_movement_win,
    post_movement_win,
    bands,
):
    """
    Three windows relative to event (t=0):
      - rest_baseline: quiet reference (e.g. [-2, -0.5])
      - pre_movement: lead-up before event (e.g. [-0.5, 0])
      - post_movement: response after event (e.g. [0, 1])
    """
    rows = []
    for _, ev in events_df.iterrows():
        fname = str(ev["filename"])
        if fname not in eeg_by_file:
            continue
        event_t = float(ev["event_time_sec"])
        trial_id = ev["trial_id"]

        eeg = eeg_by_file[fname]
        n_ch, n_samples = eeg.shape

        idx_r = window_to_samples(event_t, rest_baseline_win, fs, n_samples)
        idx_p = window_to_samples(event_t, pre_movement_win, fs, n_samples)
        idx_m = window_to_samples(event_t, post_movement_win, fs, n_samples)
        if idx_r is None or idx_p is None or idx_m is None:
            continue
        r0, r1 = idx_r
        p0, p1 = idx_p
        m0, m1 = idx_m

        for ch in channels_1based:
            x = eeg[ch - 1]
            x_rest = x[r0:r1]
            x_pre = x[p0:p1]
            x_post = x[m0:m1]

            for bname, brange in bands.items():
                p_rest = band_power_welch(x_rest, fs, brange)
                p_pre = band_power_welch(x_pre, fs, brange)
                p_post = band_power_welch(x_post, fs, brange)

                log_pre_v_rest, pct_pre_v_rest = _log_pct_pair(p_pre, p_rest)
                log_post_v_rest, pct_post_v_rest = _log_pct_pair(p_post, p_rest)
                log_post_v_pre, pct_post_v_pre = _log_pct_pair(p_post, p_pre)

                rows.append(
                    {
                        "filename": fname,
                        "trial_id": trial_id,
                        "event_time_sec": event_t,
                        "channel": ch,
                        "band": bname,
                        "power_rest": p_rest,
                        "power_pre_movement": p_pre,
                        "power_post_movement": p_post,
                        "log_ratio_pre_vs_rest": log_pre_v_rest,
                        "pct_change_pre_vs_rest": pct_pre_v_rest,
                        "log_ratio_post_vs_rest": log_post_v_rest,
                        "pct_change_post_vs_rest": pct_post_v_rest,
                        "log_ratio_post_vs_pre": log_post_v_pre,
                        "pct_change_post_vs_pre": pct_post_v_pre,
                    }
                )
    return pd.DataFrame(rows)


def summarize_stats(trial_df):
    """
    One-sample tests vs 0 on log-ratios for each contrast.
    FDR computed separately per contrast column (pre_vs_rest, post_vs_rest, post_vs_pre).
    """
    contrasts = [
        ("pre_vs_rest", "log_ratio_pre_vs_rest", "pct_change_pre_vs_rest"),
        ("post_vs_rest", "log_ratio_post_vs_rest", "pct_change_post_vs_rest"),
        ("post_vs_pre", "log_ratio_post_vs_pre", "pct_change_post_vs_pre"),
    ]
    out_rows = []
    for contrast, log_col, pct_col in contrasts:
        grp = trial_df.groupby(["channel", "band"], dropna=False)
        for (ch, band), g in grp:
            x = g[log_col].to_numpy()
            x = x[np.isfinite(x)]
            pct = g[pct_col].to_numpy()
            pct = pct[np.isfinite(pct)]
            n = len(x)
            if n == 0:
                out_rows.append(
                    {
                        "contrast": contrast,
                        "channel": ch,
                        "band": band,
                        "n_trials": 0,
                        "mean_log_ratio": np.nan,
                        "median_log_ratio": np.nan,
                        "mean_percent_change_from_log": np.nan,
                        "mean_percent_change_raw": np.nan,
                        "median_percent_change_raw": np.nan,
                        "wilcoxon_p": np.nan,
                        "ttest_p": np.nan,
                        "effect_dz": np.nan,
                    }
                )
                continue

            mean_log = float(np.mean(x))
            med_log = float(np.median(x))
            mean_pct_from_log = float((np.exp(mean_log) - 1.0) * 100.0)
            mean_pct_raw = float(np.mean(pct)) if len(pct) else np.nan
            med_pct_raw = float(np.median(pct)) if len(pct) else np.nan

            if n >= 2 and not np.allclose(x, x[0]):
                try:
                    w_p = float(
                        wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue
                    )
                except Exception:
                    w_p = np.nan
                try:
                    t_res = ttest_1samp(x, popmean=0.0, nan_policy="omit")
                    t_p = float(t_res.pvalue) if np.isfinite(t_res.pvalue) else np.nan
                except Exception:
                    t_p = np.nan
            else:
                w_p = np.nan
                t_p = np.nan

            sd = float(np.std(x, ddof=1)) if n >= 2 else np.nan
            dz = float(mean_log / sd) if np.isfinite(sd) and sd > 0 else np.nan

            out_rows.append(
                {
                    "contrast": contrast,
                    "channel": ch,
                    "band": band,
                    "n_trials": n,
                    "mean_log_ratio": mean_log,
                    "median_log_ratio": med_log,
                    "mean_percent_change_from_log": mean_pct_from_log,
                    "mean_percent_change_raw": mean_pct_raw,
                    "median_percent_change_raw": med_pct_raw,
                    "wilcoxon_p": w_p,
                    "ttest_p": t_p,
                    "effect_dz": dz,
                }
            )

    out = (
        pd.DataFrame(out_rows)
        .sort_values(["contrast", "channel", "band"])
        .reset_index(drop=True)
    )
    out["wilcoxon_q_fdr"] = np.nan
    out["ttest_q_fdr"] = np.nan
    for c in out["contrast"].dropna().unique():
        m = out["contrast"] == c
        out.loc[m, "wilcoxon_q_fdr"] = bh_fdr(out.loc[m, "wilcoxon_p"].to_numpy())
        out.loc[m, "ttest_q_fdr"] = bh_fdr(out.loc[m, "ttest_p"].to_numpy())
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Event-locked band power (Welch): rest baseline, pre-movement, post-movement. "
            "Contrasts: pre_vs_rest, post_vs_rest, post_vs_pre. "
            "Outputs trial-level CSV and summary (Wilcoxon / t-test on log-ratios, BH-FDR per contrast)."
        )
    )
    parser.add_argument("--folder", type=str, required=True, help="Folder with EEG files.")
    parser.add_argument(
        "--name-contains",
        type=str,
        default=None,
        help="Optional filename filter (case-insensitive), e.g. A35_1008_Ethan.",
    )
    parser.add_argument(
        "--events-csv",
        type=str,
        default=None,
        help="CSV with filename,event_time_sec[,trial_id]. If omitted, events are auto-generated from end.",
    )
    parser.add_argument(
        "--events-from-end-sec",
        type=float,
        default=10.0,
        help="Spacing for auto events when block not in --interval-map (default 10).",
    )
    parser.add_argument(
        "--interval-map",
        type=str,
        default=DEFAULT_INTERVAL_MAP_STR,
        help=(
            "Comma-separated BLOCK:SEC for auto events, e.g. A28:30,A29:15. "
            "Filename must contain block token (e.g. A28_). "
            "Empty string disables map and uses --events-from-end-sec for every file."
        ),
    )
    parser.add_argument(
        "--include-t0-event",
        action="store_true",
        help="Include an event exactly at t=0 when auto-generating events.",
    )
    parser.add_argument("--fs", type=float, default=250.0, help="Sampling frequency (default 250).")
    parser.add_argument("--channels", type=int, nargs="+", default=[4, 5], help="1-based channels to analyze.")
    parser.add_argument(
        "--rest-baseline-win",
        type=float,
        nargs=2,
        default=[-2.0, -0.5],
        metavar=("START", "END"),
        help="Quiet rest reference before pre-movement, seconds (default -2.0 -0.5).",
    )
    parser.add_argument(
        "--pre-movement-win",
        type=float,
        nargs=2,
        default=[-0.5, 0.0],
        metavar=("START", "END"),
        help="Lead-up window ending at event, seconds (default -0.5 0.0).",
    )
    parser.add_argument(
        "--post-movement-win",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        metavar=("START", "END"),
        help="Post-event / movement response window, seconds (default 0.0 1.0).",
    )
    parser.add_argument("--output-prefix", type=str, default="eeg_band_stats", help="Prefix for output CSV files.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Write CSV outputs here (default: same as --folder).",
    )
    args = parser.parse_args()

    run_band_stats(args)


def run_band_stats(args):
    """Run band-power extraction and stats. `args` is an argparse.Namespace from this module."""
    channels = parse_channels(args.channels, total_channels=8)
    rest_win = (float(args.rest_baseline_win[0]), float(args.rest_baseline_win[1]))
    pre_win = (float(args.pre_movement_win[0]), float(args.pre_movement_win[1]))
    post_win = (float(args.post_movement_win[0]), float(args.post_movement_win[1]))
    for label, w in [
        ("rest_baseline", rest_win),
        ("pre_movement", pre_win),
        ("post_movement", post_win),
    ]:
        if w[0] >= w[1]:
            raise ValueError(f"Invalid {label} window: START must be < END.")

    if args.events_from_end_sec <= 0:
        raise ValueError("--events-from-end-sec must be > 0.")

    interval_map = {}
    if args.interval_map and str(args.interval_map).strip():
        interval_map = parse_interval_map(args.interval_map)

    files = discover_files(args.folder, name_contains=args.name_contains)
    eeg_by_file = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        eeg_raw = load_eeg_file(fpath, n_channels=8)
        eeg = bandpass_1_45(car_reref(eeg_raw), args.fs)
        eeg_by_file[fname] = eeg

    if args.events_csv:
        events_df = load_events(args.events_csv)
    else:
        rows = []
        for fname, eeg in eeg_by_file.items():
            duration = eeg.shape[1] / float(args.fs)
            block = infer_block_from_filename(fname)
            if block and block in interval_map:
                step_sec = interval_map[block]
            else:
                step_sec = float(args.events_from_end_sec)
            ev = make_events_from_end(duration, step_sec)
            if args.include_t0_event and 0.0 not in ev:
                ev = sorted(set(ev + [0.0]))
            for i, event_time_sec in enumerate(ev, start=1):
                rows.append(
                    {
                        "filename": fname,
                        "event_time_sec": event_time_sec,
                        "trial_id": i,
                    }
                )
        events_df = pd.DataFrame(rows)

    trial_df = compute_trial_rows(
        eeg_by_file=eeg_by_file,
        events_df=events_df,
        fs=args.fs,
        channels_1based=channels,
        rest_baseline_win=rest_win,
        pre_movement_win=pre_win,
        post_movement_win=post_win,
        bands=EEG_BANDS,
    )
    if trial_df.empty:
        raise RuntimeError("No valid trials were computed. Check filenames, event_time_sec, and windows.")

    stats_df = summarize_stats(trial_df)

    out_dir = getattr(args, "out_dir", None) or args.folder
    os.makedirs(out_dir, exist_ok=True)

    trial_out = os.path.join(out_dir, f"{args.output_prefix}_trial_level.csv")
    stats_out = os.path.join(out_dir, f"{args.output_prefix}_summary_stats.csv")
    events_out = os.path.join(out_dir, f"{args.output_prefix}_events_used.csv")
    trial_df.to_csv(trial_out, index=False)
    stats_df.to_csv(stats_out, index=False)
    events_df.to_csv(events_out, index=False)

    print("Done.")
    print(f"Events used: {events_out}")
    print(f"Trial-level output: {trial_out}")
    print(f"Summary stats output: {stats_out}")
    return {
        "trial_level": trial_out,
        "summary_stats": stats_out,
        "events_used": events_out,
    }


if __name__ == "__main__":
    main()
