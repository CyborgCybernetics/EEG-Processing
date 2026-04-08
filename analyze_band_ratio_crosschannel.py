"""
Cross-channel analysis for eeg_band_stats trial-level CSV (8 channels, same events).

1) BIC comparison: 1 vs 2 univariate Gaussian components per (channel, band, ratio).
   delta_bic = BIC1 - BIC2; positive favors two components (possible "clustering" /
   multimodality in log-ratio). Rule of thumb: delta_bic > 6 is nontrivial; > 10 strong.

2) Friedman test per (band, ratio): same events paired across all 8 channels.
   Benjamini-Hochberg FDR across the 5*3 = 15 tests.

3) If friedman_q_fdr < 0.05: pairwise paired Wilcoxon on channel differences,
   Holm correction within that (band, ratio) family (28 pairs).

Outputs three CSVs next to the trial CSV (or under --output-dir).
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, friedmanchisquare, wilcoxon
from sklearn.mixture import GaussianMixture

BAND_ORDER = ("delta", "theta", "alpha", "beta", "gamma")
RATIOS = (
    ("post_vs_baseline", "log_ratio_post_vs_rest"),
    ("pre_vs_baseline", "log_ratio_pre_vs_rest"),
    ("post_vs_pre", "log_ratio_post_vs_pre"),
)
EVENT_KEYS = ("filename", "trial_id", "event_time_sec")
CHANNELS = tuple(range(1, 9))


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    if m == 0:
        return pvals.copy()
    order = np.argsort(pvals)
    sp = pvals[order]
    adj = sp * (m - np.arange(m))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    return out


def gmm_bic_delta(x: np.ndarray) -> dict:
    """Fit 1- and 2-component full-covariance Gaussians in 1D; return BIC difference."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    out = {
        "n_valid": n,
        "bic1": np.nan,
        "bic2": np.nan,
        "delta_bic": np.nan,
        "favor_two_moderate": False,
        "favor_two_strong": False,
    }
    if n < 30:
        return out
    X = x.reshape(-1, 1)
    g1 = GaussianMixture(
        n_components=1,
        covariance_type="full",
        random_state=0,
        n_init=5,
        max_iter=500,
    )
    g2 = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
        n_init=5,
        max_iter=500,
    )
    try:
        g1.fit(X)
        g2.fit(X)
        bic1 = float(g1.bic(X))
        bic2 = float(g2.bic(X))
        d = bic1 - bic2
        out["bic1"] = bic1
        out["bic2"] = bic2
        out["delta_bic"] = d
        out["favor_two_moderate"] = bool(d > 6.0)
        out["favor_two_strong"] = bool(d > 10.0)
    except Exception:
        pass
    return out


def channel_matrix(df: pd.DataFrame, band: str, col: str) -> tuple[np.ndarray, int]:
    """Rows = events with all 8 channels finite; shape (n, 8)."""
    dfb = df[df["band"] == band]
    wide = dfb.pivot_table(
        index=list(EVENT_KEYS),
        columns="channel",
        values=col,
        aggfunc="first",
    )
    wide = wide.reindex(columns=list(CHANNELS))
    if wide.isna().any().any():
        # drop events missing any channel
        ok = wide.notna().all(axis=1).to_numpy()
        wide = wide.loc[ok]
    M = wide.to_numpy(dtype=float)
    return M, int(M.shape[0])


def run_crosschannel_analysis(
    trial_csv: str,
    *,
    output_dir: str | None = None,
    prefix: str | None = None,
    pairwise_nominal_p: float | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run GMM/BIC, Friedman, and pairwise Wilcoxon exports. Returns paths and summary counts.
    """
    trial_csv = os.path.abspath(trial_csv)
    stem = prefix or os.path.splitext(os.path.basename(trial_csv))[0]
    out_dir = output_dir or os.path.dirname(trial_csv)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, stem)

    df = pd.read_csv(trial_csv)

    # --- 1) GMM / BIC per channel, band, ratio ---
    gmm_rows = []
    for ch in CHANNELS:
        for band in BAND_ORDER:
            for rname, col in RATIOS:
                sub = df[(df["channel"] == ch) & (df["band"] == band)]
                st = gmm_bic_delta(sub[col].to_numpy(dtype=float))
                gmm_rows.append(
                    {
                        "channel": ch,
                        "band": band,
                        "ratio": rname,
                        **st,
                    }
                )
    gmm_df = pd.DataFrame(gmm_rows)
    gmm_path = f"{base}_gmm_bic_1vs2.csv"
    gmm_df.to_csv(gmm_path, index=False)

    # --- 2) Friedman + 3) pairwise ---
    fried_rows = []
    pair_rows = []

    for band in BAND_ORDER:
        for rname, col in RATIOS:
            M, n_ev = channel_matrix(df, band, col)
            if n_ev < 8:
                fried_rows.append(
                    {
                        "band": band,
                        "ratio": rname,
                        "n_events": n_ev,
                        "friedman_stat": np.nan,
                        "friedman_p": np.nan,
                    }
                )
                continue
            stat, p = friedmanchisquare(*[M[:, j] for j in range(8)])
            fried_rows.append(
                {
                    "band": band,
                    "ratio": rname,
                    "n_events": n_ev,
                    "friedman_stat": float(stat),
                    "friedman_p": float(p),
                }
            )

    fried_df = pd.DataFrame(fried_rows)
    pv = fried_df["friedman_p"].to_numpy(dtype=float)
    q = np.full_like(pv, np.nan, dtype=float)
    m = np.isfinite(pv)
    if np.any(m):
        q[m] = false_discovery_control(pv[m], method="bh")
    fried_df["friedman_q_fdr"] = q

    for _, row in fried_df.iterrows():
        fq = row["friedman_q_fdr"]
        if not np.isfinite(fq) or fq >= 0.05:
            continue
        band = row["band"]
        rname = row["ratio"]
        col = next(c for rn, c in RATIOS if rn == rname)
        M, _ = channel_matrix(df, band, col)
        pairs = list(itertools.combinations(range(8), 2))
        raw_ps = []
        for i, j in pairs:
            d = M[:, i] - M[:, j]
            try:
                w = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
                raw_ps.append(float(w.pvalue))
            except Exception:
                raw_ps.append(np.nan)
        raw_ps = np.asarray(raw_ps, dtype=float)
        holm = holm_adjust(raw_ps)
        for (i, j), p_raw, p_h in zip(pairs, raw_ps, holm):
            pair_rows.append(
                {
                    "band": band,
                    "ratio": rname,
                    "channel_a": int(CHANNELS[i]),
                    "channel_b": int(CHANNELS[j]),
                    "wilcoxon_p": p_raw,
                    "wilcoxon_p_holm": p_h,
                }
            )

    pair_df = pd.DataFrame(pair_rows)

    pair_nominal_rows: list[dict] = []
    p_thr = pairwise_nominal_p
    if p_thr is not None and p_thr > 0:
        for _, row in fried_df.iterrows():
            fp = row["friedman_p"]
            if not np.isfinite(fp) or fp >= p_thr:
                continue
            band = row["band"]
            rname = row["ratio"]
            col = next(c for rn, c in RATIOS if rn == rname)
            M, _ = channel_matrix(df, band, col)
            pairs = list(itertools.combinations(range(8), 2))
            raw_ps = []
            for i, j in pairs:
                d = M[:, i] - M[:, j]
                try:
                    w = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
                    raw_ps.append(float(w.pvalue))
                except Exception:
                    raw_ps.append(np.nan)
            raw_ps = np.asarray(raw_ps, dtype=float)
            holm = holm_adjust(raw_ps)
            for (i, j), p_raw, p_h in zip(pairs, raw_ps, holm):
                pair_nominal_rows.append(
                    {
                        "band": band,
                        "ratio": rname,
                        "channel_a": int(CHANNELS[i]),
                        "channel_b": int(CHANNELS[j]),
                        "wilcoxon_p": p_raw,
                        "wilcoxon_p_holm": p_h,
                    }
                )

    fried_path = f"{base}_friedman_across_channels.csv"
    pair_path = f"{base}_pairwise_channel_wilcoxon_holm.csv"
    fried_df.to_csv(fried_path, index=False)
    pair_df.to_csv(pair_path, index=False)
    pair_nominal_path = None
    if pair_nominal_rows:
        pair_nominal_path = (
            f"{base}_pairwise_channel_wilcoxon_holm_nominal_lt{p_thr:g}.csv"
        )
        pd.DataFrame(pair_nominal_rows).to_csv(pair_nominal_path, index=False)

    n_mod = int(gmm_df["favor_two_moderate"].sum())
    n_str = int(gmm_df["favor_two_strong"].sum())
    n_fried = int((fried_df["friedman_q_fdr"] < 0.05).sum())
    if verbose:
        print(gmm_path)
        print(fried_path)
        print(pair_path)
        if pair_nominal_path:
            print(pair_nominal_path)
        print(
            f"GMM: {n_mod}/120 with delta_BIC>6 (two-component favored); "
            f"{n_str}/120 with delta_BIC>10."
        )
        print(
            f"Friedman FDR q<0.05: {n_fried}/15 band*ratio tests; "
            f"pairwise (FDR-triggered) rows: {len(pair_df)}."
        )
        if pair_nominal_path:
            print(
                f"Exploratory pairwise (Friedman p<{p_thr:g}) rows: {len(pair_nominal_rows)}."
            )

    return {
        "gmm_bic": gmm_path,
        "friedman": fried_path,
        "pairwise_holm_fdr": pair_path,
        "pairwise_holm_nominal": pair_nominal_path,
        "n_gmm_moderate": n_mod,
        "n_gmm_strong": n_str,
        "n_friedman_fdr_sig": n_fried,
        "n_pairwise_nominal_rows": len(pair_nominal_rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-csv", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Folder for CSV outputs (default: same folder as trial CSV).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filename stem for outputs (default: stem of trial-csv).",
    )
    parser.add_argument(
        "--pairwise-nominal-p",
        type=float,
        default=None,
        metavar="P",
        help=(
            "If set (e.g. 0.05), also write exploratory pairwise Holm tables "
            "when Friedman raw p < P even if FDR q >= 0.05. "
            "Saved as *_pairwise_channel_wilcoxon_holm_nominalP.csv"
        ),
    )
    args = parser.parse_args()
    run_crosschannel_analysis(
        args.trial_csv,
        output_dir=args.output_dir,
        prefix=args.prefix,
        pairwise_nominal_p=args.pairwise_nominal_p,
        verbose=True,
    )


if __name__ == "__main__":
    main()
