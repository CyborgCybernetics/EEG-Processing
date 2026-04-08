"""
Per-trial triplet scatter from eeg_band_stats trial-level CSV.

Each event is one index on the x-axis. Three colored markers = three power ratios
(Welch band power), baseline = **rest** window [-2,-0.5]s pre-event:

  Post vs baseline:  ln(P_post/P_rest)
  Pre vs baseline:   ln(P_pre/P_rest)
  Post vs pre:       ln(P_post/P_pre)

Each is a **log power difference**: ln(P_a/P_b) = ln(P_a) - ln(P_b). Values far
from 0 = large relative change between those two windows for that event.

Also supports --y log10_power (raw log10 power in rest / pre / post).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BAND_ORDER = ("delta", "theta", "alpha", "beta", "gamma")

# Order matches: post–baseline, pre–baseline, post–pre
LOG_RATIO_COLORS = ("#ea580c", "#2563eb", "#7c3aed")  # orange, blue, violet


def _log_ratio_yvals_labels(sub: pd.DataFrame):
    """Columns from eeg_band_stats trial CSV; baseline = rest window."""
    yvals = [
        sub["log_ratio_post_vs_rest"].to_numpy(dtype=float),
        sub["log_ratio_pre_vs_rest"].to_numpy(dtype=float),
        sub["log_ratio_post_vs_pre"].to_numpy(dtype=float),
    ]
    labels = (
        r"Post vs baseline: $\ln(P_{post}/P_{rest})$",
        r"Pre vs baseline: $\ln(P_{pre}/P_{rest})$",
        r"Post vs pre: $\ln(P_{post}/P_{pre})$",
    )
    return yvals, labels, LOG_RATIO_COLORS


def _sort_events(sub: pd.DataFrame) -> pd.DataFrame:
    return sub.sort_values(
        ["filename", "trial_id", "event_time_sec"], kind="mergesort"
    ).reset_index(drop=True)


def _triplet_scatter_ax(
    ax,
    sub: pd.DataFrame,
    *,
    y_mode: str,
    show_legend: bool,
    xlabel: str | None,
):
    n = len(sub)
    x = np.arange(n, dtype=float)
    jit = np.array([-0.22, 0.0, 0.22], dtype=float)

    if y_mode == "log10_power":
        pr = sub["power_rest"].to_numpy(dtype=float)
        pp = sub["power_pre_movement"].to_numpy(dtype=float)
        pm = sub["power_post_movement"].to_numpy(dtype=float)
        eps = np.finfo(float).eps
        yvals = [
            np.log10(np.maximum(pr, eps)),
            np.log10(np.maximum(pp, eps)),
            np.log10(np.maximum(pm, eps)),
        ]
        labels = ("Rest baseline", "Pre-movement", "Post-movement")
        colors = ("#4c566a", "#5e81ac", "#d08700")
        ylabel = r"$\log_{10}$ band power (Welch)"
    else:
        yvals, labels, colors = _log_ratio_yvals_labels(sub)
        ylabel = r"$\ln$(power ratio); 0 = same power"

    for k in range(3):
        ax.scatter(
            x + jit[k],
            yvals[k],
            s=12,
            alpha=0.72,
            c=colors[k],
            edgecolors="none",
            label=labels[k],
        )

    if y_mode == "log_ratio":
        ax.axhline(
            0.0, color="#64748b", linewidth=0.9, linestyle="--", alpha=0.75, zorder=0
        )
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_legend:
        ax.legend(
            frameon=True,
            loc="upper right",
            fontsize=8,
            markerscale=1.2,
            handletextpad=0.4,
        )
    ax.grid(True, axis="y", alpha=0.25)


def save_all_bands_log_ratio_figure(
    df: pd.DataFrame,
    channel: int,
    out_path: str,
    *,
    title: str | None = None,
) -> str:
    """Five stacked bands, three ln-ratios per event; save PNG to out_path."""
    n_expect = None
    fig_y = 2.85 * len(BAND_ORDER)
    n_guess = max(
        100,
        len(df[(df["channel"] == channel) & (df["band"] == "delta")]),
    )
    fig_x = max(11.0, n_guess / 32.0)
    fig, axes = plt.subplots(
        len(BAND_ORDER),
        1,
        sharex=True,
        figsize=(fig_x, fig_y),
        constrained_layout=True,
    )
    for i, band in enumerate(BAND_ORDER):
        sub = df[(df["channel"] == channel) & (df["band"] == band)].copy()
        if sub.empty:
            plt.close(fig)
            raise SystemExit(f"No rows for channel={channel} band={band}")
        sub = _sort_events(sub)
        if n_expect is None:
            n_expect = len(sub)
        elif len(sub) != n_expect:
            plt.close(fig)
            raise SystemExit(
                f"Channel {channel} band {band}: {len(sub)} rows; "
                f"expected {n_expect} (event alignment)."
            )
        ax = axes[i]
        _triplet_scatter_ax(
            ax,
            sub,
            y_mode="log_ratio",
            show_legend=(i == 0),
            xlabel=None,
        )
        ax.set_title(f"{band.capitalize()} — Ch {channel} — n={len(sub)} events")

    axes[-1].set_xlabel(
        "Event index (sorted: file, trial_id, event time); 3 ln-ratios per event"
    )
    fig.suptitle(
        title
        or (
            f"Power ratios vs baseline (rest) — ch {channel} | "
            r"$\ln(P_a/P_b)$ = log-power difference; $|y|$ large $\Rightarrow$ strong contrast"
        ),
        fontsize=11,
        y=1.01,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_all_channels_plots(
    trial_csv: str,
    output_dir: str,
    *,
    file_stem: str,
) -> list[str]:
    """
    Write eight PNGs (ch 1–8), five bands each, log-ratio triplets per event.
    Filenames: {file_stem}_ch{N}_ALL_BANDS_log_ratio_triplets.png
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(trial_csv)
    paths = []
    for ch in range(1, 9):
        out = os.path.join(
            output_dir,
            f"{file_stem}_ch{ch}_ALL_BANDS_log_ratio_triplets.png",
        )
        paths.append(save_all_bands_log_ratio_figure(df, ch, out))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trial-csv",
        type=str,
        required=True,
        help="Path to *_trial_level.csv from eeg_band_stats.py",
    )
    parser.add_argument("--channel", type=int, default=5, help="1-based EEG channel.")
    parser.add_argument(
        "--band",
        type=str,
        default="beta",
        help="Band name when not using --all-bands.",
    )
    parser.add_argument(
        "--all-bands",
        action="store_true",
        help="One stacked panel per band (delta…gamma), same events on x.",
    )
    parser.add_argument(
        "--all-channels",
        action="store_true",
        help="With --all-bands --y log_ratio: save eight PNGs (ch 1…8). Ignores --channel.",
    )
    parser.add_argument(
        "--y",
        choices=("log10_power", "log_ratio"),
        default="log10_power",
        help="Y values: window powers (rest/pre/post) or the three log-ratio contrasts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="PNG path (default: alongside CSV with suffix).",
    )
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.trial_csv)

    if args.all_bands:
        if args.y != "log_ratio":
            raise SystemExit("--all-bands is intended for --y log_ratio (log triplets per event).")
        base, _ = os.path.splitext(args.trial_csv)
        if args.all_channels:
            paths = []
            for ch in range(1, 9):
                if args.output:
                    root, ext = os.path.splitext(args.output)
                    if not ext:
                        ext = ".png"
                    out = f"{root}_ch{ch}{ext}"
                else:
                    out = f"{base}_ch{ch}_ALL_BANDS_log_ratio_triplets.png"
                paths.append(save_all_bands_log_ratio_figure(df, ch, out, title=args.title))
            for p in paths:
                print(p)
            return

        out = args.output
        if not out:
            out = f"{base}_ch{args.channel}_ALL_BANDS_log_ratio_triplets.png"
        print(save_all_bands_log_ratio_figure(df, args.channel, out, title=args.title))
        return

    sub = df[(df["channel"] == args.channel) & (df["band"] == args.band.lower())].copy()
    if sub.empty:
        raise SystemExit(
            f"No rows for channel={args.channel} band={args.band}. "
            f"Columns: {list(df.columns)}"
        )

    sub = _sort_events(sub)
    n = len(sub)
    x = np.arange(n, dtype=float)

    jit = np.array([-0.22, 0.0, 0.22], dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.0, n / 40.0), 4.5), constrained_layout=True)

    if args.y == "log10_power":
        pr = sub["power_rest"].to_numpy(dtype=float)
        pp = sub["power_pre_movement"].to_numpy(dtype=float)
        pm = sub["power_post_movement"].to_numpy(dtype=float)
        eps = np.finfo(float).eps
        yvals = [
            np.log10(np.maximum(pr, eps)),
            np.log10(np.maximum(pp, eps)),
            np.log10(np.maximum(pm, eps)),
        ]
        labels = ("Rest baseline", "Pre-movement", "Post-movement")
        colors = ("#4c566a", "#5e81ac", "#d08700")
        ylabel = r"$\log_{10}$ band power (Welch)"
    else:
        yvals, labels, colors = _log_ratio_yvals_labels(sub)
        ylabel = r"$\ln$(power ratio); 0 = same power"

    for k in range(3):
        ax.scatter(
            x + jit[k],
            yvals[k],
            s=14,
            alpha=0.65,
            c=colors[k],
            edgecolors="none",
            label=labels[k],
        )

    if args.y == "log_ratio":
        ax.axhline(0.0, color="#3b4252", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Trial index (sorted: filename, trial_id, event time)")
    ax.set_ylabel(ylabel)
    ttl = args.title or (
        f"Channel {args.channel}, {args.band} — {n} trials — {args.y}"
    )
    ax.set_title(ttl)
    ax.legend(frameon=True, loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    out = args.output
    if not out:
        base, _ = os.path.splitext(args.trial_csv)
        out = f"{base}_ch{args.channel}_{args.band}_{args.y}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
