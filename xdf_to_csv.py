import argparse
import csv
import os
import sys


def load_xdf_streams(xdf_path, deps_path=None):
    if deps_path:
        sys.path.insert(0, deps_path)
    import pyxdf  # noqa: PLC0415

    streams, _ = pyxdf.load_xdf(xdf_path)
    return streams


def find_stream(streams, name):
    for stream in streams:
        sname = stream["info"].get("name", [""])[0]
        if sname == name:
            return stream
    return None


def write_eeg_csv(stream, out_path):
    rows = stream["time_series"]
    ts = stream["time_stamps"]
    ch = int(stream["info"]["channel_count"][0])
    headers = ["timestamp"] + [f"eeg_{i+1}" for i in range(ch)]

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for t, vals in zip(ts, rows):
            w.writerow([t, *vals])


def write_label_csv(stream, out_path):
    rows = stream["time_series"]
    ts = stream["time_stamps"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "label"])
        for t, vals in zip(ts, rows):
            label = vals[0] if hasattr(vals, "__len__") else vals
            w.writerow([t, label])


def write_merged_csv(eeg_stream, label_stream, out_path):
    eeg_rows = eeg_stream["time_series"]
    eeg_ts = eeg_stream["time_stamps"]
    label_rows = label_stream["time_series"]
    label_ts = label_stream["time_stamps"]
    ch = int(eeg_stream["info"]["channel_count"][0])

    headers = ["timestamp"] + [f"eeg_{i+1}" for i in range(ch)] + ["label"]

    # Label stream was pushed at same fs/timestamps; use nearest-neighbor by index.
    n = min(len(eeg_ts), len(label_ts), len(eeg_rows), len(label_rows))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(n):
            label = (
                label_rows[i][0]
                if hasattr(label_rows[i], "__len__")
                else label_rows[i]
            )
            w.writerow([eeg_ts[i], *eeg_rows[i], label])


def main():
    parser = argparse.ArgumentParser(description="Convert XDF EEG+label to CSV files")
    parser.add_argument("xdf_path", help="Path to .xdf file")
    parser.add_argument(
        "--deps-path",
        default="",
        help="Optional local deps path containing pyxdf (e.g. ./ .deps)",
    )
    parser.add_argument("--eeg-name", default="CytonEEG")
    parser.add_argument("--label-name", default="MyoWareLabel")
    args = parser.parse_args()

    xdf_path = os.path.abspath(args.xdf_path)
    out_dir = os.path.dirname(xdf_path)
    base = os.path.splitext(os.path.basename(xdf_path))[0]

    streams = load_xdf_streams(xdf_path, args.deps_path or None)
    eeg = find_stream(streams, args.eeg_name)
    label = find_stream(streams, args.label_name)

    if eeg is None or label is None:
        names = [s["info"].get("name", [""])[0] for s in streams]
        raise RuntimeError(
            f"Could not find streams eeg={args.eeg_name}, label={args.label_name}. "
            f"Found: {names}"
        )

    eeg_csv = os.path.join(out_dir, f"{base}_CytonEEG.csv")
    label_csv = os.path.join(out_dir, f"{base}_MyoWareLabel.csv")
    merged_csv = os.path.join(out_dir, f"{base}_merged_eeg_label.csv")

    write_eeg_csv(eeg, eeg_csv)
    write_label_csv(label, label_csv)
    write_merged_csv(eeg, label, merged_csv)

    print("Wrote:")
    print(f"  {eeg_csv}")
    print(f"  {label_csv}")
    print(f"  {merged_csv}")


if __name__ == "__main__":
    main()
