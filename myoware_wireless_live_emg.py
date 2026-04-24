import time
from collections import deque
import re

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import serial
from serial.tools import list_ports

# =========================
# SETTINGS
# =========================
PORT = "/dev/tty.usbserial-110"
BAUD = 115200
WINDOW_SECONDS = 5
SAMPLE_RATE = 200
SMOOTH = 10
Y_MIN = 0.44
Y_MAX = 0.48
# Interpreting "4.56" as normalized 0.456 for this signal scale.
GRASP_THRESHOLD = 0.456
MERGE_GAP_SECONDS = 0.5
EVENT_MIN_GAP_SECONDS = 5.0
LABEL_MIN_ON_SECONDS = 0.35
BASELINE_WINDOW = 200
ARTIFACT_DROP_BELOW_BASELINE = 0.08

MAX_POINTS = WINDOW_SECONDS * SAMPLE_RATE

# =========================
# SMOOTHING
# =========================
def moving_avg(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def resolve_port(preferred_port):
    available = [p.device for p in list_ports.comports()]
    if preferred_port in available:
        return preferred_port, available

    candidates = [
        p
        for p in available
        if ("usb" in p.lower()) or ("wch" in p.lower()) or ("serial" in p.lower())
    ]
    if candidates:
        return candidates[0], available

    return None, available


def main():
    # =========================
    # SETUP SERIAL
    # =========================
    port, available_ports = resolve_port(PORT)
    if not port:
        print("No compatible serial port found.")
        print(f"Configured port: {PORT}")
        print("Available ports:")
        for p in available_ports:
            print(f"  - {p}")
        return

    if port != PORT:
        print(f"Configured port {PORT} not found. Using {port} instead.")

    ser = serial.Serial(port, BAUD, timeout=1)
    time.sleep(2)

    # =========================
    # DATA STORAGE
    # =========================
    data = deque(maxlen=MAX_POINTS)
    labels = deque(maxlen=MAX_POINTS)
    times = deque(maxlen=MAX_POINTS)
    bad_lines = 0
    samples = 0
    peak_count = 0
    max_peak = 0.0
    last_emg_print = 0.0
    last_grasp_state = None
    last_raw_grasp_time = None
    last_event_start_time = None
    candidate_on_since = None
    artifact_count = 0
    baseline_history = deque(maxlen=BASELINE_WINDOW)
    last_good_norm = None

    start = time.time()

    # =========================
    # PLOT SETUP
    # =========================
    fig, (ax_emg, ax_label) = plt.subplots(2, 1, sharex=True)
    emg_line, = ax_emg.plot([], [])
    label_line, = ax_label.plot([], [], drawstyle="steps-post")

    ax_emg.set_ylim(Y_MIN, Y_MAX)
    ax_emg.set_xlim(0, WINDOW_SECONDS)
    ax_emg.set_title("Live EMG (MyoWare Wireless)")
    ax_emg.set_ylabel("Normalized EMG")
    ax_emg.axhline(GRASP_THRESHOLD, linestyle="--")

    ax_label.set_ylim(-0.1, 1.1)
    ax_label.set_yticks([0, 1])
    ax_label.set_yticklabels(["OPEN", "GRASP"])
    ax_label.set_xlabel("Time (s)")
    ax_label.set_ylabel("Label")

    # =========================
    # UPDATE LOOP
    # =========================
    def update(frame):
        nonlocal bad_lines, samples, peak_count, max_peak, last_emg_print
        nonlocal last_grasp_state
        nonlocal last_raw_grasp_time, last_event_start_time
        nonlocal candidate_on_since
        nonlocal artifact_count, last_good_norm

        try:
            waiting = ser.in_waiting
        except (OSError, serial.SerialException):
            ax_emg.set_title("Serial disconnected - reconnect device and restart script")
            print("Serial connection lost. Close plot, reconnect device, and rerun.")
            return emg_line, label_line

        while waiting:
            try:
                val = ser.readline().decode(errors="ignore").strip()
                if val:
                    # Accept either "1234" or lines that contain a number.
                    match = re.search(r"\d+", val)
                    if not match:
                        bad_lines += 1
                        if bad_lines <= 5:
                            print(f"Skipping non-numeric line: {val}")
                        continue

                    raw = int(match.group(0))
                    norm = raw / 4095.0  # ESP32 is 12-bit (0-4095)

                    t = time.time() - start
                    # Ignore sharp downward artifacts below rolling baseline.
                    if len(baseline_history) >= 25:
                        baseline = float(np.median(baseline_history))
                        cutoff = baseline - ARTIFACT_DROP_BELOW_BASELINE
                        if norm < cutoff and last_good_norm is not None:
                            artifact_count += 1
                            if artifact_count <= 8:
                                print(
                                    f"Artifact ignored norm={norm:0.3f} "
                                    f"(baseline={baseline:0.3f}, cutoff={cutoff:0.3f})"
                                )
                            norm = last_good_norm
                        else:
                            baseline_history.append(norm)
                            last_good_norm = norm
                    else:
                        baseline_history.append(norm)
                        last_good_norm = norm

                    raw_label = 1 if norm > GRASP_THRESHOLD else 0
                    if raw_label == 1:
                        last_raw_grasp_time = t

                    # Merge grasp peaks that are separated by <= 0.5s.
                    if (
                        last_raw_grasp_time is not None
                        and (t - last_raw_grasp_time) <= MERGE_GAP_SECONDS
                    ):
                        merged_candidate = 1
                    else:
                        merged_candidate = 0

                    # Candidate must stay high for at least LABEL_MIN_ON_SECONDS.
                    if merged_candidate == 1:
                        if candidate_on_since is None:
                            candidate_on_since = t
                        candidate_ready = (t - candidate_on_since) >= LABEL_MIN_ON_SECONDS
                    else:
                        candidate_on_since = None
                        candidate_ready = False

                    label = 1 if candidate_ready else 0

                    # Enforce minimum time between event starts.
                    if label == 1 and last_grasp_state != 1:
                        if (
                            last_event_start_time is not None
                            and (t - last_event_start_time) < EVENT_MIN_GAP_SECONDS
                        ):
                            label = 0
                        else:
                            last_event_start_time = t

                    if label != last_grasp_state:
                        print("LABEL=GRASP" if label == 1 else "LABEL=OPEN")
                        last_grasp_state = label

                    data.append(norm)
                    labels.append(label)
                    times.append(t)
                    samples += 1

                    # Print live EMG output periodically without flooding.
                    if t - last_emg_print >= 0.25:
                        print(f"EMG raw={raw:4d} norm={norm:0.3f}")
                        last_emg_print = t
            except (ValueError, UnicodeDecodeError):
                # Skip malformed serial lines.
                pass
            except (OSError, serial.SerialException):
                ax_emg.set_title("Serial disconnected - reconnect device and restart script")
                print("Serial connection lost while reading.")
                return emg_line, label_line

            try:
                waiting = ser.in_waiting
            except (OSError, serial.SerialException):
                ax_emg.set_title("Serial disconnected - reconnect device and restart script")
                print("Serial connection lost. Close plot, reconnect device, and rerun.")
                return emg_line, label_line

        elapsed = time.time() - start
        ax_emg.set_title(
            f"Live EMG (MyoWare Wireless)  t={elapsed:0.1f}s  n={samples}  peaks={peak_count}"
        )

        if len(data) > SMOOTH:
            smoothed = moving_avg(list(data), SMOOTH)
            t = list(times)[-len(smoothed):]
            label_view = list(labels)[-len(smoothed):]

            if t:
                latest = t[-1]
                shifted = [x - latest + WINDOW_SECONDS for x in t]
                emg_line.set_data(shifted, smoothed)
                label_line.set_data(shifted, label_view)

                # Simple local-maximum peak detection on smoothed data.
                peaks = (smoothed[1:-1] > smoothed[:-2]) & (
                    smoothed[1:-1] > smoothed[2:]
                )
                if np.any(peaks):
                    local_peaks = smoothed[1:-1][peaks]
                    highest_local = float(np.max(local_peaks))
                    if highest_local > max_peak + 0.02:
                        max_peak = highest_local
                        peak_count += 1
                        print(f"PEAK norm={max_peak:0.3f}")

        return emg_line, label_line

    ani = animation.FuncAnimation(fig, update, interval=20, cache_frame_data=False)

    print(f"Connected on {port} @ {BAUD}. Waiting for EMG samples...")

    try:
        plt.show()
    finally:
        # Keep a live reference until the window exits.
        del ani
        ser.close()


if __name__ == "__main__":
    main()
