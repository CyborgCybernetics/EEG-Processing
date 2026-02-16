import time
import collections
import numpy as np
import serial

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import (
    DataFilter, DetrendOperations, NoiseTypes, WindowOperations, FilterTypes
)


CYTON_PORT = "/dev/ttyUSB0"   # RF dongle → Pi
ARDUINO_PORT = "/dev/ttyACM0" # Arduino Nano / Uno
ARDUINO_BAUD = 115200

# Notch filter (60 Hz region)
EN_NOTCH = NoiseTypes.SIXTY.value

# Classifier thresholds (tuned for alpha rhythm)
RATIO_THR     = 4.0      # posterior / frontal alpha ratio
POST_SUM_THR  = 10.0     # absolute posterior alpha threshold
ABS_CAP       = 5000.0   # hard ceiling on alpha power
CLIP_PERC     = 99.0     # winsorization percentile
USE_LOG_CLIP  = True

# History config (seconds)
HIST_SEC      = 60       # seconds of alpha history per channel
SMOOTH_SEC    = 3        # label majority window (streak smoothing)
WARMUP_SEC    = 20       # wait before trusting peak ratios
PEAK_THR      = 2.0      # peak ratio threshold for "bonus" in GOAT (we keep it)

DECISION_INTERVAL = 1.0  # seconds between classifier evaluations


def alpha_band_power_8_12(x, fs):
    """
    Compute alpha-band power (8–12 Hz) using BrainFlow's DataFilter.
    Adapted from GOAT script.
    """
    DataFilter.detrend(x, DetrendOperations.CONSTANT.value)
    DataFilter.remove_environmental_noise(x, fs, EN_NOTCH)
    DataFilter.perform_highpass(x, fs, 1.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
    DataFilter.perform_lowpass(x, fs, 40.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)

    n = len(x)
    nfft = max(32, 2 ** int(np.floor(np.log2(n))))
    overlap = min(nfft // 2, nfft - 1)

    psd, freqs = DataFilter.get_psd_welch(
        x, nfft, overlap, fs, WindowOperations.HANNING.value
    )
    idx = (freqs >= 8.0) & (freqs <= 12.0)
    return float(np.trapz(psd[idx], freqs[idx]))


def robust_clip(curr_val, hist_deque):
    """
    Robust outlier clipping / capping of alpha power,
    adapted from GOAT. Prevents huge spikes from dominating.
    """
    val = float(curr_val)
    val = min(val, ABS_CAP)

    # Not enough history yet
    if len(hist_deque) < max(10, int(0.2 * HIST_SEC)):
        return val

    arr = np.array(hist_deque, float)

    if USE_LOG_CLIP:
        arr = np.log10(np.clip(arr, 1e-9, None))
        p99 = np.percentile(arr, CLIP_PERC)
        cap = 10 ** p99
    else:
        p99 = np.percentile(arr, CLIP_PERC)
        cap = p99

    cap = min(cap * 2.0, ABS_CAP)
    return min(val, cap)


def hist_pstats(hist_deque):
    """
    Compute median and 99th percentile for peak-ratio debug.
    """
    if len(hist_deque) < max(10, int(0.2 * HIST_SEC)):
        return None
    arr = np.array(hist_deque, float)
    med = np.median(arr)
    p99 = np.percentile(arr, 99.0)
    return med, p99


def classify(alpha_vals_robust, peak_ratio_list, history_ready):
    """
    Core eyes-open / eyes-closed classifier from GOAT:

    Channel layout (assumed index mapping):
        0=Fpz, 1=Fz, 2=Cz, 3=C4, 4=C3, 5=Pz, 6=P3, 7=P4
    """
    Fpz, Fz, Cz, C4, C3, Pz, P3, P4 = alpha_vals_robust

    posterior = Pz + P3 + P4
    frontal   = Fpz + Fz + 1e-6
    ratio = posterior / frontal

    closed_by_ratio = (ratio > RATIO_THR)
    closed_by_sum   = (posterior > POST_SUM_THR)

       bonus = 0.0
    if history_ready:
        pz_peak = peak_ratio_list[5] if len(peak_ratio_list) > 5 else 1.0
        p3_peak = peak_ratio_list[6] if len(peak_ratio_list) > 6 else 1.0
        p4_peak = peak_ratio_list[7] if len(peak_ratio_list) > 7 else 1.0
        post_peak = (pz_peak + p3_peak + p4_peak) / 3.0
        if post_peak >= PEAK_THR:
            bonus = 1.0

    is_closed = (closed_by_ratio or closed_by_sum)
    label = "Eyes Closed" if is_closed else "Eyes Open"

    combo = [ratio, posterior, frontal, bonus]
    return label, combo


# HARDWARE

def init_brainflow():
    """
    Initialize BrainFlow for Cyton over USB.
    """
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = CYTON_PORT

    board_id = BoardIds.CYTON_BOARD.value
    board = BoardShim(board_id, params)

    board.prepare_session()
    fs = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)

    print(f"[BF] Prepared Cyton session on {CYTON_PORT}")
    print(f"[BF] Sampling rate: {fs} Hz, EEG channels: {eeg_channels}")

    board.start_stream()
    print("[BF] Stream started.")
    return board, board_id, fs, eeg_channels


def connect_arduino():
    """
    Open serial connection to Arduino.
    """
    print(f"[ARDUINO] Connecting on {ARDUINO_PORT} @ {ARDUINO_BAUD}...")
    ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
    time.sleep(2.0)  # allow Arduino to reset
    print("[ARDUINO] Connected.")
    return ser

# Main Loop
def main():
    board, board_id, fs, eeg_channels = init_brainflow()
    arduino = connect_arduino()

    # Rolling 1-second buffers per channel
    # (We will treat the 8 EEG channels in the assumed Fpz..P4 layout)
    n_eeg = len(eeg_channels)
    if n_eeg < 8:
        print(f"[ERR] Expected at least 8 EEG channels, got {n_eeg}")
        board.release_session()
        arduino.close()
        return

    # only 8 EEG channels for the classifier
    chan_indices = eeg_channels[:8]

    bufs = [collections.deque(maxlen=int(fs)) for _ in range(8)]
    alpha_hist = [collections.deque(maxlen=HIST_SEC) for _ in range(8)]

    last_decision_time = time.time()
    start_time = time.time()

    prev_label = None
    label_streak = 0
    last_state = None  # OPEN / CLOSE / REST

    print("[i] BCI hand controller running. Ctrl+C to stop.")

    try:
        while True:
            data = board.get_board_data()
            if data.shape[1] == 0:
                time.sleep(0.01)
                continue

            # Extract EEG rows for our chosen channels
            eeg_chunk = data[chan_indices, :]  # shape: (8, n_samples)

            # Append new samples into per-channel deques
            for ci in range(8):
                bufs[ci].extend(eeg_chunk[ci, :])

            now = time.time()
            # Only classify once per DECISION_INTERVAL and
            # IF we have at least 1 second of data in every buffer.
            if (now - last_decision_time) < DECISION_INTERVAL:
                continue
            if any(len(b) < int(fs) for b in bufs):
                continue

            last_decision_time = now

            # ----- Compute alpha power per channel -----
            alpha_raw_vals = []
            for ci in range(8):
                x = np.array(bufs[ci], dtype=float, copy=True)
                a_raw = alpha_band_power_8_12(x, fs)
                alpha_raw_vals.append(a_raw)

            # Update raw alpha history
            for ci in range(8):
                alpha_hist[ci].append(alpha_raw_vals[ci])

            # Robust clipped alpha & peak ratios
            alpha_vals = []
            peak_ratio = [1.0] * 8
            for ci in range(8):
                a_clip = robust_clip(alpha_raw_vals[ci], alpha_hist[ci])
                alpha_vals.append(a_clip)

                stats = hist_pstats(alpha_hist[ci])
                if stats is not None:
                    _, p99 = stats
                    if p99 > 0.0:
                        peak_ratio[ci] = alpha_raw_vals[ci] / p99
                    else:
                        peak_ratio[ci] = 1.0
                else:
                    peak_ratio[ci] = 1.0

            history_ready = (now - start_time) >= WARMUP_SEC
            curr_label, combo = classify(alpha_vals, peak_ratio, history_ready)

            # Simple streak-based smoothing: require label to persist
            if curr_label == prev_label:
                label_streak += 1
            else:
                prev_label = curr_label
                label_streak = 1

            # Only act if label is stable for a few decisions
            if label_streak >= 3:
                # Map label → hand state
                if curr_label == "Eyes Closed":
                    state = "CLOSE"
                else:
                    state = "OPEN"
            else:
                state = "NO_CHANGE"

            ratio, posterior, frontal, bonus = combo
            print(
                f"[DEBUG] alpha: "
                + "  ".join(f"{a:.3f}" for a in alpha_vals)
                + f" | label={curr_label} (streak={label_streak}) "
                  f"| state={state} | ratio={ratio:.2f}, post={posterior:.2f}, front={frontal:.2f}, bonus={bonus:.1f}"
            )

            # Send to ARUINO eyes open or closed
            if state != "NO_CHANGE" and state != last_state:
                print(f"[STATE] {last_state} → {state}  (sending to Arduino)")
                if state == "OPEN":
                    arduino.write(b"OPEN\n")
                elif state == "CLOSE":
                    arduino.write(b"CLOSE\n")
                last_state = state

    except KeyboardInterrupt:
        print("\n[i] Keyboard interrupt. Stopping.")
    finally:
        try:
            arduino.write(b"REST\n")
        except Exception:
            pass
        try:
            board.stop_stream()
        except Exception:
            pass
        board.release_session()
        arduino.close()
        print("[i] Clean exit.")


if __name__ == "__main__":
    main()
