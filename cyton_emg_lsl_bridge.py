import re
import time
from collections import deque

import serial
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from pylsl import StreamInfo, StreamOutlet
from serial.tools import list_ports

# =========================
# CONFIG
# =========================
CYTON_PORT = "/dev/cu.usbserial-DP05I8FL"
CYTON_BOARD_ID = BoardIds.CYTON_BOARD.value

EMG_PORT = "/dev/tty.usbserial-110"
EMG_BAUD = 115200

# EMG label logic (normalized EMG units)
GRASP_THRESHOLD = 0.456
MERGE_GAP_SECONDS = 0.5
EVENT_MIN_GAP_SECONDS = 5.0
LABEL_MIN_ON_SECONDS = 0.35
BASELINE_WINDOW = 200
ARTIFACT_DROP_BELOW_BASELINE = 0.08

# LSL stream names
EEG_STREAM_NAME = "CytonEEG"
EEG_STREAM_TYPE = "EEG"
LABEL_STREAM_NAME = "MyoWareLabel"
LABEL_STREAM_TYPE = "Markers"


def resolve_serial_port(preferred_port):
    available = [p.device for p in list_ports.comports()]
    if preferred_port in available:
        return preferred_port

    candidates = [
        p
        for p in available
        if ("usb" in p.lower()) or ("wch" in p.lower()) or ("serial" in p.lower())
    ]
    if candidates:
        return candidates[0]
    return None


class EMGLabeler:
    def __init__(self):
        self.baseline_history = deque(maxlen=BASELINE_WINDOW)
        self.last_good_norm = None
        self.last_raw_grasp_time = None
        self.last_event_start_time = None
        self.candidate_on_since = None
        self.current_label = 0
        self.bad_lines = 0
        self.artifact_count = 0

    def _filter_artifact(self, norm):
        if len(self.baseline_history) >= 25:
            baseline = sorted(self.baseline_history)[len(self.baseline_history) // 2]
            cutoff = baseline - ARTIFACT_DROP_BELOW_BASELINE
            if norm < cutoff and self.last_good_norm is not None:
                self.artifact_count += 1
                if self.artifact_count <= 8:
                    print(
                        f"[EMG] Artifact ignored norm={norm:0.3f} "
                        f"(baseline={baseline:0.3f}, cutoff={cutoff:0.3f})"
                    )
                return self.last_good_norm

        self.baseline_history.append(norm)
        self.last_good_norm = norm
        return norm

    def update_from_raw(self, raw_value, t_now):
        norm = float(raw_value) / 4095.0
        norm = self._filter_artifact(norm)

        raw_label = 1 if norm > GRASP_THRESHOLD else 0
        if raw_label == 1:
            self.last_raw_grasp_time = t_now

        # Merge nearby grasp peaks into one candidate event.
        if (
            self.last_raw_grasp_time is not None
            and (t_now - self.last_raw_grasp_time) <= MERGE_GAP_SECONDS
        ):
            merged_candidate = 1
        else:
            merged_candidate = 0

        # Candidate must stay high long enough.
        if merged_candidate == 1:
            if self.candidate_on_since is None:
                self.candidate_on_since = t_now
            candidate_ready = (t_now - self.candidate_on_since) >= LABEL_MIN_ON_SECONDS
        else:
            self.candidate_on_since = None
            candidate_ready = False

        label = 1 if candidate_ready else 0

        # Enforce minimum inter-event gap (refractory period).
        if label == 1 and self.current_label != 1:
            if (
                self.last_event_start_time is not None
                and (t_now - self.last_event_start_time) < EVENT_MIN_GAP_SECONDS
            ):
                label = 0
            else:
                self.last_event_start_time = t_now

        if label != self.current_label:
            print("[LABEL] GRASP" if label == 1 else "[LABEL] OPEN")
            self.current_label = label

        return norm, label

    def ingest_serial(self, ser):
        while ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            match = re.search(r"\d+", line)
            if not match:
                self.bad_lines += 1
                if self.bad_lines <= 5:
                    print(f"[EMG] Skipping non-numeric line: {line}")
                continue

            raw = int(match.group(0))
            norm, _ = self.update_from_raw(raw, time.time())
            # Keep debug light.
            if self.bad_lines % 100 == 0:
                _ = norm


def init_brainflow():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = CYTON_PORT

    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    fs = BoardShim.get_sampling_rate(CYTON_BOARD_ID)
    eeg_channels = BoardShim.get_eeg_channels(CYTON_BOARD_ID)
    ts_channel = BoardShim.get_timestamp_channel(CYTON_BOARD_ID)

    if len(eeg_channels) < 8:
        raise RuntimeError(f"Expected at least 8 EEG channels, got {len(eeg_channels)}")

    board.start_stream()
    print(f"[BF] Cyton stream started on {CYTON_PORT}, fs={fs} Hz")
    return board, fs, eeg_channels[:8], ts_channel


def init_emg_serial():
    port = resolve_serial_port(EMG_PORT)
    if not port:
        raise RuntimeError("No EMG serial port found.")
    if port != EMG_PORT:
        print(f"[EMG] Configured port {EMG_PORT} not found. Using {port}.")
    ser = serial.Serial(port, EMG_BAUD, timeout=0)
    time.sleep(1.0)
    print(f"[EMG] Connected on {port} @ {EMG_BAUD}")
    return ser


def init_lsl_outlets(fs):
    eeg_info = StreamInfo(EEG_STREAM_NAME, EEG_STREAM_TYPE, 8, fs, "float32", "cyton_eeg_8")
    eeg_outlet = StreamOutlet(eeg_info)

    label_info = StreamInfo(
        LABEL_STREAM_NAME, LABEL_STREAM_TYPE, 1, fs, "float32", "myoware_label_1"
    )
    label_outlet = StreamOutlet(label_info)

    print(f"[LSL] EEG stream: {EEG_STREAM_NAME} ({fs} Hz, 8ch)")
    print(f"[LSL] Label stream: {LABEL_STREAM_NAME} ({fs} Hz, 1ch)")
    return eeg_outlet, label_outlet


def main():
    board = None
    ser = None
    try:
        board, fs, eeg_channels, ts_channel = init_brainflow()
        ser = init_emg_serial()
        eeg_outlet, label_outlet = init_lsl_outlets(fs)
        labeler = EMGLabeler()

        print("[i] Running LSL bridge. Ctrl+C to stop.")
        while True:
            # Update label state from any waiting EMG serial lines.
            labeler.ingest_serial(ser)

            data = board.get_board_data()
            if data.shape[1] == 0:
                time.sleep(0.002)
                continue

            eeg_chunk = data[eeg_channels, :]
            ts_chunk = data[ts_channel, :]
            n = eeg_chunk.shape[1]

            for i in range(n):
                ts = float(ts_chunk[i]) if i < len(ts_chunk) else time.time()
                eeg_sample = [float(eeg_chunk[ch, i]) for ch in range(8)]
                label_sample = [float(labeler.current_label)]

                eeg_outlet.push_sample(eeg_sample, timestamp=ts)
                label_outlet.push_sample(label_sample, timestamp=ts)

    except KeyboardInterrupt:
        print("\n[i] Keyboard interrupt. Stopping.")
    finally:
        if board is not None:
            try:
                board.stop_stream()
            except Exception:
                pass
            try:
                board.release_session()
            except Exception:
                pass
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        print("[i] Clean exit.")


if __name__ == "__main__":
    main()
