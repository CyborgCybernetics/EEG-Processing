# Cyton + MyoWare LSL Bridge

Stream `8-channel OpenBCI Cyton EEG` and a `1-channel EMG-derived label` at the same sampling rate to LSL, then record both streams into `.xdf` (e.g., with LabRecorder).

This repo includes:
- Live EMG thresholding and artifact handling logic
- Combined EEG + label LSL streaming at 250 Hz (Cyton rate)
- XDF to CSV conversion utilities for downstream modeling

## What This Produces

- LSL stream `CytonEEG` (`8 channels`, nominal `250 Hz`)
- LSL stream `MyoWareLabel` (`1 channel`, nominal `250 Hz`)
- Label values: `0 = OPEN`, `1 = GRASP`

Both streams are timestamp-aligned by pushing one label sample per EEG sample using the same timestamp.

## Files

- `cyton_emg_lsl_bridge.py`
  - Main runtime script
  - Reads Cyton via BrainFlow
  - Reads MyoWare serial and computes labels
  - Publishes both LSL outlets

- `myoware_wireless_live_emg.py`
  - Standalone EMG visualization/debug script
  - Includes thresholding, label logic, artifact suppression, and live plots

- `xdf_to_csv.py`
  - Converts recorded `.xdf` files into:
    - `*_CytonEEG.csv`
    - `*_MyoWareLabel.csv`
    - `*_merged_eeg_label.csv`

## Labeling Logic (EMG Thresholding)

Current label pipeline:
- Raw threshold: `GRASP_THRESHOLD = 0.456` (normalized)
- Artifact suppression: ignore sudden drops below rolling baseline (`ARTIFACT_DROP_BELOW_BASELINE`)
- Peak merge window: combine nearby peaks within `MERGE_GAP_SECONDS = 0.5`
- Minimum sustain: candidate must remain active for `LABEL_MIN_ON_SECONDS = 0.35`
- Refractory period: new events must be at least `EVENT_MIN_GAP_SECONDS = 5.0` apart

## Requirements

Install dependencies:

```bash
pip3 install brainflow pylsl pyserial matplotlib numpy pyxdf
```

Notes:
- `LabRecorder` is recommended for recording LSL streams to `.xdf`.
- On macOS, Cyton/MyoWare ports are usually `/dev/cu.usbserial-*`.

## Run: Stream EEG + Label to LSL

1. Update ports in `cyton_emg_lsl_bridge.py`:
   - `CYTON_PORT`
   - `EMG_PORT`
2. Start bridge:

```bash
python3 cyton_emg_lsl_bridge.py
```

3. Confirm logs show:
   - `CytonEEG (250 Hz, 8ch)`
   - `MyoWareLabel (250 Hz, 1ch)`

## Record to XDF

1. Open LabRecorder
2. Select both streams:
   - `CytonEEG`
   - `MyoWareLabel`
3. Start/stop recording to create `.xdf`

## Convert XDF to CSV

```bash
python3 xdf_to_csv.py "/path/to/recording.xdf"
```

This creates CSV files in the same folder as the XDF.

## Typical Workflow

1. Run `cyton_emg_lsl_bridge.py`
2. Record streams in LabRecorder
3. Convert `.xdf` to CSV using `xdf_to_csv.py`
4. Train classifier from merged EEG+label CSV

## Troubleshooting

- `UNABLE_TO_OPEN_PORT_ERROR`:
  - Check serial port path
  - Close OpenBCI GUI and any app holding the same port
- No EMG port found:
  - Replug device/cable
  - Recheck `/dev/cu.*` ports
- Label too noisy:
  - Tune `GRASP_THRESHOLD`, `LABEL_MIN_ON_SECONDS`, and artifact parameters
