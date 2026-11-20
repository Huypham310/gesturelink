# GestureLink

Control an Arduino with hand gestures from a webcam.

MediaPipe finds the 21 hand landmarks in each frame, a small PyTorch classifier trained on a
self-collected landmark dataset turns them into a gesture, and a debounced command pipeline sends
the matching action (servo angle, LEDs, LCD text) to an Arduino Uno over a tiny serial protocol.
A FastAPI + React dashboard shows the live view, lets you remap gestures to actions, and reports
end-to-end latency.

> Status: early work in progress.

## Planned pieces

- `src/gesturelink/`: capture, landmark extraction, dataset tools, classifier, command pipeline
- `firmware/`: PlatformIO project for the Arduino Uno (servo, LEDs, 16x2 LCD)
- `web/`: React + TypeScript dashboard

Everything runs locally: no cloud APIs, the model is small enough for CPU inference.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
gesturelink info          # versions + cameras that open
```

Firmware (needs [PlatformIO](https://platformio.org/), details in [Firmware](#firmware)):

```bash
cd firmware
pio run -t upload         # Arduino Uno on the first USB serial port
```

## Data

Gestures (`configs/labels.json`): `open_palm`, `fist`, `thumbs_up`, `peace`, `point`, `ok`, `none`.

Each sample is one hand: 21 MediaPipe landmarks x (x, y, z) = 63 floats. Raw detections are stored
as CSV (`label,label_id,session_id,timestamp_ms,handedness,x0,y0,z0,...,z20`); `prepare` normalizes
them (subtract the wrist, divide by the wrist -> middle-knuckle distance) so position and distance
from the camera don't matter. Landmarks come from the MediaPipe **Tasks** `HandLandmarker`
(the old `mp.solutions` API is not in mediapipe 1.x).

```bash
gesturelink download-model                         # free hand_landmarker.task -> models/
gesturelink collect --label fist --samples 200     # webcam, one CSV per session in data/raw/
gesturelink collect --label ok --images photos/ok  # same, but from a folder of images
gesturelink prepare                                # data/raw -> data/processed/{dataset.npz,meta.json}
gesturelink synthesize --out data/raw/syn.csv      # fake landmarks, no camera needed
```

`prepare` does a stratified 80/20 train/val split (seed 42) and writes `dataset.npz`
(`X_train`, `y_train`, `X_val`, `y_val`) plus `meta.json` (labels, per-class counts, sources).
Options: `--val-fraction`, `--seed`, `--mirror-left`.

Real recordings live in `data/` and are gitignored. `tests/fixtures/landmarks/samples.csv` holds
140 synthetic samples (20 per gesture) so tests and CI run without a camera. See
`models/README.md` for the model download.

## Model

A small MLP (`63 -> 128 -> 64 -> 7`, 16,903 parameters) on the normalized landmark vector. It
runs on CPU; training on the fixture takes about a second.

```bash
gesturelink prepare                                        # data/raw -> data/processed
gesturelink train                                          # configs/train.json defaults
gesturelink train --epochs 100 --lr 0.0005 --seed 1        # override any config key
gesturelink predict --input data/raw/syn.csv -v            # classify raw CSV rows
```

`train` keeps the epoch with the best val accuracy and writes `models/gesturelink.pt` and
`models/metrics.json` (accuracy, macro F1, per-class report, confusion matrix, loss history).
Training augments with small gaussian noise and random left/right mirroring. From Python:

```python
from gesturelink.model import GesturePredictor

predictor = GesturePredictor("models/gesturelink.pt")
pred = predictor.predict(landmarks)   # raw (21, 3) MediaPipe landmarks
print(pred.label, pred.confidence)    # None if the hand is degenerate
```

Measured on synthetic data: 0.964 val accuracy on the 140-row test fixture (28 val samples),
1.000 on 1,050 synthetic samples. Synthetic hands are easy; real numbers come after collecting
real recordings. Details in `models/README.md`.

## Firmware

The Uno firmware (`firmware/`, PlatformIO + Arduino framework) drives a servo, 4 LEDs and a 16x2
LCD from newline-terminated text commands. It prints `READY gesturelink 0.2` after a reset and
answers every command with one `OK ...` or `ERR <code>` line. Full reference:
[`firmware/PROTOCOL.md`](firmware/PROTOCOL.md).

| Command | Reply |
|---|---|
| `PING` | `OK PONG gesturelink 0.2` |
| `SERVO <0-180>` | `OK SERVO <deg>` |
| `LED <mask 0-15 \| on \| off>` | `OK LED <mask>` |
| `LCD <row 0-1> <text up to 16 chars>` | `OK LCD <row>` |
| `STATUS` | `OK STATUS servo=<deg> led=<mask> uptime_ms=<ms>` |

Wiring (pins can be changed with `-DPIN_xxx` build flags, see `firmware/src/config.h`):

| Part | Uno pins |
|---|---|
| Servo signal | D9 (5 V and GND from the board; use an external supply for bigger servos) |
| LEDs 0-3 | D2, D3, D8, D10, each through 220 ohm to GND |
| LCD 16x2, 4-bit | RS D12, EN D11, D4-D7 on D7, D6, D5, D4; RW to GND; V0 to a pot or GND |

```bash
cd firmware
pio run                   # build (libraries Servo + LiquidCrystal are downloaded on first run)
pio run -t upload         # flash the board
```

Talk to the board from the PC with the `board` subcommands (the port is auto-detected, or pass
`--port COM3`):

```bash
gesturelink board ping
gesturelink board servo 135
gesturelink board led 0x5          # or: on / off / 0-15
gesturelink board lcd 0 thumbs up
gesturelink board status
```

From Python: `with BoardClient("COM3") as board: board.servo(90)` (`gesturelink.serial_client`).

No board? `firmware/wokwi/` has a [Wokwi](https://wokwi.com/) circuit (`diagram.json`) and a
`wokwi.toml` that loads the `pio run` build. With the Wokwi VS Code extension or `wokwi-cli`
running, its serial port is forwarded to `rfc2217://localhost:4000`:

```bash
gesturelink board --port rfc2217://localhost:4000 ping
```

Tests: `tests/test_serial_protocol.py` covers command encoding, reply parsing, the client (against
a fake serial port) and the CLI; `tests/test_firmware_parser.py` compiles the real firmware parser
(`firmware/src/protocol.cpp`) with `g++` and runs a table of good and bad commands through it
(skipped if there is no compiler). The one test that needs hardware is marked `board` and only runs
when `GESTURELINK_PORT` is set:

```bash
GESTURELINK_PORT=COM3 pytest -m board
```

## License

MIT
