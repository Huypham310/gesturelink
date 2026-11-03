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

Firmware (needs [PlatformIO](https://platformio.org/)):

```bash
cd firmware
pio run -t upload         # Arduino Uno on the first USB serial port
```

## License

MIT
