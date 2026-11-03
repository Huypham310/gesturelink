"""Command line entry point (``gesturelink`` or ``python -m gesturelink``)."""

from __future__ import annotations

import argparse
import platform

from gesturelink import __version__


def probe_cameras(max_index: int = 4) -> list[tuple[int, int, int]]:
    """Indexes of cameras that open and return a frame, with the frame size."""
    import cv2

    found = []
    for i in range(max_index):
        # CAP_DSHOW opens much faster than the default MSMF backend on Windows
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(i, backend)
        try:
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            if ok and frame is not None:
                h, w = frame.shape[:2]
                found.append((i, w, h))
        finally:
            cap.release()
    return found


def probe_serial_ports() -> list[str]:
    from serial.tools import list_ports

    return [f"{p.device} ({p.description})" for p in list_ports.comports()]


def _info() -> int:
    import cv2
    import mediapipe as mp

    print(f"gesturelink {__version__}")
    print(f"python      {platform.python_version()}")
    print(f"opencv      {cv2.__version__}")
    print(f"mediapipe   {mp.__version__}")
    cams = probe_cameras()
    print("cameras     " + (", ".join(f"#{i} {w}x{h}" for i, w, h in cams) or "none"))
    ports = probe_serial_ports()
    print("serial      " + (", ".join(ports) or "none (plug in the Arduino)"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gesturelink", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("info", help="show versions, cameras and serial ports")

    args = parser.parse_args(argv)
    if args.command == "info":
        return _info()
    parser.print_help()
    return 0
