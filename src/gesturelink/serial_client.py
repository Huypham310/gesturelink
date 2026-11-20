"""Host side of the Arduino serial protocol (see firmware/PROTOCOL.md).

Two layers:

* pure functions (``encode_*``, ``parse_reply``, ``parse_status``) that turn Python values into
  command lines and reply lines into objects. No hardware needed, fully unit tested.
* ``BoardClient``, a thin pyserial wrapper that uses those functions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

BAUD = 115200
SERVO_MAX = 180
LED_COUNT = 4
LED_MASK_ALL = (1 << LED_COUNT) - 1  # 0b1111
LCD_COLS = 16
LCD_ROWS = 2

# USB vendor ids of the usual Uno / clone chips (Arduino, CH340, FTDI, Silicon Labs)
_BOARD_VIDS = {0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4}


class ProtocolError(ValueError):
    """A reply line that doesn't follow the protocol."""


class BoardError(RuntimeError):
    """The board answered ``ERR <code>`` (the code is in ``.code``)."""

    def __init__(self, code: str, command: str = ""):
        super().__init__(f"board replied ERR {code}" + (f" to {command!r}" if command else ""))
        self.code = code
        self.command = command


class BoardTimeout(TimeoutError):
    """No reply from the board in time."""


# ---------------------------------------------------------------- encoding


def encode_ping() -> bytes:
    return b"PING\n"


def encode_status() -> bytes:
    return b"STATUS\n"


def encode_servo(degrees: int) -> bytes:
    """``SERVO <deg>`` for an integer angle 0-180."""
    # bool là subclass của int nên phải loại ra riêng, True != 1 độ
    if isinstance(degrees, bool) or not isinstance(degrees, int):
        raise ValueError(f"servo angle must be an int, got {degrees!r}")
    if not 0 <= degrees <= SERVO_MAX:
        raise ValueError(f"servo angle {degrees} outside 0-{SERVO_MAX}")
    return f"SERVO {degrees}\n".encode("ascii")


def parse_led_value(value: int | str | bool) -> int:
    """Turn ``on`` / ``off`` / ``5`` / ``0x5`` / True into a validated 4-bit LED mask."""
    if isinstance(value, bool):
        return LED_MASK_ALL if value else 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "on":
            return LED_MASK_ALL
        if text == "off":
            return 0
        try:
            value = int(text, 0)  # base 0 accepts decimal, 0x.. and 0b.. (tự nhận dạng hệ số)
        except ValueError:
            raise ValueError(f"bad LED value {value!r} (use on, off or a mask 0-15)") from None
    if not isinstance(value, int):
        raise ValueError(f"bad LED value {value!r}")
    if not 0 <= value <= LED_MASK_ALL:
        raise ValueError(f"LED mask {value} outside 0-{LED_MASK_ALL}")
    return value


def encode_led(value: int | str | bool) -> bytes:
    """``LED <mask>`` (always sent as a plain decimal number)."""
    return f"LED {parse_led_value(value)}\n".encode("ascii")


def encode_lcd(row: int, text: str, truncate: bool = True) -> bytes:
    """``LCD <row> <text>``.

    Text must be printable ASCII. Longer than 16 chars is cut (the display can't show more)
    unless ``truncate=False``, then it raises.
    """
    if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < LCD_ROWS:
        raise ValueError(f"LCD row must be 0 or 1, got {row!r}")
    if not all(0x20 <= ord(c) <= 0x7E for c in text):
        raise ValueError(f"LCD text must be printable ASCII: {text!r}")
    if len(text) > LCD_COLS:
        if not truncate:
            raise ValueError(f"LCD text is {len(text)} chars, max {LCD_COLS}")
        text = text[:LCD_COLS]
    return f"LCD {row} {text}\n".encode("ascii")


# ----------------------------------------------------------------- parsing


@dataclass(frozen=True)
class Reply:
    """One parsed reply line. ``tag`` is PONG/SERVO/... on success, the error code on ERR."""

    ok: bool
    tag: str
    args: tuple[str, ...] = ()
    raw: str = ""

    @property
    def fields(self) -> dict[str, str]:
        """``key=value`` args as a dict (used by STATUS)."""
        return dict(a.split("=", 1) for a in self.args if "=" in a)


@dataclass(frozen=True)
class BoardStatus:
    servo: int
    led: int
    uptime_ms: int


def parse_reply(line: str | bytes) -> Reply:
    """Parse ``OK ...`` / ``ERR ...``. Raises ProtocolError for anything else."""
    if isinstance(line, bytes):
        line = line.decode("ascii", errors="replace")
    raw = line.strip()
    parts = raw.split()
    if len(parts) >= 2 and parts[0] in ("OK", "ERR"):
        return Reply(parts[0] == "OK", parts[1], tuple(parts[2:]), raw)
    raise ProtocolError(f"not a protocol reply: {raw!r}")


def is_ready_banner(line: str | bytes) -> bool:
    """True for the ``READY gesturelink x.y`` line printed after a reset."""
    if isinstance(line, bytes):
        line = line.decode("ascii", errors="replace")
    parts = line.split()
    return len(parts) >= 2 and parts[0] == "READY" and parts[1] == "gesturelink"


def parse_status(reply: Reply) -> BoardStatus:
    """Build a BoardStatus from an ``OK STATUS servo=.. led=.. uptime_ms=..`` reply."""
    if not reply.ok or reply.tag != "STATUS":
        raise ProtocolError(f"not a STATUS reply: {reply.raw!r}")
    f = reply.fields
    try:
        return BoardStatus(int(f["servo"]), int(f["led"]), int(f["uptime_ms"]))
    except (KeyError, ValueError) as exc:
        raise ProtocolError(f"malformed STATUS reply: {reply.raw!r}") from exc


# ------------------------------------------------------------ port helpers


def find_board_port() -> str | None:
    """First serial port that looks like an Arduino, or None."""
    from serial.tools import list_ports

    for p in list_ports.comports():
        desc = (p.description or "") + " " + (p.manufacturer or "")
        if p.vid in _BOARD_VIDS or "arduino" in desc.lower() or "ch340" in desc.lower():
            return p.device
    return None


@dataclass
class BoardClient:
    """Talks to the firmware. Use as a context manager.

    ``port`` can be a device name (``COM3``) or any pyserial URL, e.g.
    ``rfc2217://localhost:4000`` for the Wokwi simulator. ``serial_factory`` lets tests
    inject a fake port.
    """

    port: str
    baud: int = BAUD
    timeout: float = 2.0  # seconds to wait for one reply
    boot_timeout: float = 3.0  # seconds to wait for the READY banner after opening
    serial_factory: Any = None
    _ser: Any = field(default=None, init=False, repr=False)

    def open(self) -> BoardClient:
        if self._ser is not None:
            return self
        if self.serial_factory is not None:
            self._ser = self.serial_factory(self.port, self.baud, self.timeout)
        else:
            import serial

            self._ser = serial.serial_for_url(self.port, baudrate=self.baud, timeout=self.timeout)
        self._wait_ready()
        return self

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def __enter__(self) -> BoardClient:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _read_line(self) -> str:
        return self._ser.readline().decode("ascii", errors="replace").strip()

    def _wait_ready(self) -> None:
        """Wait for the READY banner (the Uno resets when the port opens).

        If it never shows up we carry on anyway: over RFC2217 (Wokwi) or when the board was
        already running there is no reset, and the first command will tell us if it's alive.
        """
        # Uno reset mất ~1-2s sau khi mở cổng, nên chờ banner thay vì gửi lệnh ngay
        deadline = time.monotonic() + self.boot_timeout
        while time.monotonic() < deadline:
            line = self._read_line()
            if line and is_ready_banner(line):
                return

    def command(self, data: bytes) -> Reply:
        """Send one encoded command and return the OK reply. Raises BoardError on ERR."""
        if self._ser is None:
            raise RuntimeError("port is not open")
        self._ser.write(data)
        self._ser.flush()
        name = data.decode("ascii").strip()
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            line = self._read_line()
            if not line:
                continue
            try:
                reply = parse_reply(line)
            except ProtocolError:
                continue  # stray line (e.g. a late READY banner), keep waiting; dòng lạ thì bỏ qua
            if not reply.ok:
                raise BoardError(reply.tag, name)
            return reply
        raise BoardTimeout(f"no reply to {name!r} within {self.timeout}s")

    # one method per command
    def ping(self) -> Reply:
        return self.command(encode_ping())

    def servo(self, degrees: int) -> Reply:
        return self.command(encode_servo(degrees))

    def led(self, value: int | str | bool) -> Reply:
        return self.command(encode_led(value))

    def lcd(self, row: int, text: str) -> Reply:
        return self.command(encode_lcd(row, text))

    def status(self) -> BoardStatus:
        return parse_status(self.command(encode_status()))
