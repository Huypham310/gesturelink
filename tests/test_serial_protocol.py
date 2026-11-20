"""Serial protocol tests that run without any hardware.

The live-board test at the bottom is marked ``board`` and is skipped unless the
GESTURELINK_PORT env var points at a port (see conftest.py).
"""

from __future__ import annotations

import os

import pytest

from gesturelink import cli
from gesturelink.serial_client import (
    BoardClient,
    BoardError,
    BoardTimeout,
    ProtocolError,
    Reply,
    encode_lcd,
    encode_led,
    encode_ping,
    encode_servo,
    encode_status,
    is_ready_banner,
    parse_led_value,
    parse_reply,
    parse_status,
)

# ------------------------------------------------------------------ encoding


def test_simple_commands():
    assert encode_ping() == b"PING\n"
    assert encode_status() == b"STATUS\n"


@pytest.mark.parametrize("deg", [0, 1, 90, 180])
def test_servo_ok(deg):
    assert encode_servo(deg) == f"SERVO {deg}\n".encode()


@pytest.mark.parametrize("bad", [-1, 181, 1000, 90.5, "90", None, True])
def test_servo_rejects_bad_angles(bad):
    with pytest.raises(ValueError):
        encode_servo(bad)


@pytest.mark.parametrize(
    ("value", "mask"),
    [("on", 15), ("OFF", 0), (True, 15), (False, 0), (5, 5), ("5", 5), ("0x5", 5), ("0b101", 5),
     (" On ", 15), (15, 15), (0, 0)],
)  # fmt: skip
def test_led_values(value, mask):
    assert parse_led_value(value) == mask
    assert encode_led(value) == f"LED {mask}\n".encode()


@pytest.mark.parametrize("bad", [16, -1, "16", "0x10", "blue", "", 1.5, None])
def test_led_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        encode_led(bad)


def test_lcd_encoding():
    assert encode_lcd(0, "thumbs_up") == b"LCD 0 thumbs_up\n"
    assert encode_lcd(1, "") == b"LCD 1 \n"  # empty text clears the row
    assert encode_lcd(1, "a  b") == b"LCD 1 a  b\n"  # inner spaces survive


def test_lcd_truncates_or_raises():
    long_text = "x" * 20
    assert encode_lcd(0, long_text) == b"LCD 0 " + b"x" * 16 + b"\n"
    with pytest.raises(ValueError):
        encode_lcd(0, long_text, truncate=False)


@pytest.mark.parametrize("row", [-1, 2, "0", None])
def test_lcd_rejects_bad_rows(row):
    with pytest.raises(ValueError):
        encode_lcd(row, "hi")


@pytest.mark.parametrize("text", ["caf\u00e9", "tab\there", "line\nbreak"])
def test_lcd_rejects_non_printable_text(text):
    with pytest.raises(ValueError):
        encode_lcd(0, text)


# ------------------------------------------------------------------- parsing


def test_parse_ok_replies():
    r = parse_reply("OK PONG gesturelink 0.2\r\n")
    assert r.ok and r.tag == "PONG" and r.args == ("gesturelink", "0.2")
    r = parse_reply(b"OK SERVO 90\n")
    assert r.ok and r.tag == "SERVO" and r.args == ("90",)


def test_parse_error_reply():
    r = parse_reply("ERR out_of_range")
    assert not r.ok and r.tag == "out_of_range"


@pytest.mark.parametrize("line", ["", "OK", "ERR", "READY gesturelink 0.2", "hello", "okay SERVO"])
def test_parse_reply_rejects_garbage(line):
    with pytest.raises(ProtocolError):
        parse_reply(line)


def test_ready_banner():
    assert is_ready_banner("READY gesturelink 0.2")
    assert is_ready_banner(b"READY gesturelink 0.2\r\n")
    assert not is_ready_banner("OK PONG gesturelink 0.2")
    assert not is_ready_banner("READY other 1.0")


def test_parse_status():
    status = parse_status(parse_reply("OK STATUS servo=135 led=5 uptime_ms=8123"))
    assert (status.servo, status.led, status.uptime_ms) == (135, 5, 8123)


@pytest.mark.parametrize(
    "line",
    [
        "OK PONG gesturelink 0.2",
        "OK STATUS servo=1 led=2",
        "OK STATUS servo=x led=1 uptime_ms=3",
    ],
)
def test_parse_status_rejects_other_replies(line):
    with pytest.raises(ProtocolError):
        parse_status(parse_reply(line))


# --------------------------------------------------------------- client logic


class FakeSerial:
    """Stands in for a pyserial port: replays scripted lines, records what was written."""

    def __init__(self, lines: list[str]):
        self.lines = [ln.encode() + b"\r\n" for ln in lines]
        self.written: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> int:
        self.written.append(data)
        return len(data)

    def flush(self) -> None:
        pass

    def readline(self) -> bytes:
        if not self.lines:
            return b""
        # like a real board: OK/ERR lines only appear after a command was written,
        # banner/noise lines can show up any time
        if not self.written and self.lines[0].startswith((b"OK", b"ERR")):
            return b""
        return self.lines.pop(0)

    def close(self) -> None:
        self.closed = True


def make_client(lines: list[str], **kw) -> tuple[BoardClient, FakeSerial]:
    fake = FakeSerial(lines)
    client = BoardClient(
        "FAKE", timeout=0.05, boot_timeout=0.05, serial_factory=lambda *a: fake, **kw
    )
    return client, fake


def test_client_waits_for_banner_then_pings():
    client, fake = make_client(["noise", "READY gesturelink 0.2", "OK PONG gesturelink 0.2"])
    with client as board:
        reply = board.ping()
    assert reply.tag == "PONG"
    assert fake.written == [b"PING\n"]
    assert fake.closed


def test_client_commands_send_expected_bytes():
    client, fake = make_client(
        ["OK SERVO 45", "OK LED 5", "OK LCD 1", "OK STATUS servo=45 led=5 uptime_ms=10"]
    )
    with client as board:
        board.servo(45)
        board.led("0x5")
        board.lcd(1, "fist")
        status = board.status()
    assert fake.written == [b"SERVO 45\n", b"LED 5\n", b"LCD 1 fist\n", b"STATUS\n"]
    assert status.servo == 45 and status.led == 5


def test_client_works_without_banner():
    # e.g. over RFC2217 (Wokwi) there is no reset, so no READY line
    client, _ = make_client(["OK PONG gesturelink 0.2"])
    with client as board:
        assert board.ping().ok


def test_client_skips_stray_lines():
    client, _ = make_client(["READY gesturelink 0.2", "READY gesturelink 0.2", "OK SERVO 10"])
    with client as board:
        assert board.servo(10).tag == "SERVO"


def test_client_raises_board_error():
    client, _ = make_client(["READY gesturelink 0.2", "ERR out_of_range"])
    with client as board, pytest.raises(BoardError) as info:
        board.servo(90)
    assert info.value.code == "out_of_range"


def test_client_timeout():
    client, _ = make_client(["READY gesturelink 0.2"])
    with client as board, pytest.raises(BoardTimeout):
        board.ping()


def test_client_validates_before_sending():
    client, fake = make_client(["READY gesturelink 0.2"])
    with client as board, pytest.raises(ValueError):
        board.servo(500)
    assert fake.written == []  # nothing reached the wire


def test_command_before_open_fails():
    client, _ = make_client([])
    with pytest.raises(RuntimeError):
        client.command(encode_ping())


def test_reply_fields_helper():
    assert Reply(True, "STATUS", ("a=1", "b=2", "x")).fields == {"a": "1", "b": "2"}


# ----------------------------------------------------------------------- CLI


@pytest.fixture
def fake_board(monkeypatch):
    """Patch BoardClient so CLI tests talk to a FakeSerial."""
    import gesturelink.serial_client as sc

    state = {"lines": [], "fake": None}
    real = sc.BoardClient

    def factory(port, baud=115200, timeout=2.0, **kw):
        def make_serial(*a):
            state["fake"] = FakeSerial(["READY gesturelink 0.2", *state["lines"]])
            return state["fake"]

        return real(port, baud=baud, timeout=0.05, boot_timeout=0.01, serial_factory=make_serial)

    monkeypatch.setattr(sc, "BoardClient", factory)
    return state


def test_cli_ping(fake_board, capsys):
    fake_board["lines"] = ["OK PONG gesturelink 0.2"]
    assert cli.main(["board", "--port", "X", "ping"]) == 0
    assert "PONG" in capsys.readouterr().out


def test_cli_lcd_joins_words(fake_board, capsys):
    fake_board["lines"] = ["OK LCD 0"]
    assert cli.main(["board", "--port", "X", "lcd", "0", "hello", "world"]) == 0
    assert fake_board["fake"].written == [b"LCD 0 hello world\n"]


def test_cli_status(fake_board, capsys):
    fake_board["lines"] = ["OK STATUS servo=90 led=0 uptime_ms=5"]
    assert cli.main(["board", "--port", "X", "status"]) == 0
    assert "servo=90 led=0 uptime_ms=5" in capsys.readouterr().out


def test_cli_board_error_exit_code(fake_board, capsys):
    fake_board["lines"] = ["ERR out_of_range"]
    assert cli.main(["board", "--port", "X", "led", "7"]) == 1
    assert "out_of_range" in capsys.readouterr().out


def test_cli_rejects_bad_argument_without_io(fake_board, capsys):
    assert cli.main(["board", "--port", "X", "servo", "999"]) == 2
    assert "invalid argument" in capsys.readouterr().out


def test_cli_timeout(fake_board, capsys):
    fake_board["lines"] = []
    assert cli.main(["board", "--port", "X", "ping"]) == 1
    assert "no reply" in capsys.readouterr().out


def test_cli_needs_subcommand(capsys):
    assert cli.main(["board", "--port", "X"]) == 2


# ------------------------------------------------------------------ live board


@pytest.mark.board
def test_live_board_roundtrip():
    """Real hardware (or Wokwi via rfc2217://localhost:4000): set GESTURELINK_PORT to run."""
    port = os.environ["GESTURELINK_PORT"]
    with BoardClient(port) as board:
        assert board.ping().tag == "PONG"
        board.servo(120)
        board.led(5)
        board.lcd(0, "pytest")
        status = board.status()
        assert (status.servo, status.led) == (120, 5)
        with pytest.raises(BoardError):
            board.command(b"SERVO 181\n")  # bypass host validation on purpose
