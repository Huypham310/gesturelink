"""Runs the real firmware parser (firmware/src/protocol.cpp) on the PC.

The harness is compiled with g++/clang++ if one is installed, otherwise these tests skip.
No board and no Arduino toolchain involved.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

FIRMWARE = Path(__file__).parent.parent / "firmware"
COMPILER = shutil.which("g++") or shutil.which("clang++")

pytestmark = pytest.mark.skipif(COMPILER is None, reason="no C++ compiler on PATH")


@pytest.fixture(scope="module")
def harness(tmp_path_factory) -> Path:
    exe = tmp_path_factory.mktemp("parser") / "parser_harness.exe"
    cmd = [
        COMPILER, "-std=c++17", "-Wall", "-Wextra", "-Werror",
        str(FIRMWARE / "test" / "parser_harness.cpp"), str(FIRMWARE / "src" / "protocol.cpp"),
        "-o", str(exe),
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return exe


def run(harness: Path, *lines: str) -> list[str]:
    out = subprocess.run(
        [str(harness)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True
    )
    return out.stdout.splitlines()


CASES = [
    # --- simple commands, case-insensitive
    ("PING", "PING"),
    ("ping", "PING"),
    ("  Status  ", "STATUS"),
    ("PING now", "ERR bad_args"),
    ("STATUS 1", "ERR bad_args"),
    # --- servo
    ("SERVO 0", "SERVO 0"),
    ("SERVO 90", "SERVO 90"),
    ("servo   180", "SERVO 180"),
    ("SERVO 181", "ERR out_of_range"),
    ("SERVO -1", "ERR out_of_range"),
    ("SERVO", "ERR bad_args"),
    ("SERVO abc", "ERR bad_args"),
    ("SERVO 12abc", "ERR bad_args"),
    ("SERVO 1 2", "ERR bad_args"),
    ("SERVO 999999999999", "ERR bad_args"),
    # --- leds
    ("LED 0", "LED 0"),
    ("LED 5", "LED 5"),
    ("LED 15", "LED 15"),
    ("LED 0xA", "LED 10"),
    ("LED 0xa", "LED 10"),
    ("LED on", "LED 15"),
    ("LED OFF", "LED 0"),
    ("LED 16", "ERR out_of_range"),
    ("LED 0x10", "ERR out_of_range"),
    ("LED -1", "ERR out_of_range"),
    ("LED", "ERR bad_args"),
    ("LED maybe", "ERR bad_args"),
    ("LED 0x", "ERR bad_args"),
    ("LED onn", "ERR bad_args"),
    # --- lcd
    ("LCD 0 hello", "LCD 0 [hello]"),
    ("LCD 1 thumbs up", "LCD 1 [thumbs up]"),
    ("LCD 1   indented", "LCD 1 [  indented]"),  # only the first space is a separator
    ("LCD 0 ", "LCD 0 []"),
    ("LCD 0", "LCD 0 []"),
    ("LCD 0 1234567890123456", "LCD 0 [1234567890123456]"),
    ("LCD 0 12345678901234567", "ERR too_long"),
    ("LCD 2 x", "ERR out_of_range"),
    ("LCD 10 x", "ERR bad_args"),
    ("LCD x hi", "ERR bad_args"),
    ("LCD", "ERR bad_args"),
    ("LCD 0\tbad", "LCD 0 [bad]"),
    # --- junk
    ("", "ERR unknown_command"),
    ("   ", "ERR unknown_command"),
    ("DANCE", "ERR unknown_command"),
    ("PINGS", "ERR unknown_command"),
    ("SERVOS 90", "ERR unknown_command"),
    ("x" * 49, "ERR line_too_long"),
]


def test_parser_table(harness):
    # one process for the whole table: each input line yields exactly one output line
    got = run(harness, *(line for line, _ in CASES))
    assert len(got) == len(CASES)
    for (line, expected), actual in zip(CASES, got, strict=True):
        assert actual == expected, f"{line!r}"


def test_parser_rejects_non_ascii_lcd_text(harness):
    # raw byte 0xE9 isn't printable ASCII, the HD44780 would show garbage
    exe = subprocess.run(
        [str(harness)], input=b"LCD 0 caf\xe9\n", capture_output=True, check=True
    )
    assert exe.stdout.decode().strip() == "ERR bad_args"
