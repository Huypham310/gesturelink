# GestureLink serial protocol

Plain ASCII over USB serial, 115200 baud, 8N1. One command per line, one reply per command.

- Host -> board: `<COMMAND> [args]\n` (a `\r\n` ending is accepted too).
- Board -> host: exactly one line per command, starting with `OK` or `ERR`.
- Commands are case-insensitive (`servo 90` works), arguments are strict: junk after a number
  is an error, not ignored.
- Lines longer than 48 bytes are dropped and answered with `ERR line_too_long`.
- The Uno resets when a serial port is opened (DTR), so the board prints a banner once the
  firmware is up. The host should wait for it before sending commands:

```
READY gesturelink 0.2
```

## Commands

| Command | Reply on success | Notes |
|---|---|---|
| `PING` | `OK PONG gesturelink 0.2` | Also flashes the on-board LED (pin 13) for 50 ms. |
| `SERVO <deg>` | `OK SERVO <deg>` | `deg` is an integer 0-180. |
| `LED <mask>` | `OK LED <mask>` | `mask` is 0-15 (bit 0 = first LED), decimal or `0x` hex. |
| `LED on` / `LED off` | `OK LED 15` / `OK LED 0` | Shortcuts for all LEDs on / off. |
| `LCD <row> <text>` | `OK LCD <row>` | `row` is 0 or 1. `text` is up to 16 printable ASCII characters (0x20-0x7E); the rest of the row is cleared. Empty text clears the row. |
| `STATUS` | `OK STATUS servo=<deg> led=<mask> uptime_ms=<ms>` | Last values that were set. |

For `LCD`, everything after the single space that follows the row number is the text, so
inner and leading spaces are kept (`LCD 1   hi` shows `  hi`). Trailing spaces are trimmed.

## Errors

`ERR <code>` where `<code>` is one of:

| Code | Meaning |
|---|---|
| `unknown_command` | Empty or unrecognised command word. |
| `bad_args` | Missing, non-numeric or extra arguments; non-printable character in LCD text. |
| `out_of_range` | Number parsed fine but outside the allowed range (`SERVO 181`, `LED 16`, `LCD 2 x`). |
| `too_long` | LCD text longer than 16 characters. |
| `line_too_long` | Input line longer than 48 bytes. |

## Example session

```
< READY gesturelink 0.2
> PING
< OK PONG gesturelink 0.2
> SERVO 135
< OK SERVO 135
> LED 0x5
< OK LED 5
> LCD 0 thumbs_up
< OK LCD 0
> LCD 1 conf 0.97
< OK LCD 1
> STATUS
< OK STATUS servo=135 led=5 uptime_ms=8123
> SERVO 200
< ERR out_of_range
```

## Pins (Arduino Uno)

| Part | Pins |
|---|---|
| Servo signal | D9 |
| LEDs 0-3 | D2, D3, D8, D10 (220 ohm to GND) |
| LCD 16x2 (4-bit) | RS D12, EN D11, D4-D7 -> D7, D6, D5, D4 |

Override any of them with `-DPIN_SERVO=6` etc. in `build_flags` (see `src/config.h`).

The parser is in `src/protocol.cpp` and has no Arduino dependencies, so
`tests/test_firmware_parser.py` compiles it with `g++` on the PC and checks it without a board.
