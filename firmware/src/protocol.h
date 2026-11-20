// Command parser for the GestureLink serial protocol (see firmware/PROTOCOL.md).
//
// Pure C++ on purpose: no Arduino headers, so the same file is compiled with g++ on the PC
// by tests/test_firmware_parser.py and checked without a board.
#pragma once

#include <stdint.h>

namespace protocol {

constexpr uint8_t kLedCount = 4;                       // LEDs on the board (bits of the mask)
constexpr uint8_t kLedMaskAll = (1u << kLedCount) - 1; // 0b1111
constexpr uint8_t kLcdCols = 16;
constexpr uint8_t kLcdRows = 2;
constexpr int kServoMax = 180;
constexpr uint8_t kLineMax = 48;  // longest accepted line, "LCD 1 " + 16 chars fits easily

enum class CmdType : uint8_t { Ping, Servo, Led, Lcd, Status };

// error names that go after "ERR " on the wire, see errorName()
enum class ParseError : uint8_t { None, UnknownCommand, BadArgs, OutOfRange, TooLong };

struct Command {
    CmdType type;
    int16_t value;            // servo degrees or LED mask
    uint8_t row;              // LCD row
    char text[kLcdCols + 1];  // LCD text, NUL terminated, not padded
};

// Parses one line (without the newline). The buffer is modified in place (trailing spaces
// are trimmed). Commands are case-insensitive, arguments are strict.
ParseError parseCommand(char* line, Command& out);

// "unknown_command", "bad_args", ... (the text after "ERR ")
const char* errorName(ParseError e);

}  // namespace protocol
