#include "protocol.h"

#include <string.h>

namespace protocol {

namespace {

bool isSpace(char c) { return c == ' ' || c == '\t'; }

char upper(char c) { return (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c; }

// case-insensitive compare of a token (length n) against an upper-case word
bool tokenIs(const char* tok, size_t n, const char* word) {
    if (strlen(word) != n) return false;
    for (size_t i = 0; i < n; i++) {
        if (upper(tok[i]) != word[i]) return false;
    }
    return true;
}

bool isDigit(char c) { return c >= '0' && c <= '9'; }

int hexValue(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// parse số nguyên chặt chẽ, "12abc" bị từ chối chứ không đọc dở.
// Strict integer: optional '-', then decimal digits, or 0x/0X + hex digits. The whole string has
// to be consumed ("12abc" is rejected). Capped at 5 digits so we can't overflow a long.
bool parseInt(const char* s, long& out) {
    bool neg = false;
    if (*s == '-') {
        neg = true;
        s++;
    }
    if (*s == '\0') return false;
    long v = 0;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        s += 2;
        if (*s == '\0') return false;
        for (int n = 0; *s != '\0'; s++, n++) {
            const int h = hexValue(*s);
            if (h < 0 || n >= 4) return false;
            v = v * 16 + h;
        }
    } else {
        for (int n = 0; *s != '\0'; s++, n++) {
            if (!isDigit(*s) || n >= 5) return false;
            v = v * 10 + (*s - '0');
        }
    }
    out = neg ? -v : v;
    return true;
}

}  // namespace

ParseError parseCommand(char* line, Command& out) {
    // cắt khoảng trắng đầu/cuối dòng
    // trim trailing whitespace, then skip leading whitespace
    size_t len = strlen(line);
    while (len > 0 && isSpace(line[len - 1])) line[--len] = '\0';
    while (isSpace(*line)) line++;
    if (*line == '\0') return ParseError::UnknownCommand;

    // split the verb from the rest of the line
    char* verb = line;
    char* p = line;
    while (*p != '\0' && !isSpace(*p)) p++;
    const size_t verbLen = static_cast<size_t>(p - verb);
    while (isSpace(*p)) p++;
    char* args = p;  // may be empty

    memset(&out, 0, sizeof(out));

    if (tokenIs(verb, verbLen, "PING") || tokenIs(verb, verbLen, "STATUS")) {
        if (*args != '\0') return ParseError::BadArgs;
        out.type = tokenIs(verb, verbLen, "PING") ? CmdType::Ping : CmdType::Status;
        return ParseError::None;
    }

    if (tokenIs(verb, verbLen, "SERVO")) {
        long v;
        if (!parseInt(args, v)) return ParseError::BadArgs;
        if (v < 0 || v > kServoMax) return ParseError::OutOfRange;
        out.type = CmdType::Servo;
        out.value = static_cast<int16_t>(v);
        return ParseError::None;
    }

    if (tokenIs(verb, verbLen, "LED")) {
        long v;
        if (strlen(args) == 2 && upper(args[0]) == 'O' && upper(args[1]) == 'N') {
            v = kLedMaskAll;
        } else if (strlen(args) == 3 && upper(args[0]) == 'O' && upper(args[1]) == 'F' &&
                   upper(args[2]) == 'F') {
            v = 0;
        } else if (!parseInt(args, v)) {
            return ParseError::BadArgs;
        }
        if (v < 0 || v > kLedMaskAll) return ParseError::OutOfRange;
        out.type = CmdType::Led;
        out.value = static_cast<int16_t>(v);
        return ParseError::None;
    }

    if (tokenIs(verb, verbLen, "LCD")) {
        // "LCD <row> <text>": row is one digit, text is everything after ONE separator space
        // so the host can send leading spaces to align text. Text may be empty (clears the row).
        if (!isDigit(*args)) return ParseError::BadArgs;
        const int row = *args - '0';
        char* afterRow = args + 1;
        if (*afterRow != '\0' && !isSpace(*afterRow)) return ParseError::BadArgs;  // e.g. "LCD 10 hi"
        if (row >= kLcdRows) return ParseError::OutOfRange;
        const char* text = (*afterRow == '\0') ? afterRow : afterRow + 1;
        const size_t n = strlen(text);
        if (n > kLcdCols) return ParseError::TooLong;
        for (size_t i = 0; i < n; i++) {
            // printable ASCII only, the HD44780 ROM has no UTF-8
            if (text[i] < 0x20 || text[i] > 0x7E) return ParseError::BadArgs;
        }
        out.type = CmdType::Lcd;
        out.row = static_cast<uint8_t>(row);
        memcpy(out.text, text, n);
        out.text[n] = '\0';
        return ParseError::None;
    }

    return ParseError::UnknownCommand;
}

const char* errorName(ParseError e) {
    switch (e) {
        case ParseError::UnknownCommand: return "unknown_command";
        case ParseError::BadArgs: return "bad_args";
        case ParseError::OutOfRange: return "out_of_range";
        case ParseError::TooLong: return "too_long";
        default: return "none";
    }
}

}  // namespace protocol
