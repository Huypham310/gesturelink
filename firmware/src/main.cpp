// GestureLink firmware, Arduino Uno.
//
// Listens for newline-terminated text commands on the serial port and drives a servo,
// 4 LEDs and a 16x2 LCD. Parsing lives in protocol.cpp, this file only touches hardware.
// Protocol reference: firmware/PROTOCOL.md
#include <Arduino.h>
#include <LiquidCrystal.h>
#include <Servo.h>

#include "config.h"
#include "protocol.h"

namespace {

using protocol::Command;
using protocol::CmdType;
using protocol::ParseError;

const uint8_t kLedPins[protocol::kLedCount] = {PIN_LED0, PIN_LED1, PIN_LED2, PIN_LED3};

LiquidCrystal lcd(PIN_LCD_RS, PIN_LCD_EN, PIN_LCD_D4, PIN_LCD_D5, PIN_LCD_D6, PIN_LCD_D7);
Servo servo;

// trạng thái hiện tại, STATUS sẽ trả lại các giá trị này
// current state, reported by STATUS
int servoDeg = kServoStart;
uint8_t ledMask = 0;

// one input line at a time; if it overflows we drop the whole line (see loop)
char line[protocol::kLineMax + 1];
uint8_t len = 0;
bool overflow = false;

// the built-in LED (pin 13) flashes briefly on PING, switched off again from loop()
unsigned long pingBlinkUntil = 0;

void setLeds(uint8_t mask) {
    ledMask = mask;
    for (uint8_t i = 0; i < protocol::kLedCount; i++) {
        digitalWrite(kLedPins[i], (mask >> i) & 1 ? HIGH : LOW);
    }
}

// Writes text into one LCD row, padded with spaces so old characters get overwritten.
void showRow(uint8_t row, const char* text) {
    lcd.setCursor(0, row);
    uint8_t n = 0;
    for (; text[n] != '\0' && n < protocol::kLcdCols; n++) lcd.write(text[n]);
    for (; n < protocol::kLcdCols; n++) lcd.write(' ');
}

void replyError(const char* name) {
    Serial.print(F("ERR "));
    Serial.println(name);
}

void execute(const Command& cmd) {
    switch (cmd.type) {
        case CmdType::Ping:
            Serial.print(F("OK PONG "));
            Serial.print(kFirmwareName);
            Serial.print(' ');
            Serial.println(kFirmwareVersion);
            digitalWrite(LED_BUILTIN, HIGH);
            pingBlinkUntil = millis() + 50;
            break;
        case CmdType::Servo:
            servoDeg = cmd.value;
            servo.write(servoDeg);
            Serial.print(F("OK SERVO "));
            Serial.println(servoDeg);
            break;
        case CmdType::Led:
            setLeds(static_cast<uint8_t>(cmd.value));
            Serial.print(F("OK LED "));
            Serial.println(ledMask);
            break;
        case CmdType::Lcd:
            showRow(cmd.row, cmd.text);
            Serial.print(F("OK LCD "));
            Serial.println(cmd.row);
            break;
        case CmdType::Status:
            Serial.print(F("OK STATUS servo="));
            Serial.print(servoDeg);
            Serial.print(F(" led="));
            Serial.print(ledMask);
            Serial.print(F(" uptime_ms="));
            Serial.println(millis());
            break;
    }
}

void handleLine(char* text) {
    Command cmd;
    const ParseError err = protocol::parseCommand(text, cmd);
    if (err == ParseError::None) {
        execute(cmd);
    } else {
        replyError(protocol::errorName(err));
    }
}

}  // namespace

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    for (uint8_t i = 0; i < protocol::kLedCount; i++) pinMode(kLedPins[i], OUTPUT);
    setLeds(0);

    // servo về giữa (90 độ) lúc khởi động để khỏi giật
    servo.attach(PIN_SERVO);
    servo.write(servoDeg);

    lcd.begin(protocol::kLcdCols, protocol::kLcdRows);
    showRow(0, "GestureLink");
    showRow(1, "waiting for host");

    Serial.begin(kBaud);
    Serial.print(F("READY "));
    Serial.print(kFirmwareName);
    Serial.print(' ');
    Serial.println(kFirmwareVersion);
}

void loop() {
    if (pingBlinkUntil != 0 && static_cast<long>(millis() - pingBlinkUntil) >= 0) {
        digitalWrite(LED_BUILTIN, LOW);
        pingBlinkUntil = 0;
    }

    while (Serial.available() > 0) {
        const char c = static_cast<char>(Serial.read());
        if (c == '\r') continue;  // accept CRLF from terminals (bỏ qua \r)
        if (c == '\n') {
            if (overflow) {
                replyError("line_too_long");
            } else if (len > 0) {
                line[len] = '\0';
                handleLine(line);
            }
            len = 0;
            overflow = false;
        } else if (len < protocol::kLineMax) {
            line[len++] = c;
        } else {
            // too long: keep eating until the newline, then answer with an error
            // instead of running a half-cut command
            overflow = true;
        }
    }
}
