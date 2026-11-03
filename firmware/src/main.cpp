// GestureLink firmware, Arduino Uno.
//
// For now: announces itself and answers a ping so the host can find the right port.
// The command protocol (servo, LEDs, LCD) comes in the firmware phase.
#include <Arduino.h>

namespace {

constexpr unsigned long kBaud = 115200;
constexpr uint8_t kLed = LED_BUILTIN;  // pin 13 on the Uno

// one line at a time; 32 bytes is plenty for "PING" and later commands
char line[32];
uint8_t len = 0;

void handle(const char* cmd) {
    if (strcmp(cmd, "PING") == 0) {
        Serial.println(F("PONG gesturelink 0.1"));
        // quick blink so you can see on the board which port answered
        digitalWrite(kLed, HIGH);
        delay(50);
        digitalWrite(kLed, LOW);
    } else {
        Serial.print(F("ERR unknown "));
        Serial.println(cmd);
    }
}

}  // namespace

void setup() {
    pinMode(kLed, OUTPUT);
    Serial.begin(kBaud);
    Serial.println(F("READY gesturelink 0.1"));
}

void loop() {
    while (Serial.available() > 0) {
        const char c = static_cast<char>(Serial.read());
        if (c == '\r') continue;
        if (c == '\n') {
            line[len] = '\0';
            if (len > 0) handle(line);
            len = 0;
        } else if (len < sizeof(line) - 1) {
            line[len++] = c;
        } else {
            // overlong line: drop it rather than act on half a command
            len = 0;
        }
    }
}
