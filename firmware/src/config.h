// Pin map and constants for the Arduino Uno build.
//
// Every pin can be overridden from platformio.ini with -DPIN_xxx=n; the defaults below match
// the wiring in README.md and firmware/wokwi/diagram.json.
#pragma once

#include <Arduino.h>

#ifndef PIN_SERVO
#define PIN_SERVO 9  // PWM pin for the servo signal wire
#endif

// 4 LEDs (each with a ~220 ohm resistor to GND); bit 0 of the mask is the first pin
#ifndef PIN_LED0
#define PIN_LED0 2
#endif
#ifndef PIN_LED1
#define PIN_LED1 3
#endif
#ifndef PIN_LED2
#define PIN_LED2 8
#endif
#ifndef PIN_LED3
#define PIN_LED3 10
#endif

// 16x2 HD44780 LCD in 4-bit mode (RW tied to GND, V0 to a contrast pot or GND)
#ifndef PIN_LCD_RS
#define PIN_LCD_RS 12
#endif
#ifndef PIN_LCD_EN
#define PIN_LCD_EN 11
#endif
#ifndef PIN_LCD_D4
#define PIN_LCD_D4 7
#endif
#ifndef PIN_LCD_D5
#define PIN_LCD_D5 6
#endif
#ifndef PIN_LCD_D6
#define PIN_LCD_D6 5
#endif
#ifndef PIN_LCD_D7
#define PIN_LCD_D7 4
#endif

constexpr unsigned long kBaud = 115200;
constexpr int kServoStart = 90;  // centre position at boot
constexpr char kFirmwareName[] = "gesturelink";
constexpr char kFirmwareVersion[] = "0.2";
