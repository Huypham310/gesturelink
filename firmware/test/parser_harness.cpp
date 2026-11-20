// Host-side harness for the firmware command parser.
//
// Reads one command per line from stdin, runs it through protocol::parseCommand (the same
// code that runs on the Uno) and prints what the firmware would act on or reply:
//   PING | STATUS | SERVO <deg> | LED <mask> | LCD <row> [<text>] | ERR <code>
// tests/test_firmware_parser.py builds this with g++ and checks the output.
#include <stdio.h>
#include <string.h>

#include "../src/protocol.h"

int main() {
    char buf[256];
    while (fgets(buf, sizeof(buf), stdin) != nullptr) {
        // strip the newline like the firmware main loop does
        buf[strcspn(buf, "\r\n")] = '\0';
        // the firmware never passes more than kLineMax bytes to the parser
        if (strlen(buf) > protocol::kLineMax) {
            puts("ERR line_too_long");
            continue;
        }
        protocol::Command cmd;
        const protocol::ParseError err = protocol::parseCommand(buf, cmd);
        if (err != protocol::ParseError::None) {
            printf("ERR %s\n", protocol::errorName(err));
            continue;
        }
        switch (cmd.type) {
            case protocol::CmdType::Ping: puts("PING"); break;
            case protocol::CmdType::Status: puts("STATUS"); break;
            case protocol::CmdType::Servo: printf("SERVO %d\n", cmd.value); break;
            case protocol::CmdType::Led: printf("LED %d\n", cmd.value); break;
            case protocol::CmdType::Lcd: printf("LCD %d [%s]\n", cmd.row, cmd.text); break;
        }
    }
    return 0;
}
