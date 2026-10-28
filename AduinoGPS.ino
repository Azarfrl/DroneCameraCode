String command = "";
bool gpsStreaming = false;

void setup() {
  Serial.begin(9600);
  Serial1.begin(9600);
  Serial.println("GPS controller ready");
}

void loop() {
  // Receive START and STOP commands from Python
  while (Serial.available()) {
    char character = Serial.read();

    if (character == '\n' || character == '\r') {
      command.trim();

      if (command == "START") {
        gpsStreaming = true;
        Serial.println("GPS streaming started");
      }
      else if (command == "STOP") {
        gpsStreaming = false;
        Serial.println("GPS streaming stopped");
      }

      command = "";
    }
    else {
      command += character;
    }
  }

  // Forward raw GPS data without converting coordinates
  
  while (Serial1.available()) {
    char character = Serial1.read();

    if (gpsStreaming) {
      Serial.write(character);
    }
  }
}
