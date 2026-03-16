#include <Mouse.h>

// 定義波特率，需與 Python 端一致
const unsigned long BAUD_RATE = 115200;

void setup() {
  // 初始化 HID 滑鼠功能
  Mouse.begin();
  
  // 初始化序列埠通訊
  Serial.begin(BAUD_RATE);
  
  // 為了安全，LED 亮起表示準備就緒
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW); // 預設關閉 LED
}

void loop() {
  // 檢查是否收到完整的 3 bytes 數據 (Command, Arg1, Arg2)
  if (Serial.available() >= 3) {
    char cmd = (char)Serial.read();
    char arg1 = (char)Serial.read();
    char arg2 = (char)Serial.read();
    
    if (cmd == 1) {
      // 移動指令
      char dx = arg1;
      char dy = arg2;
      if (dx != 0 || dy != 0) {
        Mouse.move(dx, dy, 0);
      }
    } else if (cmd == 2) {
      // 點擊指令
      char action = arg1; // 1=click, 2=press, 3=release
      if (action == 1) {
        Mouse.click(MOUSE_LEFT);
      } else if (action == 2) {
        Mouse.press(MOUSE_LEFT);
      } else if (action == 3) {
        Mouse.release(MOUSE_LEFT);
      }
    }
    // 未知指令則直接丟棄，因為已經從 buffer 中讀出
  }
}
