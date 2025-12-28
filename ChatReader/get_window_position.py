import time
import pyautogui
import keyboard

while True:
    if keyboard.is_pressed("k"):
        x, y = pyautogui.position()
        print(f"Mouse is at ({x}, {y})")
        time.sleep(1)
