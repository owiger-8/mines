import pyautogui
import time


SMILEY_IMAGE_PATH = 'images/facesmile.png'
CONFIDENCE_LEVEL = 0.7


print("--- Vision Debugger Running ---")
print(f"Attempting to find '{SMILEY_IMAGE_PATH}' on the screen.")
print("Switch to your Minesweeper window now. The script will check every 2 seconds.")
print("Press Ctrl+C to stop.")

try:
    while True:
        try:
            location = pyautogui.locateOnScreen(SMILEY_IMAGE_PATH, confidence=CONFIDENCE_LEVEL)
            
            if location:
                print(f"SUCCESS! Smiley FOUND at coordinates: {location}")
             
            else:
                print("Smiley NOT found. Checking again...")

        except pyautogui.ImageNotFoundException:

            print("Smiley NOT found (Exception). Checking again...")
        

        time.sleep(2)

except KeyboardInterrupt:
    print("\nDebugger stopped.")
