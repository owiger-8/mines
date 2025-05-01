import pyautogui
import time

# --- CONFIGURATION ---
SMILEY_IMAGE_PATH = 'images/facesmile.png'
CONFIDENCE_LEVEL = 0.7# Try lowering this to 0.7 if it still fails
# -------------------

print("--- Vision Debugger Running ---")
print(f"Attempting to find '{SMILEY_IMAGE_PATH}' on the screen.")
print("Switch to your Minesweeper window now. The script will check every 2 seconds.")
print("Press Ctrl+C to stop.")

try:
    while True:
        try:
            # Try to find the image on the screen
            location = pyautogui.locateOnScreen(SMILEY_IMAGE_PATH, confidence=CONFIDENCE_LEVEL)
            
            if location:
                print(f"SUCCESS! Smiley FOUND at coordinates: {location}")
                # Optional: Uncomment the line below to move the mouse to the found location
                # pyautogui.moveTo(pyautogui.center(location), duration=0.5)
            else:
                print("Smiley NOT found. Checking again...")

        except pyautogui.ImageNotFoundException:
            # This specific exception is another way of saying it wasn't found
            print("Smiley NOT found (Exception). Checking again...")
        
        # Wait for 2 seconds before trying again
        time.sleep(2)

except KeyboardInterrupt:
    print("\nDebugger stopped.")
