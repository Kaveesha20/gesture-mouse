import ctypes
import time

print("Moving in 3 seconds...")
time.sleep(3)
ctypes.windll.user32.SetCursorPos(500, 500)
print("Done")