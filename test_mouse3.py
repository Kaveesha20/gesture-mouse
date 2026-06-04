import ctypes
import time

# Get current cursor position
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

point = POINT()
ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
print(f"Current cursor position: {point.x}, {point.y}")

print("Moving to 500, 500 in 3 seconds...")
time.sleep(3)

result = ctypes.windll.user32.SetCursorPos(500, 500)
print(f"SetCursorPos result: {result}")  # 1 = success, 0 = failed

time.sleep(1)
ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
print(f"Cursor position after move: {point.x}, {point.y}")