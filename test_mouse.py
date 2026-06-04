from pynput.mouse import Controller
import time

mouse = Controller()
print("Moving in 3 seconds...")
time.sleep(3)
mouse.position = (500, 500)
print("Done")