import cv2
import numpy as np
import pickle
import os

map_file_path = "/Users/shivampankajyadav/Desktop/Open CV/InteractiveMap/Step1-GetCornerPoints/map.p"
cam_id = 0   # your camera index
width, height = 1920, 1080

# Variables
points = []  # list to store corner points

# Mouse callback
def mousepoints(event, x, y, flags, params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Selected point: {x, y}")

# Camera setup
cap = cv2.VideoCapture(cam_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

if not cap.isOpened():
    raise IOError(" Cannot open webcam. Check if it's in use by another app.")

print("👉 Instructions: Click 4 points on the map.")
print("   Press 'r' to reset, 's' to save, 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        break

    # Draw selected points
    for p in points:
        cv2.circle(img, p, 5, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Select 4 Corners", img)
    cv2.setMouseCallback("Select 4 Corners", mousepoints)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        points = []
        print(" Points reset")

    elif key == ord("s"):
        if len(points) == 4:
            with open(map_file_path, "wb") as f:
                pickle.dump(points, f)
            print(f" Points saved to {map_file_path}")
        else:
            print(" Please select 4 points before saving.")

    elif key == ord("q"):
        print(" Exiting without saving.")
        break

cap.release()
cv2.destroyAllWindows()