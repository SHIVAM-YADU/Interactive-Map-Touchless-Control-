import cv2
import numpy as np
import pickle
import os

map_file_path = "/Users/shivampankajyadav/Desktop/Open CV/InteractiveMap/Step1-GetCornerPoints/map.p"
countries_file_path = "countries.p"
cam_id = 0   # your camera index
width, height = 1920, 1080

# open a connection to the camera
cap = cv2.VideoCapture(cam_id)  # for webcam
cap.set(3, width)
cap.set(4, height)

# load map points
with open(map_file_path, "rb") as file_obj:
    map_points = pickle.load(file_obj)
print("Loaded map coordinates:", map_points)

# Temporary list to store the points of the current polygon being marked
current_polygon = []
# Counter to keep track of number of countries marked
counter = 0

# Load previously saved polygons if file exists
if os.path.exists(countries_file_path):
    with open(countries_file_path, "rb") as file_obj:
        polygons = pickle.load(file_obj)
    print(f"Loaded {len(polygons)} states.")
else:
    polygons = []    


# Warp function
def warp_image(img, points, size=[1920, 1080]):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (size[0], size[1]))
    return imgOutput, matrix


# Mouse click to add polygon points
def mouse_points(event, x, y, flags, params):
    global counter, current_polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))


while True:
    success, img = cap.read()
    if not success:
        break

    # always check key (moved here so it's in scope everywhere)
    key = cv2.waitKey(1) & 0xFF

    # draw the loaded corner points
    for (x, y) in map_points:
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), cv2.FILLED)

    # show warped image
    if len(map_points) == 4:
        imgWarp, matrix = warp_image(img, map_points)

        # If we have collected polygon points and press "s"
        if key == ord("s") and len(current_polygon) > 2:
            state_name = input("Enter state name: ")
            polygons.append([current_polygon, state_name])  
            current_polygon = []  
            counter += 1  
            print("Number of states saved: ", len(polygons))

        if key == ord("q"): 
            with open(countries_file_path, "wb") as file_obj:
                pickle.dump(polygons, file_obj)
            print(f"Saved {len(polygons)} states")
            break

        if key == ord("d") and polygons:  
            polygons.pop()
            print("Last polygon deleted")

        # Draw current polygon (red)
        if current_polygon:
            cv2.polylines(imgWarp, [np.array(current_polygon)], isClosed=True, color=(0, 0, 255), thickness=2)

        # Overlay saved polygons (green)
        overlay = imgWarp.copy()
        for polygon, name in polygons:
            cv2.polylines(imgWarp, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(overlay, [np.array(polygon)], color=(0, 255, 0))
        cv2.addWeighted(overlay, 0.35, imgWarp, 0.65, 0, imgWarp)    

        cv2.imshow("Warped Image", imgWarp)
        cv2.imshow("Original Image", img)

    # Set mouse callback
    cv2.setMouseCallback("Warped Image", mouse_points)

    # Exit button -> Esc key
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()