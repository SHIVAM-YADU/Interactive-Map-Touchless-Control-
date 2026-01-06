import pickle 
import cv2
import numpy as np
import os
import cvzone
from cvzone.HandTrackingModule import HandDetector

map_file_path = "/Users/shivampankajyadav/Desktop/Open CV/InteractiveMap/Step1-GetCornerPoints/map.p"
countries_file_path = "countries.p"
cam_id = 0   # your camera index
width, height = 1920, 1080

# Load previously saved polygons of interest (ROIs)
if countries_file_path and os.path.exists(countries_file_path):
    with open(countries_file_path, "rb") as file_obj:
        polygons = pickle.load(file_obj)
    print(f"Loaded {len(polygons)} states.")
else:
    polygons = []  

#  Load map points (corners for warping)
with open(map_file_path, "rb") as file_obj:
    map_points = pickle.load(file_obj)
print("Loaded map coordinates:", map_points)

# Camera setup
cap = cv2.VideoCapture(cam_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

detector = HandDetector(staticMode=False,
                         maxHands=1,
                         modelComplexity=1,
                         detectionCon=0.5,
                         minTrackCon=0.5)

# Warp function
def warp_image(img, points, size=[1920, 1080]):
    pts1 = np.float32(points)  # convert points to float32
    pts2 = np.float32([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # calculate perspective transform matrix
    imgOutput = cv2.warpPerspective(img, matrix, (size[0], size[1]))  # warp the image 
    return imgOutput, matrix

def warp_single_point(point, matrix):
    point_homogeneous = np.array([[point[0], point[1], 1]], dtype=np.float32) # convert to homogeneous coordinates
    point_homogeneous_transformed = np.dot(matrix, point_homogeneous.T).T  # apply the transformation
    point_warped = point_homogeneous_transformed[0, :2] / point_homogeneous_transformed[0, 2]  # convert back to Cartesian
    return point_warped

def get_finger_location(img, imgWarped, matrix):
    hands, img = detector.findHands(img, draw=False, flipType=True)

    if hands:
        hand1 = hands[0]
        indexFinger = hand1["lmList"][8][0:2]  # x,y of index finger tip
        #v2.circle(img, indexFinger, 5, (255, 0, 255), cv2.FILLED)
        warped_point = warp_single_point(indexFinger, matrix)
        warped_point = int(warped_point[0]), int(warped_point[1])
        print(indexFinger, warped_point)
        cv2.circle(imgWarped, warped_point, 5, (255, 0, 0), cv2.FILLED)
    else:
        warped_point = None

    return warped_point 

def create_overlay_image(polygons, point, imgOverlay):
    for polygon, name in polygons:
        polygon_np = np.array(polygon, np.int32).reshape((-1, 1, 2))
        result = cv2.pointPolygonTest(polygon_np, point, False)
        if result >= 0:  # point is inside the polygon
            cv2.polylines(imgOverlay, [polygon_np], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(imgOverlay, [polygon_np], color=(0, 255, 0))
            cvzone.putTextRect(imgOverlay, name, polygon[0], scale=1, thickness=1)
            cvzone.putTextRect(imgOverlay, name, (0, 100), scale=8, thickness=5)
    return imgOverlay  

def inverse_warp_image(img, imgOverlay, map_points):
    map_points = np.array(map_points, dtype=np.float32)
    destination_points = np.array(
        [[0, 0], [imgOverlay.shape[1]-1, 0], [0, imgOverlay.shape[0]-1], [imgOverlay.shape[1]-1, imgOverlay.shape[0]-1]],
        dtype=np.float32
    )

    M = cv2.getPerspectiveTransform(destination_points, map_points)
    warped_overlay = cv2.warpPerspective(imgOverlay, M, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(img, 1, warped_overlay, 0.65, 0)
    return result

while True:
    success, img = cap.read()
    if not success:
        break
    imgOutput = img.copy()

    imgWarped, matrix = warp_image(img, map_points)

    # find hands and landmarks
    warped_points = get_finger_location(img, imgWarped, matrix)

    h, w, _ = imgWarped.shape
    imgOverlay = np.zeros((h, w, 3), dtype=np.uint8)  

    if warped_points:
        imgOverlay = create_overlay_image(polygons, warped_points, imgOverlay)
        imgOutput = inverse_warp_image(img, imgOverlay, map_points)

    #stackedImage = cvzone.stackImages([img, imgWarped, imgOutput, imgOverlay], 2, 0.3)

    #cv2.imshow("Stacked Image", stackedImage)
    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1)
    if key == 27:  # Esc key to quit
        break

cap.release()
cv2.destroyAllWindows()