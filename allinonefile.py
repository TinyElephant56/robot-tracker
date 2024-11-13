import cv2
from ultralytics import YOLO
import numpy as np
import math
import torch
import json

print("imports...")
cap = cv2.VideoCapture("sacramento.mp4")
ret, frame = cap.read()


model = YOLO("best.pt")
confidence_threshold = 0.4

bluebots = []
redbots = []

src_points = np.array([(80, 150), (350, 150), (420, 350), (10, 350)], dtype="float32")
dst_points = np.array([(0, 0), (555, 0), (555, 271), (0, 271)], dtype="float32")
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

with open ("newout.json", 'w') as f:
    json.dump([], f)


def track_robots(robots, cords, color):
    if len(robots) == 0 and len(cords) == 3: #initial robot setup
        for c in cords:
            robots.append([c, 1])
        print(f"Initial positions:", robots)

    if robots and cords: #matches the cords and robots
        incomplete = True
        assignments = 0
        while incomplete: #loop until three pairs
            closest = 999999
            bestc = None
            bestb = None

            #find the closest pair
            for b in range(len(robots)):
                if robots[b][1] != 0:
                    for c in range(len(cords)):
                        distance = math.dist(robots[b][0], cords[c])
                        if distance < closest and distance < 100:
                            bestb = b
                            bestc = c
                            closest = distance
            
            if bestc != None:
                robots[bestb] = [cords[bestc], 0]
                cords.pop(bestc)
            else:
                print("leaving a robot unassingned")
                print(bestb, bestc)
                print(robots, cords)
                
            
            assignments += 1
            if assignments == 3:
                incomplete = False
            if len(cords) == 0:
                incomplete = False

    for i in range(len(robots)): # draws circles and numbers; increments frames gone
        robots[i][1] += 1
        cv2.circle(frame, robots[i][0], 5, color, 2)
        cv2.putText(frame, f"{i}", robots[i][0], 0, 0.5, (255, 255, 0), 2)


def dewarp_robots(bluebots, redbots):
    blue_data = []
    red_data = []

    if bluebots: #blue data
        for robot in range(3):
            x, y = bluebots[robot][0]
            point = np.array([[[x, y]]], dtype="float32")
            transformed_point = cv2.perspectiveTransform(point, matrix)
            transformed_x, transformed_y = transformed_point[0][0]
            blue_data.append([ [int(transformed_x), int(transformed_y)], bluebots[robot][1]])
            cv2.circle(final, (int(transformed_x), int(transformed_y)), 5, (255, 0, 0), 2)

    if redbots: #red data
        for robot in range(3):
            x, y = redbots[robot][0]
            point = np.array([[[x, y]]], dtype="float32")
            transformed_point = cv2.perspectiveTransform(point, matrix)
            transformed_x, transformed_y = transformed_point[0][0]
            red_data.append([ [int(transformed_x), int(transformed_y)], redbots[robot][1]])
            cv2.circle(final, (int(transformed_x), int(transformed_y)), 5, (0, 0, 255), 2)

    #write the data to the big file
    with open("newout.json", 'r') as file:
        data = json.load(file)
    compressed = [blue_data, red_data]
    data.append(compressed)
    with open("newout.json", 'w') as file:
        json.dump(data, file)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #four corners, visualize
    cv2.circle(frame, (80, 150), 5, (0, 0, 0), -1)
    cv2.circle(frame, (350, 150), 5, (0, 0, 0), -1)
    cv2.circle(frame, (420, 350), 5, (0, 0, 0), -1)
    cv2.circle(frame, (10, 350), 5, (0, 0, 0), -1)

    height, width, _ = frame.shape

    final = cv2.imread("darkfield2023.png")

    results = model(frame, device="mps", verbose=False, iou=0.8)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    confidences = np.array(result.boxes.conf.cpu())
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    
    bluecords = []
    redcords = []

    for cls, bbox, conf in zip(classes, bboxes, confidences):
        (x1, y1, x2, y2) = bbox
        cord = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        if conf > confidence_threshold:
            cv2.putText(frame, f"{conf*100:.0f}", (x1, y1 - 5), 0, 0.5, (255, 255, 255), 2)
            if cls == 0:  # Blue robot
                cv2.rectangle(frame, (x1, y1), (x2, y2), (225, 0, 0), 2)
                bluecords.append(cord)
            elif cls == 1:  # Red robot
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 225), 2)
                redcords.append(cord)

    track_robots(bluebots, bluecords, (255, 0, 0))
    track_robots(redbots, redcords, (0, 0, 225))

    dewarp_robots(bluebots, redbots)

    cv2.imshow("sussy baka ohio skibidi sigma", frame)
    cv2.imshow("woahh", final)

    key = cv2.waitKey(1)
    if key == 27:
        break
    

print("done!")
cap.release()
cv2.destroyAllWindows()
