import json
import cv2
import numpy as np
import math

# Load data from JSON
with open("newout.json", 'r') as f:
    data = json.load(f)

# Define colors and initialize images
colors = [[(255, 0, 0), (255, 0, 179), (255, 225, 0)], [(0, 0, 255), (0, 68, 255), (0, 149, 255)]]
trail = np.zeros((271, 555, 3), dtype=np.uint8)
darkness = np.zeros_like(trail)
size = 15
alpha = 0.98
beta = 1 - alpha

lastpoints = None

# Helper function to draw elements for each robot
def draw_robot_elements(robot_idx, row_data, team_idx):
    global trail
    x, y = row_data[team_idx][robot_idx][0]
    color = colors[team_idx][robot_idx]
    
    if lastpoints and lastpoints[team_idx]:
        last_x, last_y = lastpoints[team_idx][robot_idx][0]
        if math.dist((x, y), (last_x, last_y)) < 150:
            cv2.line(trail, (x, y), (last_x, last_y), color, 2)
            cv2.putText(final, str(robot_idx), (x, y), 0, 0.5, color, 1)
            cv2.rectangle(final, (x - size, y - size), (x + size, y + size), color, 2)

# Main loop to process data
for row in data:
    final = cv2.imread("darkfield2023.png")
    
    # Process team 0 (blue) robots
    if row[0]:
        for robot in range(3):
            draw_robot_elements(robot, row, team_idx=0)
    else:
        cv2.putText(final, "no blue found", (10, 10), 0, 0.5, (225, 255, 0), 1)

    # Process team 1 (red) robots
    if row[1]:
        for robot in range(3):
            draw_robot_elements(robot, row, team_idx=1)
    else:
        cv2.putText(final, "no red found", (300, 10), 0, 0.5, (0, 0, 255), 1)
    
    # Update last points
    lastpoints = row
    
    # Fade trail and blend with the final image
    trail = cv2.addWeighted(trail, alpha, darkness, beta, 0)
    final = cv2.add(final, trail)
    
    # Display the result
    cv2.imshow('yo', final)
    key = cv2.waitKey(1)
    if key == 27:  # Exit on ESC key
        break

cv2.destroyAllWindows()
