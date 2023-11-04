from ultralytics import YOLO
import cv2
from djitellopy import Tello
import numpy as np


def initialize_tello():
    """
    Initialize Tello object and return it
    """

    # Connect to Tello drone
    tello = Tello()
    tello.connect()

    # Check battery
    print(f"Battery: {tello.get_battery()}%")

    # Set speed
    tello.for_back_velocity = 0
    tello.left_right_velocity = 0
    tello.up_down_velocity = 0
    tello.yaw_velocity = 0
    tello.speed = 0

    # Get frame
    tello.streamoff()
    tello.streamon()

    return tello


def tello_get_frame(tello, width, height):
    """
    Get frame from tello drone
    """

    # Get frame
    frame = tello.get_frame_read().frame

    # Resize frame
    frame = cv2.resize(frame, (width, height))

    # Set bgr to rgb
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


def detect_object(model, frame, conf):
    """
    Detect object in frame
    """

    # Detect object
    results = model.track(frame)

    # List to store object center and area
    center_list = []
    area_list = []

    # Loop through results
    for r in results:
        boxes = r.boxes.cpu().numpy()
        names = r.names

        if boxes.id is None:
            print("No object detected")
            return frame, [[0, 0], 0]
        else:
            for box in boxes:
                # Get bounding box coordinates, confidence score and class id
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = box.conf[0].astype(float)
                class_id = box.cls[0].astype(int)
                track_id = box.id[0].astype(int)

                # check if confidence is greater than conf
                if confidence > conf:
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 128), 2)

                    # Overlay mask on frame
                    overlay = frame.copy()
                    alpha = 0.8
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (128, 0, 128), -1)
                    frame = cv2.addWeighted(frame, alpha, overlay, 1 - alpha, 0)

                    # Draw confidence score and class id
                    cv2.putText(frame, f"id:{track_id} {names[class_id]} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)

                    # Get center of bounding box
                    center_x = x1 + ((x2 - x1) // 2)
                    center_y = y1 + ((y2 - y1) // 2)

                    # Draw object center
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                    # Calculate object area
                    area = (x2 - x1) * (y2 - y1)

                    # Store object center and area
                    center_list.append([center_x, center_y])
                    area_list.append(area)

                # Return the image, center, and area of largest detected object
                if area_list:
                    index = area_list.index(max(area_list))
                    return frame, [center_list[index], area_list[index]]

                # If no object detected, return zero
                else:
                    return frame, [[0, 0], 0]


def pid_control(error, prev_error, pid):
    """
    PID control 
    """""

    # Calculate pid
    speed = pid[0] * error + pid[1] * (error - prev_error)

    return speed, error
