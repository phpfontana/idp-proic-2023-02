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
    print(f"Battery: {tello.get_battery()}")

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


def initialize_model(weights):
    """
    Initialize YOLO model and return it
    """
    # Initialize YOLO model
    model = YOLO(weights)

    return model


def find_object(model, img):
    """
    find object in frame
    """

    # Detect objects in the input frame
    results = model(img)

    # Store object center and area
    center_list = []
    area_list = []

    # Loop over the detected objects
    for r in results:
        boxes = r.boxes.cpu().numpy()
        names = r.names

        for box in boxes:
            # Get bounding box coordinates, confidence score and class id
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0].astype(float) * 100
            class_id = box.cls[0].astype(int)

            # Check if confidence is greater than 50% and class id is 0 (person)
            if conf > 50 and class_id == 0:
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Calculate object center
                center_x = x1 + ((x2 - x1) // 2)
                center_y = y1 + ((y2 - y1) // 2)

                # Calculate object area
                area = (x2 - x1) * (y2 - y1)

                # Draw object center
                cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)

                # Draw confidence score and class name
                cv2.putText(img, f"{str(names[class_id])} {str(conf.astype(int))}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Store object center and area
                center_list.append([center_x, center_y])
                area_list.append(area)

    # Return the image, center, and area of largest detected object
    if area_list:
        index = area_list.index(max(area_list))
        return img, [center_list[index], area_list[index]]

    # if no object detected, return zero
    else:
        return img, [[0, 0], 0]


def track_object(tello, info, w, pid, pError, frame):
    """
    Track object in frame
    """

    # Calculate error
    error = int(info[0][0] - w // 2)

    # Calculate speed
    speed = pid[0] * error + pid[1] * (error - pError)

    # Calculate yaw
    speed = int(np.clip(speed, -100, 100))

    # if object detected
    if info[0][0] != 0:
        # set yaw velocity
        tello.yaw_velocity = speed
    else:
        # Stop
        tello.left_right_velocity = 0
        tello.for_back_velocity = 0
        tello.up_down_velocity = 0
        tello.yaw_velocity = 0
        error = 0

    # Send RC control to Tello
    if tello.send_rc_control:
        tello.send_rc_control(tello.left_right_velocity,
                              tello.for_back_velocity,
                              tello.up_down_velocity,
                              tello.yaw_velocity)

    return error, frame






