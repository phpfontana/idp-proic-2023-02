from utils import *
import cv2


def main():
    # Parameters
    w, h = 640, 480
    pid = [0.4, 0.4, 0.0]  # PID coefficients
    pError = 0
    counter = 0

    # Initialize Tello and YOLO model
    tello = initialize_tello()
    model = initialize_model('yolov8n.pt')

    while True:
        # Takeoff
        tello.takeoff()

        # Get frame
        frame = tello_get_frame(tello, w, h)

        # Find object
        frame, info = find_object(model, frame)

        # Draw axes on frame
        cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)

        # Track object
        pError, frame = track_object(tello, info, w, pid, pError, frame)

        # Display frame
        cv2.imshow('Image', frame)

        # Wait for 1ms
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Land
            tello.land()

            # Break loop
            break

    # Release the capture
    tello.streamoff()
    cv2.destroyAllWindows()
    tello.end()


if __name__ == '__main__':
    main()