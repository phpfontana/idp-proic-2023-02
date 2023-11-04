from utils import *
import cv2
import matplotlib.pyplot as plt


def main():
    # Parameters
    w, h = 640, 480
    pid_x = [0.6, 0.6, 0.0]  # PID coefficients
    pid_y = [0.4, 0.4, 0.0]  # PID coefficients
    prev_error_x = 0
    prev_error_y = 0
    store_speed_x = []
    store_speed_y = []
    store_speed_z = []

    # Initialize Tello and YOLO model
    tello = initialize_tello()
    model = YOLO('best.pt')

    # Takeoff


    while True:
        # Get frame
        frame = tello_get_frame(tello, w, h)

        # Find object
        frame, info = detect_object(model, frame, 0.8)

        # Draw X and Y axis
        cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 0, 255), 2)
        cv2.line(frame, (0, h // 2), (w, h // 2), (0, 0, 255), 2)

        # Plot line distance from object center to X and Y axis
        if info[0][0] != 0:
            cv2.line(frame, (info[0][0], info[0][1]), (w // 2, info[0][1]), (128, 0, 128), 2)
            cv2.line(frame, (info[0][0], info[0][1]), (info[0][0], h // 2), (128, 0, 128), 2)

        # X PID Control
        error_x = int(info[0][0] - w // 2)  # center_x - width // 2
        speed_x, prev_error_x = pid_control(error_x, prev_error_x, pid_x)
        speed_x = int(np.clip(speed_x, -50, 50))

        # Y PID Control
        error_y = int(h // 2 - info[0][1])  # center y - height // 2
        speed_y, prev_error_y = pid_control(error_y, prev_error_y, pid_y)
        speed_y = int(np.clip(speed_y, -50, 50))

        # Z axis PID Control
        if info[1] != 0:  # if object detected
            if info[1] < (h * w) // 4:  # if area is less than 1/4 of the frame
                speed_z = 20  # move forward
            elif info[1] > h * w - (h * w) // 4:  # if area is more than 3/4 of the frame
                speed_z = -20  # move backward
            else:  # if area is between 1/4 and 3/4 of the frame
                speed_z = 0  # stop
        else:  # if no object detected
            speed_z = 0  # stop

        # if object detected
        if info[0][0] != 0:
            # Set speed
            tello.left_right_velocity = speed_x
            tello.up_down_velocity = speed_y
            tello.for_back_velocity = speed_z
        else:
            # Stop
            tello.left_right_velocity = 0
            tello.for_back_velocity = 0
            tello.up_down_velocity = 0
            tello.yaw_velocity = 0
            prev_error_x = 0
            prev_error_y = 0

        # Send RC control to Tello
        if tello.send_rc_control:
            tello.send_rc_control(tello.left_right_velocity,
                                  tello.for_back_velocity,
                                  tello.up_down_velocity,
                                  tello.yaw_velocity)

        # display speed
        cv2.putText(frame, f"Speed X: {tello.left_right_velocity}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2)
        cv2.putText(frame, f"Speed Y: {tello.up_down_velocity}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Speed Z: {tello.for_back_velocity}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2)

        # store information
        store_speed_x.append(tello.left_right_velocity)
        store_speed_y.append(tello.up_down_velocity)
        store_speed_z.append(tello.for_back_velocity)

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

    # Plot speed
    plt.plot(store_speed_x, label='Speed X')
    plt.plot(store_speed_y, label='Speed Y')
    plt.plot(store_speed_z, label='Speed Z')

    plt.xlabel('Time')
    plt.ylabel('Speed')
    plt.title('Speed')
    plt.legend()
    plt.show()

    # save plot
    plt.savefig('speed.png')


if __name__ == '__main__':
    main()
