import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import webbrowser

# Open web page
webbrowser.open('https://en.wikipedia.org/wiki/Black_hole')


def detect_point_down():
    # Load cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Initialize variables for face positions
    face_center = 0.5
    face_top = 0
    face_bottom = 0

    # Set scroll speed and smoothing factor
    scroll_speed = 1
    smoothing_factor = 0.1

    # Set video capture device and resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Main loop
    while True:
        # Read frame from video stream
        ret, frame = cap.read()

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in grayscale frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # If face detected, calculate face positions
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_center = (x + w / 2) / frame.shape[1]
            face_top = y / frame.shape[0]
            face_bottom = (y + h) / frame.shape[0]

        # Move mouse cursor based on face positions
        if face_top < 0.2:
            pyautogui.scroll(scroll_speed)
        elif face_bottom > 0.8:
            pyautogui.scroll(-scroll_speed)

        # Smooth face positions over time
        face_center = (1 - smoothing_factor) * face_center + smoothing_factor * 0.5
        face_top = (1 - smoothing_factor) * face_top
        face_bottom = (1 - smoothing_factor) * face_bottom + smoothing_factor

        # Show video stream with face positions
        cv2.rectangle(frame, (int(face_center * frame.shape[1]), int(face_top * frame.shape[0])),
                      (int(face_center * frame.shape[1]), int(face_bottom * frame.shape[0])), (0, 0, 255), 2)
        cv2.imshow("frame", frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture device and close window
    cap.release()
    cv2.destroyAllWindows()
    return False


while True:
    if not detect_point_down():
        break
exit()
