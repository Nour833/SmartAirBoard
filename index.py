import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize drawing canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
canvas.fill(255)

color_red = (0, 0, 255)
color_blue = (255, 0, 0)
color_green = (0, 255, 0)
default_color = color_green
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    pen_color = default_color

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get index finger and thumb positions
                pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                ring = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Convert normalized coordinates to pixel coordinates
                middle_x, middle_y = int(middle.x * frame.shape[1]), int(middle.y * frame.shape[0])
                index_tip_x, index_tip_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                index_dip_x, index_dip_y = int(index_dip.x * frame.shape[1]), int(index_dip.y * frame.shape[0])
                thumb_x, thumb_y = int(thumb.x * frame.shape[1]), int(thumb.y * frame.shape[0])
                pinky_x, pinky_y = int(pinky.x * frame.shape[1]), int(pinky.y * frame.shape[0])

                # Update the circle position on the frame
                cv2.circle(frame, (middle_x, middle_y), 10, color_green, -1)
                cv2.circle(frame, (index_tip_x, index_tip_y), 10, color_red, -1)
                cv2.circle(frame, (thumb_x, thumb_y), 10, color_blue, -1)

                # Implement drawing and color selection logic
                # ...
                if pinky.y < ring.y < middle.y < index_tip.y < thumb.y:
                    if (abs(pinky.x - index_tip.x) * 1000) > 150:
                        # Get the color of the pixel at the position of the pinky finger
                        # get color of pixel on the left of pinky finger
                        b, g, r = frame[pinky_y, pinky_x - 30]

                        # set pen color to the color of the object
                        pen_color = (int(b), int(g), int(r))

                        cv2.circle(frame, (pinky_x - 30, pinky_y), 10, pen_color, -1)
                    else:
                        cv2.circle(frame, (index_tip_x, index_tip_y), 20, (255, 255, 255), -1)
                        try:
                            cv2.line(canvas, (prev_index_tip_x, prev_index_tip_y), (index_tip_x, index_tip_y),
                                     (255, 255, 255),
                                     thickness=40)
                        except:
                            prev_index_tip_x, prev_index_tip_y = index_tip_x, index_tip_y
                            cv2.line(canvas, (prev_index_tip_x, prev_index_tip_y), (index_tip_x, index_tip_y),
                                     (255, 255, 255),
                                     thickness=40)
                elif index_tip.y < middle.y and (abs(middle.x - index_tip.x) * 1000) > 30 and index_dip_y > index_tip_y:
                    try:
                        cv2.line(canvas, (prev_index_tip_x, prev_index_tip_y), (index_tip_x, index_tip_y), pen_color,
                                 thickness=5)
                    except:
                        prev_index_tip_x, prev_index_tip_y = index_tip_x, index_tip_y
                        cv2.line(canvas, (prev_index_tip_x, prev_index_tip_y), (index_tip_x, index_tip_y), pen_color,
                                 thickness=5)

                prev_index_tip_x, prev_index_tip_y = index_tip_x, index_tip_y
        alpha = 0.5
        overlay = canvas.copy()
        output = frame.copy()
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        # Show the video feed and drawing canvas
        cv2.imshow('Drawing Canvas', canvas)
        cv2.imshow('Video Feed', output)

        key = cv2.waitKey(5)

        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord("c"):
            canvas.fill(255)

cap.release()
cv2.destroyAllWindows()
