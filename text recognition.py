import cv2
import mediapipe as mp
import numpy as np
import pytesseract

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


def get_hand_landmarks(landmarks_list, hand_to_use):
    for hand_landmark, label_hand in zip(landmarks_list.multi_hand_landmarks, landmarks_list.multi_handedness):
        if hand_to_use == "R":
            if label_hand.classification[0].label == "Right":
                return hand_landmark
        elif hand_to_use == "L":
            if label_hand.classification[0].label == "Left":
                return hand_landmark
    return None


def recognize_text(img, text_color):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = img[y:y + h, x:x + w]
            text = pytesseract.image_to_string(roi, lang='eng', config='--psm 7')
            if text:
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    return img


hand_to_use = input("Enter 'R' for right hand or 'L' for left hand: ")

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
            hand_landmarks = get_hand_landmarks(result, hand_to_use)
            if hand_landmarks:

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
                elif index_tip.y < middle.y and (abs(middle.y - index_tip.y) * 1000) > 150 and (
                        abs(middle.x - index_tip.x) * 1000) > 20 and index_dip_y > index_tip_y:
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
            cv2.imshow('Frame', output)

            key = cv2.waitKey(5)

            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord("c"):
                canvas.fill(255)
            elif key & 0xFF == ord('d'):
                # Draw canvas on frame
                frame = cv2.add(frame, canvas)

                frame = recognize_text(frame, color_red)

                cv2.imshow('Frame_new', frame)
cap.release()
cv2.destroyAllWindows()
