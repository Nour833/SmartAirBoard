import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

color_red = (0, 0, 255)
color_blue = (255, 0, 0)
color_green = (0, 255, 0)


def main():
    # Ask the user which hand to use
    hand_to_use = input("Enter 'R' for right hand or 'L' for left hand: ")

    # Set up camera capture
    cap = cv2.VideoCapture(0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas.fill(255)

    pen_previous_point = None

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the image horizontally for a better user experience
            frame = cv2.flip(frame, 1)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.multi_hand_landmarks:
                hand_landmarks = get_hand_landmarks(result, hand_to_use)
                if hand_landmarks is not None:
                    pen_detected, pen_point = detect_pen(frame, hand_landmarks)

                    if pen_detected:
                        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        index_finger_tip_point = (int(index_finger_tip.x * width), int(index_finger_tip.y * height))

                        if is_touching_pen(pen_point, index_finger_tip_point):
                            if pen_previous_point is not None:
                                cv2.line(canvas, pen_previous_point, index_finger_tip_point, (0, 0, 255), 5)
                                cv2.line(frame, pen_previous_point, index_finger_tip_point, (0, 0, 255), 5)
                            else:
                                cv2.line(canvas, index_finger_tip_point, index_finger_tip_point, (0, 0, 255), 5)
                                cv2.line(frame, index_finger_tip_point, index_finger_tip_point, (0, 0, 255), 5)

                            pen_previous_point = index_finger_tip_point
                        else:
                            pen_previous_point = None

            alpha = 0.5
            overlay = canvas.copy()
            output = frame.copy()
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
            # Show the video feed and drawing canvas
            cv2.imshow('Drawing Canvas', canvas)
            cv2.imshow('Video Feed', output)

            key = cv2.waitKey(5)
            if key & 0xFF == ord("q"):
                break
            elif key & 0xFF == ord("c"):
                canvas.fill(255)

    cap.release()
    cv2.destroyAllWindows()


def get_hand_landmarks(landmarks_list, hand_to_use):
    for hand_landmarks, label_hand in zip(landmarks_list.multi_hand_landmarks, landmarks_list.multi_handedness):
        if hand_to_use == "R":
            if label_hand.classification[0].label == "Right":
                return hand_landmarks
        elif hand_to_use == "L":
            if hand_landmarks.classification[0].label == "Left":
                return hand_landmarks
    return None


def detect_pen(frame, hand_landmarks):
    # Define the ROI of the hand
    width, height = frame.shape[1], frame.shape[0]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    x_min = int(min(thumb_tip.x, middle_tip.x, index_tip.x) * width)
    x_max = int(max(thumb_tip.x, middle_tip.x, index_tip.x) * width)
    y_min = int(min(thumb_tip.y, middle_tip.y, index_tip.y) * height)
    y_max = int(max(thumb_tip.y, middle_tip.y, index_tip.y) * height)

    # Detect the pen in the ROI
    roi = frame[y_min:y_max, x_min:x_max]
    if roi.any():
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    else:
        hsv = None
    lower_blue = np.array([100, 120, 120])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 100:
            moments = cv2.moments(max_contour)
            cx = int(moments["m10"] / moments["m00"]) + x_min
            cy = int(moments["m01"] / moments["m00"]) + y_min
            return True, (cx, cy)

    return False, None


def is_touching_pen(pen_point, index_finger_tip_point):
    distance = np.linalg.norm(np.array(pen_point) - np.array(index_finger_tip_point))
    if distance < 30:
        return True
    else:
        return False


main()
