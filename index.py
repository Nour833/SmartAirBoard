import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
confirming = 0
LEVEL = 1
hand_to_use = None
hand_not_to_use = None

# Initialize drawing canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
canvas.fill(255)
canvas_bak = np.zeros((480, 640, 3), dtype=np.uint8)
canvas_bak.fill(255)

color_red = (0, 0, 255)
color_blue = (255, 0, 0)
color_green = (0, 255, 0)
default_color = color_green
cap = cv2.VideoCapture(0)
pen_color = default_color


def draw_line(hand_landmarks, frame, canvas):
    global prev_index_tip_x, prev_index_tip_y, pen_color
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
            cv2.line(canvas, (prev_index_tip_x, prev_index_tip_y), (index_tip_x, index_tip_y),
                     pen_color,
                     thickness=5)
        except:
            prev_index_tip_x, prev_index_tip_y = index_tip_x, index_tip_y
            cv2.line(canvas, (prev_index_tip_x, prev_index_tip_y), (index_tip_x, index_tip_y),
                     pen_color,
                     thickness=5)

    prev_index_tip_x, prev_index_tip_y = index_tip_x, index_tip_y
    return frame, canvas


def get_hand_landmarks(landmarks_list, hand_to_use):
    for hand_landmark, label_hand in zip(landmarks_list.multi_hand_landmarks, landmarks_list.multi_handedness):
        if hand_to_use == "Right":
            if label_hand.classification[0].label == "Right":
                return hand_landmark
        elif hand_to_use == "Left":
            if label_hand.classification[0].label == "Left":
                return hand_landmark
    return None


def draw_border(img, pt1=(375, 125), pt2=(625, 375), color=(0, 255, 0), thickness=4, r=10,
                d=20):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def confirming_hand():
    global hand_to_use, hand_not_to_use, frame, confirming, handLabel, LEVEL
    cv2.putText(frame, 'Put Your Hand', (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (255, 0, 0), 10)
    draw_border(frame)
    h = 0
    for i in hand_landmarks.landmark:
        if 375 <= i.x * 640 <= 625 and 125 <= i.y * 480 <= 375:
            h += 1
    if h == 21:
        confirming += 4
        hand_to_use = handLabel
        if hand_to_use == 'Right':
            hand_not_to_use = 'Left'
        else:
            hand_not_to_use = 'Right'
        cv2.putText(frame, 'Confirming {}%'.format(int(confirming)), (380, 420), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 5)
        if confirming == 100:
            LEVEL = 5
    else:
        confirming = 0


with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
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
            if LEVEL == 1 and not (confirming != 0 and handLabel != hand_to_use):
                for hand_landmarks in result.multi_hand_landmarks:
                    handIndex = result.multi_hand_landmarks.index(hand_landmarks)
                    handLabel = result.multi_handedness[handIndex].classification[0].label
                    confirming_hand()

            else:
                hand_landmarks = get_hand_landmarks(result, hand_to_use)
                hand_landmarks_n = get_hand_landmarks(result, hand_not_to_use)
                if hand_landmarks:
                    frame, canvas = draw_line(hand_landmarks, frame, canvas)
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
            LEVEL = 1
            confirming = 0
        elif key & 0xFF == ord("r"):
            canvas.fill(255)

cap.release()
cv2.destroyAllWindows()
