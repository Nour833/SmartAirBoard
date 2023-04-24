import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# define the upper and lower boundaries of the HSV color of the blue pen
lower_color_boundary = (100, 50, 50)
upper_color_boundary = (130, 255, 255)

# define the distance threshold for object detection
distance_threshold = 80

# define the ROI for the left and right sides of the frame
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
roi_left = np.array([[0, 0], [int(frame_width / 2), 0], [int(frame_width / 2), frame_height], [0, frame_height]])
roi_right = np.array(
    [[int(frame_width / 2), 0], [frame_width, 0], [frame_width, frame_height], [int(frame_width / 2), frame_height]])

# define the font for the text overlay
font = cv2.FONT_HERSHEY_SIMPLEX
color_red = (0, 0, 255)
color_blue = (255, 0, 0)
color_green = (0, 255, 0)
pen_color = color_green
# initialize the tracker
tracker = None
prev_center = None

# create a blank canvas to draw lines on
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
canvas.fill(255)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # convert the frame from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_color_boundary, upper_color_boundary)

    # apply a series of dilations and erosions to remove any small blobs left in the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current (x, y) center of the pen
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # if contours are found
    if len(contours) > 0:
        # find the largest contour in the mask
        c = max(contours, key=cv2.contourArea)

        # compute the minimum enclosing circle and centroid of the largest contour
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # if the radius is large enough, track the object
        if radius > 10:
            # draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 255), -1)

            # initialize the tracker
            if tracker is None:
                tracker = cv2.TrackerCSRT_create()
                bbox = (int(x - radius), int(y - radius), int(2 * radius), int(2 * radius))
                tracker.init(frame, bbox)
            # if the radius is not large enough, reset the tracker
            else:
                tracker = None

            # add the new line segment to the canvas
            if prev_center is not None:
                cv2.line(canvas, prev_center, center, (0, 255, 255), 5)

            prev_center = center

    # update the tracker if it exists
    if tracker is not None:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(val) for val in bbox]
            center = (int(x + w / 2), int(y + h / 2))
            # add the new line segment to the canvas
            if prev_center is not None:
                cv2.line(canvas, prev_center, center, (0, 255, 255), 5)
            prev_center = center
            # draw therectangle around the tracked object on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # add the updated canvas to the frame

    # display the frame
    alpha = 0.5
    overlay = canvas.copy()
    output = frame.copy()
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    # Show the video feed and drawing canvas
    cv2.imshow('Drawing Canvas', canvas)
    cv2.imshow('Video Feed', output)

    # update the previous center point
    previous_center = center

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
