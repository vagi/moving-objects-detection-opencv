import cv2
from tracker import *


# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("newtest720.mp4")

# 1. Object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    # print(height, width)

    # Extract region of interest (ROI)
    roi = frame[100: 500, 700: 1250]

    # Object detection
    mask = object_detector.apply(roi)    # (frame)
    # Remove shadow
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    # We use ComputerVision 3 here, in case of CV4 output should be only two valaues: contours, _ =
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        # We limit area of detected objects with this value
        if area > 400:
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)    # (frame)
            # Putting objects into bounding rectangular boxes
            x, y, w, h = cv2.boundingRect(cnt)

            detections.append([x, y, w, h])

    # 2. Object tracking
    boxes_ids = tracker.update(detections)
    for boxes_id in boxes_ids:
        x, y, w, h, id = boxes_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 4)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)


    # Show video in the frame
    cv2.imshow("Frame", frame)
    # Show mask over the video in the frame
    cv2.imshow("Mask", mask)
    # Show ROI in frame
    cv2.imshow("roi", roi)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
