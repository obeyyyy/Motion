import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
icon = cv2.imread('R.png')
# Define the minimum area of motion
min_area = 500

# Define the threshold for motion detection
motion_thresh = 0.01

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Apply thresholding to segment the foreground
    thresh = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY)[1]

    # Find contours of the segmented foreground
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and detect motion
    motion_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Motion Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Calculate the percentage of the image that has changed
    total_pixels = np.prod(fgmask.shape[:2])
    changed_pixels = np.count_nonzero(fgmask)
    percent_changed = changed_pixels / total_pixels

    # If motion is detected and the percentage of the image that has changed is above the threshold, trigger the motion alert
    if motion_detected and percent_changed > motion_thresh:
        icon_resized = cv2.resize(icon, (frame.shape[1] // 4, frame.shape[0] // 4))
        
        # Blend the icon and the frame using alpha blending
        alpha = 0.5
        beta = 1 - alpha
        foreground = cv2.addWeighted(frame[:icon_resized.shape[0], :icon_resized.shape[1]], alpha, icon_resized, beta, 0)
        frame[:icon_resized.shape[0], :icon_resized.shape[1]] = foreground

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
