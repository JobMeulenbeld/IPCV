import numpy as np
import cv2


class HandDetector:
    def __init__(self, calibration_interval=30):
        # Used YCrCb threshold values based on following paper: https://www.ee.cuhk.edu.hk/~knngan/TCSVT_v9_n4_p551-564.pdf.
        # These change, based on the adaptive threshold function for calibration based on face. But initial guess.
        self.y_min = 54
        self.y_max = 163
        self.cr_min = 131
        self.cr_max = 157
        self.cb_min = 110
        self.cb_max = 135

        # Calibration variables
        self.calibrated = False
        self.calibration_interval = calibration_interval  # calibrate to new YCrCB threshold every 30 frames
        self.frame_counter = 0

        # Lucas Kanade Tracking  variables
        self.prev_frame = None
        self.prev_points = None
        self.mask = None

    def detect_face(self, frame):
        # We detect the face, based on the Haar Cascade xml file implemented in CV2
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face
        face = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return face

    def calibrate_from_face(self, frame, face, std_multiplier=1.6, shrink_ratio=0.1):
        if len(face) == 0:
            return False

        x, y, w, h = face[0]

        # Perform shrinkage, as now background is also included, before calibration
        dx = int(w * shrink_ratio)
        dy = int(h * shrink_ratio)

        # apply shrinkage
        x = x + dx
        y = y + dy
        w = w - 2 * dx
        h = h - 2 * dy

        # Extract face from frame
        face_roi = frame[y:y + h, x:x + w]

        # Use the mean and standard deviation in YCrCb to set the new thresholds for calibration
        ycrcb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)

        y_mean = np.mean(ycrcb_face[:, :, 0])
        y_std = np.std(ycrcb_face[:, :, 0])
        cr_mean = np.mean(ycrcb_face[:, :, 1])
        cr_std = np.std(ycrcb_face[:, :, 1])
        cb_mean = np.mean(ycrcb_face[:, :, 2])
        cb_std = np.std(ycrcb_face[:, :, 2])

        # Set the adaptive thresholds for YCrCb.
        # If std_multiplier = 1, it covers 68% of data in a normal distribution (so the skin pixels), if 1.5 it is already 87%
        # If set too big, background and look a like objects will also be seen as skin, so do not set it too high.
        y_range = int(y_std * std_multiplier)
        cr_range = int(cr_std * std_multiplier)
        cb_range = int(cb_std * std_multiplier)

        # Set the new calibration threshold rules
        self.y_min = max(0, int(y_mean - y_range))
        self.y_max = min(255, int(y_mean + y_range))
        self.cr_min = max(0, int(cr_mean - cr_range))
        self.cr_max = min(255, int(cr_mean + cr_range))
        self.cb_min = max(0, int(cb_mean - cb_range))
        self.cb_max = min(255, int(cb_mean + cb_range))

        return True

    def detect_skin(self, frame):
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # Create the skin mask, based on the threshold values
        lower = np.array([self.y_min, self.cr_min, self.cb_min])
        upper = np.array([self.y_max, self.cr_max, self.cb_max])

        skin_mask = cv2.inRange(ycrcb, lower, upper)

        # Apply post processing, with morphology. To remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        _, skin_mask = cv2.threshold(skin_mask, 127, 255, cv2.THRESH_BINARY)

        return skin_mask

    def remove_not_interested_roi(self, skin_mask, face):
        if len(face) == 0:
            return skin_mask

        x, y, w, h = face[0]

        # Remove everything above the top of the face
        skin_mask[:y, :] = 0

        # Remove the face itself
        skin_mask[y:y + h, x:x + w] = 0

        # Remove a small area directly below the face, which is the neck, so it wont get detected as hand
        neck_y1 = y + h
        neck_y2 = min(skin_mask.shape[0], int(y + 1.6 * h))
        skin_mask[neck_y1:neck_y2, x:x + w] = 0

        return skin_mask

    def detect_hand_contours(self, frame, skin_mask, min_area=4500, max_area=200000):
        valid_contours = []
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        screen_height, screen_width = frame.shape[:2]
        right_screen_boundary = screen_width // 2

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    if cx > right_screen_boundary:
                        valid_contours.append(contour)

        if not valid_contours:
            return None

        largest_contour = max(valid_contours, key=cv2.contourArea)
        return largest_contour

    # Based on https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
    def optical_flow_lucas_kanade(self, frame, roi):
        direction = "none"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if roi is None:
            return direction

        # Normalize ROI to [x1, y1, x2, y2]
        x, y, w, h = roi
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Sometimes bounding box fails, idk why
        x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1),
                                   min(frame.shape[1], x2),
                                   min(frame.shape[0], y2)])

        # ROI mask
        roi_mask = np.zeros_like(gray)
        roi_mask[y1:y2, x1:x2] = 255

        # Initialize points and frame if needed
        if self.prev_frame is None or self.prev_points is None or len(self.prev_points) == 0:
            self.prev_frame = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7, mask=roi_mask)
            self.mask = np.zeros_like(frame)
            return direction

        # Filter previous points within ROI
        pts = self.prev_points.reshape(-1, 2)
        inside = (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
        self.prev_points = self.prev_points[inside].reshape(-1, 1, 2)

        # If no points left, redetect
        if len(self.prev_points) == 0:
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7, mask=roi_mask)
            self.prev_frame = gray
            return direction

        # Compute optical flow
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, self.prev_points, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Filter valid points
        if next_pts is None or status is None:
            self.prev_points, self.prev_frame = None, gray
            return direction

        good_prev = self.prev_points[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]

        # Compute average motion vector
        if len(good_prev) > 0:
            movement = good_next - good_prev
            avg_dx, avg_dy = np.mean(movement, axis=0).ravel()
            mag = np.hypot(avg_dx, avg_dy)

            # Direction determination
            if mag > 1.0:
                angle = (np.degrees(np.arctan2(avg_dy, avg_dx)) + 360) % 360
                if 45 <= angle < 135:
                    direction = "DOWN"
                elif 135 <= angle < 225:
                    direction = "LEFT"
                elif 225 <= angle < 315:
                    direction = "UP"
                else:
                    direction = "RIGHT"

        # Update for next frame
        self.prev_points, self.prev_frame = good_next.reshape(-1, 1, 2), gray
        return direction

    def determine_fingertips(self, contour, max_fingertips=5):
        # No contour provided
        if contour is None or len(contour) == 0:
            return [], None  # Return empty fingertips and None center

        # Use the convex hull for determining position fingertips, need at least 3 points for convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) < 3:
            return [], None

        # Compute convexity defects
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return [], None

        # Calculate the center of the hand
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return [], None

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center = (cX, cY)

        # Find the hull points that are far from defects and above the center
        candidates = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start_pt = tuple(contour[s][0])
            end_pt = tuple(contour[e][0])

            # Only consider hull points above center
            for pt in [start_pt, end_pt]:
                pt_arr = np.array(pt)
                if pt_arr[1] < center[1] + 40:  # small tolerance for thumb and pinky
                    candidates.append(pt)

        # Remove duplicates
        candidates = list(set(candidates))

        # Sort by distance to center, descending
        candidates.sort(key=lambda pt: np.linalg.norm(np.array(pt) - np.array(center)), reverse=True)

        # Filter close points
        fingertips = []
        min_distance_between_fingers = 20
        for pt in candidates:
            if all(np.linalg.norm(np.array(pt) - np.array(fpt)) >= min_distance_between_fingers for fpt in fingertips):
                fingertips.append(pt)
            if len(fingertips) >= max_fingertips:
                break

        # Sort it based on x position
        fingertips.sort(key=lambda pt: pt[0])

        return fingertips, center

    def process_frame(self, frame, tracker, gesture_detector):
        # Face always needed, so assign it directly
        detected_face = self.detect_face(frame)

        # this is basically the main that is running every frame
        if not self.calibrated:
            self.calibrated = self.calibrate_from_face(frame, detected_face)

        # Check if automatic calibration is needed again
        if self.frame_counter >= self.calibration_interval:
            self.frame_counter = 0
            self.calibrated = self.calibrate_from_face(frame, detected_face)

        # Follow the standard procedure, first detect the skin
        skin_mask = self.detect_skin(frame)

        # Removing roi, where the hand should not be, so detection becomes more stable and less jitter.
        self.remove_not_interested_roi(skin_mask, detected_face)

        # Get the right hand contour
        right_hand_contour = self.detect_hand_contours(frame, skin_mask)
        frame = self.draw_contour(frame, right_hand_contour)

        # Get the fingerpoints
        fingertips, center = self.determine_fingertips(right_hand_contour)

        # Track the bounding box
        smoothed_boundingbox, smoothed_center, smoothed_fingertips = tracker.update(
            cv2.boundingRect(right_hand_contour), center, fingertips)
        frame = self.draw_fingertips(frame, smoothed_fingertips, smoothed_center)

        # Start with motion tracking based on lucas kanade
        direction = self.optical_flow_lucas_kanade(frame, smoothed_boundingbox)

        # Check the gesture
        frame = gesture_detector.process_frame(frame, smoothed_fingertips, smoothed_center, direction,
                                               smoothed_boundingbox)

        self.frame_counter += 1

        return frame

    # ----------------- Visualization Methods ------------------
    def draw_contour(self, frame, contour):
        if contour is not None:
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 2)
            return frame
        else:
            return frame

    def draw_fingertips(self, frame, fingertips, center):
        if fingertips is None or center is None:
            return frame

        # Ensure center is a tuple of ints
        center = tuple(map(int, center))

        # Draw center point
        cv2.circle(frame, center, 6, (0, 0, 255), -1)  # Red center

        # Draw fingertips
        for i, fingertip in enumerate(fingertips):
            if fingertip is None:
                continue

            # Ensure fingertip is a tuple of ints
            fingertip = tuple(map(int, fingertip))

            cv2.circle(frame, fingertip, 5, (255, 0, 0), -1)  # Blue fingertips
            cv2.putText(
                frame, str(i + 1),
                (fingertip[0] + 5, fingertip[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        return frame

    # Draw the bounding box around the hand contour
    # x, y, w, h = cv2.boundingRect(right_hand_contour)
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
