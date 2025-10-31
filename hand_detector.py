import math

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

    def get_contour_info(self, largest_contour):
        # This function is based on the post of: https://www.javaadvent.com/2012/12/hand-and-finger-detection-using-javacv.html
        # get info about the contour's center of gravity and the principal axis (contour_axis_angle)
        # needed to compute relative positions and normalize orientation, tells the dominant orientation
        cog_point = (0, 0)

        # Compute raw image moments
        moments = cv2.moments(largest_contour)

        # Center of gravity calculation
        m00 = moments['m00']
        m10 = moments['m10']
        m01 = moments['m01']

        if m00 != 0:
            x_center = int(round(m10 / m00))
            y_center = int(round(m01 / m00))
            cog_point = (x_center, y_center)

        # Calculate central moments for orientation
        m11 = moments['mu11']  # Central moment (1,1)
        m20 = moments['mu20']  # Central moment (2,0)
        m02 = moments['mu02']  # Central moment (0,2)

        contour_axis_angle = self.calculate_tilt(m11, m20, m02)
        contour_axis_angle = 180 - contour_axis_angle

        # Normalize angle to 0-360 range
        contour_axis_angle %= 360

        return cog_point, contour_axis_angle

    def calculate_tilt(self, m11, m20, m02):
        diff = m20 - m02

        # Handle special cases where denominator would be zero
        if diff == 0:
            if m11 == 0:
                return 0
            elif m11 > 0:
                return 45
            else:  # m11 < 0
                return -45

        # Calculate the angle using the formula: θ = 0.5 * arctan(2*m11 / (m20 - m02))
        theta = 0.5 * math.atan2(2 * m11, diff)
        tilt = int(round(math.degrees(theta)))

        # Handle different quadrants based on moment signs
        if (diff > 0) and (m11 == 0):
            return 0
        elif (diff < 0) and (m11 == 0):
            return -90
        elif (diff > 0) and (m11 > 0):  # 0 to 45 degrees
            return tilt
        elif (diff > 0) and (m11 < 0):  # -45 to 0
            return 180 + tilt  # Change to counter-clockwise angle
        elif (diff < 0) and (m11 > 0):  # 45 to 90
            return tilt
        elif (diff < 0) and (m11 < 0):  # -90 to -45
            return 180 + tilt  # Change to counter-clockwise angle
        return 0

    def cluster_points(self, points, threshold=20):
        """
        Merge points that are closer than threshold into a single point (centroid).
        """
        if not points:
            return []

        clustered = []
        used = [False] * len(points)

        for i, pt in enumerate(points):
            if used[i]:
                continue
            cluster = [pt]
            used[i] = True
            for j, other in enumerate(points):
                if not used[j]:
                    dist = math.hypot(pt[0] - other[0], pt[1] - other[1])
                    if dist < threshold:
                        cluster.append(other)
                        used[j] = True
            # Take average as representative
            x_avg = int(sum(p[0] for p in cluster) / len(cluster))
            y_avg = int(sum(p[1] for p in cluster) / len(cluster))
            clustered.append((x_avg, y_avg))
        return clustered

    def find_fingertips(self, contour, cog_point, contour_axis_angle, min_distance=20):
        if contour is None or len(contour) == 0:
            return []

        # Get convex hull and convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        fingertips = []

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])

                for pt in [start, end]:
                    # Keep points far enough from center of gravity
                    if math.hypot(pt[0] - cog_point[0], pt[1] - cog_point[1]) > min_distance:
                        fingertips.append(pt)

        # Remove duplicates / cluster nearby points
        def cluster_points(points, threshold=25):
            clustered = []
            for pt in points:
                if not any(math.hypot(pt[0] - c[0], pt[1] - c[1]) < threshold for c in clustered):
                    clustered.append(pt)
            return clustered

        fingertips = cluster_points(fingertips)

        # --- Simple thumb detection based on side ---
        if contour_axis_angle > 0:
            thumb_candidates = [pt for pt in fingertips if pt[0] < cog_point[0] and pt[1] >= cog_point[1] - 20]
        else:
            thumb_candidates = [pt for pt in fingertips if pt[0] > cog_point[0] and pt[1] >= cog_point[1] - 20]

        thumb = max(thumb_candidates, key=lambda pt: abs(pt[0] - cog_point[0])) if thumb_candidates else None

        if thumb in fingertips:
            fingertips.remove(thumb)

        # Keep up to max 4 fingers + thumb
        fingertips = sorted(fingertips, key=lambda pt: pt[1])[:4]
        # Put thumb at the beginning
        if thumb:
            fingertips = [thumb] + fingertips

        return fingertips

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

        # Get contour info
        cog_point, contour_axis_angle = self.get_contour_info(right_hand_contour)
        fingertips = self.find_fingertips(right_hand_contour, cog_point, contour_axis_angle)

        # Start with the tracking
        smoothed_points, smoothed_cog_point = tracker.update(fingertips, cog_point)

        # Visualize it
        frame = self.visualize_fingertips(frame, right_hand_contour, smoothed_cog_point, contour_axis_angle,
                                          smoothed_points)
        print(contour_axis_angle)

        print(fingertips)

        frame = gesture_detector.process_frame(frame, smoothed_points, smoothed_cog_point, contour_axis_angle)

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
        # Draw center point
        cv2.circle(frame, center, 6, (0, 0, 255), -1)  # Red center

        # Draw fingertips
        for i, fingertip in enumerate(fingertips):
            cv2.circle(frame, fingertip, 5, (255, 0, 0), -1)  # Blue fingertips
            cv2.putText(frame, str(i + 1), (fingertip[0] + 5, fingertip[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def visualize_fingertips(self, image, contour, cog_point, angle, fingertips):
        if image is None or contour is None or cog_point is None or angle is None or fingertips is None:
            return image

        vis_img = image.copy()

        # Draw contour
        cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 2)

        # Draw COG
        cv2.circle(vis_img, cog_point, 5, (255, 0, 0), -1)

        # Draw major axis line
        length = 100  # line length for visualization
        theta = math.radians(angle)
        x2 = int(cog_point[0] + length * math.cos(theta))
        y2 = int(cog_point[1] - length * math.sin(theta))  # OpenCV y-axis points down
        cv2.line(vis_img, cog_point, (x2, y2), (255, 0, 255), 2)

        # Draw fingertips
        if fingertips is not None:
            for tip in fingertips:
                cv2.circle(vis_img, tip, 8, (0, 0, 255), -1)
                cv2.putText(vis_img, f"{tip}", (tip[0] + 5, tip[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return vis_img
