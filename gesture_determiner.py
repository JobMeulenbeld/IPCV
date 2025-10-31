import math
from collections import deque
import cv2


class GestureDeterminer:
    def __init__(self):
        self.base_pinch_distance = None
        self.zoom_factor_smooth = deque(maxlen=5)
        self.how_long_open = 0
        self.choose_gesture = False
        self.hand_rotation_counter = 0
        self.pinch_counter = 0
        # New attribute: can be "none", "open_hand", "pinch", "rotation"
        self.current_gesture_state = "none"

    def detect_open_hand(self, fingertips, cog_point, open_threshold=80, closed_threshold=60):
        if not fingertips or cog_point is None:
            return "unknown"

        # Compute average fingertip-to-COG distance
        distances = [math.hypot(pt[0] - cog_point[0], pt[1] - cog_point[1]) for pt in fingertips]
        avg_distance = sum(distances) / len(distances)

        # Classify gesture
        if avg_distance > open_threshold:
            return "open"
        elif avg_distance < closed_threshold:
            return "closed"
        else:
            return "unknown"

    def detect_pinch_gesture(self, fingertips):
        if fingertips is None or len(fingertips) < 2:
            return False

        # Sort fingertips by x-coordinate (left to right)
        sorted_fingertips = sorted(fingertips, key=lambda point: point[0])

        # Calculate distance between the two leftmost fingers
        left_finger1 = sorted_fingertips[0]
        left_finger2 = sorted_fingertips[1]
        distance = math.hypot(left_finger2[0] - left_finger1[0],
                              left_finger2[1] - left_finger1[1])

        # Simple threshold: if the two leftmost fingers are close, it's a pinch
        is_pinching = distance < 50  # Adjust this threshold as needed

        return is_pinching

    def do_pinch_gesture(self, fingertips, smooth=True):
        if fingertips is None or len(fingertips) < 2:
            return 1.0, 0  # default zoom factor

        thumb = fingertips[0]
        index = fingertips[1]

        # Calculate distance
        current_distance = math.hypot(index[0] - thumb[0], index[1] - thumb[1])

        # Reset baseline, when hands disappears or when fingers seperate too far
        if current_distance > 200:
            self.base_pinch_distance = None
            return 1.0, 0

        # Initialize baseline if not set
        if self.base_pinch_distance is None:
            self.base_pinch_distance = current_distance
            return 1.0, current_distance

        # Compute zoom factor (ratio to baseline)
        zoom_factor = current_distance / self.base_pinch_distance

        # Smooth the zoom factor over last few frames (optional)
        if smooth:
            self.zoom_factor_smooth.append(zoom_factor)
            zoom_factor = sum(self.zoom_factor_smooth) / len(self.zoom_factor_smooth)

        return zoom_factor, current_distance

    def determine_hand_to_left_or_right_or_center(self, contour_axis_angle, left_threshold=100, right_threshold=70):
        if contour_axis_angle is None:
            return "unknown"

        if contour_axis_angle == 180:
            return "center"
        elif contour_axis_angle > left_threshold:
            return "left"
        elif contour_axis_angle < right_threshold:
            return "right"
        else:
            return "center"

    def process_frame(self, frame, fingertips, cog_point, contour_axis_angle):
        # Determine hand gesture
        command = self.detect_open_hand(fingertips, cog_point) if fingertips else "none"

        # Reset counters if hand disappears
        if not fingertips:
            self.how_long_open = 0
            self.pinch_counter = 0
            self.hand_rotation_counter = 0
            self.current_gesture_state = "none"
            self.base_pinch_distance = None
            self.zoom_factor_smooth.clear()
            draw_text_top_left(frame, "No hand detected", y_offset=0)
            return frame

        # ----- Open hand detection -----
        if command == "open":
            self.how_long_open += 1
            if self.how_long_open >= 30:
                self.choose_gesture = True
                self.current_gesture_state = "open_hand"
                self.how_long_open = 0
        else:
            self.how_long_open = 0

        # ----- Gesture choosing mode -----
        if self.choose_gesture:
            draw_text_top_left(frame, "Choose the gesture", y_offset=0)

            # Pinch gesture
            pinch_detection = self.detect_pinch_gesture(fingertips)
            if pinch_detection:
                self.pinch_counter += 1
                if self.pinch_counter >= 20:
                    self.current_gesture_state = "pinch"
                    draw_text_top_left(frame, "Pinch detected!", y_offset=30)
                    zoom_factor, current_distance = self.do_pinch_gesture(fingertips)
                    draw_text_top_left(frame, f"Zoom factor: {zoom_factor:.2f}", y_offset=60)
                    draw_text_top_left(frame, f"Pinch distance: {current_distance:.1f}", y_offset=90)
            else:
                if self.current_gesture_state == "pinch":
                    # Keep in pinch mode until hand changes (e.g., fist)
                    pass
                else:
                    self.pinch_counter = 0

            # Hand rotation for filter switching
            hand_rotation_command = self.determine_hand_to_left_or_right_or_center(contour_axis_angle)
            if hand_rotation_command == "left":
                self.hand_rotation_counter += 1
            elif hand_rotation_command == "right":
                self.hand_rotation_counter -= 1
            else:
                if self.current_gesture_state != "rotation":
                    self.hand_rotation_counter = 0

            self.hand_rotation_counter = max(min(self.hand_rotation_counter, 20), -20)

            if self.hand_rotation_counter >= 10:
                draw_text_top_left(frame, "Filter switch to the left", y_offset=120)
                self.current_gesture_state = "rotation"
            elif self.hand_rotation_counter <= -10:
                draw_text_top_left(frame, "Filter switch to the right", y_offset=120)
                self.current_gesture_state = "rotation"

        else:
            draw_text_top_left(frame, f"Open hand counter: {self.how_long_open}", y_offset=0)

        return frame


def draw_text_top_left(frame, text, y_offset=0, font_scale=1, color=(0, 255, 0), thickness=2):
    cv2.putText(frame, text, (10, 30 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame
