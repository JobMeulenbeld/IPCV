import math
import cv2


class GestureDeterminer:
    def __init__(self, choosing_gesture_threshold=20, direction_counter_threshold=5):
        self.open_counter = 0
        self.choose_gesture = False
        self.current_gesture_state = "none"
        self.choosing_gesture_threshold = choosing_gesture_threshold
        self.direction_counter_threshold = direction_counter_threshold
        self.direction_counters = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0}

    def detect_open_hand(self, fingertips, center, open_threshold=80, closed_threshold=70):
        if not fingertips or center is None:
            return "unknown"

        # Compute average fingertip-to-center distance
        distances = [math.hypot(pt[0] - center[0], pt[1] - center[1]) for pt in fingertips]
        avg_distance = sum(distances) / len(distances)

        # Classify gesture
        if avg_distance > open_threshold:
            return "open"
        elif avg_distance < closed_threshold:
            return "closed"
        else:
            return "unknown"

    def process_frame(self, frame, fingertips, center, direction, bounding_box):
        # Reset counters if hand dissapears
        if not fingertips:
            self.open_counter = 0
            self.choose_gesture = False
            
        self.current_gesture_state = "none"
        # Determine hand gesture and if we need to switch to choosing gesture
        hand_command = self.detect_open_hand(fingertips, center) if fingertips else "none"

        if hand_command == "open":
            self.open_counter += 1
            if self.open_counter >= self.choosing_gesture_threshold:
                self.choose_gesture = True
                self.current_gesture_state = "open_hand"
                self.open_counter = 0
                self.direction_counters = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0}

        if hand_command == "closed":
            self.open_counter = max(self.open_counter - 5, 0)

        if self.choose_gesture:
            self.draw_text_top_left(frame, "choose the gesture", y_offset=30)

            # Check if wants to change filter
            for key in self.direction_counters.keys():
                if key == direction:
                    self.direction_counters[key] += 1
                else:
                    self.direction_counters[key] = 0

                if self.direction_counters[key] >= self.direction_counter_threshold:
                    self.current_gesture_state = direction
                    self.choose_gesture = False
                    self.open_counter = 0
                    break
        else:
            self.draw_text_top_left(frame, f"current state = {self.current_gesture_state}", y_offset=0)
            self.draw_text_top_left(frame, f"open_hand counter = {self.open_counter}", y_offset=30)

        return frame, self.current_gesture_state

    def draw_text_top_left(self, frame, text, y_offset=0, font_scale=1, color=(0, 255, 0), thickness=2):
        cv2.putText(frame, text, (10, 30 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return frame