import cv2
import numpy as np
import math
from landmarks import get_landmarks
#from face_feature import detect_face, detect_eyes, detect_smile, approximate_landmarks

class face_augmentation:
    def __init__(self, image, scale_factor, x_offset, y_offset, smoothing_factor=0.8):
        self.image = image
        self.scale_factor = scale_factor
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.smoothing_factor = smoothing_factor

        self.prev_top_left = None
        self.prev_size = None

    def overlay_transparent(self, frame, overlay, x, y):
        """
        Overlay RGBA `overlay` image onto BGR `frame` at position (x, y).
        Handles alpha blending and clipping at image borders.
        """
        h, w = frame.shape[:2]
        h_o, w_o = overlay.shape[:2]

        # Clip overlay to stay within the frame
        if x >= w or y >= h:
            return frame

        w = min(w_o, w - x)
        h = min(h_o, h - y)

        if w <= 0 or h <= 0:
            return frame

        overlay = overlay[0:h, 0:w]
        overlay_img = overlay[:, :, :3]
        mask = overlay[:, :, 3:] / 255.0  # alpha channel normalized to [0,1]

        # Perform alpha blending
        frame[y:y+h, x:x+w] = (1.0 - mask) * frame[y:y+h, x:x+w] + mask * overlay_img

        return frame
    
    def smoothing(self, top_left, size):
        
        if self.prev_top_left is None:
            self.prev_top_left = top_left
            self.prev_size = size

        alpha = 1.0 - self.smoothing_factor
        self.prev_top_left = self.smoothing_factor * self.prev_top_left + alpha * top_left
        self.prev_size = self.smoothing_factor * self.prev_size + alpha * size

        smoothed_top_left = self.prev_top_left.astype(int)
        smoothed_size = self.prev_size.astype(int)

        return smoothed_top_left, smoothed_size

    def face_overlay(self, frame, landmark1, landmark2):
        height, width, channels = self.image.shape

        x1, y1 = landmark1
        x2, y2 = landmark2
        #cv2.circle(frame, (int(x1), int(y1)), radius=3, color=(0, 0, 255), thickness=-1)
        #cv2.circle(frame, (int(x2), int(y2)), radius=3, color=(0, 0, 255), thickness=-1)

        dx = int(abs(x2 - x1))
        image_width_resized = int(dx * self.scale_factor)

        ratio = image_width_resized / width
        image_height_resized = int(height * ratio)

        image_x_offset = (image_width_resized-dx)/2 + (self.x_offset * ratio)
        image_y_offset = (self.y_offset * ratio)

        top_left = np.array([x1 - image_x_offset, y1 - image_y_offset], dtype=np.float32)
        #bottom_right = (int(top_left[0] + image_width_resized), int(top_left[1] + image_height_resized))

        size = np.array([image_width_resized, image_height_resized], dtype=np.float32)

        smoothed_top_left, smoothed_size = self.smoothing(top_left, size)

        image_resized = cv2.resize(self.image, tuple(smoothed_size))

        if smoothed_top_left[1] < 0:
            crop_top = abs(smoothed_top_left[1])
            image_resized = image_resized[crop_top:, :, :]  # remove the top rows
            top_left = (smoothed_top_left[0], 0)

        frame = self.overlay_transparent(frame, image_resized, int(smoothed_top_left[0]), int(smoothed_top_left[1]))
        return frame


'''
# open webcam
cap = cv2.VideoCapture(0)
frame_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    strength = 1.0 + 0.5 * math.sin(frame_counter * 0.05)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = detect_face(gray)
    if face is not None:
        x,y,w,h = face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        eyes = detect_eyes(gray, face)
        # for (ex,ey,ew,eh) in eyes:
        #    cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        smiles = detect_smile(gray, face)
        # for (sx,sy,sw,sh) in smiles:
        #     cv2.rectangle(frame, (sx,sy), (sx+sw, sy+sh), (0,0,255), 2)
        landmarks = approximate_landmarks(face, eyes, smiles)
        # for (lx, ly) in landmarks:
        #     cv2.circle(frame, (int(lx), int(ly)), 3, (0,255,255), -1)
        # Apply squish effect

        frame = face_overlay(frame, glasses, landmarks[8], landmarks[9], 2.2, 0, 160)
        frame = face_overlay(frame, hat, landmarks[0], landmarks[2], 1, 0, 4500)
    
    frame_counter += 1

    cv2.imshow("Real-time Facial Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
'''