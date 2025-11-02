import cv2
import numpy as np
import math
from landmarks import get_landmarks
#from face_feature import detect_face, detect_eyes, detect_smile, approximate_landmarks

glasses = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
hat = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)

landmarks = None

def overlay_transparent(frame, overlay, x, y):
    """
    Overlay RGBA `overlay` image onto BGR `frame` at position (x, y).
    Handles alpha blending and clipping at image borders.
    """
    h, w = frame.shape[:2]
    h_o, w_o = overlay.shape[:2]

    # If overlay is completely outside frame
    if x >= w or y >= h or x + w_o <= 0 or y + h_o <= 0:
        return frame

    # Clip overlay region
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w_o, w)
    y2 = min(y + h_o, h)

    overlay_x1 = max(0, -x)
    overlay_y1 = max(0, -y)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    # Extract valid regions
    overlay_cropped = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    overlay_img = overlay_cropped[:, :, :3]
    mask = overlay_cropped[:, :, 3:] / 255.0

    # Blend
    frame[y1:y2, x1:x2] = (1.0 - mask) * frame[y1:y2, x1:x2] + mask * overlay_img
    return frame

def face_overlay(frame, image, landmark1, landmark2, scale_factor, x_offset, y_offset):
    height, width, channels = image.shape

    x1, y1 = landmark1
    x2, y2 = landmark2
    #cv2.circle(frame, (int(x1), int(y1)), radius=3, color=(0, 0, 255), thickness=-1)
    #cv2.circle(frame, (int(x2), int(y2)), radius=3, color=(0, 0, 255), thickness=-1)

    dx = int(abs(x2 - x1))
    image_width_resized = int(dx * scale_factor)

    ratio = image_width_resized / width
    image_height_resized = int(height * ratio)

    image_x_offset = (image_width_resized-dx)/2 + (x_offset * ratio)
    image_y_offset = (y_offset * ratio)

    image_resized = cv2.resize(image, (image_width_resized, image_height_resized))

    top_left = (int(x1 - image_x_offset), int(y1 - image_y_offset))
    #bottom_right = (int(top_left[0] + image_width_resized), int(top_left[1] + image_height_resized))

    frame = overlay_transparent(frame, image_resized, int(top_left[0]), int(top_left[1]))
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