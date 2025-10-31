import cv2
import numpy as np
import math
from landmarks import get_landmarks
from face_feature import detect_face, detect_eyes, detect_smile, approximate_landmarks

glasses = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)

landmarks = None

def overlay_transparent(frame, overlay, x, y):
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

def face_augmemtation(frame, landmarks):
    height, width, channels = glasses.shape

    x1, y1 = landmarks[8]
    x2, y2 = landmarks[9]

    x_difference = int(abs(x2 - x1))
    y_difference = int(abs(y2 - y1))

    ratio = x_difference / width
    height_resized = int(height * ratio)

    glasses_resized = cv2.resize(glasses, (x_difference, height_resized))

    top_left = (int(landmarks[0][0]), int(landmarks[0][1]))
    bottom_right = (int(top_left[0] + x_difference), int(top_left[1] + height_resized))

    frame = overlay_transparent(frame, glasses_resized, int(top_left[0]), int(top_left[1]))

    frame = cv2.flip(frame, 1)

    return frame

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
        #     cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        smiles = detect_smile(gray, face)
        # for (sx,sy,sw,sh) in smiles:
        #     cv2.rectangle(frame, (sx,sy), (sx+sw, sy+sh), (0,0,255), 2)
        landmarks = approximate_landmarks(face, eyes, smiles)
        # for (lx, ly) in landmarks:
        #     cv2.circle(frame, (int(lx), int(ly)), 3, (0,255,255), -1)
        # Apply squish effect

        frame = face_augmemtation(frame, landmarks)
    
    frame_counter += 1

    cv2.imshow("Real-time Facial Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

'''
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    landmarks = get_landmarks(frame, smooth_landmarks=landmarks, face_net=face_net, facemark=facemark, alpha=0.3, count_points=True)

    if landmarks is None:
        continue

    frame = face_augmemtation(frame, landmarks)

    cv2.imshow("Real-time Facial Landmarks (DNN + LBF)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
'''