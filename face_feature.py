import cv2
import numpy as np
from face_warp import squish_features
from face_augmentation import face_overlay
import math

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

glasses = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
hat = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)

def detect_face(gray):
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100,100))
    if len(faces) == 0: return None
    # pick largest
    x,y,w,h = max(faces, key=lambda r: r[2]*r[3])
    return (x,y,w,h)

def detect_eyes(gray, face):
    x,y,w,h = face
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(30,30))
    return [(ex+x, ey+y, ew, eh) for (ex,ey,ew,eh) in eyes]

def detect_smile(gray, face):
    x,y,w,h = face
    my0 = int(y + h*0.6)
    roi_gray = gray[my0:y+h, x:x+w]
    smiles = smile_cascade.detectMultiScale(roi_gray, 1.1, 50, minSize=(25,25))
    return [(sx+x, sy+my0, sw, sh) for (sx,sy,sw,sh) in smiles]

def approximate_landmarks(face, eyes, smiles):
    x, y, w, h = face

    # --- Basic face corners
    landmarks = [
        (x, y),               # top-left
        (x+w//2, y),          # top-center
        (x+w, y),             # top-right
        (x+w, y+h//2),        # right-center
        (x+w, y+h),           # bottom-right
        (x+w//2, y+h),        # bottom-center
        (x, y+h),             # bottom-left
        (x, y+h//2),          # left-center
    ]

    # Eye centers
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])  # left-to-right
        left_eye = (eyes[0][0]+eyes[0][2]//2, eyes[0][1]+eyes[0][3]//2)
        right_eye = (eyes[1][0]+eyes[1][2]//2, eyes[1][1]+eyes[1][3]//2)
    else:
        left_eye = (x + int(0.3*w), y + int(0.35*h))
        right_eye = (x + int(0.7*w), y + int(0.35*h))
    landmarks += [left_eye, right_eye]

    # Mouth center / corners
    if len(smiles) > 0:
        sx, sy, sw, sh = max(smiles, key=lambda s: s[2])
        mouth_center = (sx + sw//2, sy + sh//2)
        mouth_left = (sx, sy + sh//2)
        mouth_right = (sx + sw, sy + sh//2)
    else:
        mouth_center = (x + w//2, y + int(0.75*h))
        mouth_left = (x + int(0.4*w), y + int(0.75*h))
        mouth_right = (x + int(0.6*w), y + int(0.75*h))
    landmarks += [mouth_left, mouth_center, mouth_right]

    # Nose tip (approx halfway)
    nose = (x + w//2, y + int(0.55*h))
    landmarks.append(nose)

    return np.array(landmarks, np.float32)

# open webcam

# cap = cv2.VideoCapture(0)
# frame_counter = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     strength = 1.0 + 0.5 * math.sin(frame_counter * 0.05)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     face = detect_face(gray)
#     if face is not None:
#         x,y,w,h = face
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
#         eyes = detect_eyes(gray, face)
#         # for (ex,ey,ew,eh) in eyes:
#         #     cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
#         smiles = detect_smile(gray, face)
#         # for (sx,sy,sw,sh) in smiles:
#         #     cv2.rectangle(frame, (sx,sy), (sx+sw, sy+sh), (0,0,255), 2)
#         landmarks = approximate_landmarks(face, eyes, smiles)
#         # for (lx, ly) in landmarks:
#         #     cv2.circle(frame, (int(lx), int(ly)), 3, (0,255,255), -1)
#         # Apply squish effect
#         frame, landmarks = squish_features(frame, landmarks, strength=strength, debug=False)

#         frame = face_overlay(frame, glasses, landmarks[8], landmarks[9], 2.2, 0, 160)
#         frame = face_overlay(frame, hat, landmarks[0], landmarks[2], 1, 0, 4500)
    
#     frame_counter += 1

#     cv2.imshow("Real-time Facial Landmarks", frame)
#     if cv2.waitKey(1) & 0xFF == 27:  # ESC
#         break

# cap.release()
# cv2.destroyAllWindows()
