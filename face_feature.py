import cv2
import numpy as np
#from face_warp import FaceWarp
import math

class FaceFeature:
    def __init__(self):
        self.previous_landmarks = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    def smooth_landmarks(self, prev_pts, new_pts, alpha=0.7):
        if prev_pts is None:
            return new_pts
        return (alpha * prev_pts) + ((1 - alpha) * new_pts)

    def detect_face(self, gray):
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100,100))
        if len(faces) == 0: return None
        # pick largest
        x,y,w,h = max(faces, key=lambda r: r[2]*r[3])
        return (x,y,w,h)

    def detect_eyes(self, gray, face):
        x,y,w,h = face
        roi_gray = gray[y:y+h//2, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(30,30))
        return [(ex+x, ey+y, ew, eh) for (ex,ey,ew,eh) in eyes]

    def detect_smile(self, gray, face):
        x,y,w,h = face
        my0 = int(y + h*0.6)
        roi_gray = gray[my0:y+h, x:x+w]
        smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.1, 50, minSize=(25,25))
        return [(sx+x, sy+my0, sw, sh) for (sx,sy,sw,sh) in smiles]

    def approximate_landmarks(self, face, eyes, smiles):
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
            left_eye = (x + int(0.3*w), y + int(0.4*h))
            right_eye = (x + int(0.7*w), y + int(0.4*h))
        landmarks += [left_eye, right_eye]

        # Mouth center / corners
        if len(smiles) > 0:
            sx, sy, sw, sh = max(smiles, key=lambda s: s[2])
            mouth_center = (sx + sw//2, sy + sh//2)
            mouth_left = (sx, sy + sh//2)
            mouth_right = (sx + sw, sy + sh//2)
        else:
            mouth_center = (x + w//2, y + int(0.77*h))
            mouth_left = (x + int(0.3*w), y + int(0.77*h))
            mouth_right = (x + int(0.7*w), y + int(0.77*h))
        landmarks += [mouth_left, mouth_center, mouth_right]

        # Nose tip (approx halfway)
        nose = (x + w//2, y + int(0.55*h))
        landmarks.append(nose)

        return np.array(landmarks, np.float32)

    def process_frame(self, frame, debug=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = self.detect_face(gray)
        if face is not None:
            x,y,w,h = face
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            eyes = self.detect_eyes(gray, face)
            if debug:
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
            smiles = self.detect_smile(gray, face)
            if debug:
                for (sx,sy,sw,sh) in smiles:
                    cv2.rectangle(frame, (sx,sy), (sx+sw, sy+sh), (0,0,255), 2)
            landmarks = self.approximate_landmarks(face, eyes, smiles)
            landmarks  = self.smooth_landmarks(self.previous_landmarks, landmarks, alpha=0.7)
            if debug:
                for (lx, ly) in landmarks:
                    cv2.circle(frame, (int(lx), int(ly)), 3, (0,255,255), -1)
            self.previous_landmarks = landmarks
            # Apply squish effect
            #frame, landmarks = self.face_warp.squish_features(frame, landmarks, strength=strength, debug=debug)
        return frame, self.previous_landmarks

