import cv2
import numpy as np

from face_warp import squish_features
from face_feature import detect_face, detect_eyes, detect_smile, approximate_landmarks

# --- CONFIG ---
FLOW_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)
THRESHOLD = 1.0  # Minimum average flow to consider as movement

# Load Haar cascade once
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_hand_gesture(prev_gray, gray, frame):
    h, w = gray.shape

    # Detect face(s) and mask them out
    mask = np.ones_like(gray, dtype=np.uint8)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w_face, h_face) in faces:
        expand_y = int(h_face * 0.5)
        expand_x = int(w_face * 0.2)
        y1 = max(y - expand_y, 0)
        y2 = min(y + h_face, h)
        x1 = max(x - expand_x, 0)
        x2 = min(x + w_face + expand_x, w)
        mask[y1:y2, x1:x2] = 0  # mask out face region
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Apply mask before computing flow
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    prev_masked = cv2.bitwise_and(prev_gray, prev_gray, mask=mask)

    # Define hand ROIs
    top_height = int(h * 0.3)
    left_x_end = int(w / 3)
    right_x_start = int(w * 2 / 3)

    # Compute flow in ROIs only
    flow_left = cv2.calcOpticalFlowFarneback(
        prev_masked[:top_height, :left_x_end],
        gray_masked[:top_height, :left_x_end],
        None, **FLOW_PARAMS
    )
    flow_right = cv2.calcOpticalFlowFarneback(
        prev_masked[:top_height, right_x_start:],
        gray_masked[:top_height, right_x_start:],
        None, **FLOW_PARAMS
    )

    # Compute mean horizontal motion
    left_mean_x = np.mean(flow_left[..., 0])
    right_mean_x = np.mean(flow_right[..., 0])

    # Draw visualization boxes
    cv2.rectangle(frame, (0, 0), (left_x_end, top_height), (0, 255, 0), 2)
    cv2.rectangle(frame, (right_x_start, 0), (w, top_height), (0, 255, 0), 2)

    # Determine direction for each side
    left_dir = "right" if left_mean_x > THRESHOLD else "left" if left_mean_x < -THRESHOLD else "-"
    right_dir = "right" if right_mean_x > THRESHOLD else "left" if right_mean_x < -THRESHOLD else "-"

    cv2.putText(frame, f"L: {left_dir}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"R: {right_dir}", (w//2 + 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


    return left_dir, right_dir, frame


cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

strength = 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    left_dir, right_dir, frame = detect_hand_gesture(prev_gray, gray, frame)

    if left_dir == "left" and right_dir == "left":
        print("Both hands moving left!")
    elif left_dir == "right" and right_dir == "right":
        print("Both hands moving right!")
    elif left_dir == "left" and right_dir == "right":
        if(strength <= 1.4): 
            strength += 0.1
        print("Hands moving apart!")
    elif left_dir == "right" and right_dir == "left":
        if(strength >= 0.5): 
            strength -= 0.1
        print("Hands moving together!")

    print(f"Current squish strength: {strength:.2f}")
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
        frame, landmarks = squish_features(frame, landmarks, strength=strength, debug=False)

    cv2.imshow("Gesture Detection", frame)
    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
