import cv2
import numpy as np
from landmarks import get_landmarks


modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
face_net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")

glasses = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)

# open webcam
cap = cv2.VideoCapture(0)

landmarks = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    landmarks = get_landmarks(frame, smooth_landmarks=landmarks, face_net=face_net, facemark=facemark, alpha=0.3, count_points=True)

    width, height = glasses.shape[:2]

    x1, y1 = landmarks[0]
    x2, y2 = landmarks[16]

    x_difference = int(abs(x2 - x1))
    ratio = x_difference / width if width > 0 else 1
    height_resized = int(height * ratio)

    glasses_resized = cv2.resize(glasses, (x_difference, height_resized))

    # Ensure integer coordinates
    top_left = (int(landmarks[0][0]), int(landmarks[0][1]))
    bottom_right = (int(top_left[0] + x_difference), int(top_left[1] + height_resized))

    # Clip coordinates to frame boundaries
    h, w = frame.shape[:2]
    x1, y1 = max(0, top_left[0]), max(0, top_left[1])
    x2, y2 = min(w, bottom_right[0]), min(h, bottom_right[1])

    # Resize overlay if necessary to fit the clipped region
    overlay_w, overlay_h = x2 - x1, y2 - y1
    glasses_resized = cv2.resize(glasses_resized, (overlay_w, overlay_h))

    # Blend or overlay
    if glasses_resized.shape[2] == 4:
        alpha = glasses_resized[:, :, 3] / 255.0
        rgb = glasses_resized[:, :, :3]
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (
                alpha * rgb[:, :, c] +
                (1 - alpha) * frame[y1:y2, x1:x2, c]
            )
    else:
        frame[y1:y2, x1:x2] = glasses_resized

    cv2.imshow("Real-time Facial Landmarks (DNN + LBF)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()