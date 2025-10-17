import cv2
import numpy as np
import math
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

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    landmarks = get_landmarks(frame, smooth_landmarks=landmarks, face_net=face_net, facemark=facemark, alpha=0.3, count_points=True)

    height, width, channels = glasses.shape

    x1, y1 = landmarks[0]
    x2, y2 = landmarks[16]

    x_difference = int(abs(x2 - x1))
    y_difference = int(abs(y2 - y1))

    ratio = x_difference / width
    height_resized = int(height * ratio)

    glasses_resized = cv2.resize(glasses, (x_difference, height_resized))

    dy = y2 - y1
    dx = x2 - x1
    angle_rad = -math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    glasses_rotated = rotate_image(glasses_resized, angle_deg)

    top_left = (int(landmarks[0][0]), int(landmarks[0][1] - 30))
    bottom_right = (int(top_left[0] + x_difference), int(top_left[1] + height_resized))

    frame = overlay_transparent(frame, glasses_rotated, int(top_left[0]), int(top_left[1]))

    #frame = cv2.flip(frame, 1)

    cv2.imshow("Real-time Facial Landmarks (DNN + LBF)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()