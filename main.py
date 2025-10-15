import cv2
import numpy as np
from landmarks import get_landmarks


modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
face_net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")

# open webcam
cap = cv2.VideoCapture(0)

landmarks = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    #hello
    get_landmarks(frame, smooth_landmarks=landmarks, face_net=face_net, facemark=facemark, alpha=0.3)

    cv2.imshow("Real-time Facial Landmarks (DNN + LBF)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()