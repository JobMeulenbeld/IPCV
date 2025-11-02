import cv2

from gesture_determiner import GestureDeterminer
from hand_detector import HandDetector
from tracker import Tracker
from face_feature import detect_face, detect_eyes, detect_smile, approximate_landmarks
from face_warp import squish_features
from face_augmentation import face_overlay

modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
face_net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
glasses = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
hat = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)

#facemark = cv2.face.createFacemarkLBF()
#facemark.loadModel("lbfmodel.yaml")

# open webcam
cap = cv2.VideoCapture(0)

landmarks = None

hand_detector = HandDetector()
tracker = Tracker()
gesture_detector = GestureDeterminer()

strength = 1.0
state = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Flip horizontally

    #landmarks = get_landmarks(frame, smooth_landmarks=landmarks, face_net=face_net, facemark=facemark, alpha=0.3)

    frame, gesture = hand_detector.process_frame(frame, tracker, gesture_detector)

    print("Current gesture state:", gesture)

    if gesture == "LEFT":
        if(state < 4): 
            state += 1
        elif state ==4:
            state = 1
    elif gesture == "RIGHT":
        if(state > 1):
            state -= 1
        elif state == 1:
            state = 4
    elif gesture == "UP":
        if(strength <= 1.5): 
            strength += 0.25
    elif gesture == "DOWN":
        if(strength >= 0.5): 
            strength -= 0.25

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
        frame, landmarks = squish_features(frame, landmarks, strength=strength, debug=False)

    if state == 1:
        #nothing
        pass
    elif state == 2:
        frame = face_overlay(frame, glasses, landmarks[8], landmarks[9], 2.2, 0, 160)
    elif state == 3:
        frame = face_overlay(frame, hat, landmarks[0], landmarks[2], 1, 0, 4500)
    elif state == 4:
        frame = face_overlay(frame, glasses, landmarks[8], landmarks[9], 2.2, 0, 160)
        frame = face_overlay(frame, hat, landmarks[0], landmarks[2], 1, 0, 4500)

    cv2.imshow("Real-time Facial Landmarks (DNN + LBF)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()