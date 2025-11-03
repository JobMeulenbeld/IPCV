import cv2
from gesture_determiner import GestureDeterminer
from hand_detector import HandDetector
from tracker import Tracker
from face_feature import FaceFeature
from face_augmentation import face_augmentation
from face_warp import FaceWarp

def handle_gesture(gesture, state, strength, closed_hand_counter):
    """Update the overlay state and strength based on the detected gesture."""
    if gesture == "LEFT":
        state = 1 if state == 4 else state + 1
    elif gesture == "RIGHT":
        state = 4 if state == 1 else state - 1
    elif gesture == "UP":
        if strength <= 1.5:
            strength += 0.25
    elif gesture == "DOWN":
        if strength >= 0.5:
            strength -= 0.25
    elif gesture == "closed_hand":
        closed_hand_counter += 1
        if closed_hand_counter >= 25:
            state = 1
            strength = 1
            closed_hand_counter = 0
    else:
        closed_hand_counter = 0  # Reset counter if gesture changes

    return state, strength, closed_hand_counter


def main():
    #Load overlay images and initialize them
    glasses = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
    hat = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)
    glasses_overlay = face_augmentation(glasses, 2.2, 0, 160)
    hat_overlay = face_augmentation(hat, 1, 0, 4500)

    #Initialize detectors and trackers
    hand_detector = HandDetector()
    tracker = Tracker()
    gesture_detector = GestureDeterminer()
    face_feature_detector = FaceFeature()
    face_warping = FaceWarp()

    #Initialze starting variables
    strength = 1.0
    state = 1
    closed_hand_counter = 0

    #Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Flip horizontally

        #Detect hand and gesture
        frame, gesture = hand_detector.process_frame(frame, tracker, gesture_detector)
        print("Current gesture state:", gesture)

        #Update state, strength, and closed hand counter based on gesture
        state, strength, closed_hand_counter = handle_gesture(gesture, state, strength, closed_hand_counter)

        #Detect facial landmarks
        frame, landmarks = face_feature_detector.process_frame(frame)

        #Warp facial features based on strength
        if landmarks is not None:
            frame, landmarks = face_warping.squish_features(frame, landmarks, strength=strength, debug=False)

        #Overlay augmentations based on current state
        if state == 2:
            frame = glasses_overlay.face_overlay(frame, landmarks[8], landmarks[9])
        elif state == 3:
            frame = hat_overlay.face_overlay(frame, landmarks[0], landmarks[2])
        elif state == 4:
            frame = glasses_overlay.face_overlay(frame, landmarks[8], landmarks[9])
            frame = hat_overlay.face_overlay(frame, landmarks[0], landmarks[2])

        # Display the resulting frame
        cv2.imshow("Real-time Facial Landmarks (DNN + LBF)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__': 
    main()