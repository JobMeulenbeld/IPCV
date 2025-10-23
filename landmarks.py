import cv2
import numpy as np

def add_forehead_arc(frame, landmarks, num_points=10, height_ratio=0.08, color=(0, 0, 255), draw=False):
    if landmarks is None:
        return None
    # Start and end points
    p0 = landmarks[0]
    p16 = landmarks[16]

    # Estimate face height for curvature
    y_min, y_max = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
    face_height = y_max - y_min

    arc_height = face_height * height_ratio

    # Generate arc points
    arc_pts = []
    for t in np.linspace(0, 1, num_points):
        # Linear interpolation between p0 and p8
        x = (1 - t) * (p0[0]+5) + t * (p16[0]-5)
        y = (1 - t) * p0[1] + t * p16[1]
        # Add upward curvature (parabolic)
        curve = -4 * arc_height * (t - 0.5) ** 2 + arc_height  # max at t=0.5
        arc_pts.append([x, (y - 20) - curve])

    arc_pts = np.array(arc_pts, dtype=np.float32)

    # Append to landmarks
    extended = np.vstack((landmarks, arc_pts))

    if draw:
        # Draw points
        start_index = len(landmarks)
        for i, (x, y) in enumerate(arc_pts.astype(int)):
            cv2.circle(frame, (x, y), 2, color, -1)
            cv2.putText(frame, str(start_index + i), (x + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return extended

def get_landmarks(frame, smooth_landmarks, face_net, facemark, alpha=0.3, count_points = False):
    h, w = frame.shape[:2]
    # --- Step 4: Detect faces with DNN ---
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            faces.append((x1, y1, x2 - x1, y2 - y1))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    if len(faces) > 0:
        # Before facemark.fit():
        faces_np = []
        for (x, y, w_, h_) in faces:
            y_adj = y + int(0.1 * h_)        # move top down a bit
            h_adj = int(h_ * 0.9)            # cut off lower 10%
            faces_np.append([x, y_adj, w_, h_adj])
        faces_np = np.array(faces_np)
        ok, landmarks = facemark.fit(gray, faces_np)
        if ok:
            # Simple exponential smoothing
            current_landmarks = landmarks[0][0]
            if smooth_landmarks is None:
                smooth_landmarks = current_landmarks
            else:
                smooth_landmarks = alpha * smooth_landmarks + (1 - alpha) * current_landmarks

            # Draw the points
            for index, (x, y) in enumerate(smooth_landmarks.astype(int)):
                # cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                if count_points:
                    label_pos = (x + 5, y)
                    cv2.putText(frame, str(index), label_pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2, cv2.LINE_AA)
                    
                # for (x, y, w_, h_) in faces:
                #     cv2.rectangle(frame, (x, y), (x + w_, y + h_), (255, 0, 0), 2)
            return smooth_landmarks
    return None

