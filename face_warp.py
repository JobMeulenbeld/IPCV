import cv2
import numpy as np
from landmarks import get_landmarks, add_forehead_arc

triangles_cache = None   # global variable

def get_delaunay_triangles(coords, width, height):
    # Create subdiv object that covers the image
    rect = (0, 0, width, height)
    subdiv = cv2.Subdiv2D(rect)

    # Insert all points (your landmarks)
    for p in coords:
        subdiv.insert(tuple(p))

    # Get list of triangles as flat arrays [x1, y1, x2, y2, x3, y3]
    triangleList = subdiv.getTriangleList()

    # Convert to landmark indices
    triangles = []
    for t in triangleList:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        for p in pts:
            for i in range(len(coords)):
                if np.linalg.norm(p - coords[i]) < 1.0:
                    idx.append(i)
                    break
        if len(idx) == 3:
            triangles.append(tuple(idx))

    return triangles

def squish_features(frame, landmarks, strength=0.7, debug=False, NeuralNet=False):
    global triangles_cache

    if landmarks is None:
        return frame, landmarks
    
    try:
        h, w = frame.shape[:2]
        src_pts = landmarks.astype(np.float32)
        dst_pts = src_pts.copy()

        if NeuralNet:
            inner_idxs = list(range(17, 68))  # full inner face
            # eyebrows = list(range(17, 27))    # eyebrows
            # inner_idxs += eyebrows            # include brows explicitly
            center = np.mean(src_pts[inner_idxs], axis=0)

            # Move inner points toward center
            for i in inner_idxs:
                direction = center - src_pts[i]
                dst_pts[i] = src_pts[i] + direction * (1 - strength)
        else:
            outer_idxs = [0, 1, 2, 3, 4, 5 , 6, 7]     # face corners
            inner_idxs = list(range(8, len(src_pts)))  # eyes, mouth, nose
            center = np.mean(src_pts[inner_idxs], axis=0)
            for i in inner_idxs:
                direction = center - src_pts[i]
                dst_pts[i] = src_pts[i] + direction * (1 - strength)

        #Compute Delaunay triangulation for warp
        if triangles_cache is None:
            triangles_cache = get_delaunay_triangles(src_pts, w, h)
        
        result = frame.astype(np.float32)
        for tri in triangles_cache:
            src_tri = np.float32([src_pts[i] for i in tri])
            dst_tri = np.float32([dst_pts[i] for i in tri])
            if debug:
                pts = np.int32([src_pts[i] for i in tri])
                cv2.polylines(frame, [pts], True, (0, 255, 0), 1)

            r1 = cv2.boundingRect(src_tri)
            r2 = cv2.boundingRect(dst_tri)

            src_roi = frame[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]

            src_off = np.array([[p[0]-r1[0], p[1]-r1[1]] for p in src_tri], np.float32)
            dst_off = np.array([[p[0]-r2[0], p[1]-r2[1]] for p in dst_tri], np.float32)

            M = cv2.getAffineTransform(src_off, dst_off)
            warped_roi = cv2.warpAffine(src_roi, M, (r2[2], r2[3]),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT_101)

            # Create mask for triangle area
            mask = np.zeros((r2[3], r2[2], 3), np.float32)
            cv2.fillConvexPoly(mask, np.int32(dst_off), (1,1,1), 16, 0)

            # 🔹Write directly into frame — no blending
            y1, y2 = r2[1], r2[1]+r2[3]
            x1, x2 = r2[0], r2[0]+r2[2]

            existing = result[y1:y2, x1:x2]
            existing[mask > 0] = warped_roi[mask > 0]
            result[y1:y2, x1:x2] = existing

        return np.clip(result, 0, 255).astype(np.uint8), dst_pts
    except Exception as e:
        print("Error in squish_features:", e)
        return frame, landmarks


# # *********************************************************************************

# modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
# configFile = "deploy.prototxt"
# face_net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


# facemark = cv2.face.createFacemarkLBF()
# facemark.loadModel("lbfmodel.yaml")

# # open webcam
# cap = cv2.VideoCapture(0)

# landmarks = None
# frame_count = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     landmarks = get_landmarks(frame, smooth_landmarks=landmarks, face_net=face_net, facemark=facemark, alpha=0.3, count_points=False)
#     extended_landmarks = add_forehead_arc(frame, landmarks, 20, 0.25, color=(255,0,0), draw=False)

#     frame, dst_points = squish_features(frame, extended_landmarks, strength=0.7, NeuralNet=True, debug=False)

#     print(dst_points)

#     cv2.imshow("Real-time Facial Landmarks (DNN + LBF)", frame)
#     if cv2.waitKey(1) & 0xFF == 27:  # ESC
#         break

#     frame_count += 1

# cap.release()
# cv2.destroyAllWindows()