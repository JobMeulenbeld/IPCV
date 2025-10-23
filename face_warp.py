import cv2
import numpy as np
from landmarks import get_landmarks

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

def squish_features(frame, landmarks, strength=0.7):
    global triangles_cache

    if landmarks is None:
        return frame
    
    try:
        h, w = frame.shape[:2]
        src_pts = landmarks.astype(np.float32)
        dst_pts = src_pts.copy()

        inner_idxs = list(range(17, 68))  # full inner face
        eyebrows = list(range(17, 27))    # eyebrows
        inner_idxs += eyebrows            # include brows explicitly
        center = np.mean(src_pts[inner_idxs], axis=0)

        # Move inner points toward center
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

        return np.clip(result, 0, 255).astype(np.uint8)
    except Exception as e:
        print("Error in squish_features:", e)
        return frame

def face_affine_squish(frame, landmarks, strength=0.8):
    h, w = frame.shape[:2]
    pts = landmarks.astype(np.float32)

    # --- choose 3 stable points for affine reference ---
    # corners of eyes + tip of nose (good triangle for alignment)
    src_tri = np.float32([
        pts[36],   # left eye corner
        pts[45],   # right eye corner
        pts[33]    # nose tip
    ])

    # Compute the center of the face region
    face_center = np.mean(pts[17:68], axis=0)

    # Move those points inward toward center
    dst_tri = src_tri.copy()
    for i in range(3):
        direction = face_center - src_tri[i]
        dst_tri[i] = src_tri[i] + direction * (1 - strength)

    # --- get full-face ROI (convex hull of all facial landmarks) ---
    face_hull = cv2.convexHull(pts[0:68])
    x, y, w_box, h_box = cv2.boundingRect(face_hull)
    roi = frame[y:y+h_box, x:x+w_box]
    mask = np.zeros_like(roi, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(face_hull - [x, y]), (255, 255, 255))

    # --- compute affine transform ---
    print(src_tri.shape, dst_tri.shape, x, y)
    src_local = (src_tri - [x, y]).astype(np.float32)
    dst_local = (dst_tri - [x, y]).astype(np.float32)
    M = cv2.getAffineTransform(src_local, dst_local)

    # --- warp the entire ROI ---
    warped_roi = cv2.warpAffine(roi, M, (w_box, h_box),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)

    # --- blend it back (so only inside the face hull is replaced) ---
    inv_mask = cv2.bitwise_not(mask)
    face_bg = cv2.bitwise_and(roi, inv_mask)
    face_fg = cv2.bitwise_and(warped_roi, mask)
    combined = cv2.add(face_bg, face_fg)
    frame[y:y+h_box, x:x+w_box] = combined

    return frame

def local_face_squish(frame, landmarks, strength=0.7):
    if landmarks is None:
        return frame
    
    h, w = frame.shape[:2]
    pts = landmarks.astype(np.float32)

    # --- Face region (convex hull of facial landmarks) ---
    face_hull = cv2.convexHull(pts[0:68])
    x, y, w_box, h_box = cv2.boundingRect(face_hull)
    roi = frame[y:y+h_box, x:x+w_box]
    roi_h, roi_w = roi.shape[:2]

    # --- Center of face region ---
    cx, cy = roi_w / 2, roi_h / 2

    # --- Affine matrix that scales toward center ---
    M = cv2.getRotationMatrix2D((cx, cy), 0, strength)

    # Warp the **ROI only**, not the whole image
    squished = cv2.warpAffine(roi, M, (roi_w, roi_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    # --- Mask to isolate face region ---
    mask = np.zeros((roi_h, roi_w), np.uint8)
    cv2.fillConvexPoly(mask, np.int32(face_hull - [x, y]), 255)
    mask = cv2.GaussianBlur(mask, (25,25), 10)

    # --- Blend warped ROI into frame ---
    mask3 = cv2.merge([mask, mask, mask]) / 255.0
    result = frame.copy().astype(np.float32)
    result[y:y+h_box, x:x+w_box] = \
        result[y:y+h_box, x:x+w_box] * (1 - mask3) + squished * mask3

    return np.clip(result, 0, 255).astype(np.uint8)

def show_debug(title, img, scale=0.5):
    small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    cv2.imshow(title, small)

def face_morph_squish(frame, landmarks, strength=0.7):
    global triangles_cache

    if landmarks is None:
        return frame

    src_pts = landmarks.astype(np.float32)
    h, w = frame.shape[:2]

    # Convex hull mask (face area)
    hull = cv2.convexHull(src_pts)
    center = np.mean(hull, axis=0)
    scale = 0.95  # e.g., 0.95 makes it about 5% smaller (≈4 px for a ~100 px face)

    hull = center + (hull - center) * scale
    hull = hull.astype(np.int32)
    face_mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(face_mask, np.int32(hull), 255)

    show_debug("Face mask", face_mask)

    # Compute face center and squished landmarks
    center = np.mean(src_pts, axis=0)
    dst_pts = src_pts + (center - src_pts) * (1 - strength)

    hull = cv2.convexHull(dst_pts)
    transformed_mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(transformed_mask, np.int32(hull), 255)

    # Compute the ring area (where fill color should come from)
    ring_mask = cv2.bitwise_xor(face_mask, transformed_mask)

    # Take mean color from that ring area in the original frame
    ring_pixels = frame[ring_mask == 255]
    mean_color = np.mean(ring_pixels, axis=0).astype(np.uint8)
    fill_color = tuple(int(c) for c in mean_color)

    # Build Delaunay triangulation (once)
    if triangles_cache is None:
        triangles_cache = get_delaunay_triangles(src_pts, w, h)

    warped = np.zeros_like(frame)

    for tri in triangles_cache:
        src_tri = np.float32([src_pts[i] for i in tri])
        dst_tri = np.float32([dst_pts[i] for i in tri])

        r1 = cv2.boundingRect(src_tri)
        r2 = cv2.boundingRect(dst_tri)

        x, y, w_rect, h_rect = r1
        x2, y2, w2, h2 = r2

        src_off = np.array([[p[0]-r1[0], p[1]-r1[1]] for p in src_tri], np.float32)
        dst_off = np.array([[p[0]-r2[0], p[1]-r2[1]] for p in dst_tri], np.float32)

        M = cv2.getAffineTransform(src_off, dst_off)
        roi_src = frame[y:y+h_rect, x:x+w_rect]
        warp = cv2.warpAffine(
            roi_src, M, (w2, h2),
            None, flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        mask_tri = np.zeros((h2, w2), np.uint8)
        cv2.fillConvexPoly(mask_tri, np.int32(dst_off), 255)

        roi_dst = warped[y2:y2+h2, x2:x2+w2]
        mask3 = mask_tri[..., None] / 255.0
        roi_dst[:] = roi_dst * (1 - mask3) + warp * mask3

    show_debug("Warped face", warped)

    # --- Create background filled with skin color ---
    background = np.full_like(frame, fill_color, dtype=np.uint8)
    show_debug("Background fill", background)

    # --- Invert mask for blending (outside of face gets fill color) ---
    inv_mask = cv2.bitwise_not(face_mask)
    inv_mask3 = cv2.merge([inv_mask, inv_mask, inv_mask]) / 255.0
    face_mask3 = cv2.merge([face_mask, face_mask, face_mask]) / 255.0

    show_debug("Inverse mask3", inv_mask3)
    show_debug("face mask3", face_mask3)
    # Combine warped face + filled background
    result = warped * face_mask3 + background * inv_mask3
    black_pixels = np.all(result == [0, 0, 0], axis=-1)
    result[black_pixels] = fill_color

    show_debug("Result before blur", result)

    # Feather the edge slightly for smooth transition
    edge_mask = cv2.GaussianBlur(face_mask, (9, 9), 15)
    edge_mask3 = cv2.merge([edge_mask, edge_mask, edge_mask]) / 255.0
    result = result * edge_mask3 + frame * (1 - edge_mask3)

    show_debug("Result before return", result)
    return np.clip(result, 0, 255).astype(np.uint8)


# *********************************************************************************

modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
face_net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")

# open webcam
cap = cv2.VideoCapture(0)

landmarks = None
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    landmarks = get_landmarks(frame, smooth_landmarks=landmarks, face_net=face_net, facemark=facemark, alpha=0.3, count_points=False)

    frame = squish_features(frame, landmarks, strength=0.9 - frame_count * 0.003 if frame_count < 200 else 0.3)
    #frame = local_face_squish(frame, landmarks, strength=0.4)
    #frame = face_morph_squish(frame, landmarks, strength=0.5)

    cv2.imshow("Real-time Facial Landmarks (DNN + LBF)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()