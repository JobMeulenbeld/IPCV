import cv2
import numpy as np

class face_augmentation:
    def __init__(self, image, scale_factor, x_offset, y_offset, smoothing_factor=0.8):
        """Initialize the face augmentation with the overlay image and parameters."""
        self.image = image
        self.scale_factor = scale_factor
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.smoothing_factor = smoothing_factor

        self.prev_top_left = None
        self.prev_size = None

    def overlay_transparent(self, frame, overlay, x, y):
        """
        Overlay RGBA `overlay` image onto BGR `frame` at position (x, y).
        Handles alpha blending and clipping at image borders.
        """

        h, w = frame.shape[:2]
        h_o, w_o = overlay.shape[:2]

        # Clip overlay to stay within the frame
        if x >= w or y >= h:
                return frame
            
        # If overlay is larger than frame return the frame
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
    
    def smoothing(self, top_left, size):
        """Smooth the position and size of the overlay using exponential moving average."""

        # Initialize previous values if they don't exist
        if self.prev_top_left is None:
            self.prev_top_left = top_left
            self.prev_size = size

        # Apply exponential moving average
        alpha = 1.0 - self.smoothing_factor
        self.prev_top_left = self.smoothing_factor * self.prev_top_left + alpha * top_left
        self.prev_size = self.smoothing_factor * self.prev_size + alpha * size

        #convert to integer for pixel coordinates
        smoothed_top_left = self.prev_top_left.astype(int)
        smoothed_size = self.prev_size.astype(int)

        return smoothed_top_left, smoothed_size

    def face_overlay(self, frame, landmark1, landmark2):
        """Overlay the image onto the frame based on two facial landmarks."""
        height, width, channels = self.image.shape

        x1, y1 = landmark1
        x2, y2 = landmark2

        #scale the overlay image based on distance between landmarks
        dx = int(abs(x2 - x1))
        image_width_resized = int(dx * self.scale_factor)

        #determine resized height to maintain aspect ratio
        ratio = image_width_resized / width
        image_height_resized = int(height * ratio)

        #determine the offstets based on the resized image
        image_x_offset = (image_width_resized-dx)/2 + (self.x_offset * ratio)
        image_y_offset = (self.y_offset * ratio)

        #determine the top-left position and size of the overlay
        top_left = np.array([x1 - image_x_offset, y1 - image_y_offset], dtype=np.float32)
        size = np.array([image_width_resized, image_height_resized], dtype=np.float32)

        # Apply smoothing
        smoothed_top_left, smoothed_size = self.smoothing(top_left, size)

        # Resize the overlay image
        image_resized = cv2.resize(self.image, tuple(smoothed_size))

        # Handle cropping if the overlay goes above the frame
        if smoothed_top_left[1] < 0:
            crop_top = int(-smoothed_top_left[1])
            image_resized = image_resized[crop_top:, :, :]  # remove the top rows
            smoothed_top_left[1] = 0

        print(smoothed_top_left)

        # Overlay the image onto the frame
        frame = self.overlay_transparent(frame, image_resized, int(smoothed_top_left[0]), int(smoothed_top_left[1]))
        return frame