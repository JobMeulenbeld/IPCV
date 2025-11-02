class Tracker:
    def __init__(self, alpha=0.2, max_missing_frames=10):
        self.alpha = alpha
        self.max_missing_frames = max_missing_frames

        # Tracked / smoothed entities
        self.tracked_bbox = None
        self.tracked_center = None
        self.tracked_fingertips = []

        self.missing_count = 0

    def update(self, bbox, center, fingertips):
        if bbox is None or center is None:
            self.missing_count += 1
            if self.missing_count > self.max_missing_frames:
                self.reset()
            return self.tracked_bbox, self.tracked_center, self.tracked_fingertips

        # Detection available, means that we can reset the missing counter
        self.missing_count = 0

        # smooth the bounding box
        if self.tracked_bbox is None:
            self.tracked_bbox = bbox
        else:
            self.tracked_bbox = self.ema_update_box(self.tracked_bbox, bbox)

        # Smooth the center
        if self.tracked_center is None:
            self.tracked_center = center
        else:
            self.tracked_center = self.ema_update_point(self.tracked_center, center)

        # Smooth the fingertips
        smoothed_fingertips = []
        if not fingertips:
            self.tracked_fingertips = smoothed_fingertips
        else:
            for i, current_tip in enumerate(fingertips):
                if i < len(self.tracked_fingertips):
                    smoothed_tip = self.ema_update_point(self.tracked_fingertips[i], current_tip)
                else:
                    smoothed_tip = current_tip
                smoothed_fingertips.append(smoothed_tip)
            self.tracked_fingertips = smoothed_fingertips

        return self.tracked_bbox, self.tracked_center, self.tracked_fingertips

    def ema_update_box(self, previous, current):
        return tuple(
            self.alpha * c + (1 - self.alpha) * p
            for p, c in zip(previous, current)
        )

    def ema_update_point(self, previous, current):
        return (
            self.alpha * current[0] + (1 - self.alpha) * previous[0],
            self.alpha * current[1] + (1 - self.alpha) * previous[1]
        )

    def reset(self):
        self.tracked_bbox = None
        self.tracked_center = None
        self.tracked_fingertips = []
        self.missing_count = 0
