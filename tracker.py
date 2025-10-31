class Tracker:
    def __init__(self, alpha=0.2, max_missing_frames=30):
        self.alpha = alpha
        self.max_missing_frames = max_missing_frames
        self.tracked_fingertips = []  # List of smoothed fingertip positions
        self.tracked_center = None  # Smoothed center position
        self.missing_count = 0  # How many frames no hand was detected

    def update(self, fingertips, center):
        # If no hand detected in this frame
        if not fingertips or center is None:
            self.missing_count += 1

            # If we've been missing too many frames, reset everything
            if self.missing_count > self.max_missing_frames:
                self.tracked_fingertips = []
                self.tracked_center = None
            return self.tracked_fingertips, self.tracked_center

        # Reset missing counter since we have a detection
        self.missing_count = 0

        # Smooth the center point
        if self.tracked_center is None:
            self.tracked_center = center
        else:
            self.tracked_center = self.ema_update(self.tracked_center, center)

        # Smooth the fingertips
        smoothed_fingertips = []

        for i, current_tip in enumerate(fingertips):
            if i < len(self.tracked_fingertips):
                # Update existing fingertip with EMA
                smoothed_tip = self.ema_update(self.tracked_fingertips[i], current_tip)
                smoothed_fingertips.append(smoothed_tip)
            else:
                # New fingertip - just use current position
                smoothed_fingertips.append(current_tip)

        # Update tracked fingertips
        self.tracked_fingertips = smoothed_fingertips

        return self.tracked_fingertips, self.tracked_center

    def ema_update(self, previous, current):
        return (
            int(self.alpha * current[0] + (1 - self.alpha) * previous[0]),
            int(self.alpha * current[1] + (1 - self.alpha) * previous[1])
        )