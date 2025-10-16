import cv2
import numpy as np
import time
from app.tasks.base_workers import BaseWorker
from app.utils.logger import log

class SmokeDetectorWorker(BaseWorker):
    """
    Classical smoke detector (non-YOLO).
    Uses background subtraction + color filtering + motion ratio.
    """

    def __init__(self, state):
        super().__init__(name="SmokeDetectorWorker", input_queue=state.roi_queue)
        self.state = state

        # Background subtractor tuned for soft, slow smoke motion
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=25,
            detectShadows=True
        )

        # Parameters
        self.motion_threshold = self.state.config.get("smoke_motion_threshold", 0.03)
        self.min_area_ratio = self.state.config.get("smoke_min_area_ratio", 0.02)
        self.max_area_ratio = self.state.config.get("smoke_max_area_ratio", 0.5)

    # -------------------------------------------------------------
    def detect_smoke_in_roi(self, frame, roi_bbox):
        """Hybrid detection using background subtraction + color segmentation."""

        x1, y1, x2, y2 = map(int, roi_bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        # 1️⃣ Background subtraction
        fgmask = self.bg_subtractor.apply(roi)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # 2️⃣ Color mask for grayish/whitish smoke
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 70])
        upper = np.array([180, 50, 255])
        color_mask = cv2.inRange(hsv, lower, upper)

        # 3️⃣ Combine motion + color
        combined = cv2.bitwise_and(fgmask, color_mask)

        # 4️⃣ Compute ratio of motion pixels to ROI area
        ratio = np.sum(combined > 0) / (roi.shape[0] * roi.shape[1])

        # 5️⃣ Apply thresholds
        return self.min_area_ratio < ratio < self.max_area_ratio

    # -------------------------------------------------------------
    def process(self, item):
        """Consumes RoI frames and detects smoke per ROI."""

        frame_id = getattr(item, "frame_id", None) or item.get("frame_id")
        if not frame_id:
            return

        frame_data = self.state.frame_store.get(frame_id)
        if not frame_data:
            log(f"[{self.name}] Frame {frame_id} not found in frame_store")
            return

        frame = frame_data.get("frame")
        rois = frame_data.get("rois", [])

        if frame is None or not rois:
            log(f"[{self.name}] Missing frame or RoIs for {frame_id}")
            return

        smokes = []
        for idx, roi in enumerate(rois):
            bbox = roi.get("bbox")
            if not bbox:
                continue

            try:
                detected = self.detect_smoke_in_roi(frame, bbox)
            except Exception as e:
                log(f"[{self.name}] Smoke detection failed on ROI {idx}: {e}")
                continue

            if detected:
                smokes.append({
                    "bbox": bbox,
                    "source": roi.get("source", f"ROI-{idx}")
                })
                log(f"[{self.name}] Smoke detected for {roi.get('source', f'ROI-{idx}')}")
            else:
                log(f"[{self.name}] No smoke for {roi.get('source', f'ROI-{idx}')}")

        # Update frame store (even if no smokes → store empty list)
        self.state.frame_store.update(frame_id, smokes=smokes)

        # Push forward to smoke_queue for color detection
        self.state.smoke_queue.put({"frame_id": frame_id})
        log(f"[{self.name}] Processed frame {frame_id} with {len(smokes)} smokes")
