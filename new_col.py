import cv2
import numpy as np
import time
from app.tasks.base_workers import BaseWorker
from app.utils.logger import log

class SmokeDetectorWorker(BaseWorker):
    """
    Classical smoke detector with color-independence and motion analysis.
    Detects smoke based on:
      - Continuous motion (via background subtraction)
      - Upward flow (via optical flow)
      - Texture softness (blurriness)
      - LAB color deviation (gray, brown, black, yellow, etc.)
    """

    def __init__(self, state):
        super().__init__(name="SmokeDetectorWorker", input_queue=state.roi_queue)
        self.state = state

        # Background subtractor: tuned for slow, continuous motion
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=16,
            detectShadows=True
        )

        # Store last gray frame per ROI for optical flow
        self.prev_gray_map = {}

        # Detection parameters (tunable)
        self.motion_threshold = 0.03      # % of pixels in motion
        self.blur_threshold = 150         # Laplacian variance (low = blurry)
        self.color_dev_threshold = 15     # deviation in LAB A/B channels
        self.upward_motion_ratio = 0.4    # ratio of upward motion pixels
        self.min_persistence = 3          # consecutive frames before confirming
        self.persistence_counter = {}

    # --------------------------------------------------------------------------
    def detect_smoke_in_roi(self, frame, roi_bbox, roi_key):
        x1, y1, x2, y2 = map(int, roi_bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False, None

        # 1️⃣ Background motion detection
        fgmask = self.bg_subtractor.apply(roi)
        if fgmask is None:
            log(f"[{self.name}] Warning: fgmask is None for ROI")
            return False, None
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        motion_ratio = np.sum(fgmask > 0) / (roi.shape[0] * roi.shape[1])

        # 2️⃣ Color deviation (LAB)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        color_var = cv2.addWeighted(cv2.absdiff(A, np.median(A)), 0.5,
                                    cv2.absdiff(B, np.median(B)), 0.5, 0)
        _, color_mask = cv2.threshold(color_var, self.color_dev_threshold, 255, cv2.THRESH_BINARY)

        # 3️⃣ Texture softness
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_mask = np.zeros_like(gray, dtype=np.uint8)
        if blur_score < self.blur_threshold:
            blur_mask[:] = 255  # Mark entire ROI as "blurry" (potential smoke)

        # 4️⃣ Ensure all masks are uint8
        fgmask = fgmask.astype(np.uint8)
        color_mask = color_mask.astype(np.uint8)
        blur_mask = blur_mask.astype(np.uint8)

        # 5️⃣ Combine all masks safely
        combined = cv2.bitwise_and(fgmask, color_mask)
        combined = cv2.bitwise_and(combined, blur_mask)

        # 5️⃣ Directional motion (optical flow)
        upward_ratio = 0.0
        if roi_key in self.prev_gray_map:
            prev_gray = self.prev_gray_map[roi_key]
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # Upward = ~90° (pi/2), mag > 1.0
            upward = (ang > np.pi / 3) & (ang < 2 * np.pi / 3) & (mag > 1.0)
            upward_ratio = np.sum(upward) / (roi.shape[0] * roi.shape[1])

        # Store current frame for next iteration
        self.prev_gray_map[roi_key] = gray

        # Combine decisions
        detected = (
            motion_ratio > self.motion_threshold and
            ratio > 0.02 and
            upward_ratio > self.upward_motion_ratio
        )

        return detected, {
            "motion_ratio": motion_ratio,
            "blur_score": blur_score,
            "upward_ratio": upward_ratio,
            "area_ratio": ratio
        }

    # --------------------------------------------------------------------------
    def process(self, item):
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
            log(f"[{self.name}] Missing frame/rois for {frame_id}")
            return

        smokes = []
        for idx, roi in enumerate(rois):
            bbox = roi.get("bbox")
            roi_key = f"{frame_id}-{idx}"
            if not bbox:
                continue

            detected, metrics = self.detect_smoke_in_roi(frame, bbox, roi_key)

            # Handle temporal persistence
            prev_count = self.persistence_counter.get(roi_key, 0)
            if detected:
                self.persistence_counter[roi_key] = prev_count + 1
            else:
                self.persistence_counter[roi_key] = max(0, prev_count - 1)

            persistent = self.persistence_counter[roi_key] >= self.min_persistence

            if persistent:
                smokes.append({
                    "bbox": bbox,
                    "source": roi.get("source", f"ROI-{idx}"),
                    "metrics": metrics
                })
                log(f"[{self.name}] Smoke confirmed near {roi.get('source', f'ROI-{idx}')}: {metrics}")

        # Update frame_store
        self.state.frame_store.update(frame_id, smokes=smokes)

        # Always push forward, even if no smokes
        self.state.smoke_queue.put({"frame_id": frame_id})
        log(f"[{self.name}] Frame {frame_id} processed — {len(smokes)} active smokes")
