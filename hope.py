import cv2
import numpy as np
from app.tasks.base_workers import BaseWorker
from app.utils.logger import log

class SmokeDetectorWorker(BaseWorker):
    """
    Robust classical smoke detector (color-independent, motion + blur + direction).
    """

    def __init__(self, state):
        super().__init__(name="SmokeDetectorWorker", input_queue=state.roi_queue)
        self.state = state

        # Background subtractor for smooth motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=16, detectShadows=True
        )

        self.prev_gray_map = {}

        # Detection parameters
        self.motion_threshold = 0.03      # 3% pixels in motion
        self.blur_threshold = 150         # lower = blurry
        self.color_dev_threshold = 15     # LAB A/B deviation
        self.upward_motion_ratio = 0.4    # % of upward motion
        self.min_persistence = 3          # consecutive frames
        self.persistence_counter = {}

    # --------------------------------------------------------------------------
    def detect_smoke_in_roi(self, frame, roi_bbox, roi_key):
        """
        Safely detect smoke inside a single ROI using:
        - background subtraction
        - LAB color deviation
        - texture softness
        - optical flow (upward motion)
        """
        try:
            x1, y1, x2, y2 = map(int, roi_bbox)
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                log(f"[{self.name}] Skipping tiny/empty ROI {roi_key}")
                return False, None

            # 1️⃣ Motion mask
            fgmask = self.bg_subtractor.apply(roi)
            if fgmask is None or fgmask.size == 0:
                log(f"[{self.name}] fgmask invalid for {roi_key}")
                return False, None

            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            motion_ratio = np.sum(fgmask > 0) / (roi.shape[0] * roi.shape[1] + 1e-6)

            # 2️⃣ LAB color deviation
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)

            # Safety for np.median → always uint8
            med_A = np.median(A).astype(np.uint8) if hasattr(np.median(A), "astype") else int(np.median(A))
            med_B = np.median(B).astype(np.uint8) if hasattr(np.median(B), "astype") else int(np.median(B))

            # Absolute differences (safe version)
            try:
                diffA = cv2.absdiff(A, np.full_like(A, med_A))
                diffB = cv2.absdiff(B, np.full_like(B, med_B))
                color_var = cv2.addWeighted(diffA, 0.5, diffB, 0.5, 0)
            except cv2.error as e:
                log(f"[{self.name}] absdiff error for {roi_key}: {e}")
                color_var = np.zeros_like(A)

            _, color_mask = cv2.threshold(color_var, self.color_dev_threshold, 255, cv2.THRESH_BINARY)

            # 3️⃣ Blurriness (texture softness)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_mask = np.zeros_like(gray, dtype=np.uint8)
            if blur_score < self.blur_threshold:
                blur_mask[:] = 255

            # 4️⃣ Ensure all masks are valid
            for mask_name, mask in zip(["fgmask", "color_mask", "blur_mask"], [fgmask, color_mask, blur_mask]):
                if mask is None or mask.shape[:2] != gray.shape:
                    log(f"[{self.name}] Mask shape mismatch in {mask_name} for {roi_key}")
                    return False, None

            fgmask = fgmask.astype(np.uint8)
            color_mask = color_mask.astype(np.uint8)
            blur_mask = blur_mask.astype(np.uint8)

            # 5️⃣ Combine all masks
            try:
                combined = cv2.bitwise_and(fgmask, color_mask)
                combined = cv2.bitwise_and(combined, blur_mask)
            except cv2.error as e:
                log(f"[{self.name}] bitwise_and error for {roi_key}: {e}")
                return False, None

            ratio = np.sum(combined > 0) / (roi.shape[0] * roi.shape[1] + 1e-6)

            # 6️⃣ Optical flow (motion direction)
            upward_ratio = 0.0
            if roi_key in self.prev_gray_map:
                prev_gray = self.prev_gray_map[roi_key]
                if prev_gray.shape == gray.shape:
                    try:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        upward = (ang > np.pi / 3) & (ang < 2 * np.pi / 3) & (mag > 1.0)
                        upward_ratio = np.sum(upward) / (roi.shape[0] * roi.shape[1] + 1e-6)
                    except cv2.error as e:
                        log(f"[{self.name}] Optical flow error for {roi_key}: {e}")
            self.prev_gray_map[roi_key] = gray

            # 7️⃣ Decision logic
            detected = (
                motion_ratio > self.motion_threshold
                and ratio > 0.02
                and upward_ratio > self.upward_motion_ratio
            )

            log(
                f"[{self.name}] ROI {roi_key}: motion={motion_ratio:.3f}, ratio={ratio:.3f}, "
                f"upward={upward_ratio:.3f}, blur={blur_score:.1f}, detected={detected}"
            )

            return detected, {
                "motion_ratio": float(motion_ratio),
                "blur_score": float(blur_score),
                "upward_ratio": float(upward_ratio),
                "area_ratio": float(ratio),
            }

        except Exception as e:
            log(f"[{self.name}] Unexpected error in detect_smoke_in_roi({roi_key}): {e}")
            return False, None

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
                log(f"[{self.name}] ✅ Smoke confirmed near {roi.get('source', f'ROI-{idx}')}")

        # Save results and pass forward
        self.state.frame_store.update(frame_id, smokes=smokes)
        self.state.smoke_queue.put({"frame_id": frame_id})
        log(f"[{self.name}] Frame {frame_id} processed — {len(smokes)} smokes detected")


# which one?

[SmokeDetectorWorker] ROI 4f6a073f-b08a-4608-a72e-7abd8346b729-0: motion=1.000, ratio=0.000, upward=0.000, blur=166.3, detected=False
[SmokeDetectorWorker] ROI 4f6a073f-b08a-4608-a72e-7abd8346b729-1: motion=1.000, ratio=0.000, upward=0.000, blur=516.7, detected=False
[SmokeDetectorWorker] ROI 4f6a073f-b08a-4608-a72e-7abd8346b729-2: motion=1.000, ratio=0.000, upward=0.000, blur=108.0, detected=False
[SmokeDetectorWorker] ROI 4f6a073f-b08a-4608-a72e-7abd8346b729-3: motion=1.000, ratio=0.000, upward=0.000, blur=726.6, detected=False
[SmokeDetectorWorker] ROI 4f6a073f-b08a-4608-a72e-7abd8346b729-4: motion=1.000, ratio=0.000, upward=0.000, blur=106.8, detected=False
[SmokeDetectorWorker] ROI 4f6a073f-b08a-4608-a72e-7abd8346b729-5: motion=1.000, ratio=0.000, upward=0.000, blur=1482.3, detected=False
