from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from utils.logger import get_logger
import config as cfg

log = get_logger(__name__)

@dataclass
class FaceFrameScore:
    """Facial action scores for a single video frame."""
    frame_no:       int
    time_sec:       float
    ear_score:      float      # normalised Eye Aspect Ratio delta
    mar_score:      float      # normalised Mouth Aspect Ratio delta
    brow_score:     float      # normalised Brow Raise
    composite:      float      # weighted combination of the above
    face_detected:  bool
    num_faces:      int


@dataclass
class VisualScore:
    """Aggregated visual reaction score for one audio event timestamp."""
    query_time_sec:     float
    reaction_score:     float           # [0, 1]
    num_valid_frames:   int
    peak_frame_time:    Optional[float]
    frame_scores:       List[FaceFrameScore] = field(default_factory=list)
    confidence:         str = "low"     # low | medium | high
    note:               str = ""

class VisualReactionDetector:
    def __init__(
        self,
        min_detection_confidence: float = cfg.MEDIAPIPE_DETECTION_CONFIDENCE,
        min_tracking_confidence:  float = cfg.MEDIAPIPE_TRACKING_CONFIDENCE,
    ):
        self._det_conf  = min_detection_confidence
        self._trk_conf  = min_tracking_confidence
        self._face_mesh = None   # lazy-loaded per-video

    # ── Public API ──────────────────────────────────────────────────────────

    def analyse(
        self,
        video_path:  str,
        timestamps:  List[float],
    ) -> Dict[float, VisualScore]:
        log.info("[M2] Opening video: %s", video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log.error("[M2] Cannot open video: %s", video_path)
            return {t: self._null_score(t, "video_open_failed") for t in timestamps}

        fps           = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec  = total_frames / fps

        log.info("[M2] Video: fps=%.1f  frames=%d  duration=%.2f s",
                 fps, total_frames, duration_sec)

        # Load MediaPipe once for the whole video
        self._load_face_mesh()

        results: Dict[float, VisualScore] = {}

        for ts in timestamps:
            log.debug("[M2] Processing window around t=%.2f s", ts)
            score = self._score_window(cap, ts, fps, total_frames, duration_sec)
            results[ts] = score
            log.debug(
                "[M2]   ts=%.2f  reaction=%.3f  valid_frames=%d  conf=%s",
                ts, score.reaction_score, score.num_valid_frames, score.confidence,
            )

        cap.release()
        self._release_face_mesh()

        log.info("[M2] Visual analysis complete — %d timestamps processed",
                 len(results))
        return results

    def _score_window(
        self,
        cap:           cv2.VideoCapture,
        query_time:    float,
        fps:           float,
        total_frames:  int,
        duration_sec:  float,
    ) -> VisualScore:
        """
        Sample frames in [query_time - BEFORE, query_time + AFTER],
        score each, and aggregate.
        """
        t_start = max(0.0, query_time - cfg.VISUAL_WINDOW_BEFORE_SEC)
        t_end   = min(duration_sec,  query_time + cfg.VISUAL_WINDOW_AFTER_SEC)

        # Build evenly-spaced sample timestamps
        n        = cfg.VISUAL_MAX_FRAMES_PER_WINDOW
        sample_t = np.linspace(t_start, t_end, n)

        frame_scores: List[FaceFrameScore] = []
        for t in sample_t:
            fn = int(t * fps)
            fn = max(0, min(fn, total_frames - 1))
            frame = self._seek_frame(cap, fn)
            if frame is None:
                continue
            fs = self._score_frame(frame, fn, t)
            frame_scores.append(fs)

        # Aggregate
        return self._aggregate(query_time, frame_scores)

    @staticmethod
    def _seek_frame(
        cap: cv2.VideoCapture, frame_no: int
    ) -> Optional[np.ndarray]:
        """
        Seek to *frame_no* and return the frame as a numpy BGR array,
        or None on failure.
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        return frame if ret else None

    def _score_frame(
        self,
        frame:    np.ndarray,
        frame_no: int,
        time_sec: float,
    ) -> FaceFrameScore:

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return FaceFrameScore(
                frame_no=frame_no, time_sec=time_sec,
                ear_score=0.0, mar_score=0.0, brow_score=0.0,
                composite=0.0, face_detected=False, num_faces=0,
            )

        # Evaluate all detected faces, take the highest-reacting one
        best_composite = -1.0
        best_ear = best_mar = best_brow = 0.0

        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark

            def pt(idx):
                return np.array([lm[idx].x * w, lm[idx].y * h])

            ear   = self._compute_ear(pt)
            mar   = self._compute_mar(pt)
            brow  = self._compute_brow_raise(pt, h)

            ear_norm  = max(0.0, (ear  - cfg.EAR_BASELINE)  / (1.0 - cfg.EAR_BASELINE))
            mar_norm  = max(0.0, (mar  - cfg.MAR_BASELINE)   / (1.0 - cfg.MAR_BASELINE))
            brow_norm = max(0.0, brow / 0.10)   # 0.10 is practical max raise

            # Clamp each to [0, 1]
            ear_norm  = min(ear_norm,  1.0)
            mar_norm  = min(mar_norm,  1.0)
            brow_norm = min(brow_norm, 1.0)

            composite = (
                cfg.VISUAL_WEIGHT_EAR  * ear_norm +
                cfg.VISUAL_WEIGHT_MAR  * mar_norm +
                cfg.VISUAL_WEIGHT_BROW * brow_norm
            )

            if composite > best_composite:
                best_composite = composite
                best_ear  = ear_norm
                best_mar  = mar_norm
                best_brow = brow_norm

        return FaceFrameScore(
            frame_no     = frame_no,
            time_sec     = time_sec,
            ear_score    = round(best_ear,  4),
            mar_score    = round(best_mar,  4),
            brow_score   = round(best_brow, 4),
            composite    = round(best_composite, 4),
            face_detected = True,
            num_faces    = len(results.multi_face_landmarks),
        )

    @staticmethod
    def _compute_ear(pt) -> float:
        """
        Eye Aspect Ratio (EAR) — averaged across both eyes.
        EAR = (vertical_distance) / (2 * horizontal_distance)
        A wide-open eye has EAR ≈ 0.35; closed eye ≈ 0.
        """
        ears = []
        for side in ("left", "right"):
            idx = cfg.EYE_LANDMARKS[side]
            v   = np.linalg.norm(pt(idx["top"]) - pt(idx["bottom"]))
            h   = np.linalg.norm(pt(idx["inner"]) - pt(idx["outer"]))
            if h > 1e-6:
                ears.append(v / h)

        return float(np.mean(ears)) if ears else cfg.EAR_BASELINE

    @staticmethod
    def _compute_mar(pt) -> float:
        """
        Mouth Aspect Ratio (MAR).
        Larger values → more open mouth.
        """
        idx = cfg.MOUTH_LANDMARKS
        v1 = np.linalg.norm(pt(idx["top"])  - pt(idx["bottom"]))
        v2 = np.linalg.norm(pt(idx["top2"]) - pt(idx["bottom2"]))
        h  = np.linalg.norm(pt(idx["left"]) - pt(idx["right"]))
        if h < 1e-6:
            return cfg.MAR_BASELINE
        return float((v1 + v2) / (2.0 * h))

    @staticmethod
    def _compute_brow_raise(pt, face_height: int) -> float:
        """
        Brow raise: average distance from brow landmarks to eye-top,
        normalised by face height.  Higher → more raised brows.
        """
        def _side_raise(brow_idxs, eye_top_idx):
            brow_y = np.mean([pt(i)[1] for i in brow_idxs])
            eye_y  = pt(eye_top_idx)[1]
            # brow is above eye → brow_y < eye_y in image coords
            return max(0.0, float(eye_y - brow_y)) / max(face_height, 1)

        left_raise  = _side_raise(
            cfg.BROW_LANDMARKS["left_brow"],
            cfg.BROW_LANDMARKS["left_eye_top"],
        )
        right_raise = _side_raise(
            cfg.BROW_LANDMARKS["right_brow"],
            cfg.BROW_LANDMARKS["right_eye_top"],
        )
        return (left_raise + right_raise) / 2.0

    def _aggregate(
        self,
        query_time:   float,
        frame_scores: List[FaceFrameScore],
    ) -> VisualScore:
        valid = [f for f in frame_scores if f.face_detected]

        if len(valid) < cfg.VISUAL_MIN_VALID_FRAMES:
            note = (
                "too_few_valid_frames"
                if valid else "no_face_detected"
            )
            return VisualScore(
                query_time_sec   = query_time,
                reaction_score   = 0.0,
                num_valid_frames = len(valid),
                peak_frame_time  = None,
                frame_scores     = frame_scores,
                confidence       = "low",
                note             = note,
            )

        # Temporal weights
        weights = np.array([
            math.exp(-abs(f.time_sec - query_time) / 0.5)
            for f in valid
        ])
        composites = np.array([f.composite for f in valid])
        w_sum = weights.sum()
        if w_sum < 1e-9:
            reaction = float(np.mean(composites))
        else:
            reaction = float(np.dot(weights, composites) / w_sum)

        reaction = min(1.0, max(0.0, reaction))

        peak = valid[int(np.argmax(composites))]
        conf = (
            "high"   if len(valid) >= 5 else
            "medium" if len(valid) >= 3 else
            "low"
        )

        return VisualScore(
            query_time_sec   = query_time,
            reaction_score   = round(reaction, 4),
            num_valid_frames = len(valid),
            peak_frame_time  = peak.time_sec,
            frame_scores     = frame_scores,
            confidence       = conf,
            note             = "ok",
        )

    def _load_face_mesh(self) -> None:
        if self._face_mesh is not None:
            return
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode        = False,    # video mode → faster tracking
                max_num_faces            = 4,        # handle group reactions
                refine_landmarks         = True,     # iris landmarks for better EAR
                min_detection_confidence = self._det_conf,
                min_tracking_confidence  = self._trk_conf,
            )
            log.info("[M2] MediaPipe Face Mesh loaded")
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is required for Module 2.  "
                "Run: pip install mediapipe"
            ) from exc

    def _release_face_mesh(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    @staticmethod
    def _null_score(ts: float, note: str) -> VisualScore:
        return VisualScore(
            query_time_sec   = ts,
            reaction_score   = 0.0,
            num_valid_frames = 0,
            peak_frame_time  = None,
            confidence       = "low",
            note             = note,
        )