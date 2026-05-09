"""
modules/fusion_engine.py
========================
Module 3 — Fusion Decision Engine

Responsibility
--------------
Combine the outputs of Module 1 (AudioEvent list) and Module 2
(Dict[timestamp → VisualScore]) into a final list of CaptionEntry objects
that get written to the SRT file.

Decision logic (in order)
--------------------------
1.  Compute a weighted fusion score for every candidate audio event:
        fusion = AUDIO_WEIGHT × audio_confidence
               + VISUAL_WEIGHT × visual_reaction_score

2.  Apply a priority-aware threshold:
        HIGH   events (scream, explosion)  → threshold = 0.28  (lenient)
        MEDIUM events (laughter, animals)  → threshold = 0.40
        LOW    events (ambient noise)      → threshold = 0.60  (strict)

3.  Audio-only safety net:
        If Module 2 found no face in the video, the fusion formula becomes
        100 % audio-driven, and the threshold is lowered by 20 % to
        compensate for the missing visual signal.

4.  Temporal deduplication:
        If two accepted captions share the same category and their start
        times are within CAPTION_DEDUP_SEC of each other, keep only the
        higher-scoring one.

5.  SRT gap enforcement:
        Ensure consecutive captions never overlap on the timeline, and
        maintain at least SRT_MIN_GAP_SEC between any two entries.

6.  Frame annotation (optional):
        Save annotated JPEG frames to demo_results/frames/ at each accepted
        caption timestamp so the pipeline output is visually inspectable.

Design notes
------------
•   All magic numbers come from config.py — nothing is hard-coded here.
•   The engine is pure-Python (no ML model), so it's fast and deterministic.
•   VisualScore.reaction_score = 0.0 when no face was found; the engine
    treats this gracefully rather than crashing.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from modules.sound_detector  import AudioEvent
from modules.visual_detector import VisualScore
from utils.logger            import get_logger
from utils.srt_writer        import CaptionEntry

import config as cfg

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal decision record (richer than CaptionEntry — used for logging)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Decision:
    audio_event:   AudioEvent
    visual_score:  VisualScore
    fusion_score:  float
    threshold:     float
    accepted:      bool
    reject_reason: str   # "" when accepted


# ─────────────────────────────────────────────────────────────────────────────
# FusionEngine
# ─────────────────────────────────────────────────────────────────────────────

class FusionEngine:
    """
    Fuses Module 1 and Module 2 outputs into final CC subtitle entries.

    Parameters
    ----------
    audio_weight  : weight of audio confidence in the fusion formula
    visual_weight : weight of visual reaction score in the fusion formula
    output_dir    : directory for annotated frames and logs
    """

    def __init__(
        self,
        audio_weight:  float = cfg.FUSION_AUDIO_WEIGHT,
        visual_weight: float = cfg.FUSION_VISUAL_WEIGHT,
        output_dir:    str   = cfg.OUTPUT_DIR,
    ):
        if abs(audio_weight + visual_weight - 1.0) > 1e-6:
            raise ValueError(
                f"audio_weight ({audio_weight}) + visual_weight "
                f"({visual_weight}) must sum to 1.0"
            )
        self.audio_weight  = audio_weight
        self.visual_weight = visual_weight
        self.output_dir    = Path(output_dir)
        self.frames_dir    = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────

    def decide(
        self,
        audio_events:  List[AudioEvent],
        visual_scores: Dict[float, VisualScore],
        video_path:    Optional[str] = None,
    ) -> List[CaptionEntry]:
        """
        Run the full fusion pipeline.

        Parameters
        ----------
        audio_events  : sorted list from Module 1
        visual_scores : {timestamp: VisualScore} from Module 2
        video_path    : optional video path for frame annotation

        Returns
        -------
        List of CaptionEntry objects ready for SRT output, sorted by
        start_sec.
        """
        if not audio_events:
            log.warning("[M3] No audio events received — nothing to decide.")
            return []

        log.info("[M3] Evaluating %d candidate audio events …",
                 len(audio_events))

        # Detect whether any window actually found a face → drives audio-only mode
        has_visual = self._any_face_detected(visual_scores)
        if not has_visual:
            log.warning(
                "[M3] No faces detected in any visual window. "
                "Switching to audio-only fusion mode (thresholds lowered 20 %%)."
            )

        # ── Step 1: Score every event ────────────────────────────────────────
        decisions = self._score_all_events(
            audio_events, visual_scores, has_visual
        )

        # ── Step 2: Log the decision table ──────────────────────────────────
        self._log_decisions(decisions)

        # ── Step 3: Accept / reject ──────────────────────────────────────────
        accepted = [d for d in decisions if d.accepted]
        log.info(
            "[M3] %d / %d events accepted for captioning.",
            len(accepted), len(decisions),
        )

        # ── Step 4: Temporal deduplication ──────────────────────────────────
        deduped = self._deduplicate(accepted)
        if len(deduped) < len(accepted):
            log.info("[M3] Deduplication removed %d duplicate(s).",
                     len(accepted) - len(deduped))

        # ── Step 5: Build CaptionEntry list ─────────────────────────────────
        entries = self._build_caption_entries(deduped)

        # ── Step 6: Enforce SRT timeline gaps ───────────────────────────────
        entries = self._enforce_srt_gaps(entries)

        # ── Step 7: Optional frame annotation ───────────────────────────────
        if video_path and Path(video_path).exists():
            self._annotate_frames(entries, video_path)

        log.info("[M3] Fusion complete — %d caption entries ready.", len(entries))
        return entries

    # ── Step 1: Score all events ─────────────────────────────────────────────

    def _score_all_events(
        self,
        audio_events:  List[AudioEvent],
        visual_scores: Dict[float, VisualScore],
        has_visual:    bool,
    ) -> List[_Decision]:
        """
        Compute fusion score and threshold, then accept/reject each event.
        """
        decisions: List[_Decision] = []

        for event in audio_events:
            vscore = self._lookup_visual(event.timestamp_sec, visual_scores)
            fusion, threshold = self._compute_fusion(event, vscore, has_visual)
            accepted  = fusion >= threshold
            reject_reason = "" if accepted else (
                f"fusion {fusion:.3f} < threshold {threshold:.3f}"
            )
            decisions.append(_Decision(
                audio_event   = event,
                visual_score  = vscore,
                fusion_score  = fusion,
                threshold     = threshold,
                accepted      = accepted,
                reject_reason = reject_reason,
            ))

        return decisions

    def _lookup_visual(
        self,
        timestamp:     float,
        visual_scores: Dict[float, VisualScore],
    ) -> VisualScore:
        """
        Find the VisualScore for *timestamp*.
        Falls back to a zero-score placeholder if the key is missing.
        (Handles floating-point near-matches within 0.05 s.)
        """
        # Exact match first
        if timestamp in visual_scores:
            return visual_scores[timestamp]

        # Near match (floating-point drift tolerance)
        for ts, vs in visual_scores.items():
            if abs(ts - timestamp) <= 0.05:
                return vs

        # Nothing found — construct a null VisualScore
        return VisualScore(
            query_time_sec   = timestamp,
            reaction_score   = 0.0,
            num_valid_frames = 0,
            peak_frame_time  = None,
            confidence       = "low",
            note             = "not_queried",
        )

    def _compute_fusion(
        self,
        event:      AudioEvent,
        vscore:     VisualScore,
        has_visual: bool,
    ) -> Tuple[float, float]:
        """
        Returns (fusion_score, threshold).

        Audio-only mode:
            If no face was found anywhere, visual weight collapses to 0
            and the threshold is reduced by 20 % to compensate.

        Category-aware threshold:
            HIGH   → 0.28   (screams / explosions must not be missed)
            MEDIUM → 0.40
            LOW    → 0.60
        """
        priority  = event.priority
        threshold = cfg.FUSION_THRESHOLD.get(priority, 0.45)

        if not has_visual:
            # Pure audio mode
            fusion    = float(event.confidence)
            threshold = threshold * 0.80
        else:
            fusion = (
                self.audio_weight  * event.confidence +
                self.visual_weight * vscore.reaction_score
            )

        fusion = round(min(1.0, max(0.0, fusion)), 4)
        return fusion, round(threshold, 4)

    # ── Step 2: Logging decision table ──────────────────────────────────────

    def _log_decisions(self, decisions: List[_Decision]) -> None:
        """Pretty-print each decision to the logger."""
        header = (
            f"{'Time':>7}  {'Category':<16} {'Pri':<7} "
            f"{'Audio':>6} {'Visual':>7} {'Fusion':>7} {'Thresh':>7}  {'Decision'}"
        )
        log.info("[M3] Decision table:\n         %s", header)

        for d in decisions:
            ev = d.audio_event
            vs = d.visual_score
            verdict = "✓ EMIT" if d.accepted else f"✗ SKIP ({d.reject_reason})"
            row = (
                f"{ev.timestamp_sec:>6.2f}s  "
                f"{ev.category:<16} "
                f"{ev.priority:<7} "
                f"{ev.confidence:>6.3f} "
                f"{vs.reaction_score:>7.3f} "
                f"{d.fusion_score:>7.3f} "
                f"{d.threshold:>7.3f}  "
                f"{verdict}"
            )
            if d.accepted:
                log.info("         %s", row)
            else:
                log.warning("         %s", row)

    # ── Step 4: Temporal deduplication ──────────────────────────────────────

    def _deduplicate(self, accepted: List[_Decision]) -> List[_Decision]:
        """
        Within each sound category, suppress events that start within
        CAPTION_DEDUP_SEC of a higher-scoring event of the same category.
        """
        by_category: Dict[str, List[_Decision]] = defaultdict(list)
        for d in accepted:
            by_category[d.audio_event.category].append(d)

        kept: List[_Decision] = []

        for cat_decisions in by_category.values():
            # Sort by timestamp
            cat_decisions.sort(key=lambda d: d.audio_event.timestamp_sec)
            survivors: List[_Decision] = []

            for current in cat_decisions:
                suppress = False
                for survivor in survivors:
                    gap = abs(
                        current.audio_event.timestamp_sec -
                        survivor.audio_event.timestamp_sec
                    )
                    if gap < cfg.CAPTION_DEDUP_SEC:
                        # Keep the higher-scoring one
                        if current.fusion_score > survivor.fusion_score:
                            survivors.remove(survivor)
                            # Will be added below
                        else:
                            suppress = True
                        break

                if not suppress:
                    survivors.append(current)

            kept.extend(survivors)

        # Re-sort by timestamp
        kept.sort(key=lambda d: d.audio_event.timestamp_sec)
        return kept

    # ── Step 5: Build CaptionEntry list ─────────────────────────────────────

    def _build_caption_entries(
        self,
        decisions: List[_Decision],
    ) -> List[CaptionEntry]:
        """
        Convert accepted _Decision objects into CaptionEntry objects,
        computing display duration from priority tier.
        """
        entries: List[CaptionEntry] = []

        for idx, d in enumerate(decisions, start=1):
            ev       = d.audio_event
            priority = ev.priority
            duration = cfg.SRT_DISPLAY_DURATION.get(priority, 2.0)

            start_sec = ev.timestamp_sec
            # Prefer the event's own end_sec if it gives a sensible duration
            natural_dur = ev.end_sec - ev.timestamp_sec
            if 0.3 <= natural_dur <= 5.0:
                end_sec = ev.timestamp_sec + max(natural_dur, duration)
            else:
                end_sec = start_sec + duration

            entries.append(CaptionEntry(
                index        = idx,
                start_sec    = round(start_sec, 3),
                end_sec      = round(end_sec,   3),
                caption_text = ev.display_label,
                category     = ev.category,
                priority     = priority,
                audio_score  = ev.confidence,
                visual_score = d.visual_score.reaction_score,
                fusion_score = d.fusion_score,
            ))

        return entries

    # ── Step 6: SRT gap enforcement ──────────────────────────────────────────

    def _enforce_srt_gaps(
        self,
        entries: List[CaptionEntry],
    ) -> List[CaptionEntry]:
        """
        Ensure no two captions overlap and that there is at least
        SRT_MIN_GAP_SEC between the end of one and the start of the next.
        Adjusts end_sec of the earlier caption where needed.
        """
        if len(entries) < 2:
            return entries

        for i in range(len(entries) - 1):
            curr = entries[i]
            nxt  = entries[i + 1]
            required_end = nxt.start_sec - cfg.SRT_MIN_GAP_SEC
            if curr.end_sec > required_end:
                curr.end_sec = max(
                    curr.start_sec + 0.2,   # keep at least 0.2 s visible
                    required_end,
                )
                curr.end_sec = round(curr.end_sec, 3)

        return entries

    # ── Step 7: Frame annotation ─────────────────────────────────────────────

    def _annotate_frames(
        self,
        entries:    List[CaptionEntry],
        video_path: str,
    ) -> None:
        """
        For each accepted caption, grab the nearest video frame and save it
        as a JPEG with the caption text burned in.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log.warning("[M3] Cannot open video for frame annotation: %s",
                        video_path)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for entry in entries:
            frame_no = min(
                int(entry.start_sec * fps),
                total_frames - 1,
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            self._draw_annotation(frame, entry, frame_no)

            safe_cat = entry.category.lower().replace("/", "_")
            filename = (
                f"frame_{frame_no:05d}_"
                f"t{entry.start_sec:.2f}s_"
                f"{safe_cat}.jpg"
            )
            out_path = self.frames_dir / filename
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            log.debug("[M3] Annotated frame saved → %s", out_path)

        cap.release()
        log.info("[M3] %d annotated frames saved to %s",
                 len(entries), self.frames_dir)

    @staticmethod
    def _draw_annotation(
        frame:   np.ndarray,
        entry:   CaptionEntry,
        frame_no: int,
    ) -> None:
        """
        Burn caption text, scores, and a timestamp into a video frame
        in-place (modifies the numpy array directly).
        """
        h, w = frame.shape[:2]

        # ── Subtitle bar at bottom ───────────────────────────────────────────
        bar_h  = max(52, h // 10)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.6, w / 900)
        thickness  = max(1, int(font_scale * 1.5))

        # Caption text centred
        text = entry.caption_text
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        tx = max(8, (w - tw) // 2)
        ty = h - bar_h // 2 + th // 2
        cv2.putText(frame, text, (tx, ty), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

        # ── Top-left info badge ──────────────────────────────────────────────
        PRIORITY_COLOURS = {
            "HIGH":   (0,   80, 220),   # red-ish
            "MEDIUM": (30, 160,  30),   # green
            "LOW":    (180, 100,  0),   # blue-ish
        }
        badge_colour = PRIORITY_COLOURS.get(entry.priority, (100, 100, 100))
        badge_lines = [
            f"t={entry.start_sec:.2f}s  frm#{frame_no}",
            f"audio={entry.audio_score:.3f}  visual={entry.visual_score:.3f}",
            f"fusion={entry.fusion_score:.3f}  [{entry.priority}]",
        ]
        small_scale = max(0.35, w / 1800)
        small_thick = 1
        for i, line in enumerate(badge_lines):
            y = 20 + i * 18
            cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        small_scale, badge_colour, small_thick, cv2.LINE_AA)

    # ── Utility ──────────────────────────────────────────────────────────────

    @staticmethod
    def _any_face_detected(visual_scores: Dict[float, VisualScore]) -> bool:
        """Return True if at least one visual window found a face."""
        return any(
            vs.num_valid_frames > 0
            for vs in visual_scores.values()
        )

    # ── Reporting helper ─────────────────────────────────────────────────────

    def summary(self, entries: List[CaptionEntry]) -> str:
        """
        Return a human-readable summary string of the caption output.
        Useful for the final console banner in main.py.
        """
        if not entries:
            return "No captions generated."

        lines = ["Caption Summary:", "─" * 54]
        for e in entries:
            lines.append(
                f"  {e.start_sec:6.2f}s → {e.end_sec:6.2f}s  "
                f"{e.caption_text:<28}  "
                f"(fusion={e.fusion_score:.3f})"
            )
        lines.append("─" * 54)
        lines.append(f"  Total: {len(entries)} caption(s)")
        return "\n".join(lines)