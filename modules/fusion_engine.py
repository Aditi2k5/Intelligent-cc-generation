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

@dataclass
class _Decision:
    audio_event:   AudioEvent
    visual_score:  VisualScore
    fusion_score:  float
    threshold:     float
    accepted:      bool
    reject_reason: str   

class FusionEngine:

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


    def decide(
        self,
        audio_events:  List[AudioEvent],
        visual_scores: Dict[float, VisualScore],
        video_path:    Optional[str] = None,
    ) -> List[CaptionEntry]:
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

        decisions = self._score_all_events(
            audio_events, visual_scores, has_visual
        )

        self._log_decisions(decisions)

        accepted = [d for d in decisions if d.accepted]
        log.info(
            "[M3] %d / %d events accepted for captioning.",
            len(accepted), len(decisions),
        )

        deduped = self._deduplicate(accepted)
        if len(deduped) < len(accepted):
            log.info("[M3] Deduplication removed %d duplicate(s).",
                     len(accepted) - len(deduped))

        entries = self._build_caption_entries(deduped)

        entries = self._enforce_srt_gaps(entries)

        if video_path and Path(video_path).exists():
            self._annotate_frames(entries, video_path)

        log.info("[M3] Fusion complete — %d caption entries ready.", len(entries))
        return entries


    def _score_all_events(
        self,
        audio_events:  List[AudioEvent],
        visual_scores: Dict[float, VisualScore],
        has_visual:    bool,
    ) -> List[_Decision]:
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

    def _log_decisions(self, decisions: List[_Decision]) -> None:
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


    def _deduplicate(self, accepted: List[_Decision]) -> List[_Decision]:
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


    def _build_caption_entries(
        self,
        decisions: List[_Decision],
    ) -> List[CaptionEntry]:
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

    def _enforce_srt_gaps(
        self,
        entries: List[CaptionEntry],
    ) -> List[CaptionEntry]:
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


    def _annotate_frames(
        self,
        entries:    List[CaptionEntry],
        video_path: str,
    ) -> None:
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


    @staticmethod
    def _any_face_detected(visual_scores: Dict[float, VisualScore]) -> bool:
        """Return True if at least one visual window found a face."""
        return any(
            vs.num_valid_frames > 0
            for vs in visual_scores.values()
        )


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