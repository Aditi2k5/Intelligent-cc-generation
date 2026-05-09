from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from utils.logger import get_logger
import config as cfg

log = get_logger(__name__)


@dataclass
class AudioEvent:
    timestamp_sec:  float           # centre of the detection window
    end_sec:        float           # estimated end time
    category:       str             # key from SOUND_CATEGORIES
    display_label:  str             # human-readable CC text
    priority:       str             # HIGH | MEDIUM | LOW
    confidence:     float           # boosted & normalised score [0, 1]
    raw_class:      str             # best-matching YAMNet class name
    raw_score:      float           # original YAMNet score before boost


def _build_lookup() -> Tuple[List[str], Dict[str, dict]]:
    blacklist = [s.lower() for s in cfg.YAMNET_BLACKLIST]

    class_to_cat: Dict[str, dict] = {}
    for cat_key, cat_info in cfg.SOUND_CATEGORIES.items():
        for token in cat_info["yamnet"]:
            class_to_cat[token.lower()] = {
                "key":      cat_key,
                "display":  cat_info["display"],
                "priority": cat_info["priority"],
                "boost":    cat_info["boost"],
            }

    return blacklist, class_to_cat


_BLACKLIST_TOKENS, _CLASS_TO_CAT = _build_lookup()


class SoundEventDetector:
 
    def __init__(self, model_handle: str = cfg.YAMNET_MODEL_PATH):
        self._model_handle = model_handle
        self._yamnet       = None          # lazy-loaded
        self._class_names  = None


    def detect(self, video_path: str) -> List[AudioEvent]:

        log.info("[M1] Starting sound detection on: %s", video_path)

        # Step 1: extract audio waveform
        waveform = self._extract_audio(video_path)
        if waveform is None or len(waveform) == 0:
            log.error("[M1] Could not extract audio from %s", video_path)
            return []

        duration_sec = len(waveform) / cfg.AUDIO_SAMPLE_RATE
        log.info("[M1] Audio duration: %.2f s  |  samples: %d",
                 duration_sec, len(waveform))

        # Step 2: load model once
        self._load_model()

        # Step 3: sliding-window inference
        raw_events = self._sliding_window_inference(waveform)
        log.info("[M1] Sliding window produced %d candidate events",
                 len(raw_events))

        # Step 4: filter, map, boost
        filtered = self._filter_and_map(raw_events)
        log.info("[M1] After filtering: %d events remain", len(filtered))

        # Step 5: merge nearby duplicates
        merged = self._merge_events(filtered)
        log.info("[M1] After merging: %d events", len(merged))

        # Step 6: cap per-category count
        capped = self._cap_per_category(merged)
        log.info("[M1] Final audio events: %d", len(capped))

        for ev in capped:
            log.debug(
                "  [M1] %.2fs  %-15s  conf=%.3f  raw='%s'",
                ev.timestamp_sec, ev.category, ev.confidence, ev.raw_class,
            )

        return sorted(capped, key=lambda e: e.timestamp_sec)


    def _extract_audio(self, video_path: str) -> Optional[np.ndarray]:
        video_path = str(Path(video_path).resolve())

        try:
            return self._extract_via_ffmpeg(video_path)
        except Exception as exc:
            log.warning("[M1] ffmpeg extraction failed (%s), trying librosa", exc)

        try:
            import librosa
            waveform, _ = librosa.load(
                video_path,
                sr=cfg.AUDIO_SAMPLE_RATE,
                mono=True,
            )
            return waveform.astype(np.float32)
        except Exception as exc:
            log.error("[M1] librosa fallback failed: %s", exc)
            return None

    def _extract_via_ffmpeg(self, video_path: str) -> np.ndarray:
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            ffmpeg_bin = shutil.which("ffmpeg")
            if not ffmpeg_bin:
                # Fallback to the bundled binary from imageio-ffmpeg when PATH has no ffmpeg.
                try:
                    import imageio_ffmpeg
                    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception as exc:
                    raise RuntimeError(
                        "ffmpeg executable not found. Install ffmpeg or imageio-ffmpeg."
                    ) from exc

            cmd = [
                ffmpeg_bin, "-y", "-loglevel", "error",
                "-i", video_path,
                "-ar", str(cfg.AUDIO_SAMPLE_RATE),
                "-ac", "1",
                "-f", "wav",
                tmp_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, check=True, timeout=120
            )
            data, _ = sf.read(tmp_path, dtype="float32")
            return data
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


    def _load_model(self) -> None:
        if self._yamnet is not None:
            return
        log.info("[M1] Loading YAMNet from %s …", self._model_handle)
        try:
            import tensorflow_hub as hub
            self._yamnet = hub.load(self._model_handle)
            # Retrieve class names from the model asset
            import csv
            class_map_path = self._yamnet.class_map_path().numpy().decode()
            with open(class_map_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header row
                self._class_names = [row[2] for row in reader if len(row) >= 3]

            if not self._class_names:
                raise RuntimeError(f"YAMNet class map is empty: {class_map_path}")
            log.info("[M1] YAMNet loaded — %d classes", len(self._class_names))
        except ImportError as exc:
            raise RuntimeError(
                "tensorflow_hub is required for Module 1. "
                "Run: pip install tensorflow tensorflow-hub"
            ) from exc


    def _sliding_window_inference(
        self,
        waveform: np.ndarray,
    ) -> List[dict]:
        import tensorflow as tf

        window_samples = int(cfg.AUDIO_WINDOW_SEC * cfg.AUDIO_SAMPLE_RATE)
        hop_samples    = int(cfg.AUDIO_HOP_SEC    * cfg.AUDIO_SAMPLE_RATE)

        results = []
        n = len(waveform)
        start = 0

        while start < n:
            end = start + window_samples
            chunk = waveform[start:end]

            # Zero-pad the last chunk if needed
            if len(chunk) < window_samples:
                chunk = np.pad(chunk, (0, window_samples - len(chunk)))

            # YAMNet expects shape [num_samples]
            scores_tensor, _, _ = self._yamnet(
                tf.constant(chunk, dtype=tf.float32)
            )

            # scores_tensor shape: [num_patches, 521]
            # Average across time patches for a single window-level score
            mean_scores = tf.reduce_mean(scores_tensor, axis=0).numpy()

            # Top-5 classes
            top_k = np.argsort(mean_scores)[::-1][:5]
            top_pairs = [
                (self._class_names[i], float(mean_scores[i]))
                for i in top_k
                if float(mean_scores[i]) >= cfg.YAMNET_RAW_THRESHOLD
            ]

            timestamp = start / cfg.AUDIO_SAMPLE_RATE
            end_time  = min(end / cfg.AUDIO_SAMPLE_RATE,
                            n   / cfg.AUDIO_SAMPLE_RATE)

            if top_pairs:
                results.append({
                    "timestamp_sec": timestamp,
                    "end_sec":       end_time,
                    "scores":        top_pairs,
                })

            start += hop_samples

        return results


    def _filter_and_map(self, raw_events: List[dict]) -> List[AudioEvent]:
        events: List[AudioEvent] = []

        for raw in raw_events:
            best_event = self._best_category_match(
                raw["scores"],
                raw["timestamp_sec"],
                raw["end_sec"],
            )
            if best_event is not None:
                events.append(best_event)

        return events

    def _best_category_match(
        self,
        scores:        List[Tuple[str, float]],
        timestamp_sec: float,
        end_sec:       float,
    ) -> Optional[AudioEvent]:
        best_score    = 0.0
        best_cat      = None
        best_raw_cls  = ""
        best_raw_score = 0.0

        for class_name, raw_score in scores:
            cname_lower = class_name.lower()

            if any(bl in cname_lower for bl in _BLACKLIST_TOKENS):
                continue

            matched_cat = None
            for token, cat_info in _CLASS_TO_CAT.items():
                if token in cname_lower:
                    matched_cat = cat_info
                    break

            if matched_cat is None:
                continue

            boosted = min(raw_score * matched_cat["boost"], 1.0)

            if boosted > best_score:
                best_score     = boosted
                best_cat       = matched_cat
                best_raw_cls   = class_name
                best_raw_score = raw_score

        if best_cat is None:
            return None

        if best_score < cfg.AUDIO_EMIT_THRESHOLD:
            return None

        return AudioEvent(
            timestamp_sec = timestamp_sec,
            end_sec       = end_sec,
            category      = best_cat["key"],
            display_label = best_cat["display"],
            priority      = best_cat["priority"],
            confidence    = round(best_score, 4),
            raw_class     = best_raw_cls,
            raw_score     = round(best_raw_score, 4),
        )


    def _merge_events(self, events: List[AudioEvent]) -> List[AudioEvent]:
        if not events:
            return events

        # Group by category
        from collections import defaultdict
        by_cat: Dict[str, List[AudioEvent]] = defaultdict(list)
        for ev in events:
            by_cat[ev.category].append(ev)

        merged_all: List[AudioEvent] = []

        for cat_events in by_cat.values():
            cat_events.sort(key=lambda e: e.timestamp_sec)
            groups: List[List[AudioEvent]] = []
            current_group: List[AudioEvent] = [cat_events[0]]

            for ev in cat_events[1:]:
                gap = ev.timestamp_sec - current_group[-1].end_sec
                if gap <= cfg.EVENT_MERGE_GAP_SEC:
                    current_group.append(ev)
                else:
                    groups.append(current_group)
                    current_group = [ev]
            groups.append(current_group)

            for group in groups:
                # Representative = highest confidence
                best = max(group, key=lambda e: e.confidence)
                merged_all.append(AudioEvent(
                    timestamp_sec  = group[0].timestamp_sec,
                    end_sec        = group[-1].end_sec,
                    category       = best.category,
                    display_label  = best.display_label,
                    priority       = best.priority,
                    confidence     = best.confidence,
                    raw_class      = best.raw_class,
                    raw_score      = best.raw_score,
                ))

        return merged_all

    def _cap_per_category(self, events: List[AudioEvent]) -> List[AudioEvent]:
        from collections import defaultdict
        by_cat: Dict[str, List[AudioEvent]] = defaultdict(list)
        for ev in events:
            by_cat[ev.category].append(ev)

        capped: List[AudioEvent] = []
        for cat_events in by_cat.values():
            cat_events.sort(key=lambda e: e.confidence, reverse=True)
            capped.extend(cat_events[: cfg.MAX_EVENTS_PER_CATEGORY])

        return capped