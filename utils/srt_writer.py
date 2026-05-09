"""
utils/srt_writer.py
===================
Utilities for writing SRT subtitle files and JSON event reports.

SRT format:
    1
    00:00:01,000 --> 00:00:03,000
    [ Sound Effect ]

    2
    ...
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CaptionEntry:
    """One finalized CC subtitle entry ready for SRT output."""
    index:          int
    start_sec:      float
    end_sec:        float
    caption_text:   str
    category:       str
    priority:       str
    audio_score:    float
    visual_score:   float
    fusion_score:   float


# ─────────────────────────────────────────────────────────────────────────────
# Time Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sec_to_srt_time(seconds: float) -> str:
    """Convert float seconds → SRT timestamp  HH:MM:SS,mmm."""
    seconds = max(0.0, seconds)
    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = int(seconds % 60)
    millis  = int(round((seconds % 1) * 1000))
    # Guard against rounding pushing millis to 1000
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ─────────────────────────────────────────────────────────────────────────────
# SRT Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_srt(entries: List[CaptionEntry], output_path: str) -> None:
    """
    Write a list of CaptionEntry objects as a valid .srt file.

    Entries are sorted by start time.  Overlapping end/start times are
    nudged so subtitles never overlap in the timeline.

    Parameters
    ----------
    entries     : list of finalized CaptionEntry objects
    output_path : destination file path (e.g. "demo_results/output.srt")
    """
    if not entries:
        log.warning("No caption entries to write — SRT file will be empty.")

    # Sort by start time
    sorted_entries = sorted(entries, key=lambda e: e.start_sec)

    # Prevent timeline overlaps: ensure each end <= next start
    for i in range(len(sorted_entries) - 1):
        curr = sorted_entries[i]
        nxt  = sorted_entries[i + 1]
        if curr.end_sec > nxt.start_sec:
            curr.end_sec = max(curr.start_sec + 0.1, nxt.start_sec - 0.05)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        for idx, entry in enumerate(sorted_entries, start=1):
            start_ts = _sec_to_srt_time(entry.start_sec)
            end_ts   = _sec_to_srt_time(entry.end_sec)
            fh.write(f"{idx}\n")
            fh.write(f"{start_ts} --> {end_ts}\n")
            fh.write(f"{entry.caption_text}\n")
            fh.write("\n")

    log.info("SRT written → %s  (%d entries)", out_path, len(sorted_entries))


# ─────────────────────────────────────────────────────────────────────────────
# JSON Report Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_json_report(
    entries:        List[CaptionEntry],
    audio_events:   list,
    visual_scores:  dict,
    output_path:    str,
    video_path:     Optional[str] = None,
) -> None:
    """
    Write a comprehensive JSON report containing:
      - pipeline metadata
      - raw audio events from Module 1
      - visual scores from Module 2
      - finalized CC entries from Module 3

    Parameters
    ----------
    entries       : finalized CaptionEntry list
    audio_events  : raw AudioEvent list from Module 1
    visual_scores : dict {timestamp: VisualScore} from Module 2
    output_path   : destination .json file path
    video_path    : optional path to the source video (for reference)
    """
    import datetime

    report = {
        "meta": {
            "tool":       "Intelligent CC Suggestion Tool",
            "version":    "2.0.0",
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "video_path": str(video_path) if video_path else None,
        },
        "summary": {
            "total_audio_events":   len(audio_events),
            "total_visual_windows": len(visual_scores),
            "total_captions":       len(entries),
        },
        "audio_events":  [_serialise(e) for e in audio_events],
        "visual_scores": {
            str(round(k, 3)): _serialise(v)
            for k, v in visual_scores.items()
        },
        "captions": [asdict(e) for e in entries],
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    log.info("JSON report written → %s", out_path)


def _serialise(obj) -> dict:
    """Best-effort serialisation for dataclasses or plain objects."""
    try:
        return asdict(obj)
    except TypeError:
        return vars(obj) if hasattr(obj, "__dict__") else str(obj)