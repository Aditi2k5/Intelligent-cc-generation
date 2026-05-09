from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

DEFAULT_VIDEO_PATH = "fight.mp4"



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cc_tool",
        description=(
            "Intelligent CC Suggestion Tool — "
            "generates non-speech closed captions for videos."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """,
    )

    parser.add_argument(
        "--video", "-v",
        required=False,
        default=DEFAULT_VIDEO_PATH,
        metavar="PATH",
        help=(
            "Path to the input video file (MP4, AVI, MKV, MOV, …). "
            f"Defaults to {DEFAULT_VIDEO_PATH}."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="DIR",
        help=(
            "Output directory for SRT, JSON and annotated frames. "
            "Defaults to config.OUTPUT_DIR ('demo_results/')."
        ),
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help=(
            "Skip Module 2 (visual reaction detection). "
            "Faster, but captions are based on audio confidence alone."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level log messages (very verbose).",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Skip saving annotated JPEG frames to the output folder.",
    )

    return parser.parse_args()


def _print_banner() -> None:
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║   Intelligent CC Suggestion Tool  •  PlanetRead / C4GT  ║\n"
        "║   Module 1: Sound Detection (YAMNet)                    ║\n"
        "║   Module 2: Visual Reaction (MediaPipe Face Mesh)        ║\n"
        "║   Module 3: Fusion Engine + SRT Output                  ║\n"
        "╚══════════════════════════════════════════════════════════╝\n"
    )


def _print_final_summary(
    entries,
    output_dir: Path,
    elapsed:    float,
    audio_count: int,
) -> None:
    srt_path  = output_dir / "output.srt"
    json_path = output_dir / "report.json"
    frames_dir = output_dir / "frames"
    frame_count = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0

    print(
        "\n"
        "════════════════════════════════════════════════════════════\n"
        f"  Pipeline complete in {elapsed:.1f} s\n"
        f"  Audio events detected : {audio_count}\n"
        f"  Captions emitted      : {len(entries)}\n"
        f"  Annotated frames      : {frame_count}\n"
        f"  SRT   → {srt_path}\n"
        f"  JSON  → {json_path}\n"
        "════════════════════════════════════════════════════════════\n"
    )

    if entries:
        print("  Caption preview:")
        print("  " + "─" * 54)
        for e in entries:
            print(
                f"  {e.start_sec:6.2f}s → {e.end_sec:6.2f}s  "
                f"{e.caption_text:<28}  "
                f"fusion={e.fusion_score:.3f}"
            )
        print("  " + "─" * 54)
    else:
        print(
            "  ⚠  No captions were generated.\n"
            "     Try lowering FUSION_THRESHOLD or AUDIO_EMIT_THRESHOLD\n"
            "     in config.py, or check the pipeline.log for details.\n"
        )

def run_pipeline(
    video_path:  str,
    output_dir:  str,
    skip_visual: bool = False,
    save_frames: bool = True,
) -> int:
    
    from modules.sound_detector  import SoundEventDetector
    from modules.visual_detector import VisualReactionDetector, VisualScore
    from modules.fusion_engine   import FusionEngine
    from utils.srt_writer        import write_srt, write_json_report
    from utils.logger            import get_logger, setup_file_logger
    import config as cfg

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logger(str(out_dir))

    log = get_logger("main")
    log.info("Video   : %s", video_path)
    log.info("Output  : %s", out_dir)
    log.info("Mode    : %s", "audio-only" if skip_visual else "audio+visual")

    t_start = time.perf_counter()

    # ════════════════════════════════════════════════════════════════════════
    # MODULE 1 — Sound Event Detection
    # ════════════════════════════════════════════════════════════════════════
    print("\n[1/3] Running sound event detection …")
    try:
        detector     = SoundEventDetector()
        audio_events = detector.detect(video_path)
    except Exception as exc:
        log.error("[M1] Fatal error during sound detection: %s", exc,
                  exc_info=True)
        return 1

    if not audio_events:
        log.warning(
            "No audio events passed the filter. "
            "Check AUDIO_EMIT_THRESHOLD in config.py or verify the video "
            "has a usable audio track."
        )
        # Write empty outputs so downstream tools don't crash
        _write_empty_outputs(out_dir)
        return 0

    timestamps = [ev.timestamp_sec for ev in audio_events]
    log.info("[M1] %d audio event(s) found at timestamps: %s",
             len(audio_events),
             [f"{t:.2f}s" for t in timestamps])

    # ════════════════════════════════════════════════════════════════════════
    # MODULE 2 — Visual Reaction Detection
    # ════════════════════════════════════════════════════════════════════════
    visual_scores: dict = {}

    if skip_visual:
        log.info("[M2] Skipped (--no-visual flag set).")
        print("[2/3] Visual reaction detection … SKIPPED (--no-visual)")
        # Provide zero-score placeholders so Module 3 runs in audio-only mode
        from modules.visual_detector import VisualScore
        visual_scores = {
            ts: VisualScore(
                query_time_sec   = ts,
                reaction_score   = 0.0,
                num_valid_frames = 0,
                peak_frame_time  = None,
                confidence       = "low",
                note             = "skipped_by_user",
            )
            for ts in timestamps
        }
    else:
        print("[2/3] Running visual reaction detection …")
        try:
            analyser      = VisualReactionDetector()
            visual_scores = analyser.analyse(video_path, timestamps)
        except Exception as exc:
            log.error("[M2] Fatal error during visual analysis: %s", exc,
                      exc_info=True)
            log.warning("[M2] Falling back to audio-only mode.")
            from modules.visual_detector import VisualScore
            visual_scores = {
                ts: VisualScore(
                    query_time_sec   = ts,
                    reaction_score   = 0.0,
                    num_valid_frames = 0,
                    peak_frame_time  = None,
                    confidence       = "low",
                    note             = "module2_error",
                )
                for ts in timestamps
            }

    # ════════════════════════════════════════════════════════════════════════
    # MODULE 3 — Fusion Decision Engine
    # ════════════════════════════════════════════════════════════════════════
    print("[3/3] Running fusion engine …")
    try:
        engine  = FusionEngine(output_dir=str(out_dir))
        entries = engine.decide(
            audio_events  = audio_events,
            visual_scores = visual_scores,
            video_path    = video_path if save_frames else None,
        )
    except Exception as exc:
        log.error("[M3] Fatal error in fusion engine: %s", exc, exc_info=True)
        return 1

    srt_path  = str(out_dir / "output.srt")
    json_path = str(out_dir / "report.json")

    write_srt(entries, srt_path)
    write_json_report(
        entries       = entries,
        audio_events  = audio_events,
        visual_scores = visual_scores,
        output_path   = json_path,
        video_path    = video_path,
    )

    elapsed = time.perf_counter() - t_start
    _print_final_summary(entries, out_dir, elapsed, len(audio_events))

    return 0

def _write_empty_outputs(out_dir: Path) -> None:
    """Write empty SRT and minimal JSON so callers don't get FileNotFoundError."""
    from utils.srt_writer import write_srt, write_json_report
    write_srt([], str(out_dir / "output.srt"))
    write_json_report(
        entries       = [],
        audio_events  = [],
        visual_scores = {},
        output_path   = str(out_dir / "report.json"),
    )


def _validate_video_path(path: str) -> str:
    """Resolve and verify the video file exists; exit with a message if not."""
    p = Path(path).resolve()
    if not p.exists():
        print(f"\n  ✗  Video file not found: {p}", file=sys.stderr)
        sys.exit(1)
    if p.suffix.lower() not in {
        ".mp4", ".avi", ".mkv", ".mov", ".webm",
        ".flv", ".wmv", ".m4v", ".3gp",
    }:
        print(
            f"\n  ⚠  Warning: '{p.suffix}' may not be a supported video format.\n"
            f"     Supported: .mp4 .avi .mkv .mov .webm .flv .wmv .m4v .3gp\n"
            f"     Proceeding anyway — ffmpeg may still handle it.\n",
            file=sys.stderr,
        )
    return str(p)

def main() -> None:
    args = _parse_args()

    # ── Set log level before any module imports ──────────────────────────────
    import logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    _print_banner()

    video_path = _validate_video_path(args.video)

    import config as cfg
    output_dir = args.output if args.output else cfg.OUTPUT_DIR

    exit_code = run_pipeline(
        video_path  = video_path,
        output_dir  = output_dir,
        skip_visual = args.no_visual,
        save_frames = not args.no_frames,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()