import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf
import cv2
import json
import logging
from datetime import timedelta
import argparse
from tqdm import tqdm
from panns_inference import AudioTagging

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ====================== CONFIG ======================
SAMPLE_RATE = 32000
WINDOW_SEC = 0.96
HOP_SEC = 0.20
CONFIDENCE_THRESHOLD = 0.07
MAX_EVENTS = 200
DEDUP_GAP_SEC = 0.50

FACE_WEIGHT = 0.45
BODY_WEIGHT = 0.55

GENERATE_FINAL_VIDEO = True

SCRIPT_DIR = Path(__file__).parent.resolve()
FACE_MODEL_PATH = SCRIPT_DIR / "face_landmarker.task"
POSE_MODEL_PATH = SCRIPT_DIR / "pose_landmarker_heavy.task"

if not FACE_MODEL_PATH.exists():
    raise FileNotFoundError(f"face_landmarker.task not found at: {FACE_MODEL_PATH}")
if not POSE_MODEL_PATH.exists():
    raise FileNotFoundError(f"pose_landmarker_full.task not found at: {POSE_MODEL_PATH}")

print("Loading MediaPipe models...")

face_base_options = python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH))
face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=4,
    output_face_blendshapes=True
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

pose_base_options = python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    running_mode=vision.RunningMode.IMAGE
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

print("Loading PANNs model...")
panns_model = AudioTagging(checkpoint_path=None, device='cpu')
print("All models loaded successfully!\n")


def setup_logger(video_name: str, output_dir: Path):
    logger = logging.getLogger(video_name)
    logger.setLevel(logging.DEBUG)
    log_file = output_dir / f"{video_name}_processing.log"
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def extract_audio(video_path: str, logger):
    logger.info(f"Extracting audio from: {video_path}")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [ffmpeg_bin, "-y", "-loglevel", "error", "-i", video_path,
               "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", tmp_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError("ffmpeg failed")
        data, _ = sf.read(tmp_path, dtype="float32")
        return data
    except Exception as e:
        logger.warning(f"ffmpeg failed. Trying librosa...")
        try:
            import librosa
            waveform, _ = librosa.load(video_path, sr=SAMPLE_RATE, mono=True)
            return waveform.astype(np.float32)
        except Exception as e2:
            logger.error(f"Both methods failed: {e2}")
            return None
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


def should_boost(label: str) -> bool:
    BOOST_KEYWORDS = [
        "firecracker", "firework", "explosion", "blast", "bang",
        "splash", "water", "glass", "break", "crash", "rat", "squeak"
    ]
    return any(kw in label.lower() for kw in BOOST_KEYWORDS)


def detect_audio_events(waveform: np.ndarray, logger):
    window_samples = int(WINDOW_SEC * SAMPLE_RATE)
    hop_samples = int(HOP_SEC * SAMPLE_RATE)
    events = []
    n = len(waveform)
    start = 0

    while start < n:
        end = start + window_samples
        chunk = waveform[start:end]
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)

        output = panns_model.inference(chunk)
        scores = output[0] if isinstance(output, (list, tuple)) else output['clipwise_output']
        if hasattr(scores, 'ndim') and scores.ndim > 1:
            scores = scores[0]

        top_indices = np.argsort(scores)[::-1][:6]

        for idx in top_indices:
            raw_score = float(scores[idx])
            if raw_score >= CONFIDENCE_THRESHOLD:
                label = panns_model.labels[idx]
                timestamp = start / SAMPLE_RATE
                boost = 0.18 if should_boost(label) else 0.0
                final_score = min(raw_score + boost, 1.0)

                events.append({
                    "timestamp_sec": round(timestamp, 2),
                    "label": label,
                    "audio_confidence": round(final_score, 4),
                    "raw_confidence": round(raw_score, 4)
                })
        start += hop_samples

    final_events = []
    for ev in events:
        if not final_events or (ev["timestamp_sec"] - final_events[-1]["timestamp_sec"] > DEDUP_GAP_SEC):
            final_events.append(ev)
        else:
            if ev["audio_confidence"] > final_events[-1]["audio_confidence"]:
                final_events[-1] = ev
    return final_events[:MAX_EVENTS]


def classify_reaction_type(face_blendshapes, pose_landmarks) -> dict:
    face_reaction = "neutral"
    body_reaction = "still"
    visual_score = 0.0

    if face_blendshapes:
        for face in face_blendshapes:
            eye_wide = max([b.score for b in face if b.category_name in ["eyeWideLeft", "eyeWideRight"]], default=0)
            jaw_open = max([b.score for b in face if b.category_name == "jawOpen"], default=0)
            brow_raise = max([b.score for b in face if b.category_name in ["browInnerUp", "browOuterUpLeft", "browOuterUpRight"]], default=0)

            if eye_wide > 0.55 and brow_raise > 0.45:
                face_reaction = "surprised"
                visual_score = max(visual_score, 0.72)
            elif jaw_open > 0.60 and eye_wide > 0.45:
                face_reaction = "shocked"
                visual_score = max(visual_score, 0.82)
            elif jaw_open > 0.68:
                face_reaction = "screaming"
                visual_score = max(visual_score, 0.78)
            elif brow_raise > 0.50 and eye_wide > 0.40:
                face_reaction = "scared"
                visual_score = max(visual_score, 0.68)

    if pose_landmarks:
        for pose in pose_landmarks:
            left_shoulder_y = pose[11].y
            right_shoulder_y = pose[12].y
            nose_y = pose[0].y
            movement = abs(left_shoulder_y - right_shoulder_y) + abs(nose_y - (left_shoulder_y + right_shoulder_y) / 2)

            if movement > 0.09:
                body_reaction = "sudden_flinch"
                visual_score = max(visual_score, 0.65)
            if movement > 0.13:
                body_reaction = "strong_reaction"
                visual_score = max(visual_score, 0.78)

    final_score = round(min(visual_score, 1.0), 4)

    if face_reaction in ["shocked", "screaming"] or body_reaction == "strong_reaction":
        reaction_type = face_reaction if face_reaction != "neutral" else "strong_reaction"
    elif face_reaction in ["surprised", "scared"]:
        reaction_type = face_reaction
    elif body_reaction == "sudden_flinch":
        reaction_type = "flinched"
    else:
        reaction_type = face_reaction if face_reaction != "neutral" else "no_significant_reaction"

    return {
        "reaction_type": reaction_type,
        "face_reaction": face_reaction,
        "body_reaction": body_reaction,
        "visual_score": final_score
    }


def analyze_visual_reaction(video_path: str, timestamp: float, logger):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_no = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return {"visual_score": 0.0, "reaction_type": "no_face_detected", "face_reaction": "none", "body_reaction": "none"}

    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face_result = face_detector.detect(rgb_frame)
    pose_result = pose_detector.detect(rgb_frame)

    return classify_reaction_type(
        face_result.face_blendshapes if face_result else None,
        pose_result.pose_landmarks if pose_result else None
    )


def decide_caption(audio_conf: float, visual_score: float, reaction_type: str) -> bool:
    if reaction_type in ["shocked", "screaming", "strong_reaction"]:
        return True
    if visual_score >= 0.38:
        return True
    if audio_conf >= 0.70 and visual_score >= 0.18:
        return True
    if audio_conf >= 0.80:
        return True
    return False


def annotate_frame(frame, caption_text, output_path):
    """Clean annotated frame - only caption at bottom center"""
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 55), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85
    thickness = 2
    
    text_size = cv2.getTextSize(caption_text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - 18
    
    cv2.putText(frame, caption_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.imwrite(str(output_path), frame)


def generate_final_video(video_path: str, events: list, output_path: Path):
    """Generate final video with burned-in captions + original audio using ffmpeg"""
    import tempfile
    import subprocess
    import shutil

    accepted_events = [e for e in events if e.get("should_caption")]
    if not accepted_events:
        print("No accepted events → skipping final video generation.")
        return

    print(f"Generating final video with {len(accepted_events)} captions + audio...")

    # Step 1: Create captioned video using OpenCV (silent)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create temporary silent video with captions
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        silent_video_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(silent_video_path, fourcc, fps, (width, height))

    pbar = tqdm(total=total_frames, desc="Burning captions")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        # Find active caption for this frame
        active_caption = None
        for ev in accepted_events:
            start = ev["timestamp_sec"]
            end = start + 1.8
            if start <= current_time < end:
                active_caption = f"[{ev['reaction_type'].upper()}] {ev['label']}"
                break

        if active_caption:
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 55), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.85
            thickness = 2
            text_size = cv2.getTextSize(active_caption, font, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 18
            cv2.putText(frame, active_caption, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    # Step 2: Merge captioned video + original audio using ffmpeg
    try:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", silent_video_path,      # captioned video (no audio)
            "-i", video_path,             # original video (has audio)
            "-c:v", "copy",               # copy video stream
            "-c:a", "aac",                # re-encode audio
            "-map", "0:v:0",              # take video from first input
            "-map", "1:a:0",              # take audio from second input
            "-shortest",
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"✅ Final video with audio saved: {output_path}")
        else:
            print(f"❌ ffmpeg failed: {result.stderr}")

    except Exception as e:
        print(f"❌ Error during audio merge: {e}")

    finally:
        # Clean up temporary silent video
        try:
            os.unlink(silent_video_path)
        except:
            pass

def process_video(video_path: str, output_dir: Path):
    video_name = Path(video_path).stem
    logger = setup_logger(video_name, output_dir)

    logger.info(f"{'='*70}")
    logger.info(f"PROCESSING: {video_name}")
    logger.info(f"{'='*70}")

    waveform = extract_audio(video_path, logger)
    if waveform is None:
        return

    audio_events = detect_audio_events(waveform, logger)
    logger.info(f"Detected {len(audio_events)} audio events")

    if not audio_events:
        return

    frames_dir = output_dir / "annotated_frames"
    frames_dir.mkdir(exist_ok=True)

    final_captions = []

    for i, event in enumerate(audio_events):
        visual_result = analyze_visual_reaction(video_path, event['timestamp_sec'], logger)

        event['visual_score'] = visual_result['visual_score']
        event['reaction_type'] = visual_result['reaction_type']
        event['face_reaction'] = visual_result['face_reaction']
        event['body_reaction'] = visual_result['body_reaction']

        should_caption = decide_caption(
            event['audio_confidence'],
            visual_result['visual_score'],
            visual_result['reaction_type']
        )
        event['should_caption'] = should_caption
        event['caption_class'] = event['label'] if should_caption else None

        status = "ACCEPTED" if should_caption else "SKIPPED"
        logger.info(f"{event['timestamp_sec']:6.2f}s | {event['label']:<38} | "
                    f"Audio={event['audio_confidence']:.2f} | Visual={visual_result['visual_score']:.2f} | "
                    f"Reaction={visual_result['reaction_type']:<18} | {status}")

        if should_caption:
            # Save clean annotated frame
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_no = int(event['timestamp_sec'] * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            cap.release()

            if ret:
                caption_text = f"[{visual_result['reaction_type'].upper()}] {event['label']}"
                frame_path = frames_dir / f"frame_{event['timestamp_sec']:.2f}s.png"
                annotate_frame(frame, caption_text, frame_path)

            final_captions.append(event)

    logger.info(f"\nFinal captions generated: {len(final_captions)} / {len(audio_events)}")

    # Save SRT
    def format_srt_time(seconds):
        td = timedelta(seconds=seconds)
        h, r = divmod(td.seconds, 3600)
        m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{int(td.microseconds/1000):03d}"

    with open(output_dir / "captions.srt", "w", encoding="utf-8") as f:
        for i, ev in enumerate(final_captions, 1):
            start = ev["timestamp_sec"]
            end = start + 1.8
            f.write(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n[{ev['reaction_type'].upper()}] {ev['label']}\n\n")

    # Save JSON
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "video_name": video_name,
            "total_audio_events": len(audio_events),
            "captions_generated": len(final_captions),
            "events": audio_events
        }, f, indent=2, ensure_ascii=False)

    # Generate final video
    if GENERATE_FINAL_VIDEO and final_captions:
        generate_final_video(video_path, final_captions, output_dir / "final_output.mp4")


def main():
    parser = argparse.ArgumentParser(description="Intelligent CC Pipeline - Single Video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return

    output_root = Path("panns_visual_results")
    output_root.mkdir(exist_ok=True)

    video_out_dir = output_root / video_path.stem
    video_out_dir.mkdir(exist_ok=True)

    try:
        process_video(str(video_path), video_out_dir)
        print(f"\n✅ Done! Results saved in: {video_out_dir}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()