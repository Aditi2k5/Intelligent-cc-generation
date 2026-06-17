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

from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util
from panns_inference import AudioTagging

# ====================== CONFIG ======================
SAMPLE_RATE = 32000
WINDOW_SEC = 0.96
HOP_SEC = 0.20
CONFIDENCE_THRESHOLD = 0.07
MAX_EVENTS = 2000
DEDUP_GAP_SEC = 0.50
GENERATE_FINAL_VIDEO = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*70}")
print(f"Using device: {device.upper()}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"{'='*70}\n")

print("Loading PANNs model...")
panns_model = AudioTagging(checkpoint_path=None, device=device)
print("PANNs loaded.\n")

print("Loading Sentence Transformer on GPU...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("Sentence Transformer loaded.\n")


def setup_logger(video_name: str, output_dir: Path):
    logger = logging.getLogger(video_name)
    logger.setLevel(logging.INFO)
    log_file = output_dir / f"{video_name}_processing.log"
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_blip_captions_from_log(log_path: Path, logger) -> list:
    """Parse already generated BLIP captions from previous log file."""
    if not log_path or not log_path.exists():
        logger.warning("No previous BLIP log provided. Will skip scene detection.")
        return []
    
    captions = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "BLIP Caption:" in line:
                caption = line.split("BLIP Caption:", 1)[1].strip()
                if caption and len(caption) > 5:
                    captions.append(caption)
    
    logger.info(f"Loaded {len(captions)} BLIP captions from previous log.")
    return captions


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
        logger.warning("ffmpeg failed. Trying librosa...")
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
    keywords = ["firecracker", "firework", "explosion", "blast", "splash", "glass", "break", "crash", "rat"]
    return any(kw in label.lower() for kw in keywords)


def detect_audio_events(waveform: np.ndarray, logger, scene_boosts: dict = None):
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

                if scene_boosts and label in scene_boosts:
                    final_score = min(final_score + scene_boosts[label], 1.0)

                events.append({
                    "timestamp_sec": round(timestamp, 2),
                    "label": label,
                    "audio_confidence": round(final_score, 4)
                })
        start += hop_samples

    # Deduplication
    final_events = []
    for ev in events:
        if not final_events or (ev["timestamp_sec"] - final_events[-1]["timestamp_sec"] > DEDUP_GAP_SEC):
            final_events.append(ev)
        else:
            if ev["audio_confidence"] > final_events[-1]["audio_confidence"]:
                final_events[-1] = ev
    return final_events[:MAX_EVENTS]


def decide_caption(audio_conf: float, scene_boosts: dict, label: str) -> bool:
    if audio_conf >= 0.75:
        return True
    if scene_boosts and label in scene_boosts:
        boost_value = scene_boosts[label]
        if audio_conf >= 0.55 and boost_value >= 0.30:
            return True
        if audio_conf >= 0.45 and boost_value >= 0.45:
            return True
    if audio_conf >= 0.82:
        return True
    return False


def get_scene_from_saved_captions(captions: list, logger) -> dict:
    """Use already saved BLIP captions to detect scene (no re-running BLIP)."""
    if not captions:
        return {"scene": "unknown", "boosts": {}}

    full_description = " ".join(captions)

    scene_descriptions = {
        "forest_jungle": "dense forest, jungle, trees, green vegetation, outdoor nature scene",
        "temple": "traditional Indian temple, religious place, pooja, worship",
        "marriage_wedding": "Indian wedding, marriage ceremony, bride and groom, festive",
        "grassland_rural": "grass field, rural village, dirt path, countryside",
        "street_road": "busy street, road, traffic, vehicles, urban outdoor",
        "indoor_room": "inside a room, indoor setting, people gathering",
        "mela_festival": "festival, mela, celebration, crowd, cultural event"
    }

    video_embedding = sentence_model.encode(full_description, convert_to_tensor=True)
    scene_embeddings = sentence_model.encode(list(scene_descriptions.values()), convert_to_tensor=True)

    similarities = util.cos_sim(video_embedding, scene_embeddings)[0]
    best_idx = similarities.argmax().item()
    best_scene = list(scene_descriptions.keys())[best_idx]

    logger.info(f"Detected Scene from saved captions: {best_scene}")

    scene_boosts = {}
    if best_scene == "forest_jungle":
        scene_boosts = {
            "Bird vocalization, bird call, bird song": 0.55,
            "Crow": 0.50,
            "Wind": 0.40,
            "Leaves rustling": 0.35
        }
    elif best_scene == "temple":
        scene_boosts = {"Bell": 0.55, "Chime": 0.42}
    elif best_scene in ["marriage_wedding", "mela_festival"]:
        scene_boosts = {"Shehnai": 0.50, "Dhol": 0.45, "Firecracker": 0.40}
    elif best_scene == "grassland_rural":
        scene_boosts = {"Footsteps": 0.48, "Wind": 0.40, "Grass rustling": 0.42, "Crow": 0.35}
    elif best_scene == "street_road":
        scene_boosts = {"Vehicle": 0.45, "Traffic": 0.38, "Horn": 0.35}

    if any(word in full_description.lower() for word in ["people", "group", "women", "man", "person", "crowd", "laugh", "sit"]):
        scene_boosts["Laughter"] = scene_boosts.get("Laughter", 0) + 0.45
        scene_boosts["Screaming"] = scene_boosts.get("Screaming", 0) + 0.38
        scene_boosts["Crying"] = scene_boosts.get("Crying", 0) + 0.32

    return {"scene": best_scene, "boosts": scene_boosts}


def annotate_frame(frame, caption_text, output_path):
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
    accepted_events = [e for e in events if e.get("should_caption")]
    if not accepted_events:
        print("No accepted events → skipping final video generation.")
        return

    print(f"Generating final video with {len(accepted_events)} captions + audio...")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
        active_caption = None
        for ev in accepted_events:
            start = ev["timestamp_sec"]
            end = start + 1.8
            if start <= current_time < end:
                active_caption = f"[{ev['label']}]"
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

    try:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_bin, "-y",
            "-i", silent_video_path,
            "-i", video_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
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
        try:
            os.unlink(silent_video_path)
        except:
            pass


def process_video(video_path: str, output_dir: Path, blip_log_path: Path = None):
    video_name = Path(video_path).stem
    logger = setup_logger(video_name, output_dir)

    logger.info(f"{'='*70}")
    logger.info(f"PROCESSING: {video_name} (Reusing saved BLIP captions)")
    logger.info(f"{'='*70}")

    # === Load saved BLIP captions instead of running BLIP ===
    saved_captions = load_blip_captions_from_log(blip_log_path, logger)
    scene_context = get_scene_from_saved_captions(saved_captions, logger)
    scene_boosts = scene_context.get("boosts", {})

    # === Continue from Audio Extraction ===
    waveform = extract_audio(video_path, logger)
    if waveform is None:
        return

    audio_events = detect_audio_events(waveform, logger, scene_boosts=scene_boosts)
    logger.info(f"Detected {len(audio_events)} audio events")

    if not audio_events:
        return

    frames_dir = output_dir / "annotated_frames"
    frames_dir.mkdir(exist_ok=True)

    final_captions = []

    for event in audio_events:
        should_caption = decide_caption(event['audio_confidence'], scene_boosts, event['label'])

        event['should_caption'] = should_caption
        event['caption_class'] = event['label'] if should_caption else None
        event['scene_boosts'] = scene_boosts

        status = "ACCEPTED" if should_caption else "SKIPPED"
        logger.info(f"{event['timestamp_sec']:6.2f}s | {event['label']:<38} | "
                    f"Audio={event['audio_confidence']:.2f} | Boost={scene_boosts.get(event['label'], 0):.2f} | {status}")

        if should_caption:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_no = int(event['timestamp_sec'] * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            cap.release()

            if ret:
                caption_text = f"[{event['label']}]"
                frame_path = frames_dir / f"frame_{event['timestamp_sec']:.2f}s.png"
                annotate_frame(frame, caption_text, frame_path)

            final_captions.append(event)

    logger.info(f"\nFinal captions generated: {len(final_captions)} / {len(audio_events)}")

    def format_srt_time(seconds):
        td = timedelta(seconds=seconds)
        h, r = divmod(td.seconds, 3600)
        m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{int(td.microseconds/1000):03d}"

    with open(output_dir / "captions.srt", "w", encoding="utf-8") as f:
        for i, ev in enumerate(final_captions, 1):
            start = ev["timestamp_sec"]
            end = start + 1.8
            f.write(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n[{ev['label']}]")

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "video_name": video_name,
            "total_audio_events": len(audio_events),
            "captions_generated": len(final_captions),
            "events": audio_events
        }, f, indent=2, ensure_ascii=False)

    if GENERATE_FINAL_VIDEO and final_captions:
        generate_final_video(video_path, final_captions, output_dir / "final_output.mp4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--blip-log", type=str, default=None, help="Path to previous processing log containing BLIP captions")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return

    blip_log_path = Path(args.blip_log) if args.blip_log else None

    output_root = Path("pann_with_blip")
    output_root.mkdir(exist_ok=True)
    video_out_dir = output_root / video_path.stem
    video_out_dir.mkdir(exist_ok=True)

    try:
        process_video(str(video_path), video_out_dir, blip_log_path)
        print(f"\n✅ Done! Results saved in: {video_out_dir}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()