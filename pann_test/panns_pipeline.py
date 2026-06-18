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
from collections import defaultdict
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from panns_inference import AudioTagging

# ====================== GPU CHECK ======================
assert torch.cuda.is_available(), "CUDA is required!"
device = "cuda"
print(f"\n{'='*70}")
print(f"Using device: {device.upper()} | GPU: {torch.cuda.get_device_name(0)}")
print(f"{'='*70}\n")

# ====================== CONFIG ======================
SAMPLE_RATE = 32000
WINDOW_SEC = 0.96
HOP_SEC = 0.20
FRAME_SAMPLE_INTERVAL = 1.5
SEMANTIC_THRESHOLD = 0.18          
SCENE_BOOST_WINDOW_SEC = 10.0      
DEDUP_GAP_SEC = 1.2
GENERATE_FINAL_VIDEO = True
MAX_SIMULTANEOUS_CAPTIONS = 1      
PALETTE_SIZE = 18                  
W_AUDIO = 0.55
W_PALETTE = 0.45
ACCEPT_THRESHOLD = 0.16           

# ---- Per-class thresholds -------------------------------------------------

CLASS_THRESHOLD_KEYWORDS = [
    (["bird vocalization", "bird call", "bird song"], 0.035),
    (["crow"], 0.04),
    (["caw"], 0.03),
    (["wind noise"], 0.06),
    (["wind"], 0.05),
    (["rustl"], 0.04),
    (["cricket"], 0.04),
    (["insect"], 0.05),
    (["walk", "footstep"], 0.06),
    (["music"], 0.12),
    (["musical instrument"], 0.08),
    (["drum"], 0.06),
    (["plucked string"], 0.06),
    (["singing"], 0.10),
    (["laughter"], 0.07),
    (["giggle"], 0.05),
    (["chuckle", "chortle"], 0.05),
    (["belly laugh"], 0.08),
]
DEFAULT_THRESHOLD = 0.06           # fallback for any class not explicitly listed
_resolved_threshold_cache = {}    

def resolve_label_thresholds(panns_labels: list) -> dict:
    """Builds {exact_label_string: threshold} by matching keyword groups
    against the REAL label list PANNs ships with, instead of trusting
    hand-typed exact strings that might not match the installed ontology
    version exactly."""
    global _resolved_threshold_cache
    if _resolved_threshold_cache:
        return _resolved_threshold_cache
    resolved = {}
    for label in panns_labels:
        label_lower = label.lower()
        for keywords, thresh in CLASS_THRESHOLD_KEYWORDS:
            if all(k in label_lower for k in keywords) or any(
                label_lower == k or k in label_lower for k in keywords
            ):
                if label not in resolved or thresh < resolved[label]:
                    resolved[label] = thresh
    _resolved_threshold_cache = resolved
    return resolved

SPEECH_LABELS = {"speech", "male speech", "female speech", "conversation",
                  "narration", "monologue", "dialogue", "voice", "talking",
                  "child speech", "babbling"}

def is_speech_label(label: str) -> bool:
    return any(s in label.lower() for s in SPEECH_LABELS)

def get_class_threshold(label: str) -> float:
    resolved = _resolved_threshold_cache
    return resolved.get(label, DEFAULT_THRESHOLD)


SPEECH_LABELS = {"speech", "male speech", "female speech", "conversation",
                  "narration", "monologue", "dialogue", "voice", "talking",
                  "child speech", "babbling"}

def is_speech_label(label: str) -> bool:
    return any(s in label.lower() for s in SPEECH_LABELS)

def predict_scene_palette(scene_caption: str, panns_labels: list, sentence_model,
                           class_embs, top_k: int = PALETTE_SIZE) -> dict:
    emb = sentence_model.encode(scene_caption, convert_to_tensor=True)
    sims = util.cos_sim(emb, class_embs)[0]
    candidates = []
    for i, label in enumerate(panns_labels):
        if is_speech_label(label):
            continue
        sim = float(sims[i])
        if sim >= SEMANTIC_THRESHOLD:
            candidates.append((label, sim))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return dict(candidates[:top_k])

_florence_processor = None
_florence_model = None
_sentence_model = None
_panns_model = None
_vad_model = None
_vad_utils = None

def get_florence():
    global _florence_processor, _florence_model
    if _florence_model is None:
        print("Loading Florence-2-large...")
        _florence_processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True)
        _florence_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", torch_dtype=torch.float16, trust_remote_code=True
        ).to(device)
        print("Florence-2 loaded.\n")
    return _florence_processor, _florence_model

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        print("Loading Sentence Transformer...")
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print("Sentence Transformer loaded.\n")
    return _sentence_model

def get_panns():
    global _panns_model
    if _panns_model is None:
        print("Loading PANNs...")
        _panns_model = AudioTagging(checkpoint_path=None, device=device)
        print("PANNs loaded.\n")
    return _panns_model

def get_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        print("Loading Silero VAD (CPU for stability)...")
        torch.set_num_threads(1)
        _vad_model, _vad_utils = torch.hub.load(
            'snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
        _vad_model = _vad_model.cpu()
        print("Silero VAD loaded on CPU.\n")
    return _vad_model, _vad_utils

def get_speech_mask(waveform: np.ndarray) -> np.ndarray:
    vad_model, vad_utils = get_vad()
    get_speech_timestamps = vad_utils[0]
    wav_tensor = torch.from_numpy(waveform).float()
    speech_ts = get_speech_timestamps(wav_tensor, vad_model, return_seconds=True, threshold=0.5)
    mask = np.zeros(len(waveform), dtype=bool)
    for seg in speech_ts:
        start = int(seg['start'] * SAMPLE_RATE)
        end = int(seg['end'] * SAMPLE_RATE)
        mask[start:end] = True
    return mask

def setup_logger(video_name: str, output_dir: Path):
    logger = logging.getLogger(video_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
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
        ffmpeg_bin = shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()
        cmd = [ffmpeg_bin, "-y", "-loglevel", "error", "-i", video_path,
               "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", tmp_path]
        subprocess.run(cmd, check=True, capture_output=True)
        data, _ = sf.read(tmp_path, dtype="float32")
        return data
    finally:
        try: os.unlink(tmp_path)
        except: pass

# ====================== CAPTION MAPPER ======================

CAPTION_RULES = [
    # --- specific bird types before generic "bird" ---
    (["caw"], "A crow can be heard cawing in the distance.", "any"),
    (["crow"], "A crow can be heard cawing nearby.", "any"),
    (["bird vocalization", "bird call", "bird song", "bird"],
     "Birds can be heard chirping in the background.", "any"),
    (["fowl", "duck", "rooster", "chicken", "cluck"], "Birds can be heard in the background.", "any"),

    # --- laughter: chuckle/giggle are distinct from belly laugh ---
    (["belly laugh"], "Hearty laughter can be heard.", "any"),
    (["chuckle"], "A soft chuckle can be heard nearby.", "any"),
    (["chortle"], "A soft chuckle can be heard nearby.", "any"),
    (["giggle"], "Giggling can be heard nearby.", "any"),
    (["laughter"], "A woman can be heard laughing softly.", "any"),

    # --- rustling: grass vs leaves vs generic (ALL keywords required) ---
    (["rustl", "grass"], "Grass can be heard rustling nearby.", "all"),
    (["rustl", "leaf"], "Leaves can be heard rustling.", "all"),
    (["rustl", "leaves"], "Leaves can be heard rustling.", "all"),
    (["rustl"], "A soft rustling sound can be heard.", "any"),

    (["wind noise"], "Wind can be heard blowing softly.", "any"),
    (["wind"], "Wind can be heard blowing through the trees.", "any"),
    (["footstep", "walk"], "Footsteps can be heard on the path.", "any"),
    (["cricket"], "Crickets can be heard chirping in the background.", "any"),
    (["insect", "buzz"], "Insects can be heard buzzing nearby.", "any"),
    (["water", "stream", "river"], "Water can be heard flowing nearby.", "any"),
    (["thunder"], "Thunder can be heard rumbling in the distance.", "any"),
    (["rain"], "Rain can be heard falling.", "any"),
    (["animal", "livestock", "cattle", "bull", "cow"], "An animal can be heard nearby.", "any"),
]

MUSIC_LABEL_PHRASING = [
    (["drum"], "the sound of drums"),
    (["plucked string"], "a plucked string instrument"),
    (["tabla"], "tabla"),          # only fires if a future AudioSet/PANNs version adds this class
    (["sitar"], "a sitar"),        # same — included so the mapping just works if it ever appears
    (["flute"], "a flute"),
    (["singing"], "singing"),
    (["musical instrument"], "an instrument playing"),
]

def get_music_caption(label: str) -> str:
    l = label.lower()
    for keywords, phrase in MUSIC_LABEL_PHRASING:
        if any(k in l for k in keywords):
            return f"Music can be heard playing, with {phrase} in the background."
    return "Music can be heard playing in the background."

def get_natural_caption(label: str) -> str:
    l = label.lower()
    if "music" in l:
        return get_music_caption(label)
    for keywords, caption, mode in CAPTION_RULES:
        matched = all(k in l for k in keywords) if mode == "all" else any(k in l for k in keywords)
        if matched:
            return caption
    return f"A {label.lower()} sound can be heard."

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

# ====================== (STAGE 1) VISION MODEL W FLORENCE EXTRACTION ======================
def extract_vision_log(video_path: str, output_path: Path, logger):
    florence_processor, florence_model = get_florence()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(FRAME_SAMPLE_INTERVAL * fps))

    logger.info(f"Extracting vision log: sampling every {FRAME_SAMPLE_INTERVAL}s "
                f"({total} frames @ {fps:.1f}fps)...")

    with open(output_path, "w", encoding="utf-8") as f:
        for idx in range(0, total, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            t = idx / fps
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = florence_processor(text="<MORE_DETAILED_CAPTION>", images=image,
                                         return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                ids = florence_model.generate(input_ids=inputs["input_ids"],
                                               pixel_values=inputs["pixel_values"],
                                               max_new_tokens=800)
            cap_text = florence_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            f.write(json.dumps({"timestamp_sec": round(t, 2), "caption": cap_text}, ensure_ascii=False) + "\n")
            logger.info(f"[{t:.2f}s] Florence Caption: {cap_text}")
    cap.release()
    logger.info(f"Vision log written to: {output_path}")

def load_vision_log(log_path: Path, logger):
    if not log_path or not Path(log_path).exists():
        logger.warning(f"Vision log not found at {log_path}.")
        return []
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                entries.append((float(obj["timestamp_sec"]), obj["caption"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                logger.warning(f"Skipping malformed vision log line: {line[:80]}")
    entries.sort(key=lambda e: e[0])
    logger.info(f"Loaded {len(entries)} timestamped scene captions from vision log.")
    return entries

# ====================== SCENE-FIRST PALETTE INDEX ======================
def build_scene_context_index(captions_with_ts, panns_labels, logger):
    if not captions_with_ts:
        return []

    sentence_model = get_sentence_model()
    class_embs = sentence_model.encode(panns_labels, convert_to_tensor=True)
    entries = []
    for t, cap_text in captions_with_ts:
        palette = predict_scene_palette(cap_text, panns_labels, sentence_model, class_embs)
        entries.append((t, palette, cap_text))
        logger.info(f"[{t:.2f}s] Predicted palette ({len(palette)} classes): "
                    f"{', '.join(list(palette.keys())[:6])}{'...' if len(palette) > 6 else ''}")

    logger.info(f"Built scene-first palette index from {len(entries)} vision log entries.")
    return entries

def get_local_scene_context(t: float, scene_index: list):
    if not scene_index:
        return {}, None
    best = min(scene_index, key=lambda e: abs(e[0] - t))
    if abs(best[0] - t) > SCENE_BOOST_WINDOW_SEC:
        return {}, None
    return best[1], best[2]

# ====================== DETECTION ======================
def detect_audio_events(waveform: np.ndarray, logger, scene_index: list):
    logger.info("Running PANNs against scene-predicted palettes (one best match per window)...")

    panns_model = get_panns()
    resolve_label_thresholds(panns_model.labels)   # populate the runtime threshold cache once
    speech_mask = get_speech_mask(waveform)
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

        start_sec = start / SAMPLE_RATE
        speech_ratio = float(speech_mask[start:end].mean())
        speech_penalty = speech_ratio * 0.5

        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)

        output = panns_model.inference(chunk)
        scores = output[0] if isinstance(output, (list, tuple)) else output['clipwise_output']
        if hasattr(scores, 'ndim') and scores.ndim > 1:
            scores = scores[0]
        palette, matched_scene_caption = get_local_scene_context(start_sec, scene_index)

        top_indices = np.argsort(scores)[::-1][:20]
        best_candidate = None
        best_combined = -1.0

        for idx in top_indices:
            raw_score = float(scores[idx])
            label = panns_model.labels[idx]

            if is_speech_label(label):
                continue

            class_thresh = get_class_threshold(label)
            if raw_score < class_thresh:
                continue

            adjusted_conf = raw_score * (1 - speech_penalty)
            palette_score = palette.get(label, 0.0)

            combined = (W_AUDIO * adjusted_conf) + (W_PALETTE * palette_score)
            if combined > best_combined:
                best_combined = combined
                best_candidate = {
                    "timestamp_sec": round(start_sec, 2),
                    "label": label,
                    "raw_confidence": round(raw_score, 4),
                    "speech_ratio": round(speech_ratio, 3),
                    "adjusted_confidence": round(adjusted_conf, 4),
                    "palette_score": round(palette_score, 4),
                    "combined_score": round(combined, 4),
                    "should_caption": combined >= ACCEPT_THRESHOLD,
                    "matched_scene_caption": matched_scene_caption,
                }

        if best_candidate is not None:
            events.append(best_candidate)
        start += hop_samples

    return events

def dedup_events_per_label(events: list, gap_sec: float = DEDUP_GAP_SEC):
    """Per-label dedup, replacing the old global-winner-only dedup that
    collapsed co-occurring sounds (e.g. birds + wind at the same timestamp)
    down to a single event."""
    by_label = defaultdict(list)
    for ev in events:
        label = ev["label"]
        bucket = by_label[label]
        if not bucket or (ev["timestamp_sec"] - bucket[-1]["timestamp_sec"] > gap_sec):
            bucket.append(ev)
        else:
            if ev["combined_score"] > bucket[-1]["combined_score"]:
                bucket[-1] = ev
    merged = [ev for bucket in by_label.values() for ev in bucket]
    merged.sort(key=lambda e: e["timestamp_sec"])
    return merged

# ====================== FINAL VIDEO GENERATION ======================
def generate_final_video(video_path: str, events: list, output_path: Path, logger):
    accepted = [e for e in events if e.get("should_caption")]
    if not accepted:
        logger.info("No accepted events → skipping final video.")
        return

    logger.info(f"Generating final video with {len(accepted)} caption events "
                f"(max {MAX_SIMULTANEOUS_CAPTIONS} simultaneous on screen)...")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        silent_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(silent_path, fourcc, fps, (width, height))
    timeline = [(e["timestamp_sec"], e["timestamp_sec"] + 2.5, e["caption"]) for e in accepted]
    timeline.sort(key=lambda x: x[0])

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        current_time = frame_idx / fps

        active = [c for c in timeline if c[0] <= current_time < c[1]][:MAX_SIMULTANEOUS_CAPTIONS]
        if active:
            combined_text = "  /  ".join(c[2] for c in active)
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 55), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            thickness = 2
            text_size = cv2.getTextSize(combined_text, font, font_scale, thickness)[0]
            text_x = max(5, (w - text_size[0]) // 2)
            text_y = h - 18
            cv2.putText(frame, combined_text, (text_x, text_y), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()

    try:
        ffmpeg_bin = shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()
        cmd = [ffmpeg_bin, "-y", "-i", silent_path, "-i", video_path,
               "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
               "-shortest", str(output_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Final video saved: {output_path}")
    except Exception as e:
        logger.error(f"Final video merge failed: {e}")
    finally:
        try: os.unlink(silent_path)
        except: pass

# ====================== (STAGE 2) ======================
def process_video(video_path: str, output_dir: Path, florence_log: Path = None):
    video_name = Path(video_path).stem
    logger = setup_logger(video_name, output_dir)

    logger.info("="*70)
    logger.info(f"PROCESSING: {video_name} (v2: calibrated thresholds + vision-log fusion)")
    logger.info("="*70)

    panns_model = get_panns()
    if florence_log and Path(florence_log).exists():
        logger.info(f"Using existing vision log: {florence_log}")
        captions_with_ts = load_vision_log(florence_log, logger)
    else:
        fallback_log_path = florence_log if florence_log else (output_dir / "florence_log.jsonl")
        logger.info(f"No existing vision log found — extracting live and saving to {fallback_log_path} "
                     f"for reuse next time.")
        extract_vision_log(video_path, fallback_log_path, logger)
        captions_with_ts = load_vision_log(fallback_log_path, logger)

    scene_index = build_scene_context_index(captions_with_ts, panns_model.labels, logger)

    waveform = extract_audio(video_path, logger)
    raw_events = detect_audio_events(waveform, logger, scene_index)
    logger.info(f"Raw candidate events after per-class thresholding: {len(raw_events)}")

    deduped = dedup_events_per_label(raw_events)
    logger.info(f"After per-label dedup: {len(deduped)}")

    frames_dir = output_dir / "annotated_frames"
    frames_dir.mkdir(exist_ok=True)
    final_captions = []

    for ev in deduped:
        if ev["should_caption"]:
            ev["caption"] = get_natural_caption(ev["label"])
        else:
            ev["caption"] = None

        status = "ACCEPTED" if ev["should_caption"] else "skipped"
        scene_snippet = (ev.get("matched_scene_caption") or "")[:70]

        logger.info(
            f"{ev['timestamp_sec']:6.2f}s | {ev['label']:<42} | "
            f"raw={ev['raw_confidence']:.3f} | "
            f"adj={ev['adjusted_confidence']:.3f} | "
            f"palette={ev['palette_score']:.3f} | "
            f"combined={ev['combined_score']:.3f} | "
            f"speech={ev['speech_ratio']:.2f} | "
            f"scene='{scene_snippet}' | {status}"
        )

        if ev["should_caption"]:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_no = int(ev['timestamp_sec'] * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame_path = frames_dir / f"frame_{ev['timestamp_sec']:.2f}s.png"
                annotate_frame(frame, ev['caption'], frame_path)
            final_captions.append(ev)

    logger.info(f"\nFinal captions generated: {len(final_captions)}")

    with open(output_dir / "captions.srt", "w", encoding="utf-8") as f:
        for i, ev in enumerate(final_captions, 1):
            s = ev["timestamp_sec"]
            e = s + 2.5
            f.write(f"{i}\n{str(timedelta(seconds=s))[:-3]} --> {str(timedelta(seconds=e))[:-3]}\n{ev['caption']}\n\n")

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_candidate_events": len(raw_events),
            "after_dedup": len(deduped),
            "captions_generated": len(final_captions),
            "events": final_captions,
            "all_events_for_debugging": deduped,
        }, f, indent=2, ensure_ascii=False)

    if GENERATE_FINAL_VIDEO and final_captions:
        generate_final_video(video_path, final_captions, output_dir / "final_output.mp4", logger)

    logger.info(f"\nDone! Results saved in: {output_dir}")

# ====================== CALIBRATION MODE ======================
def run_calibration(video_path: str, output_dir: Path):

    logger = setup_logger(Path(video_path).stem + "_calibration", output_dir)
    logger.info("Running calibration pass (no thresholds, no VAD gating)...")

    panns_model = get_panns()
    waveform = extract_audio(video_path, logger)
    window_samples = int(WINDOW_SEC * SAMPLE_RATE)
    hop_samples = int(HOP_SEC * SAMPLE_RATE)

    score_log = defaultdict(list)
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

        top_indices = np.argsort(scores)[::-1][:15]
        t = start / SAMPLE_RATE
        for idx in top_indices:
            label = panns_model.labels[idx]
            score_log[label].append({"t": round(t, 2), "score": round(float(scores[idx]), 4)})
        start += hop_samples

    report = {}
    for label, entries in score_log.items():
        scores = [e["score"] for e in entries]
        report[label] = {
            "count": len(scores),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "mean": round(float(np.mean(scores)), 4),
            "p50": round(float(np.percentile(scores, 50)), 4),
            "p75": round(float(np.percentile(scores, 75)), 4),
            "p90": round(float(np.percentile(scores, 90)), 4),
            "sample_timestamps": [e["t"] for e in entries[:10]],
        }

    report_path = output_dir / "calibration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Calibration report saved to {report_path}")
    logger.info("Next step: manually scrub through the clip with the timestamps "
                "above, mark which detections a human would actually caption, "
                "and set CLASS_THRESHOLDS in the script from the resulting "
                "percentiles (e.g. p75 of clips you marked 'should caption').")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--florence-log", type=str, default=None,
                         help="Path to a timestamped JSONL vision log (read in Stage 2, "
                              "or written here if --extract-vision is also passed).")
    parser.add_argument("--extract-vision", action="store_true",
                         help="Stage 1 only: run Florence-2 over the video and write a "
                              "timestamped JSONL vision log to --florence-log (or a default "
                              "path under the output dir), then exit without running PANNs.")
    parser.add_argument("--calibrate", action="store_true",
                         help="Run PANNs calibration mode instead of the full pipeline.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    output_dir = Path("panns_v2_results") / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.extract_vision:
        log_path = Path(args.florence_log) if args.florence_log else (output_dir / "florence_log.jsonl")
        logger = setup_logger(video_path.stem + "_vision_extract", output_dir)
        extract_vision_log(str(video_path), log_path, logger)
        logger.info(f"\nStage 1 complete. Reuse this log with:\n"
                     f"  python {Path(__file__).name} --video {video_path} --florence-log {log_path}")
    elif args.calibrate:
        run_calibration(str(video_path), output_dir)
    else:
        process_video(str(video_path), output_dir,
                      Path(args.florence_log) if args.florence_log else None)

if __name__ == "__main__":
    main()