import os, sys, subprocess, tempfile, shutil, json, logging, re, argparse
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import numpy as np
import soundfile as sf
import cv2
import torch
import torchaudio
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from panns_inference import AudioTagging
from dotenv import load_dotenv   # ADD THIS
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ====================== GPU ======================
assert torch.cuda.is_available(), "CUDA required"
DEVICE = "cuda"
print(f"\nGPU: {torch.cuda.get_device_name(0)}\n")

# ====================== CONFIG ======================
SAMPLE_RATE          = 32000
WINDOW_SEC           = 0.96
HOP_SEC              = 0.20
FRAMES_PER_MINUTE    = 20
SCENE_WINDOW_SEC     = 8.0
DEDUP_GAP_SEC        = 1.5
BURST_GAP_SEC        = 2.0       # events closer than this merge into one caption
PALETTE_SIZE         = 20
SEMANTIC_THRESHOLD   = 0.17
W_AUDIO              = 0.55
W_PALETTE            = 0.45
ACCEPT_THRESHOLD     = 0.12      # lowered slightly; speech gate is now hard not soft
FRAME_STABILITY_THRESHOLD = 25.0
USE_NOISE_REDUCTION  = True
GENERATE_FINAL_VIDEO = True


# ---- Per-class thresholds ----
CLASS_THRESHOLD_KEYWORDS = [
    (["caw"],                               0.022),
    (["crow"],                              0.028),
    (["bird vocalization", "bird call"],    0.028),
    (["bird"],                              0.032),
    (["rustl"],                             0.028),
    (["creak", "creaking"],                 0.028),
    (["wood"],                              0.032),
    (["cricket"],                           0.028),
    (["insect"],                            0.048),
    (["wind noise"],                        0.045),
    (["wind"],                              0.038),
    (["laughter"],                          0.030),
    (["giggle"],                            0.028),
    (["chuckle", "chortle"],                0.032),
    (["belly laugh"],                       0.035),
    (["walk", "footstep"],                  0.028),
    (["run", "jog"],                        0.042),
    (["music"],                             0.095),
    (["musical instrument"],                0.060),
    (["drum"],                              0.048),
    (["plucked string"],                    0.048),
    (["singing"],                           0.075),
    (["crowd", "cheering", "chatter"],      0.055),
    (["water", "stream", "splash", "rain"], 0.038),
    (["thunder"],                           0.045),
    (["fire", "crackling"],                 0.032),
    (["animal", "cattle", "dog", "cat"],    0.038),
]
DEFAULT_THRESHOLD = 0.052
_threshold_cache: dict = {}

def resolve_label_thresholds(panns_labels: list) -> dict:
    global _threshold_cache
    if _threshold_cache:
        return _threshold_cache
    resolved = {}
    for label in panns_labels:
        ll = label.lower()
        for keywords, thresh in CLASS_THRESHOLD_KEYWORDS:
            if any(k in ll for k in keywords):
                if label not in resolved or thresh < resolved[label]:
                    resolved[label] = thresh
    _threshold_cache = resolved
    return resolved

def get_class_threshold(label: str) -> float:
    return _threshold_cache.get(label, DEFAULT_THRESHOLD)

SPEECH_SUBSTRINGS = {
    "speech", "male speech", "female speech", "conversation", "narration",
    "monologue", "dialogue", "voice", "talking", "child speech", "babbling",
    "whispering", "shout", "yell", "screaming",
}

def is_speech_label(label: str) -> bool:
    ll = label.lower()
    return any(s in ll for s in SPEECH_SUBSTRINGS)

# ====================== SCENE VOCABULARY ======================
SCENE_SOUND_HINTS = [
    {
        "name": "forest_outdoor",
        "detect_keywords": ["forest", "jungle", "trees", "woods", "outdoor", "path",
                             "nature", "leaves", "branches", "vegetation", "grass",
                             "field", "meadow", "hill"],
        "sound_hints": ["birds chirping", "wind rustling leaves", "crickets",
                        "crow cawing", "branch creak", "insects buzzing",
                        "footsteps on dirt", "rustling grass"],
    },
    {
        "name": "indoor_room",
        "detect_keywords": ["room", "indoors", "inside", "house", "wall", "ceiling",
                             "furniture", "table", "chair", "floor", "lamp", "window"],
        "sound_hints": ["door creak", "footsteps on floor", "clock ticking",
                        "distant voices", "fan humming"],
    },
    {
        "name": "water_scene",
        "detect_keywords": ["river", "stream", "lake", "ocean", "sea", "pond",
                             "waterfall", "rain", "water", "boat", "shore", "beach"],
        "sound_hints": ["water flowing", "water splashing", "rain falling",
                        "frogs croaking", "wind over water"],
    },
    {
        "name": "crowd_public",
        "detect_keywords": ["crowd", "market", "street", "bazaar", "gathering",
                             "festival", "ceremony", "procession"],
        "sound_hints": ["crowd chatter", "music playing", "children playing",
                        "bells ringing"],
    },
    {
        "name": "night_scene",
        "detect_keywords": ["night", "dark", "moonlight", "stars", "dusk", "evening",
                             "candle", "firelight"],
        "sound_hints": ["crickets chirping", "owl hooting", "night insects",
                        "distant music", "crackling fire", "wind"],
    },
    {
        "name": "village_rural",
        "detect_keywords": ["village", "hut", "mud", "rural", "farm", "cattle",
                             "well", "bullock", "cart"],
        "sound_hints": ["cattle lowing", "rooster crowing", "bells", "wind",
                        "birds", "distant music"],
    },
    {
        "name": "temple_religious",
        "detect_keywords": ["temple", "shrine", "idol", "incense", "prayer",
                             "priest", "worship", "ritual", "sacred"],
        "sound_hints": ["bells ringing", "chanting", "music", "crowd murmur",
                        "wind", "birds"],
    },
    {
        "name": "battle_conflict",
        "detect_keywords": ["battle", "war", "fight", "weapon", "sword", "army",
                             "soldier", "attack", "conflict"],
        "sound_hints": ["crowd shouting", "metal clashing", "drums", "horses",
                        "running", "wind"],
    },
]

def detect_scene_hints(text: str) -> list:
    tl = text.lower()
    hints = []
    for scene in SCENE_SOUND_HINTS:
        if any(kw in tl for kw in scene["detect_keywords"]):
            hints.extend(scene["sound_hints"])
    return list(set(hints))

CLOTHING_WORDS = {
    "wearing", "dressed", "saari", "sari", "hoodie", "cloak", "robe", "turban",
    "garment", "cloth", "outfit", "attire", "costume", "dhoti", "kurta", "dupatta",
    "shawl", "veil", "uniform", "armor", "armour", "helmet", "crown",
    "hair", "skin", "complexion", "beard", "mustache", "bald", "tall", "short",
    "thin", "fat", "muscular", "elderly", "wrinkled",
}
CLOTHING_COLOUR_PATTERNS = [
    "wearing a", "dressed in", "clothed in", "garment is", "robe is",
    "white cloth", "red cloth", "blue cloth", "saffron robe", "yellow robe",
]
ACTION_WORDS = {
    "walking", "running", "standing", "sitting", "laughing", "smiling", "crying",
    "shouting", "fighting", "dancing", "talking", "looking", "approaching",
    "fleeing", "gesturing", "praying", "angry", "scared", "joyful", "worried",
    "surprised", "concerned", "serious", "calm", "distressed",
}

def filter_florence_for_scene(text: str) -> str:
    sentences = [s.strip() for s in text.replace(".", ". ").split(".") if len(s.strip()) > 5]
    kept = []
    for sent in sentences:
        words = set(sent.lower().split())
        has_clothing = bool(words & CLOTHING_WORDS) or any(
            p in sent.lower() for p in CLOTHING_COLOUR_PATTERNS)
        has_action = bool(words & ACTION_WORDS)
        if has_clothing and not has_action:
            continue
        kept.append(sent)
    result = ". ".join(kept).strip()
    return result if len(result) > 20 else text

# ====================== LAZY MODEL LOADING ======================
_florence_processor = None
_florence_model = None
_sentence_model = None
_panns_model = None
_silero_model = None
_silero_utils=None

def get_florence():
    global _florence_processor, _florence_model
    if _florence_model is None:
        print("Loading Florence-2-large...")
        _florence_processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True)
        _florence_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", torch_dtype=torch.float16,
            trust_remote_code=True).to(DEVICE)
        print("Florence-2 loaded.\n")
    return _florence_processor, _florence_model

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        print("Loading Sentence Transformer...")
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
        print("Sentence Transformer loaded.\n")
    return _sentence_model

def get_panns():
    global _panns_model
    if _panns_model is None:
        print("Loading PANNs...")
        _panns_model = AudioTagging(checkpoint_path=None, device=DEVICE)
        print("PANNs loaded.\n")
    return _panns_model

def get_silero_vad():
    global _silero_model, _silero_utils
    if _silero_model is None:
        print("Loading Silero VAD (reliable speech detection)...")
        torch.set_num_threads(1)
        _silero_model, _silero_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        _silero_model = _silero_model.to(DEVICE)
        print("Silero VAD loaded successfully.\n")
    return _silero_model, _silero_utils


def get_speech_segments(wav_path: str, logger) -> list:
    """
    Reliable speech segmentation using Silero VAD.
    Much more stable than pyannote on Windows.
    """
    model, utils = get_silero_vad()
    (get_speech_timestamps, _, _, _, _) = utils

    logger.info(f"Running Silero VAD on {wav_path}...")

    # Load audio
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform.to(DEVICE)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=sample_rate,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
        speech_pad_ms=400,
    )

    segments = []
    for seg in speech_timestamps:
        start = seg['start'] / sample_rate
        end = seg['end'] / sample_rate
        segments.append((start, end))

    # Merge overlapping segments
    if segments:
        segments.sort()
        merged = [segments[0]]
        for s, e in segments[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        segments = merged

    total_speech = sum(e - s for s, e in segments)
    logger.info(f"✅ Silero detected {len(segments)} speech segments "
                f"({total_speech:.1f}s total speech)")

    if len(segments) == 0:
        logger.warning("⚠️ Silero found ZERO speech segments!")

    return segments


# ====================== SPEECH DETECTION (PYANNOTE) ======================

def build_speech_mask_from_segments(segments: list, n_samples: int) -> np.ndarray:
    """Convert (start_sec, end_sec) list to a sample-level boolean mask."""
    mask = np.zeros(n_samples, dtype=bool)
    for start, end in segments:
        s = int(start * SAMPLE_RATE)
        e = min(n_samples, int(end * SAMPLE_RATE))
        mask[s:e] = True
    return mask

def is_speech_window(start_sec: float, speech_segments: list,
                     threshold_ratio: float = 0.25) -> bool:
    end_sec = start_sec + WINDOW_SEC
    overlap = 0.0
    for seg_start, seg_end in speech_segments:
        overlap += max(0.0, min(end_sec, seg_end) - max(start_sec, seg_start))
    return (overlap / WINDOW_SEC) >= threshold_ratio

# ====================== NOISE REDUCTION ======================
def apply_noise_reduction(waveform: np.ndarray) -> np.ndarray:
    if not USE_NOISE_REDUCTION:
        return waveform
    try:
        import noisereduce as nr
        n = len(waveform)
        candidates = [
            waveform[:SAMPLE_RATE],
            waveform[n//4: n//4 + SAMPLE_RATE],
            waveform[n//2: n//2 + SAMPLE_RATE],
        ]
        noise_clip = min(candidates, key=lambda c: float(np.sqrt(np.mean(c**2))))
        return nr.reduce_noise(
            y=waveform, y_noise=noise_clip, sr=SAMPLE_RATE,
            prop_decrease=0.6, stationary=True, n_fft=1024,
        ).astype(np.float32)
    except ImportError:
        return waveform

# ====================== HELPERS ======================
def setup_logger(name: str, output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(output_dir / f"{name}.log", mode='w', encoding='utf-8')
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

def extract_audio(video_path: str, logger) -> tuple:
    logger.info(f"Extracting audio: {video_path}")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    ffmpeg_bin = shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()
    subprocess.run([
        ffmpeg_bin, "-y", "-loglevel", "error", "-i", video_path,
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", tmp.name
    ], check=True, capture_output=True)
    data, _ = sf.read(tmp.name, dtype="float32")
    return data, tmp.name

# ====================== CAPTION RENDERING ======================
def render_caption(frame: np.ndarray, text: str) -> np.ndarray:
    """Renders text on frame with automatic two-line wrapping and adaptive
    font scaling. Black outline ensures readability on any background."""
    h, w = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    thick = 2

    def text_width(t, scale):
        return cv2.getTextSize(t, font, scale, thick)[0][0]

    # Find the largest scale that fits the full text in one line
    scale = 0.75
    for s in [0.75, 0.68, 0.60, 0.52]:
        if text_width(text, s) <= w - 40:
            scale = s
            break

    # If still too wide, split into two lines
    lines = [text]
    if text_width(text, scale) > w - 40:
        words = text.split()
        mid = len(words) // 2
        for attempt in range(min(4, len(words))):
            for offset in [attempt, -attempt]:
                idx = mid + offset
                if 1 <= idx < len(words):
                    l1 = " ".join(words[:idx])
                    l2 = " ".join(words[idx:])
                    if text_width(l1, scale) <= w - 40 and text_width(l2, scale) <= w - 40:
                        lines = [l1, l2]
                        break
            else:
                continue
            break

    line_h   = cv2.getTextSize("A", font, scale, thick)[0][1]
    bar_h    = len(lines) * (line_h + 14) + 18
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    for i, line in enumerate(lines):
        tw = text_width(line, scale)
        tx = max(20, (w - tw) // 2)
        ty = h - 16 - (len(lines) - 1 - i) * (line_h + 14)
        # Outline pass
        cv2.putText(frame, line, (tx, ty), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        # Text pass
        cv2.putText(frame, line, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return frame

def annotate_frame(frame: np.ndarray, text: str, path: Path):
    render_caption(frame.copy(), text)
    cv2.imwrite(str(path), frame)

# ====================== STAGE 1: VISION LOG ======================
def extract_vision_log(video_path: str, output_path: Path, logger):
    fp, fm = get_florence()
    cap  = cv2.VideoCapture(video_path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur  = tot / fps / 60.0
    n    = max(1, int(dur * FRAMES_PER_MINUTE))
    idxs = np.linspace(0, tot - 1, n, dtype=int)

    logger.info(f"Vision extraction: {n} frames over {dur:.1f} min "
                f"({FRAMES_PER_MINUTE} FPM)")

    prev_gray, written = None, 0
    with open(output_path, "w", encoding="utf-8") as f:
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret: continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
            if prev_gray is not None:
                diff = np.abs(gray - prev_gray).mean()
                if diff > FRAME_STABILITY_THRESHOLD:
                    logger.info(f"[{idx/fps:.1f}s] skip unstable frame (diff={diff:.1f})")
                    prev_gray = gray
                    continue
            prev_gray = gray

            t     = idx / fps
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            def run_florence(prompt):
                inputs = fp(text=prompt, images=image,
                            return_tensors="pt").to(DEVICE, torch.float16)
                with torch.no_grad():
                    ids = fm.generate(input_ids=inputs["input_ids"],
                                      pixel_values=inputs["pixel_values"],
                                      max_new_tokens=512)
                return fp.batch_decode(ids, skip_special_tokens=True)[0].strip()

            scene_cap  = run_florence("<MORE_DETAILED_CAPTION>")
            action_cap = run_florence("<DETAILED_CAPTION>")

            # Merge action sentences not already in scene caption
            for sent in [s.strip() for s in action_cap.replace(".", ". ").split(".")
                         if len(s.strip()) > 10]:
                if sent.lower() not in scene_cap.lower():
                    scene_cap += " " + sent

            scene_text  = filter_florence_for_scene(scene_cap)
            scene_hints = detect_scene_hints(scene_cap)
            expressions = [w for w in [
                "laughing","smiling","crying","angry","scared","worried",
                "concerned","surprised","joyful","distressed","shouting",
            ] if w in scene_cap.lower()]

            entry = {
                "timestamp_sec": round(t, 2),
                "raw_caption":   scene_cap,
                "scene_text":    scene_text,
                "scene_hints":   scene_hints,
                "expressions":   expressions,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info(f"[{t:.1f}s] scene='{scene_text[:80]}' | "
                        f"hints={scene_hints[:3]} | expr={expressions}")
            written += 1

    cap.release()
    logger.info(f"Vision log: {written} frames → {output_path}")

def load_vision_log(log_path: Path, logger) -> list:
    if not log_path or not Path(log_path).exists():
        logger.warning(f"Vision log not found: {log_path}")
        return []
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                entries.append({
                    "t":           float(obj["timestamp_sec"]),
                    "scene_text":  obj.get("scene_text") or obj.get("caption", ""),
                    "scene_hints": obj.get("scene_hints", []),
                    "expressions": obj.get("expressions", []),
                    "raw_caption": obj.get("raw_caption", ""),
                })
            except Exception as e:
                logger.warning(f"Bad log line: {line[:60]} ({e})")
    entries.sort(key=lambda e: e["t"])
    logger.info(f"Loaded {len(entries)} vision entries.")
    return entries

# ====================== PALETTE PREDICTION ======================
def predict_scene_palette(entry: dict, panns_labels: list,
                           sentence_model, class_embs) -> dict:
    scene_text  = entry["scene_text"]
    scene_hints = entry["scene_hints"]
    hint_str    = (". Ambient sounds expected: " + ", ".join(scene_hints)) if scene_hints else ""
    query       = scene_text + hint_str

    emb  = sentence_model.encode(query, convert_to_tensor=True)
    sims = util.cos_sim(emb, class_embs)[0]

    palette = {}
    for i, label in enumerate(panns_labels):
        if is_speech_label(label): continue
        sim = float(sims[i])
        if sim >= SEMANTIC_THRESHOLD:
            palette[label] = sim

    # Direct hint injection — ensures forest scene explicitly boosts crow, rustle, creak
    for hint in scene_hints:
        hl = hint.lower()
        for label in panns_labels:
            if is_speech_label(label): continue
            if any(word in label.lower() for word in hl.split() if len(word) > 3):
                if label not in palette or palette[label] < SEMANTIC_THRESHOLD:
                    palette[label] = SEMANTIC_THRESHOLD

    sorted_p = sorted(palette.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_p[:PALETTE_SIZE])

def build_scene_index(vision_entries: list, panns_labels: list, logger) -> list:
    if not vision_entries: return []
    sm         = get_sentence_model()
    class_embs = sm.encode(panns_labels, convert_to_tensor=True)
    index      = []
    for entry in vision_entries:
        palette = predict_scene_palette(entry, panns_labels, sm, class_embs)
        index.append({
            "t":           entry["t"],
            "palette":     palette,
            "scene_text":  entry["scene_text"],
            "expressions": entry.get("expressions", []),
        })
        logger.info(f"[{entry['t']:.1f}s] palette top-5: "
                    f"{list(palette.keys())[:5]} | expr={entry.get('expressions', [])}")
    logger.info(f"Scene index: {len(index)} entries")
    return index

_EMPTY_SCENE = {"palette": {}, "scene_text": "", "expressions": []}

def get_scene_at(t: float, scene_index: list) -> dict:
    if not scene_index: return _EMPTY_SCENE
    best = min(scene_index, key=lambda e: abs(e["t"] - t))
    if abs(best["t"] - t) > SCENE_WINDOW_SEC: return _EMPTY_SCENE
    return best

# ====================== DETECTION ======================
def detect_audio_events(waveform: np.ndarray, scene_index: list,
                         speech_segments: list, logger) -> list:
    logger.info("=" * 65)
    logger.info("AUDIO ANALYSIS — PANNs detection with pyannote speech gate")
    logger.info("=" * 65)

    panns_model = get_panns()
    resolve_label_thresholds(panns_model.labels)

    clean  = apply_noise_reduction(waveform)
    logger.info(f"Noise reduction: {'applied' if USE_NOISE_REDUCTION else 'skipped'}")

    win_samples = int(WINDOW_SEC * SAMPLE_RATE)
    hop_samples = int(HOP_SEC * SAMPLE_RATE)
    events      = []
    n           = len(clean)
    start       = 0

    while start < n:
        start_sec = start / SAMPLE_RATE
        chunk     = clean[start:start + win_samples]
        if len(chunk) < win_samples:
            chunk = np.pad(chunk, (0, win_samples - len(chunk)))

        # ---- HARD SPEECH GATE ----
        # If pyannote flagged this window as speech (even 25% overlap),
        # skip it entirely. No ambient caption is ever shown during speech.
        if is_speech_window(start_sec, speech_segments, threshold_ratio=0.25):
            logger.debug(f"[{start_sec:.2f}s] SPEECH — skipped")
            start += hop_samples
            continue

        scene   = get_scene_at(start_sec, scene_index)
        palette = scene["palette"]

        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        out    = panns_model.inference(chunk)
        scores = out[0] if isinstance(out, (list, tuple)) else out["clipwise_output"]
        if hasattr(scores, "ndim") and scores.ndim > 1:
            scores = scores[0]

        top_idx    = np.argsort(scores)[::-1][:20]
        best       = None
        best_score = -1.0
        candidates = []

        for idx in top_idx:
            raw   = float(scores[idx])
            label = panns_model.labels[idx]
            if is_speech_label(label): continue

            thresh = get_class_threshold(label)
            if raw < thresh: continue

            # No VAD penalty — speech windows are hard-gated above, so any
            # window reaching here is definitively non-speech. Applying a
            # speech penalty here just hurts recall for no benefit.
            palette_score = palette.get(label, 0.0)
            combined      = W_AUDIO * raw + W_PALETTE * palette_score
            passed        = combined >= ACCEPT_THRESHOLD

            candidates.append({
                "label": label, "raw": round(raw, 4),
                "palette": round(palette_score, 4),
                "combined": round(combined, 4), "pass": passed,
            })

            if combined > best_score:
                best_score = combined
                best = {
                    "timestamp_sec":      round(start_sec, 2),
                    "label":              label,
                    "raw_confidence":     round(raw, 4),
                    "palette_score":      round(palette_score, 4),
                    "combined_score":     round(combined, 4),
                    "should_caption":     passed,
                    "scene_text":         scene["scene_text"][:120],
                    "expressions":        scene.get("expressions", []),
                }

        # ---- PER-WINDOW LOG ----
        scene_snippet = scene["scene_text"][:70]
        logger.info(
            f"\n[{start_sec:6.2f}s] AMBIENT | scene='{scene_snippet}'"
        )
        if candidates:
            logger.info(f"  {'LABEL':<42} {'RAW':>6} {'PAL':>6} {'CMB':>6} PASS")
            for c in sorted(candidates, key=lambda x: -x["combined"])[:6]:
                logger.info(
                    f"  {c['label']:<42} {c['raw']:>6.3f} "
                    f"{c['palette']:>6.3f} {c['combined']:>6.3f} "
                    f"{'✓' if c['pass'] else '✗'}"
                )
        if best and best["should_caption"]:
            logger.info(f"  → WINNER: {best['label']} "
                        f"(combined={best['combined_score']:.3f})")
        else:
            logger.info("  → no winner")

        if best:
            events.append(best)

        start += hop_samples

    accepted = sum(1 for e in events if e["should_caption"])
    logger.info(f"\nDetection done: {len(events)} windows, {accepted} accepted")
    return events

# ====================== DEDUP ======================
def dedup_events(events: list) -> list:
    by_label = defaultdict(list)
    for ev in events:
        lbl    = ev["label"]
        bucket = by_label[lbl]
        if not bucket or (ev["timestamp_sec"] - bucket[-1]["timestamp_sec"] > DEDUP_GAP_SEC):
            bucket.append(ev)
        elif ev["combined_score"] > bucket[-1]["combined_score"]:
            bucket[-1] = ev
    merged = [ev for b in by_label.values() for ev in b]
    merged.sort(key=lambda e: e["timestamp_sec"])
    return merged

# ====================== OPENAI CAPTION REFINEMENT ======================
def _gender_from_scene(scene_text: str) -> str:
    st = scene_text.lower()
    w  = bool(re.search(r"\b(woman|female|girl|lady|she|her)\b", st))
    m  = bool(re.search(r"\b(man|male|boy|he|his)\b", st))
    if w and not m: return "female"
    if m and not w: return "male"
    return "unknown"

def refine_caption_with_openai(raw_caption: str, scene_text: str,
                                 detected_labels: list,
                                 expressions: list, logger) -> str:
    if not OPENAI_API_KEY:
        return raw_caption

    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        gender = _gender_from_scene(scene_text)
        gender_note = f" The person in frame appears to be {gender}." if gender != "unknown" else ""

        prompt = (
            f"You are writing closed captions for a deaf viewer watching an old Indian TV drama. "
            f"Rewrite the following as ONE natural, clear English sentence. "
            f"Maximum 12 words. Do not invent sounds not in the detected list. "
            f"Use present tense. Be specific and natural — write as a professional subtitler would.\n\n"
            f"Detected audio labels: {', '.join(detected_labels)}\n"
            f"Scene description: {scene_text[:200]}\n"
            f"Visible expressions: {', '.join(expressions) if expressions else 'none'}\n"
            f"{gender_note}\n"
            f"Draft caption: {raw_caption}\n\n"
            f"Refined caption (ONE sentence, max 12 words, no quotes):"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.2,
        )
        refined = response.choices[0].message.content.strip().strip('"').strip("'")
        # Sanity check: must be a sentence, not too long
        if refined and len(refined.split()) <= 16 and len(refined) > 5:
            logger.info(f"  GPT refined: '{raw_caption}' → '{refined}'")
            return refined
        return raw_caption
    except Exception as e:
        logger.warning(f"OpenAI refinement failed: {e} — using rule-based caption")
        return raw_caption

# ====================== RULE-BASED CAPTION (FALLBACK) ======================
SOUND_FAMILIES = {
    "crow":       ["crow", "caw"],
    "bird":       ["bird vocalization", "bird call", "bird song", "bird",
                   "fowl", "rooster", "chicken"],
    "laugh_soft": ["chuckle", "chortle", "giggle"],
    "laugh_full": ["laughter", "belly laugh"],
    "cricket":    ["cricket"],
    "insect":     ["insect", "buzz"],
    "wind":       ["wind"],
    "rustling":   ["rustl"],
    "creak":      ["creak", "wood"],
    "footstep":   ["footstep", "walk", "run", "jog"],
    "water":      ["water", "stream", "river", "splash", "rain"],
    "thunder":    ["thunder"],
    "fire":       ["fire", "crackling"],
    "music":      ["music", "musical instrument", "drum", "plucked string",
                   "singing", "flute", "string"],
    "crowd":      ["crowd", "cheering", "chatter"],
    "animal":     ["cattle", "cow", "bull", "dog", "bark", "cat", "horse",
                   "neigh", "frog", "animal"],
    "bell":       ["bell", "ring"],
}

def _family(label: str) -> str:
    ll = label.lower()
    for fam, kws in SOUND_FAMILIES.items():
        if any(k in ll for k in kws):
            return fam
    return ll

FAMILY_CAPTION_MAP = {
    "crow":       "A crow can be heard cawing in the distance.",
    "bird":       "Birds can be heard chirping in the background.",
    "laugh_soft": "A soft chuckle can be heard nearby.",
    "laugh_full": "Laughter can be heard nearby.",
    "cricket":    "Crickets can be heard chirping in the background.",
    "insect":     "Insects can be heard buzzing nearby.",
    "wind":       "Wind can be heard blowing through the trees.",
    "rustling":   "Leaves and grass can be heard rustling.",
    "creak":      "Branches can be heard creaking in the wind.",
    "footstep":   "Footsteps can be heard on the path.",
    "water":      "Water can be heard flowing nearby.",
    "thunder":    "Thunder can be heard rumbling in the distance.",
    "fire":       "A fire can be heard crackling.",
    "music":      "Music can be heard playing in the background.",
    "crowd":      "A crowd can be heard in the background.",
    "bell":       "Bells can be heard ringing in the distance.",
}

def _rule_caption(families: list, best_labels: dict,
                   scene_text: str, expressions: list) -> str:
    fset   = set(families)
    outdoor = any(x in scene_text.lower()
                  for x in ["forest","outdoor","trees","jungle","nature","path"])
    gender  = _gender_from_scene(scene_text)

    # Laughter — gender-aware
    if "laugh_soft" in fset or "laugh_full" in fset:
        soft = "laugh_soft" in fset and "laugh_full" not in fset
        if gender == "female":
            return ("A woman can be heard laughing softly."
                    if soft else "A woman can be heard laughing.")
        if gender == "male":
            return ("A man can be heard chuckling."
                    if soft else "A man can be heard laughing.")
        return "A soft chuckle can be heard." if soft else "Laughter can be heard."

    # Music
    if "music" in fset:
        ll = best_labels.get("music", "").lower()
        if "drum" in ll:            return "Music with drums can be heard."
        if "plucked string" in ll:  return "Music with a string instrument can be heard."
        if "flute" in ll:           return "Music with a flute can be heard softly."
        if "sing" in ll:            return "Singing can be heard in the background."
        return "Music can be heard playing in the background."

    # Walking through undergrowth — audio+vision synthesis
    walking = any(x in scene_text.lower()
                  for x in ["walk","path","trail","moving","strolling","approaching"])
    if outdoor and walking and fset & {"rustling", "creak", "footstep"}:
        ambient = fset & {"bird","crow","cricket","insect","wind"}
        if "crow" in ambient:
            return ("Footsteps through undergrowth can be heard, "
                    "with a crow cawing in the distance.")
        if "bird" in ambient:
            return ("Footsteps and rustling through the undergrowth, "
                    "with birds in the background.")
        return "Footsteps and rustling can be heard through the undergrowth."

    # Multi-sound combos
    if "crow" in fset and "bird" in fset:
        if "wind" in fset:
            return "Crows cawing, birds chirping, and wind can be heard."
        return "Crows cawing and birds chirping can be heard."
    if "bird" in fset and "wind" in fset:
        return "Birds can be heard chirping as wind blows through the trees."
    if "bird" in fset and "cricket" in fset:
        return "Birds chirping and crickets can be heard in the background."
    if "rustling" in fset and "wind" in fset:
        return "Wind can be heard rustling through the trees."
    if "rustling" in fset and "creak" in fset:
        return "Leaves rustling and branches creaking can be heard."
    if "crow" in fset and "wind" in fset:
        return "A crow cawing and wind blowing can be heard."
    if "water" in fset and "bird" in fset:
        return "Water flowing and birds chirping can be heard."

    # Single family
    fam = families[0] if families else "unknown"
    return FAMILY_CAPTION_MAP.get(fam, f"Ambient sounds can be heard.")

# ====================== BURST CONSOLIDATION ======================
def build_timeline(events: list, scene_index: list, logger) -> list:
    """Groups dedup'd events into bursts (BURST_GAP_SEC), generates one
    rule-based caption per burst, then refines with GPT-4o-mini."""
    accepted = [ev for ev in events if ev.get("should_caption")]
    if not accepted:
        return []

    accepted.sort(key=lambda e: e["timestamp_sec"])

    # Group into bursts
    bursts, cur = [], [accepted[0]]
    for ev in accepted[1:]:
        if ev["timestamp_sec"] - cur[-1]["timestamp_sec"] <= BURST_GAP_SEC:
            cur.append(ev)
        else:
            bursts.append(cur); cur = [ev]
    bursts.append(cur)

    logger.info(f"Burst grouping: {len(accepted)} events → {len(bursts)} bursts")
    logger.info("\nFINAL CAPTION SYNTHESIS:")
    logger.info("=" * 65)

    timeline = []
    for burst in bursts:
        start_sec = burst[0]["timestamp_sec"]
        end_sec   = burst[-1]["timestamp_sec"] + 1.8

        # Best label per family
        fam_best = {}
        for ev in sorted(burst, key=lambda e: -e["combined_score"]):
            fam = _family(ev["label"])
            if fam not in fam_best:
                fam_best[fam] = ev

        top_fams    = sorted(fam_best, key=lambda f: fam_best[f]["combined_score"],
                             reverse=True)[:3]
        best_labels = {f: fam_best[f]["label"] for f in top_fams}
        all_labels  = [fam_best[f]["label"] for f in top_fams]

        # Use scene from highest-scoring event
        best_ev    = max(burst, key=lambda e: e["combined_score"])
        scene_text = best_ev.get("scene_text", "")
        expressions= best_ev.get("expressions", [])

        # 1. Rule-based caption
        raw_caption = _rule_caption(top_fams, best_labels, scene_text, expressions)

        # 2. GPT-4o-mini refinement
        final_caption = refine_caption_with_openai(
            raw_caption, scene_text, all_labels, expressions, logger)

        timeline.append({
            "start_sec":   round(start_sec, 2),
            "end_sec":     round(end_sec, 2),
            "caption":     final_caption,
            "raw_caption": raw_caption,
            "families":    top_fams,
            "scene_text":  scene_text[:100],
        })
        logger.info(
            f"  [{start_sec:.1f}→{end_sec:.1f}s] families={top_fams}\n"
            f"    raw='{raw_caption}'\n"
            f"    final='{final_caption}'"
        )

    logger.info("=" * 65)
    return timeline

# ====================== SRT ======================
def _srt_ts(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{int((sec%1)*1000):03d}"

def write_srt(timeline: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(timeline, 1):
            f.write(f"{i}\n{_srt_ts(seg['start_sec'])} --> "
                    f"{_srt_ts(seg['end_sec'])}\n{seg['caption']}\n\n")

# ====================== FINAL VIDEO ======================
def generate_final_video(video_path: str, timeline: list,
                          output_path: Path, logger):
    if not timeline:
        logger.info("Empty timeline — skipping video."); return
    logger.info(f"Generating video with {len(timeline)} caption segments...")

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        silent_path = tmp.name
    out = cv2.VideoWriter(silent_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    tl  = sorted(timeline, key=lambda s: s["start_sec"])

    for fi in range(total):
        ret, frame = cap.read()
        if not ret: break
        t      = fi / fps
        active = next((s for s in tl if s["start_sec"] <= t < s["end_sec"]), None)
        if active:
            frame = render_caption(frame, active["caption"])
        out.write(frame)

    cap.release(); out.release()
    try:
        ffmpeg_bin = shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()
        subprocess.run([
            ffmpeg_bin, "-y", "-i", silent_path, "-i", video_path,
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(output_path)
        ], check=True, capture_output=True)
        logger.info(f"Video saved: {output_path}")
    except Exception as e:
        logger.error(f"Video merge failed: {e}")
    finally:
        try: os.unlink(silent_path)
        except: pass

# ====================== MAIN PIPELINE ======================
def process_video(video_path: str, output_dir: Path,
                   florence_log: Path = None):
    vname  = Path(video_path).stem
    logger = setup_logger(vname, output_dir)
    logger.info("=" * 70)
    logger.info(f"PROCESSING: {vname} (v4 — pyannote + GPT-4o-mini)")
    logger.info("=" * 70)

    panns_model = get_panns()

    # ---- Vision ----
    if florence_log and Path(florence_log).exists():
        vision_entries = load_vision_log(florence_log, logger)
    else:
        fallback = florence_log or (output_dir / "florence_log.jsonl")
        logger.info(f"No vision log — extracting live to {fallback}")
        extract_vision_log(video_path, str(fallback), logger)
        vision_entries = load_vision_log(fallback, logger)

    scene_index = build_scene_index(vision_entries, panns_model.labels, logger)

    # ---- Audio extraction ----
    waveform, wav_path = extract_audio(video_path, logger)

    # ---- Speech detection (pyannote) ----
    speech_segments = get_speech_segments(wav_path, logger)
    try: os.unlink(wav_path)
    except: pass

    # ---- PANNs detection ----
    raw_events = detect_audio_events(
        waveform, scene_index, speech_segments, logger)
    logger.info(f"Raw events: {len(raw_events)}, "
                f"accepted: {sum(1 for e in raw_events if e.get('should_caption'))}")

    deduped = dedup_events(raw_events)
    logger.info(f"After dedup: {len(deduped)}")

    # Attach fresh scene context to each deduped event
    for ev in deduped:
        sc = get_scene_at(ev["timestamp_sec"], scene_index)
        ev.setdefault("scene_text",  sc.get("scene_text", ""))
        ev.setdefault("expressions", sc.get("expressions", []))

    # ---- Caption synthesis (rule-based + GPT-4o-mini) ----
    timeline = build_timeline(deduped, scene_index, logger)
    logger.info(f"Timeline: {len(timeline)} caption segments")

    # ---- Annotated frames ----
    frames_dir = output_dir / "annotated_frames"
    frames_dir.mkdir(exist_ok=True)
    cap_obj = cv2.VideoCapture(video_path)
    fps_v   = cap_obj.get(cv2.CAP_PROP_FPS) or 25.0
    for seg in timeline:
        mid = (seg["start_sec"] + seg["end_sec"]) / 2
        cap_obj.set(cv2.CAP_PROP_POS_FRAMES, int(mid * fps_v))
        ret, frame = cap_obj.read()
        if ret:
            rendered = render_caption(frame.copy(), seg["caption"])
            cv2.imwrite(str(frames_dir / f"frame_{seg['start_sec']:.1f}s.png"), rendered)
    cap_obj.release()

    # ---- SRT ----
    srt_path = output_dir / "captions.srt"
    write_srt(timeline, srt_path)
    logger.info(f"SRT: {srt_path}")

    # ---- JSON ----
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "caption_segments": len(timeline),
            "timeline": timeline,
            "raw_events": len(raw_events),
            "deduped_events": len(deduped),
            "speech_segments": len(speech_segments),
        }, f, indent=2, ensure_ascii=False)

    if GENERATE_FINAL_VIDEO and timeline:
        generate_final_video(video_path, timeline,
                             output_dir / "final_output.mp4", logger)

    logger.info(f"\nDone. {len(timeline)} segments → {output_dir}")

# ====================== CALIBRATION ======================
def run_calibration(video_path: str, output_dir: Path):
    logger = setup_logger(Path(video_path).stem + "_cal", output_dir)
    logger.info("Calibration — no thresholds, logging all raw PANNs scores")
    panns_model = get_panns()
    waveform, wav_path = extract_audio(video_path, logger)
    try: os.unlink(wav_path)
    except: pass
    clean  = apply_noise_reduction(waveform)
    ws, hs = int(WINDOW_SEC * SAMPLE_RATE), int(HOP_SEC * SAMPLE_RATE)
    score_log = defaultdict(list)
    start = 0
    while start < len(clean):
        chunk = clean[start:start + ws]
        if len(chunk) < ws: chunk = np.pad(chunk, (0, ws - len(chunk)))
        if chunk.ndim == 1: chunk = chunk.reshape(1, -1)
        o = panns_model.inference(chunk)
        sc = o[0] if isinstance(o, (list, tuple)) else o["clipwise_output"]
        if hasattr(sc, "ndim") and sc.ndim > 1: sc = sc[0]
        t = start / SAMPLE_RATE
        for idx in np.argsort(sc)[::-1][:15]:
            score_log[panns_model.labels[idx]].append(
                {"t": round(t,2), "score": round(float(sc[idx]),4)})
        start += hs
    report = {}
    for lbl, entries in score_log.items():
        sc2 = [e["score"] for e in entries]
        report[lbl] = {
            "count": len(sc2), "max": round(max(sc2),4),
            "mean": round(float(np.mean(sc2)),4),
            "p50": round(float(np.percentile(sc2,50)),4),
            "p75": round(float(np.percentile(sc2,75)),4),
            "p90": round(float(np.percentile(sc2,90)),4),
            "sample_ts": [e["t"] for e in entries[:5]],
        }
    rp = output_dir / "calibration_report.json"
    with open(rp, "w") as f: json.dump(report, f, indent=2)
    logger.info(f"Calibration report: {rp}")

# ====================== MAIN ======================
def main():
    parser = argparse.ArgumentParser(
        description="Non-speech audio captioning pipeline v4 — pyannote + GPT-4o-mini")
    parser.add_argument("--video",          required=True)
    parser.add_argument("--florence-log",   default=None,
                        help="Path to JSONL vision log (Stage 1 output).")
    parser.add_argument("--extract-vision", action="store_true",
                        help="Stage 1: run Florence, write vision log, exit.")
    parser.add_argument("--calibrate",      action="store_true",
                        help="PANNs calibration pass, no thresholds.")
    parser.add_argument("--pyannote-token", default=None,
                        help="HuggingFace token for pyannote (overrides HF_TOKEN env).")
    parser.add_argument("--openai-key",     default=None,
                        help="OpenAI API key (overrides OPENAI_API_KEY env).")
    args = parser.parse_args()

    global HF_TOKEN, OPENAI_API_KEY
    if args.pyannote_token: HF_TOKEN = args.pyannote_token
    if args.openai_key:     OPENAI_API_KEY  = args.openai_key

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}"); return

    out = Path("panns_pyannote_results") / video_path.stem
    out.mkdir(parents=True, exist_ok=True)

    if args.extract_vision:
        log    = Path(args.florence_log) if args.florence_log else (out / "florence_log.jsonl")
        logger = setup_logger(video_path.stem + "_vision", out)
        extract_vision_log(str(video_path), str(log), logger)
        logger.info(f"\nStage 1 done. Stage 2:\n"
                    f"  python {Path(__file__).name} --video {video_path} "
                    f"--florence-log {log}")
    elif args.calibrate:
        run_calibration(str(video_path), out)
    else:
        process_video(str(video_path), out,
                      Path(args.florence_log) if args.florence_log else None)

if __name__ == "__main__":
    main()