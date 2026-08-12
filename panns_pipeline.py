import os, sys, subprocess, tempfile, shutil, json, logging, re, argparse
from pathlib import Path
import random
from datetime import timedelta, datetime
from collections import defaultdict
from venv import logger
import csv
import numpy as np
import soundfile as sf
import cv2
import torch
import torchaudio
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from panns_inference import AudioTagging
from dotenv import load_dotenv  
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ====================== GPU ======================
# Device selection for Mac MPS, CUDA, or CPU.
if torch.backends.mps.is_available():
    DEVICE = "mps"          # Apple Silicon GPU
    print("\nUsing Apple Silicon GPU (MPS)\n")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"\nGPU: {torch.cuda.get_device_name(0)}\n")
else:
    DEVICE = "cpu"
    print("\nNo GPU found — running on CPU (will be slow)\n")

# Shared Florence precision keeps model and vision inference consistent.
MODEL_DTYPE = torch.float16 if DEVICE in ("mps", "cuda") else torch.float32

# ====================== CONFIG ======================
SAMPLE_RATE          = 32000
WINDOW_SEC           = 0.96
HOP_SEC              = 0.20
FRAMES_PER_MINUTE    = 10
SCENE_WINDOW_SEC     = 30.0
# Florence scene gaps are wider than expected, so the pipeline uses a more tolerant window.
DEDUP_GAP_SEC        = 1.5
BURST_GAP_SEC        = 2.0       # events closer than this merge into one caption
PALETTE_SIZE         = 20
SEMANTIC_THRESHOLD   = 0.24      # short-phrase embeddings
W_AUDIO              = 0.65      # trust real audio 
W_PALETTE            = 0.35      
ACCEPT_THRESHOLD     = 0.12      
FRAME_STABILITY_THRESHOLD = 35.0
USE_NOISE_REDUCTION  = True
GENERATE_FINAL_VIDEO = True


# ---- Per-class thresholds ----
CLASS_THRESHOLD_KEYWORDS = [
    (["caw"],                               0.28),
    (["crow"],                              0.32),
    (["bird vocalization", "bird call"],    0.35),
    (["bird"],                              0.40),
    (["rustl"],                             0.028),
    (["creak", "creaking"],                 0.028),
    (["cricket"],                           0.10),
    (["insect"],                            0.13),
    (["wind noise"],                        0.045),
    (["wind"],                              0.038),
    (["laughter"],                          0.025),
    (["giggle"],                            0.022),
    (["chuckle", "chortle"],                0.025),
    (["belly laugh"],                       0.028),
    (["walk", "footstep"],                  0.028),
    (["run", "jog"],                        0.042),
    (["music"],                             0.06),
    (["musical instrument"],                0.055),
    (["tabla", "sitar", "dhol", "shehnai"], 0.030),
    (["drum"],                              0.048),
    (["owl", "hoot"],                       0.45),
    (["applause", "clapping"],              0.045),
    (["wheeze", "groan", "grunt", "pant", "gasp", "sigh",
      "battle cry", "whimper", "scream", "shout", "yell", "sob"], 0.05),
    (["whistle", "wolf-whistle", "whistling"], 0.05),
    (["vehicle", "car", "engine", "motor", "traffic"], 0.16),
    (["plucked string"],                    0.048),
    (["singing"],                           0.075),
    (["crowd", "cheering", "chatter"],      0.055),
    (["thunder"],                           0.045),
    (["fire", "crackling"],                 0.032),
    (["cattle", "sheep", "cow"],            0.025),
    (["water", "stream", "splash", "rain", "waterfall"], 0.035),
    (["white noise", "static"],             0.30),
    (["mantra", "chant"],                   0.28),
    (["wood", "knock", "chop"],              0.10),
    (["metal", "clank", "clang", "clash"],               0.032),
    (["glass", "clink", "shatter"],                      0.030),
    (["mridangam", "pakhawaj"],               0.030),
    (["sitar"],                              0.028),
    (["sarangi", "veena", "sarod"],          0.032),
    (["santoor", "santur"],                  0.032),
    (["tanpura", "tambura"],                 0.032),
    (["flute", "bansuri"],                   0.030),
    (["shehnai", "nadaswaram"],              0.030),
    (["harmonium"],                          0.032),
    (["organ", "electronic organ", "hammond"], 0.035),
    (["violin", "fiddle"],                   0.035),
    (["guitar"],                             0.038),
    (["piano", "keyboard"],                  0.040),
    (["drum kit", "bass drum", "snare"],     0.040),
    (["manjira", "percussion", "cymbal"],    0.040),
    (["bell", "chime", "wind chime"],        0.035),
    (["telephone", "ringtone", "phone"],     0.020),
    (["duck", "quack", "goose", "honk"],    0.09),
    (["wild animal", "domestic animal"],    0.09),
    (["animal", "dog", "cat"],              0.09),
    (["turkey", "fowl", "rodent", "rat", "mice", "mouse"], 0.14),
]
DEFAULT_THRESHOLD = 0.052
_threshold_cache: dict = {}

# ====================== MULTI SCENE PALETTE ======================
SCENE_PALETTES = {
    "nature": {
        "keywords": ["forest", "tree", "trees", "jungle", "woods", "bushes",
                     "leaves", "branch", "nest", "sky", "outdoor", "nature",
                     "feather", "wing", "farm", "field", "village"],
        "implausible": {"applause", "whistle", "mantra", "chant", "vehicle",
                         "human_reaction", "ringtone"},
    },
    "circus_performance": {
        "keywords": ["circus", "stage", "tent", "arena", "performer",
                     "acrobat", "trapeze", "net", "ring", "clown",
                     "rope", "big top", "high wire", "ringmaster",
                     "tightrope", "juggling", "pole", "aerial"],
        "implausible": {"crow", "bird", "cricket", "insect", "mantra",
                         "chant", "wind", "water", "vehicle", "wood"},
    },
    "crowd_audience": {
        "keywords": ["crowd", "audience", "spectator", "spectators",
                      "cheering", "watching", "bleachers", "seated",
                      "rows of people", "clapping"],
        "implausible": {"crow", "bird", "cricket", "insect", "mantra",
                         "chant", "wind", "vehicle", "wood"},
    },
    "ritual_temple": {
        "keywords": ["temple", "shrine", "idol", "priest", "ritual",
                     "prayer", "puja"],
        "implausible": {"vehicle", "applause", "whistle", "ringtone"},
    },
    "indoor_generic": {
        "keywords": ["indoor", "room", "hall", "auditorium", "studio",
                     "inside"],
        "implausible": {"crow", "bird", "cricket", "insect", "wind",
                         "water", "vehicle"},
    },
    "phone_present": {
        "keywords": ["phone", "mobile", "smartphone", "cell phone",
                     "telephone"],
        "implausible": {"crow", "bird", "cricket", "insect", "mantra",
                         "chant"},
    },
}

def _scene_palette(scene_text: str):
    """Return (category_name, implausible_family_set) for the given scene
    text. Falls back to ('unknown', empty set) when nothing matches or
    scene_text is empty — a scene we can't classify applies NO vetoes,
    which is the safe default."""
    if not scene_text:
        return "unknown", set()
    st = scene_text.lower()
    for category, cfg in SCENE_PALETTES.items():
        if any(k in st for k in cfg["keywords"]):
            return category, cfg["implausible"]
    return "unknown", set()

SCENE_CATEGORY_KEYWORDS = {
    "forest": ["forest", "tree", "trees", "jungle", "woods", "bushes",
               "leaves", "branch", "greenery", "foliage"],
    "path_walking": ["path", "walking", "walk", "trail", "road", "person walking",
                     "footpath", "dirt road"],
}

SCENE_PHRASE_BANKS = {
    "forest": [
        "पत्तों की सरसराहट सुनाई दे रही है",
        "झाड़ियों में हल्की आवाज़ है",
        "हवा से पेड़ों की टहनियाँ हिल रही हैं",
        "सूखी पत्तियों के चरमराने की आवाज़ है",
        "घास में हल्की सी आवाज़ है",
    ],
    "path_walking": [
        "कदमों की आहट सुनाई दे रही है",
        "पैरों तले मिट्टी चलने की आवाज़ है",
        "चलने की धीमी आवाज़ आ रही है",
        "रास्ते किनारे पत्तों की सरसराहट है",
        "घास और झाड़ियों की सरसराहट है",
        "चलते हुए सूखी टहनियों की आवाज़ है",
    ],
}

# ====================== EXTENDED SOUND PALETTE ======================
# These are candidate PANNs labels only; they expand the search space without forcing captions.
# Instrument names are centralized so families and Hindi labels stay consistent.
INSTRUMENT_HINDI = {
    "tabla":     "तबला",
    "dhol":      "ढोल",
    "mridangam": "मृदंगम",
    "manjira":   "मंजीरा",
    "sitar":     "सितार",
    "sarangi":   "सारंगी",
    "santoor":   "संतूर",
    "tanpura":   "तानपुरा",
    "flute":     "बांसुरी",
    "shehnai":   "शहनाई",
    "harmonium": "हारमोनियम",
    "violin":    "वायलिन",
    "guitar":    "गिटार",
    "piano":     "पियानो",
}
INSTRUMENT_FAMILIES = set(INSTRUMENT_HINDI.keys())
# Generic drum labels are handled separately from specific Indian drum families.

def classify_scene_category(scene_text: str) -> str:
    """Match the Florence-derived visual scene description to a rough
    setting category, so the ambient fallback can be grounded in what's
    actually on screen instead of being setting-agnostic filler."""
    if not scene_text:
        return None
    st = scene_text.lower()
    for category, keywords in SCENE_CATEGORY_KEYWORDS.items():
        if any(kw in st for kw in keywords):
            return category
    return None

AMBIENT_FALLBACK_NATURE = [
    "प्रकृति की धीमी आवाज़ सुनाई दे रही है",
    "बाहर से हल्की प्राकृतिक आवाज़ आ रही है",
    "पेड़-पौधों के बीच हल्की आवाज़ है",
]
AMBIENT_FALLBACK_GENERIC = [
    "दूर से हल्की आवाज़ आ रही है",
    "हल्की सी आवाज़ सुनाई दे रही है",
    "कोई हल्की आवाज़ सुनाई दे रही है",
]
NATURE_FAMILIES = {"animal", "bird", "insect", "cricket", "wind", "rustling",
                    "creak", "thunder", "crow"}
# Indoor inference is intentionally avoided for generic sound labels without visual support.

def ambient_fallback_hint(families: list = None, scene_text: str = None) -> str:
    """Plain-text (no brackets) best-guess hint for a generic/unclear
    sound, used as the 'Original raw caption' reference fed to GPT — NOT
    a final answer. This is intentionally not bracket-wrapped so it does
    NOT get treated as a deterministic bypass: for anything this codebase
    doesn't specifically recognize (a car horn, crowd chatter, a phone
    ringing — anything outside the forest/village content this was tuned
    on), GPT still gets to look at the actual raw detected label text and
    adapt, rather than the sound being silently swallowed by one of a
    handful of canned filler phrases regardless of what it really is."""
    category = classify_scene_category(scene_text)
    if category and category in SCENE_PHRASE_BANKS:
        return random.choice(SCENE_PHRASE_BANKS[category])

    fams = set(f.lower() for f in (families or []))
    if fams & NATURE_FAMILIES:
        pool = AMBIENT_FALLBACK_NATURE
    else:
        pool = AMBIENT_FALLBACK_GENERIC
    return random.choice(pool)

def ambient_fallback_caption(families: list = None, scene_text: str = None) -> str:
    """Bracket-wrapped FINAL fallback — used only as a true last resort
    when GPT itself errors out and no adaptive help is possible at all."""
    return f"[{ambient_fallback_hint(families, scene_text)}]"


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
        has_action = bool(words & ACTION_WORDS)
        if not has_action:
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
            "microsoft/Florence-2-large",
            torch_dtype=MODEL_DTYPE,
            trust_remote_code=True
        ).to(DEVICE)

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
    # threshold lowered 0.5 -> 0.35 and min_speech_duration_ms lowered
    # 250 -> 150: atypical vocalizations (quiet mumbling, a distorted
    # scream, slurred speech) can genuinely score below Silero's default
    # confidence bar and never get recorded as a speech segment at all —
    # and no amount of downstream overlap-ratio tuning in is_speech_window
    # can compensate for a segment that was never detected in the first
    # place. Erring toward over-detecting speech is the safer failure
    # mode here, same reasoning as the overlap-ratio tightening below.
    speech_timestamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=sample_rate,
        threshold=0.35,
        min_speech_duration_ms=150,
        min_silence_duration_ms=300,
        speech_pad_ms=500,
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
    logger.info(f"Silero detected {len(segments)} speech segments "
                f"({total_speech:.1f}s total speech)")

    if len(segments) == 0:
        logger.warning("⚠️ Silero found ZERO speech segments!")

    return segments


# ====================== SPEECH DETECTION ======================

def build_speech_mask_from_segments(segments: list, n_samples: int) -> np.ndarray:
    """Convert (start_sec, end_sec) list to a sample-level boolean mask."""
    mask = np.zeros(n_samples, dtype=bool)
    for start, end in segments:
        s = int(start * SAMPLE_RATE)
        e = min(n_samples, int(end * SAMPLE_RATE))
        mask[s:e] = True
    return mask

def is_speech_window(start_sec: float, speech_segments: list,
                     threshold_ratio: float = 0.10) -> bool:
    # Tightened further: threshold lowered 0.25 -> 0.15 -> 0.10, and
    # padding increased 0.3 -> 0.4s. VAD segment edges are imprecise
    # (soft/trailing speech at onset/offset often gets slightly clipped),
    # and captions were still being reported as persisting visibly into
    # dialogue in a few places. Erring toward gating more aggressively is
    # the safer failure mode — a missed sound-effect caption is far less
    # jarring than one overlapping real speech.
    PAD_SEC = 0.4
    end_sec = start_sec + WINDOW_SEC
    overlap = 0.0
    for seg_start, seg_end in speech_segments:
        seg_start -= PAD_SEC
        seg_end   += PAD_SEC
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

SDH_MAX_WORDS = 8
SDH_MAX_CHARS = 42

def enforce_caption_format(caption: str) -> str:
    """Force short length, square brackets, and remove punctuation.

    Enforces the strict OTT SDH limit (max 8 words / 42 characters,
    matching Netflix/Amazon CPS-compliant subtitle density) in code —
    GPT does not reliably obey a "max 8 words" prompt instruction on its
    own, however emphatically it's worded, so this is the actual
    guarantee, not the prompt text.
    """
    if not caption:
        return "[sound]"

    # Remove existing brackets
    caption = caption.replace("[", "").replace("]", "").strip()

    # Remove punctuation (including Devanagari danda ।)
    for ch in [".", ",", "!", "?", "।", "|", ";", ":"]:
        caption = caption.replace(ch, "" if ch not in (",",) else "§")
    # (comma temporarily marked §, used below to detect a clause boundary
    # before being stripped for good)

    words = caption.split()
    if len(words) > SDH_MAX_WORDS or len(caption) > SDH_MAX_CHARS:
        # Prefer cutting at a clause boundary (where a comma was) if the
        # first clause alone already fits — keeps grammar intact instead
        # of chopping mid-sentence.
        if "§" in caption:
            first_clause = caption.split("§")[0].strip()
            if first_clause and len(first_clause.split()) <= SDH_MAX_WORDS \
               and len(first_clause) <= SDH_MAX_CHARS:
                caption = first_clause
            else:
                caption = " ".join(first_clause.split()[:SDH_MAX_WORDS])
        else:
            caption = " ".join(words[:SDH_MAX_WORDS])
        # Hard char cap as a final backstop, trimmed at the last full word
        if len(caption) > SDH_MAX_CHARS:
            caption = caption[:SDH_MAX_CHARS].rsplit(" ", 1)[0]

    caption = caption.replace("§", "").strip()

    # Add square brackets
    if not caption.startswith("["):
        caption = "[" + caption
    if not caption.endswith("]"):
        caption = caption + "]"

    return caption

def group_into_bursts(events: list, gap_sec: float = 1.5) -> list:
    """
    Group consecutive audio events into bursts if they are close in time.
    """
    if not events:
        return []

    bursts = []
    current_burst = [events[0]]

    for ev in events[1:]:
        if ev["timestamp_sec"] - current_burst[-1]["timestamp_sec"] <= gap_sec:
            current_burst.append(ev)
        else:
            bursts.append(current_burst)
            current_burst = [ev]

    bursts.append(current_burst)
    return bursts

# ====================== CAPTION RENDERING ======================
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

# LLM-based Hindi caption refinement with tracing.
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.06,
    max_tokens=60,
    api_key=OPENAI_API_KEY,
)

@traceable(name="generate_hindi_caption")
def generate_hindi_caption(raw_caption: str, scene_text: str,
                           detected_labels, expressions: list,
                           logger) -> str:
    # detected_labels is a dict {family: raw_label} — use the actual raw
    # label values (e.g. "Moo", "Dog") for GPT, not the dict's family-name
    # keys. Joining the dict directly iterates its keys, which for some
    # families (e.g. generic "animal") doesn't match the real label text.
    label_values = list(detected_labels.values()) if isinstance(detected_labels, dict) else detected_labels
    labels_str = ", ".join(label_values) if label_values else "None"
    expr_str = ", ".join(expressions) if expressions else "None"

    # If the audio itself is generic/unclear, ground the fallback in what's
    # actually on screen (visual scene text from Florence) instead of a
    # setting-agnostic filler phrase — a forest scene suggests leaves/grass/
    # twigs, a walking scene suggests footsteps, etc.
    scene_category = classify_scene_category(scene_text)
    scene_hint = ""
    if scene_category and scene_category in SCENE_PHRASE_BANKS:
        examples = "، ".join(SCENE_PHRASE_BANKS[scene_category][:3])
        scene_hint = (f"\n7. If the sound is generic/unclear and no specific "
                      f"species/instrument is detected, the visual scene looks "
                      f"like a {scene_category.replace('_',' ')} setting — prefer "
                      f"something concrete and plausible for that setting over "
                      f"vague filler, e.g.: {examples}")

    system_prompt = f"""You are an expert Hindi SDH (Subtitles for Deaf and Hard-of-Hearing) Timed-Text Specialist for major OTT platforms (Netflix, Amazon Prime Video, Disney+ Hotstar).

Your task is to convert raw audio detection tags into a single, streaming-compliant, high-quality Hindi non-speech subtitle card.

STRICT OTT SDH COMPLIANCE RULES:
1. STRICT AUDIO GROUNDING:
   - Describe ONLY sound events explicitly supported by "Detected sounds". Never invent unlisted sound events or transcribe visual actions.
   - ANIMALS: Any animal sound (cow, dog, cat, horse, or unspecified) -> use the generic "जानवर की आवाज़ सुनाई दे रही है". Do NOT name the exact species (न गाय, न कुत्ता, न बिल्ली) — exact species names make captions too scene-specific; the generic term is preferred.
   - BIRDS: Any bird sound (crow, owl, or unspecified bird) -> use the generic "पक्षियों की चहचहाहट सुनाई दे रही है". Do NOT name the exact species (न कौआ, न उल्लू) for the same reason.
   - MULTIPLE ANIMAL/BIRD SOUNDS: If both an animal sound and a bird sound are present together, mention both generically rather than naming species: "जानवर और पक्षियों की चहचहाहट सुनाई दे रही है".

2. INSTRUMENT & MUSIC SPECIFICATIONS:
   - Faithfully translate detected instruments without substitution:
     Tabla -> "संगीत के साथ तबला बज रहा है"
     Sitar -> "संगीत के साथ सितार बज रहा है"
     Dhol -> "संगीत के साथ ढोल बज रहा है"
     Shehnai -> "संगीत के साथ शहनाई बज रहा है"
     Flute -> "संगीत के साथ बांसुरी बज रहा है"
     Harmonium -> "संगीत के साथ हारमोनियम बज रहा है"
     Violin -> "संगीत के साथ वायलिन बज रहा है"
   - Generic Drums/Percussion -> "ढोल-नगाड़े जैसी थाप सुनाई दे रही है"
   - Music without specified instrument -> "संगीत बज रहा है" (Do NOT default to tabla or any specific instrument unless explicitly detected).

3. HINDI ORTHOGRAPHY & SYNTAX:
   - Bird sounds (crow, owl, or any other): Always use "पक्षियों की चहचहाहट सुनाई दे रही है" — do not name the species.
   - Laughter: Use "हल्की हँसी सुनाई दे रही है".

4. OBJECTIVITY & BANNED WORDS (CRITICAL OTT QC RULE):
   - BANNED vague filler words: "पृष्ठभूमि" (background), "वातावरण" (environment/atmosphere), "आस-पास" (nearby/around), "हलचल" (movement/stir).
   - BANNED subjective/dramatic adjectives: "डरावनी", "भयावह", "सुरीली", "मधुर", "मनमोहक".
   - BANNED time-of-day references: Do NOT describe night/day (रात/दिन). You only know what was HEARD.

5. FORMATTING & DENSITY CONSTRAINTS:
   - Output length: Max 42 characters / 6-8 words (for CPS < 20 compliance).
   - Enclose output ENTIRELY inside square brackets: [हिंदी कैप्शन].
   - NO sentence-ending punctuation inside brackets (NO '।', '.', '!', '?').
   - STYLE: Name the sound as a short noun phrase. Do NOT end with "सुनाई दे रहा है" / "सुनाई दे रही है" / "सुनाई दे रहे हैं"/"बज रहा है"/"बज रही है"/"बज रहे हैं" ("can be heard") — this is redundant filler, not new information. Prefer "पक्षियों की चहचहाहट" over "पक्षियों की चहचहाहट सुनाई दे रही है". """
    
    human_prompt = f"""Generate a compliant Hindi SDH subtitle card for the following inputs:

Detected sounds: {labels_str}
Expressions: {expr_str}
Scene context hint: {scene_hint if scene_hint else "None"}
Original raw caption: {raw_caption}

Output ONLY the final Hindi caption inside square brackets [हिंदी पाठ]."""

    try:
        messages = [SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)]
        response = llm.invoke(messages)
        
        caption = response.content.strip()
        return caption

    except Exception as e:
        logger.error(f"LangChain GPT error in generate_hindi_caption: {e}")
        return ambient_fallback_caption(scene_text=scene_text)


@traceable(name="polish_hindi_caption")
def polish_hindi_caption(caption: str, detected_labels, logger) -> str:
    label_values = list(detected_labels.values()) if isinstance(detected_labels, dict) else detected_labels
    labels_str = ", ".join(label_values) if label_values else "None"

    system_prompt = f"""You are a Quality Control (QC) Inspector for Hindi SDH Timed-Text on major OTT platforms.

Your sole job is to audit and polish the input Hindi subtitle card to ensure strict OTT platform compliance:
1. FORMATTING: Wrap entirely in square brackets [...] with NO internal punctuation (no '।', '.', '!', '?').
2. CHARACTER LIMIT: Ensure word count does NOT exceed 6-8 words (max 42 characters) for CPS compliance.
3. ANIMAL/BIRD GENERALIZATION: If the caption names a specific animal or bird species (गाय, कुत्ता, बिल्ली, घोड़ा, कौआ, उल्लू, etc.), replace it with the generic "जानवर की आवाज़" (for animals) or "पक्षियों की चहचहाहट" (for birds) — do not name exact species.
4. INSTRUMENT INTEGRITY: Preserve named musical instruments (तबला, सितार, ढोल, शहनाई, बांसुरी, हारमोनियम, वायलिन) exactly as detected. Do NOT default to or add "तबला" unless explicitly listed.
5. SANITIZATION:
   - Remove subjective/dramatic adjectives ("सुरीली", "डरावनी", "मधुर", "भयावह").
   - Remove vague filler terms ("पृष्ठभूमि", "वातावरण", "आस-पास", "हलचल").
   - Remove the trailing phrase "सुनाई दे रहा है" / "सुनाई दे रही है" / "सुनाई दे रहे हैं/"बज रहा है"/"बज रही है"/"बज रहे हैं" wherever it appears — captions should name the sound as a noun phrase, not end with "can be heard". 
6. OUTPUT: Return ONLY the polished caption inside square brackets."""
    
    human_prompt = f"""QC Audit and polish this subtitle card:

Detected sounds: {labels_str}
Current caption: {caption}

Output ONLY the corrected caption in square brackets:"""

    try:
        messages = [SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)]
        response = llm.invoke(messages)
        
        polished = response.content.strip()
        return polished

    except Exception as e:
        logger.error(f"LangChain GPT error in polish_hindi_caption: {e}")
        return caption



def fix_hindi_issues(caption: str) -> str:
    # Fix crow word: standard spelling is कौआ (nominative) / कौए (oblique,
    # used before "की"). Normalize all common variants GPT might produce.
    for variant in ["कौवा", "कव्वा", "काऊआ"]:
        caption = caption.replace(variant, "कौआ")
    for variant in ["कौवे", "कव्वे"]:
        caption = caption.replace(variant, "कौए")

    # Fix crow's call matra: the nasalized "kaanv" sound on आ (a vowel with
    # no ascender) takes chandrabindu (ँ), not anusvara (ं).
    caption = caption.replace("कांव", "काँव")

    caption = caption.replace("कौवा काँव-काँव", "कौआ काँव-काँव")
    caption = caption.replace("कौए की काँव-काँव", "कौआ काँव-काँव")
    caption = caption.replace("कौआ की आवाज", "कौए की आवाज")
    caption = caption.replace("कौए की आवाज", "कौए की आवाज़")
    caption = caption.replace("कीटों की आवाज़", "कीड़ों की आवाज़")

    # Removes any mention of vague phrases of aas paas GPT doesn't reliably follow the prompt
    # instruction against these words, so enforce it here unconditionally on every caption rather than relying on GPT to comply.
    if "आस-पास" in caption or "आसपास" in caption or "हलचल" in caption or "इर्द-गिर्द" in caption:
        caption = caption.replace("आस-पास", "").replace("आसपास", "")
        caption = caption.replace("इर्द-गिर्द", "")
        caption = caption.replace("हलचल", "आवाज़")
        caption = caption.replace("  ", " ").strip()
        caption = caption.replace(" की आवाज़ सुनाई", " आवाज़ सुनाई")
        if caption in ("[]", "[ ]", ""):
            caption = "[हल्की सी आवाज़ सुनाई दे रही है]"

    # Fix incomplete captions (previously this patched them by appending
    # "सुनाई दे रहा/रही है" — no longer wanted, so just leave them as a
    # clean noun phrase instead).
    if caption.strip() == "[संगीत]":
        caption = "[संगीत बज रहा है]"

    # GLOBAL: strip "सुनाई दे रहा/रही/रहे है/हैं" ("can be heard") style
    # trailing verb phrases everywhere, regardless of source (hardcoded
    # deterministic phrase or GPT-generated). Captions should read as a
    # plain noun phrase naming the sound ("तबला बज रहा है", "पक्षियों की
    # चहचहाहट") rather than "this sound can be heard" every time. This is
    # a single regex applied to every caption, not a per-phrase fix, so it
    # also covers any wording GPT comes up with that we haven't seen yet.
    caption = re.sub(r"\s*सुनाई\s*दे\s*रह[ाी]\s*है\s*", " ", caption)
    caption = re.sub(r"\s*सुनाई\s*दे\s*रहे\s*हैं\s*", " ", caption)
    caption = re.sub(r"\s*सुनाई\s*दे\s*रहा\s*", " ", caption)   # dangling, no है
    caption = re.sub(r"\s*सुनाई\s*दे\s*रही\s*", " ", caption)   # dangling, no है
    caption = re.sub(r"\s*बज\s*रहा\s*है\s*", " ", caption)
    caption = re.sub(r"\s*बज\s*रही\s*है\s*", " ", caption)
    caption = re.sub(r"\s*बज\s*रहे\s*हैं\s*", " ", caption)    

    # Collapse whitespace left behind, and reattach the closing bracket
    # cleanly if it ended up separated from the last word.
    caption = re.sub(r"\s+\]", "]", caption)
    caption = re.sub(r"\s{2,}", " ", caption).strip()
    # A stray empty/near-empty result after stripping — extremely rare,
    # but guard against captioning nothing.
    if caption in ("[]", "[ ]", ""):
        caption = "[हल्की आवाज़]"

    return caption


def render_caption(frame: np.ndarray, text: str,
                   font_path: str = "TiroDevanagariHindi-Italic.ttf",
                   font_size: int = 26,
                   max_width_ratio: float = 0.82) -> np.ndarray:
    """
    Renders clean, professional Hindi subtitles on video frames.
    """
    if not text or not text.strip():
        return frame

    try:
        from PIL import Image, ImageDraw, ImageFont
        import os

        h, w = frame.shape[:2]
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        # === Smart Font Loading (prioritizes your folder) ===
        font = None
        possible_paths = [
            font_path,                                           
            os.path.join(os.getcwd(), font_path),                
            os.path.join(os.path.dirname(__file__), font_path),  
            r"C:\Windows\Fonts\TiroDevanagari-Regular.ttf",      
        ]

        for path in possible_paths:
            if path and os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, font_size)
                    break
                except:
                    continue

        if font is None:
            font = ImageFont.load_default()
            logger.warning("Tiro Devanagari font not found. Using default.")

        # === Word Wrapping (max 2 lines) ===
        max_text_width = int(w * max_width_ratio)
        lines = []
        words = text.split()
        current_line = ""

        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_text_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        if len(lines) > 2:
            lines = lines[:2]   # Force max 2 lines

        # === Calculate text block size ===
        line_heights = []
        line_widths = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])

        text_block_width = max(line_widths) if line_widths else 0
        text_block_height = sum(line_heights) + (len(lines) - 1) * 8

        # === Background Box ===
        padding_x = 28
        padding_y = 16
        box_width = text_block_width + 2 * padding_x
        box_height = text_block_height + 2 * padding_y

        margin_bottom = 48
        x = (w - box_width) // 2
        y = h - box_height - margin_bottom

        # Semi-transparent rounded background
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle(
            [x, y, x + box_width, y + box_height],
            radius=14,
            fill=(0, 0, 0, 215)
        )
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)

        # === Draw text with shadow ===
        shadow_offset = 2
        for i, line in enumerate(lines):
            line_y = y + padding_y + sum(line_heights[:i]) + i * 8 + shadow_offset
            line_x = x + padding_x + (text_block_width - line_widths[i]) // 2
            draw.text((line_x + shadow_offset, line_y), line, font=font, fill=(0, 0, 0, 180))

        for i, line in enumerate(lines):
            line_y = y + padding_y + sum(line_heights[:i]) + i * 8
            line_x = x + padding_x + (text_block_width - line_widths[i]) // 2
            draw.text((line_x, line_y), line, font=font, fill=(255, 255, 255))

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    except Exception as e:
        logger.warning(f"render_caption failed: {e}")
        return frame
    
def annotate_frame(frame, caption_text, output_path):
    rendered_frame = render_caption(frame.copy(), caption_text)
    cv2.imwrite(str(output_path), rendered_frame)

def normalize_hindi_caption(caption: str) -> str:
    """Standardize common phrases for consistency."""
    caption = caption.strip()
    
    # Standardize bird sounds
    if "पक्षी" in caption or "चहचहाहट" in caption or "चिड़िया" in caption:
        if "पेड़ों के बीच" not in caption:
            caption = caption.replace("पक्षियों की चहचहाहट गूंज रही है", "पक्षियों की चहचहाहट सुनाई दे रही है")
    
    # Standardize music
    if "संगीत" in caption:
        caption = caption.replace("संगीत की मधुर धुन गूंज रही है", "संगीत की धुन सुनाई दे रही है")
        caption = caption.replace("चारों ओर मधुर संगीत गूंज रहा है", "संगीत की धुन सुनाई दे रही है")

    # Remove repetitive or robotic phrases
    caption = caption.replace("सुनाई दे रही है।", "सुनाई दे रही है")
    caption = caption.replace("गूंज रही है।", "गूंज रही है")

    return caption

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
                inputs = fp(text=prompt, images=image, return_tensors="pt")
                inputs = {
                    k: (v.to(DEVICE, dtype=MODEL_DTYPE)
                        if torch.is_tensor(v) and torch.is_floating_point(v)
                        else v.to(DEVICE) if hasattr(v, "to") else v)
                    for k, v in inputs.items()
                }
                with torch.inference_mode():
                    ids = fm.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=64,
                        do_sample=False,
                        num_beams=1,
                    )
                return fp.batch_decode(ids, skip_special_tokens=True)[0].strip()

            scene_cap  = run_florence("<DETAILED_CAPTION>")
            action_cap = ""

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
    logger.info("AUDIO ANALYSIS — PANNs detection with SileroVAD speech gate")
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

            # No VAD penalty — speech windows are hard-gated above, so any window reaching here is definitively non-speech. 
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

        # Keep strong instrument-family hits even when "Music" wins the frame.
        # This protects all spellings of the same instrument family.
        for c in candidates:
            if not c["pass"]:
                continue
            if _family(c["label"]) not in INSTRUMENT_FAMILIES:
                continue
            if best and c["label"] == best["label"]:
                continue  # already recorded as the winner event above
            events.append({
                "timestamp_sec":      round(start_sec, 2),
                "label":              c["label"],
                "raw_confidence":     c["raw"],
                "palette_score":      c["palette"],
                "combined_score":     c["combined"],
                "should_caption":     True,
                "scene_text":         scene["scene_text"][:120],
                "expressions":        scene.get("expressions", []),
            })

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

SPECIFIC_ANIMAL_LABELS = {
    "cattle, bovinae", "moo", "cow", "livestock, farm animals, working animals",
    "dog", "bark", "bow-wow", "canidae, dogs, wolves",
    "cat", "meow", "purr",
    "horse", "neigh, whinny", "clip-clop",
    "owl", "hoot",
}

def filter_labels_for_caption(events: list, min_combined_score: float = 0.18) -> list:
    """
    Filter out weak detections before sending to LLM.
    This is the fix against random animal captions.
    """
    filtered = []
    for ev in events:
        score = ev.get("combined_score", 0)
        label = ev.get("label", "")
        fam   = _family(label)

        if fam == "animal" and label.lower() not in SPECIFIC_ANIMAL_LABELS:
            # Generic animal labels are noisy, so they need stronger evidence.
            if score < 0.35:
                continue

        # Phone/ringtone cues can be brief and still meaningful, so they get a lower bar.
        if fam == "ringtone" and score >= 0.06:
            filtered.append(ev)
            continue

        # Birds/crows are high false-positive sources — require stronger evidence
        if fam in ("bird", "crow") and score < 0.40:
            continue

        if score >= min_combined_score:
            filtered.append(ev)

    return filtered


def final_cleanup(caption: str, detected_labels, scene_text: str = None) -> str:
    """
    Post-processing safety net to catch remaining generic animal captions.
    """
    caption_lower = caption.lower()
    label_values = list(detected_labels.values()) if isinstance(detected_labels, dict) else detected_labels
    labels_str = " ".join(label_values).lower()
    
    # Keep generic animal/bird captions, but normalize exact species names back to the generic form.
    SPECIES_TO_GENERIC = {
        "गाय": "जानवर", "कुत्ते": "जानवर", "कुत्ता": "जानवर",
        "बिल्ली": "जानवर", "घोड़े": "जानवर", "घोड़ा": "जानवर",
        "कौआ": "पक्षियों", "कौवा": "पक्षियों", "कौए": "पक्षियों",
        "उल्लू": "पक्षियों", "बत्तख": "पक्षियों", "हंस": "पक्षियों",
    }
    for species, generic in SPECIES_TO_GENERIC.items():
        if species in caption:
            caption = f"[{generic} की आवाज़ सुनाई दे रही है]"
            break

    return caption

def _gender_from_scene(scene_text: str) -> str:
    st = scene_text.lower()
    w  = bool(re.search(r"\b(woman|female|girl|lady|she|her)\b", st))
    m  = bool(re.search(r"\b(man|male|boy|he|his)\b", st))
    if w and not m: return "female"
    if m and not w: return "male"
    return "unknown"

# ====================== RULE-BASED CAPTION (FALLBACK) ======================
SOUND_FAMILIES = {
    "crow":       ["crow", "caw"],
    "bird":       ["bird vocalization", "bird call", "bird song", "bird",
                   "fowl", "rooster", "chicken", "owl", "hoot", "turkey"],
    # Human reaction sounds were missing before, so they are included here.
    "human_reaction": ["wheeze", "groan", "grunt", "pant", "gasp", "sigh",
                        "battle cry", "whimper", "scream", "shout", "yell",
                        "sob", "sniff", "cough"],
    "whistle":    ["whistle", "wolf-whistle", "whistling"],
    "laugh_soft": ["chuckle", "chortle", "giggle"],
    "laugh_full": ["laughter", "belly laugh"],
    "cricket":    ["cricket"],
    "insect":     ["insect", "buzz"],
    "wind":       ["wind"],
    "rustling":   ["rustl"],
    "creak":      ["creak"],
    "footstep":   ["footstep", "walk", "run", "jog"],
    "thunder":    ["thunder"],
    "music":      ["music", "musical instrument", "plucked string",
                   "bowed string", "wind instrument", "singing"],
    "crowd":      ["crowd", "cheering", "chatter"],
    "applause":   ["applause", "clapping"],
    "vehicle":    ["vehicle", "car", "engine", "motor", "motorcycle",
                   "truck", "traffic"],
    "animal":     ["cattle", "cow", "bull", "dog", "bark", "cat", "horse",
                   "neigh", "frog", "animal", "duck", "quack", "goose",
                   "honk", "wild animal", "domestic animal"],
    # Ringing sounds are split so phone and bell cases stay distinct.
    "bell":       ["bell", "chime", "wind chime"],
    "ringtone":   ["telephone", "ringtone", "phone"],
    "siren":      ["siren", "alarm", "civil defense"],
    # Each instrument family needs its own entry to normalize varied raw labels.
    "tabla":      ["tabla"],
    "dhol":       ["dhol", "dholak"],
    "mridangam":  ["mridangam", "pakhawaj"],
    "manjira":    ["manjira", "cymbal"],
    "sitar":      ["sitar"],
    "sarangi":    ["sarangi", "veena", "sarod"],
    "santoor":    ["santoor", "santur"],
    "tanpura":    ["tanpura", "tambura"],
    "flute":      ["flute", "bansuri"],
    "shehnai":    ["shehnai", "nadaswaram"],
    "harmonium":  ["harmonium", "organ", "electronic organ", "hammond"],
    "violin":     ["violin", "fiddle"],
    "guitar":     ["guitar"],
    "piano":      ["piano", "keyboard"],
    "drum":       ["drum kit", "bass drum", "snare drum", "percussion"],
    "water":      ["water", "stream", "river", "splash", "rain", "waterfall"],
    "wood":       ["wood", "knock", "chop"],
    "metal":      ["metal", "clank", "clang", "clash"],
    "glass":      ["glass", "clink", "shatter"],
    "fire":       ["fire", "crackling"],
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
                   scene_text: str, expressions: list, fam_scores: dict = None) -> str:
    """Generate a deterministic rule-based caption.

    fam_scores: optional mapping of family -> combined_score to allow
    gating low-confidence high-risk families (e.g. mantra, bird) so we
    don't repeatedly caption them on very weak evidence.
    """
    fset = set(families)
    fam_scores = fam_scores or {}
    # Minimum combined_score required to treat certain high-risk families as deterministic
    MANTRA_MIN_SCORE = 0.40
    BIRD_MIN_SCORE = 0.40

    # If a family was detected but its score is below the safety threshold,
    # ignore it for deterministic rule decisions (avoids repeated false "mantra"/bird captions).
    if "mantra" in fset and fam_scores.get("mantra", 0.0) < MANTRA_MIN_SCORE:
        fset.remove("mantra")
    if "chant" in fset and fam_scores.get("chant", 0.0) < MANTRA_MIN_SCORE:
        fset.remove("chant")
    if "crow" in fset and fam_scores.get("crow", 0.0) < BIRD_MIN_SCORE:
        fset.remove("crow")
    if "bird" in fset and fam_scores.get("bird", 0.0) < BIRD_MIN_SCORE:
        fset.remove("bird")

    outdoor = any(x in scene_text.lower()
                  for x in ["forest","outdoor","trees","jungle","nature","path"])
    gender  = _gender_from_scene(scene_text)

    # Use raw label strings, not family names, for exact species checks.
    bl = [v.lower() for v in best_labels.values()]

    # Specific animal sounds win before the generic fallback.
    ANIMAL_SOUND_LABELS = {"moo", "cattle, bovinae", "cow",
                           "dog", "bark", "bow-wow", "canidae, dogs, wolves",
                           "cat", "meow", "purr",
                           "horse", "neigh, whinny", "clip-clop"}
    if any(b in ANIMAL_SOUND_LABELS for b in bl):
        return "[जानवर की आवाज़ सुनाई दे रही है]"

    # "Patter" was being freely interpreted by GPT as rain ("बारिश की हल्की
    # आवाज़") with no basis.
    if "patter" in bl:
        return "[हल्की थपथपाहट जैसी आवाज़ है]"

    if "animal" in families and len(families) == 1:
        return "[जानवर की आवाज़ सुनाई दे रही है]"

    # Laughter — deterministic Hindi, reported alone and takes priority
    # over co-occurring music/instrument/bird/animal in the same burst.
    # Previously this branch returned English reference text, which meant
    # it always failed the GPT-bypass check and got sent to GPT — which
    # then freely combined it with whatever else was in the burst (e.g.
    # "हँसी और पक्षियों की चहचहाहट"). Laughter is the salient, human-
    # relevant sound and should be reported on its own.
    if "laugh_soft" in fset or "laugh_full" in fset:
        soft = "laugh_soft" in fset and "laugh_full" not in fset
        return "[हल्की हँसी सुनाई दे रही है]" if soft else "[हँसी सुनाई दे रही है]"

    # Instrument detection: pick whichever instrument family ranks highest
    # instead of choosing tabla no matter what. Uses the module-level
    # INSTRUMENT_HINDI dict, which is the same one used for secondary-
    # candidate protection and persistence filtering elsewhere - a single
    # source of truth instead of three separately-maintained lists that
    # inevitably drift out of sync with each other.
    detected_instruments = [INSTRUMENT_HINDI[fam.lower()] for fam in families
                            if fam.lower() in INSTRUMENT_HINDI]

    # Keep drum cues when a named instrument is also present.
    # This prevents real percussion from being lost during grouping.
    LARGE_DRUM_FAMILIES = {"drum", "bass drum", "drum kit", "percussion"}
    has_drum = any(fam.lower() in LARGE_DRUM_FAMILIES for fam in families)
    if has_drum and len(detected_instruments) == 1 and "ढोल" not in detected_instruments:
        detected_instruments.append("ढोल")

    # Resolve mantra, instruments, and drum cues in a confidence-aware order.
    has_mantra = any(f.lower() in ("mantra", "chant") for f in families)

    if has_mantra:
        # Require a higher confidence for mantra to be considered definitive
        if fam_scores and fam_scores.get("mantra", 0.0) < MANTRA_MIN_SCORE and fam_scores.get("chant", 0.0) < MANTRA_MIN_SCORE:
            has_mantra = False
        else:
            return "[मंत्रों का उच्चारण सुनाई दे रहा है]"
    if detected_instruments:
        if len(detected_instruments) == 1:
            return f"[संगीत के साथ {detected_instruments[0]} बज रहा है]"
        joined = "-".join(detected_instruments[:2])
        return f"[{joined} के साथ संगीत बज रहा है]"
    if has_drum:
        return "[ढोल-नगाड़े जैसी थाप सुनाई दे रही है]"

    # White noise is not a meaningful caption on its own, so use a generic ambient hint.
    if fset <= {"white noise", "static"}:
        return ambient_fallback_hint(families, scene_text)

    # Audience reaction pairs are a narrow, readable exception to the normal single-sound rule.
    if "applause" in fset and "whistle" in fset:
        return "[तालियों और सीटियों की आवाज़]"

    # Outdoor walking scenes can use rustling or footsteps as a stronger visual match.
    walking = any(x in scene_text.lower()
                  for x in ["walk","path","trail","moving","strolling","approaching"])
    if outdoor and walking and fset & {"rustling", "creak", "footstep"}:
        ambient = fset & {"bird","crow","cricket","insect","wind"}
        if "crow" in ambient or "bird" in ambient:
            return "[पक्षियों की चहचहाहट सुनाई दे रही है]"
        return "[पत्तों की सरसराहट सुनाई दे रही है]"

    # After the early rules, score order decides the final phrase, with music as a normal candidate.
    SOUND_PHRASE_MAP = {
        "crow":      "[पक्षियों की चहचहाहट]",
        "bird":      "[पक्षियों की चहचहाहट]",
        "animal":    "[जानवर की आवाज़]",
        "applause":  "[तालियों की आवाज़]",
        "whistle":   "[सीटी की आवाज़]",
        "human_reaction": "[व्यक्ति की आवाज़]",
        "crowd":     "[भीड़ की आवाज़]",
        "vehicle":   "[वाहन की आवाज़]",
        "bell":      "[घंटी की आवाज़]",
        "ringtone":  "[फ़ोन बजने की आवाज़]",
        "manjira":   "[मंजीरे की आवाज़]",
        "santoor":   "[संतूर की धुन]",
        "tanpura":   "[तानपुरे की आवाज़]",
        "water":     "[पानी की आवाज़]",
        "wood":      "[लकड़ी की आवाज़]",
        "metal":     "[धातु की खनक]",
        "glass":     "[काँच की आवाज़]",
        "fire":      "[आग की चटचटाहट]",
        "cricket":   "[झींगुरों की आवाज़]",
        "insect":    "[कीड़ों की आवाज़]",
        "wind":      "[हवा चलने की आवाज़]",
        "rustling":  "[पत्तों की सरसराहट]",
        "creak":     "[टहनियों की चरमराहट]",
        "music":     "[संगीत बज रहा है]",
    }

    # Scene context can suppress implausible sounds but not real animal cases.
    _scene_category, _implausible = _scene_palette(scene_text)
    _implausible = _implausible - {"animal"}

    for fam_name in families:  # already score-sorted for this burst
        fl = fam_name.lower()
        if fl in _implausible:
            continue
        if fl in SOUND_PHRASE_MAP:
            return SOUND_PHRASE_MAP[fl]

    # Ambiguous cases stay as a hint instead of a forced caption, so GPT can adapt to the real signal.
    return ambient_fallback_hint(families, scene_text)

# ====================== BURST CONSOLIDATION ======================
def build_timeline(events: list, scene_index: list, logger, output_dir: Path = None,
                    speech_segments: list = None) -> list:
    accepted = [ev for ev in events if ev.get("should_caption")]
    if not accepted:
        return []

    accepted.sort(key=lambda e: e["timestamp_sec"])

    # Music labels need a short persistence window before being kept.
    MUSIC_PERSISTENCE_SEC = 3.0
    music_fams = {"music"} | INSTRUMENT_FAMILIES
    filtered_accepted = []
    for i, ev in enumerate(accepted):
        fam = _family(ev["label"])
        if fam not in music_fams:
            filtered_accepted.append(ev)
            continue
        # Check for ANY music-family neighbor, not strictly the same
        # instrument. A specific instrument (e.g. Tabla) only "wins" the
        # odd frame even when playing continuously throughout.
        has_neighbor = any(
            _family(other["label"]) in music_fams
            and other is not ev
            and abs(other["timestamp_sec"] - ev["timestamp_sec"]) <= MUSIC_PERSISTENCE_SEC
            for other in accepted
        )
        if has_neighbor:
            filtered_accepted.append(ev)
    accepted = filtered_accepted
    if not accepted:
        return []

    # Group nearby detections into bursts.
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
    MIN_SIREN_SPAN_SEC = 4.0
    SPEECH_BUFFER_SEC = 0.15  # small safety margin before speech onset
    for burst in bursts:
        start_sec = burst[0]["timestamp_sec"]
        end_sec   = burst[-1]["timestamp_sec"] + 1.8

        # Avoid extending captions into real speech regions.
        if speech_segments:
            for seg_start, seg_end in speech_segments:
                if seg_start > start_sec and seg_start < end_sec:
                    end_sec = max(start_sec + 0.3, seg_start - SPEECH_BUFFER_SEC)
                    break

        siren_events = [ev for ev in burst if _family(ev["label"]) == "siren"]
        if siren_events:
            span = siren_events[-1]["timestamp_sec"] - siren_events[0]["timestamp_sec"]
            if span < MIN_SIREN_SPAN_SEC:
                burst = [ev for ev in burst if _family(ev["label"]) != "siren"]
                if not burst:
                    continue

        # Best label per family
        fam_best = {}
        for ev in sorted(burst, key=lambda e: -e["combined_score"]):
            fam = _family(ev["label"])
            if fam not in fam_best:
                fam_best[fam] = ev

        # Prioritize reaction sounds ahead of generic high-score noise.
        PRIORITY_FAMILIES = {"laugh_soft", "laugh_full", "human_reaction"}
        ranked   = sorted(fam_best, key=lambda f: fam_best[f]["combined_score"], reverse=True)
        priority = [f for f in ranked if f in PRIORITY_FAMILIES]
        rest     = [f for f in ranked if f not in PRIORITY_FAMILIES]
        top_fams = (priority + rest)[:3]
        best_labels = {f: fam_best[f]["label"] for f in top_fams}
        # Pass family-level combined scores so rule-based logic can gate
        # low-confidence high-risk families (mantra, bird, etc.).
        fam_scores = {f: fam_best[f].get("combined_score", 0.0) for f in top_fams}

        # Specific animal sounds override generic animal fallback in the same burst.
        specific_present = any(
            lbl.lower() in SPECIFIC_ANIMAL_LABELS for lbl in best_labels.values()
        )

        AMBIGUOUS_VOCALIZATION_FAMILIES = {"roar", "howl", "growl", "growling",
                                            "whale vocalization"}
        if specific_present:
            for f, lbl in list(best_labels.items()):
                if _family(lbl) == "animal" and lbl.lower() not in SPECIFIC_ANIMAL_LABELS:
                    del best_labels[f]
                elif f.lower() in AMBIGUOUS_VOCALIZATION_FAMILIES:
                    del best_labels[f]

        for f, lbl in list(best_labels.items()):
            if _family(lbl) == "animal" and lbl.lower() not in SPECIFIC_ANIMAL_LABELS:
                best_labels[f] = "Ambient/unclear sound (do not name species)"

        # Use the strongest event as the scene anchor.
        best_ev     = max(burst, key=lambda e: e["combined_score"])
        scene_text  = best_ev.get("scene_text", "")
        expressions = best_ev.get("expressions", [])

        # Suppress instrument captions when speech overlaps the burst.
        if speech_segments:
            burst_dur = max(0.001, end_sec - start_sec)
            overlap = 0.0
            for seg_start, seg_end in speech_segments:
                overlap += max(0.0, min(end_sec, seg_end) - max(start_sec, seg_start))
            speech_fraction = overlap / burst_dur
            if speech_fraction > 0.05:
                # remove any music/instrument families from top_fams
                top_fams = [f for f in top_fams if f not in music_fams]
                if not top_fams:
                    # if nothing left, fall back to keeping human_reaction if present
                    top_fams = [f for f in list(fam_best.keys()) if f == 'human_reaction'] or top_fams

        # Rule-based caption comes first, then GPT only for ambiguous cases.
        raw_caption = _rule_caption(top_fams, best_labels, scene_text, expressions, fam_scores)

        is_deterministic = (
            raw_caption.startswith("[") and
            any("\u0900" <= ch <= "\u097F" for ch in raw_caption)
        )

        if is_deterministic:
            hindi_caption = raw_caption
        else:
            # 2. Generate Hindi caption (LangSmith traces this automatically)
            hindi_caption = generate_hindi_caption(
                raw_caption=raw_caption,
                scene_text=scene_text,
                detected_labels=best_labels,
                expressions=expressions,
                logger=logger
            )

            # 3. Polish the caption
            if hindi_caption and len(hindi_caption) > 5:
                hindi_caption = polish_hindi_caption(
                    caption=hindi_caption,
                    detected_labels=best_labels,
                    logger=logger
                )

        hindi_caption = enforce_caption_format(hindi_caption)
        hindi_caption = final_cleanup(hindi_caption, best_labels, scene_text)
        hindi_caption = fix_hindi_issues(hindi_caption)

        # Final consistency pass catches mismatches between family signal and caption wording.
        _fam_lower = set(f.lower() for f in top_fams)
        _consistency_checks = [
            ("पक्षियों", {"bird", "crow"} & _fam_lower,
             "[जानवर की आवाज़]" if "animal" in _fam_lower else None),
            ("मंत्रों का उच्चारण", {"mantra", "chant"} & _fam_lower, None),
            ("वाहन", {"vehicle"} & _fam_lower, None),
            ("तालियों", {"applause"} & _fam_lower, None),
            ("लकड़ी", {"wood"} & _fam_lower, None),
        ]
        for _mention, _justified_by, _fallback_override in _consistency_checks:
            if _mention in hindi_caption and not _justified_by:
                logger.warning(
                    f"CAPTION/FAMILY MISMATCH at {start_sec:.1f}s: caption "
                    f"'{hindi_caption}' mentions '{_mention}' but families="
                    f"{top_fams} does not justify it. Forcing correction."
                )
                if _fallback_override:
                    hindi_caption = _fallback_override
                elif top_fams:
                    hindi_caption = _rule_caption(top_fams, best_labels, scene_text, expressions, fam_scores)
                    if not (hindi_caption.startswith("[") and
                            any("\u0900" <= ch <= "\u097F" for ch in hindi_caption)):
                        hindi_caption = ambient_fallback_caption(top_fams, scene_text)
                break

        timeline.append({
            "start_sec":   round(start_sec, 2),
            "end_sec":     round(end_sec, 2),
            "caption":     hindi_caption,
            "families":    top_fams,
            "scene_text":  scene_text[:100],
        })

        logger.info(
            f"  [{start_sec:.1f}→{end_sec:.1f}s] families={top_fams}\n"
            f"    final='{hindi_caption}'"
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
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac",
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
    logger.info(f"PROCESSING: {vname} (PANNs + GPT-4o-mini)")
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

    speech_segments = get_speech_segments(wav_path, logger)
    try: os.unlink(wav_path)
    except: pass

    # ---- PANNs detection ----
    raw_events = detect_audio_events(
        waveform, scene_index, speech_segments, logger)
    logger.info(f"Raw events: {len(raw_events)}, "
                f"accepted: {sum(1 for e in raw_events if e.get('should_caption'))}")

    deduped = dedup_events(raw_events)
    deduped = filter_labels_for_caption(deduped, min_combined_score=0.18)

    bursts = group_into_bursts(deduped)
    logger.info(f"After dedup: {len(deduped)}")

    # Attach fresh scene context to each deduped event
    for ev in deduped:
        sc = get_scene_at(ev["timestamp_sec"], scene_index)
        ev.setdefault("scene_text",  sc.get("scene_text", ""))
        ev.setdefault("expressions", sc.get("expressions", []))

    # ---- Caption synthesis (rule-based + GPT-4o-mini) ----
    timeline = build_timeline(deduped, scene_index, logger, output_dir=output_dir,
                              speech_segments=speech_segments)
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

# ====================== MAIN ======================
def main():
    parser = argparse.ArgumentParser(
        description="Non-speech audio captioning pipeline + GPT-4o-mini")
    parser.add_argument("--video",          required=True)
    parser.add_argument("--florence-log", "-florence-log", default=None,
                        help="Path to JSONL vision log (Stage 1 output).")
    parser.add_argument("--extract-vision", action="store_true",
                        help="Stage 1: run Florence, write vision log, exit.")
    parser.add_argument("--openai-key",     default=None,
                        help="OpenAI API key (overrides OPENAI_API_KEY env).")
    args = parser.parse_args()

    global HF_TOKEN, OPENAI_API_KEY
    if args.openai_key:     OPENAI_API_KEY  = args.openai_key

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}"); return

    out = Path("results_v17") / video_path.stem
    out.mkdir(parents=True, exist_ok=True)

    if args.extract_vision:
        log    = Path(args.florence_log) if args.florence_log else (out / "florence_log.jsonl")
        logger = setup_logger(video_path.stem + "_vision", out)
        extract_vision_log(str(video_path), str(log), logger)
        logger.info(f"\nStage 1 done. Stage 2:\n"
                    f"  python {Path(__file__).name} --video {video_path} "
                    f"--florence-log {log}")
    else:
        process_video(str(video_path), out,
                      Path(args.florence_log) if args.florence_log else None)

if __name__ == "__main__":
    main() 