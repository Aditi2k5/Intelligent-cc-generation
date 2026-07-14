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
assert torch.cuda.is_available(), "CUDA required"
DEVICE = "cuda"
print(f"\nGPU: {torch.cuda.get_device_name(0)}\n")

# ====================== CONFIG ======================
SAMPLE_RATE          = 32000
WINDOW_SEC           = 0.96
HOP_SEC              = 0.20
FRAMES_PER_MINUTE    = 10
SCENE_WINDOW_SEC     = 8.0
DEDUP_GAP_SEC        = 1.5
BURST_GAP_SEC        = 2.0       # events closer than this merge into one caption
PALETTE_SIZE         = 20
SEMANTIC_THRESHOLD   = 0.24      # short-phrase embeddings
W_AUDIO              = 0.65      # trust real audio 
W_PALETTE            = 0.35      
ACCEPT_THRESHOLD     = 0.12      
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
    (["cricket"],                           0.07),
    (["insect"],                            0.09),
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
    (["owl", "hoot"],                       0.35),
    (["plucked string"],                    0.048),
    (["singing"],                           0.075),
    (["crowd", "cheering", "chatter"],      0.055),
    (["water", "stream", "splash", "rain"], 0.038),
    (["thunder"],                           0.045),
    (["fire", "crackling"],                 0.032),
    (["cattle", "sheep", "cow"],            0.025),
    # Generic / rare animal classes are AudioSet's worst false-positive
    # offenders — they fire on incidental noise, reverb, cloth rustle, etc.
    # Give them a much higher bar than specific classes like cattle/moo.
    (["duck", "quack", "goose", "honk"],    0.09),
    (["wild animal", "domestic animal"],    0.09),
    (["animal", "dog", "cat"],              0.09),
]
DEFAULT_THRESHOLD = 0.052
_threshold_cache: dict = {}

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
AMBIENT_FALLBACK_INDOOR = [
    "कमरे में हल्की आवाज़ है",
    "पास में कोई हल्की गतिविधि हो रही है",
    "कोई हल्की आवाज़ महसूस हो रही है",
]
AMBIENT_FALLBACK_GENERIC = [
    "दूर से हल्की आवाज़ आ रही है",
    "हल्की सी आवाज़ सुनाई दे रही है",
    "कोई हल्की आवाज़ सुनाई दे रही है",
]
NATURE_FAMILIES = {"animal", "bird", "insect", "cricket", "wind", "rustling",
                    "creak", "thunder", "crow"}
INDOOR_FAMILIES = {"footstep", "mechanisms", "rodents, rats, mice",
                    "environmental noise"}

def ambient_fallback_caption(families: list = None, scene_text: str = None) -> str:
    """Randomised ambient fallback so the exact same sentence doesn't
    repeat every time a low-confidence"""
    category = classify_scene_category(scene_text)
    if category and category in SCENE_PHRASE_BANKS:
        return f"[{random.choice(SCENE_PHRASE_BANKS[category])}]"

    fams = set(f.lower() for f in (families or []))
    if fams & NATURE_FAMILIES:
        pool = AMBIENT_FALLBACK_NATURE
    elif fams & INDOOR_FAMILIES:
        pool = AMBIENT_FALLBACK_INDOOR
    else:
        pool = AMBIENT_FALLBACK_GENERIC
    return f"[{random.choice(pool)}]"


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
                     threshold_ratio: float = 0.15) -> bool:
    # Lowered from 0.25 and added boundary padding: VAD segment edges are
    # imprecise (soft/trailing speech at onset/offset often gets slightly
    # clipped), and a 25% overlap requirement was letting some genuinely
    # speech-containing windows through as "AMBIENT", causing non-speech
    # captions to wrongly appear over dialogue. 
    PAD_SEC = 0.3
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

def enforce_caption_format(caption: str) -> str:
    """Force short length, square brackets, and remove punctuation."""
    if not caption:
        return "[sound]"

    # Remove existing brackets
    caption = caption.replace("[", "").replace("]", "").strip()

    # Remove punctuation
    caption = caption.replace(".", "").replace(",", "").replace("!", "").replace("?", "").strip()

    # Limit to ~10 words max
    words = caption.split()
    if len(words) > 10:
        caption = " ".join(words[:10])

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
from langchain_core.messages import HumanMessage
from langsmith import traceable

# ====================== LANGCHAIN + LANGSMITH SETUP ======================
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

    prompt = f"""You are a strict Hindi closed caption writer for non-speech sounds only.

        Rules (follow strictly):
        1. ONLY describe sounds that are clearly present in "Detected sounds".
        2. If "animal"/"duck"/"wild animal"/"domestic animal" appears BUT no specific species (cattle, moo, dog, cat, horse, owl, etc.) is ALSO listed anywhere in Detected sounds, do NOT name any animal or bird species. Also do NOT use the vague filler words "पृष्ठभूमि" (background), "वातावरण" (environment/atmosphere), "आस-पास" (nearby/around), or "हलचल" (movement/stir) — they tell the viewer nothing concrete about what the sound actually is. Instead pick from (or write something similarly concrete and varied): "हल्की सी आवाज़ सुनाई दे रही है", "दूर से हल्की आवाज़ आ रही है", "कोई हल्की आवाज़ है". If a specific species IS listed, always name that species and ignore this generic rule. If MULTIPLE specific species are listed together (e.g. Moo and Roar both present), name the one that appears first in Detected sounds specifically — do NOT collapse them into a vague collective phrase like "जानवरों की आवाज़ें" (animal sounds); naming one real species beats a vague plural covering all of them.
        3. If laughter-related labels are present, describe it as "हल्की हँसी सुनाई दे रही है".
        4. For music, name the instrument EXACTLY as it appears in "Detected sounds" (translate faithfully, do not substitute a different instrument):
        - Tabla → "संगीत के साथ तबला बज रहा है"
        - Sitar → "संगीत के साथ सितार बज रहा है"
        - Dhol → "संगीत के साथ ढोल बज रहा है"
        - Shehnai → "संगीत के साथ शहनाई बज रहा है"
        - Flute → "संगीत के साथ बांसुरी बज रहा है"
        - Harmonium → "संगीत के साथ हारमोनियम बज रहा है"
        - Violin → "संगीत के साथ वायलिन बज रहा है"
        - No specific instrument named in Detected sounds → "संगीत बज रहा है" (do NOT default to tabla when no instrument is specified)
        - Generic "Drum"/"Bass drum"/"Drum kit"/"Percussion" (not Tabla/Sitar specifically) → "ढोल-नगाड़े जैसी थाप सुनाई दे रही है"
        5. For crow-cawing sounds, ALWAYS use exactly this phrasing: "कौआ काँव-काँव कर रहा है" (do not use "कौए की", "कौवा", or any other variant/case form). For owl sounds, use "उल्लू की आवाज़ सुनाई दे रही है".
        6. Never describe time of day (रात/night, दिन/day) — you only know what was HEARD, not what time it is. Describe only the sound itself.{scene_hint}
        8. Keep captions short (max 7-8 words). No punctuation. Output ONLY inside square brackets [].
        9. Never describe visuals or actions. Only describe what can be HEARD.

        Detected sounds: {labels_str}
        Expressions: {expr_str}
        Original: {raw_caption}

        Output ONLY the caption in this exact format: [short hindi text]"""

    try:
        messages = [HumanMessage(content=prompt)]
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

    prompt = f"""You are a Hindi closed caption editor.

        Strict Rules:
        1. Keep it short and natural (max 7-8 words).
        2. Use square brackets [] only. No punctuation.
        3. If the caption names an animal/bird species (including "जानवर", "बत्तख", "हंस") but no specific species is in Detected sounds, replace it with a generic ambient description — vary the wording, don't reuse the exact same sentence every time.
        4. For music, keep whichever specific instrument (tabla/sitar/dhol/shehnai/flute/harmonium/violin) is already named in the caption — do not change it to a different instrument, and do not add "तबला" unless it was already there.
        5. Output ONLY the improved caption inside square brackets.

        Detected sounds: {labels_str}
        Current caption: {caption}

        Output ONLY the polished caption in square brackets."""

    try:
        messages = [HumanMessage(content=prompt)]
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

    # Fix incomplete captions
    if caption.strip() == "[संगीत]":
        caption = "[संगीत सुनाई दे रहा है]"
    if "सुनाई दे रही]" in caption and "है" not in caption:
        caption = caption.replace("सुनाई दे रही]", "सुनाई दे रही है]")

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
            caption = caption.replace("पक्षियों की आवाज़ गूंज रही है", "पक्षियों की चहचहाहट सुनाई दे रही है")
    
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
                inputs = fp(text=prompt, images=image,
                            return_tensors="pt").to(DEVICE, torch.float16)
                with torch.no_grad():
                    ids = fm.generate(input_ids=inputs["input_ids"],
                                      pixel_values=inputs["pixel_values"],
                                      max_new_tokens=256)
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

        NAMED_INSTRUMENTS = {"tabla", "sitar", "dhol", "shehnai",
                             "flute", "harmonium", "violin"}
        for c in candidates:
            if not c["pass"]:
                continue
            if c["label"].lower() not in NAMED_INSTRUMENTS:
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
            # Higher bar for generic/rare animal classes (Animal, Duck,
            # Wild animals, Domestic animals, etc.) — these are AudioSet's
            # noisiest classes and need strong, unambiguous evidence.
            if score < 0.28:
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
    
    # If caption mentions any animal (generic or specific, incl. duck/goose)
    # but we don't have strong, specific-species evidence, downgrade it.
    ANIMAL_MENTION_WORDS = ["जानवरों की आवाज़", "जानवर", "बत्तख", "हंस"]  # animal(s), duck, goose
    if any(w in caption for w in ANIMAL_MENTION_WORDS):
        strong_animal = any(x in labels_str for x in
                             ["cattle", "moo", "cow", "dog", "cat", "horse"])
        if not strong_animal:
            caption = ambient_fallback_caption(["animal"], scene_text)

    return caption

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
    """
    Generates natural, grammatically correct Hindi captions.
    """
    if not OPENAI_API_KEY:
        return raw_caption

    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        gender = _gender_from_scene(scene_text) if '_gender_from_scene' in globals() else "unknown"
        gender_note = f" व्यक्ति {gender} प्रतीत होता है।" if gender != "unknown" else ""

        prompt = f"""You are a professional Hindi subtitle writer for Indian TV dramas and accessibility content.

            Rewrite the following caption into **natural, correct and professional Hindi**.

            Strict Rules:
            - Use proper Hindi grammar with correct matras (मात्राएँ).
            - NEVER repeat the word "पृष्ठभूमि" more than once in the sentence.
            - Avoid robotic phrases like "पृष्ठभूमि में", "सुना जा सकता है", "आवाज़ आ रही है" repeatedly.
            - Make it sound like natural spoken Hindi used in good subtitles.
            - Maximum 12 words.
            - Do not invent any sounds that are not in the detected labels.
            - Vary the sentence structure. Make it feel human-written.

            Detected non-speech sounds: {', '.join(detected_labels)}
            Scene context: {scene_text[:200]}
            {gender_note}
            Original caption: {raw_caption}

            Write ONLY the improved Hindi caption. Do not add any explanation."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.3,
        )

        hindi_caption = response.choices[0].message.content.strip().strip('"').strip("'")

        # Safety check: if still repetitive, fall back
        if hindi_caption.count("पृष्ठभूमि") >= 2:
            logger.warning("GPT still produced repetitive 'पृष्ठभूमि'. Using cleaned version.")
            return _clean_hindi_caption(raw_caption)

        return hindi_caption

    except Exception as e:
        logger.warning(f"OpenAI Hindi generation failed: {e}")
        return raw_caption

def _clean_hindi_caption(caption: str) -> str:
    """Strong cleanup for repetitive, robotic, English-mixed, or vague Hindi.

    पृष्ठभूमि ("background") is grammatically correct but tells the viewer
    almost nothing — it could mean literally any ambient sound. Replace it
    with phrasing that at least signals *what kind* of ambient texture it
    is (natural vs. mechanical/crowd), picked from a varied pool so it
    doesn't just become a new repeated filler word.
    """
    NATURE_ALTS = ["प्रकृति की हल्की आवाज़ें हैं",
                   "हल्की प्राकृतिक ध्वनियाँ हैं",
                   "दूर से हल्की सी आवाज़ आ रही है"]
    caption = caption.replace("पृष्ठभूमि में हल्की आवाज़ें हैं", random.choice(NATURE_ALTS))
    caption = caption.replace("पृष्ठभूमि में ", "चारों ओर ")
    caption = caption.replace("पृष्ठभूमि ", "हल्की ")
    caption = caption.replace("सुना जा सकता है", "")
    caption = caption.replace("आवाज़ आ रही है", "")
    caption = caption.replace("buzzing", "भिनभिनाहट")
    caption = caption.replace("music", "संगीत")
    caption = caption.replace("Music", "संगीत")
    caption = caption.replace("background", "हल्की")
    caption = caption.replace("आस-पास", "")
    caption = caption.replace("आसपास", "")
    caption = caption.replace("हलचल", "आवाज़")
    caption = caption.replace("  ", " ").strip()

    if caption:
        caption = caption[0].upper() + caption[1:]
    return caption

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
    "music":      ["music", "musical instrument", "plucked string",
                   "singing", "flute", "string"],
    "crowd":      ["crowd", "cheering", "chatter"],
    "animal":     ["cattle", "cow", "bull", "dog", "bark", "cat", "horse",
                   "neigh", "frog", "animal", "duck", "quack", "goose",
                   "honk", "wild animal", "domestic animal"],
    "bell":       ["bell", "ring"],
    "siren":      ["siren", "alarm", "civil defense"],
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

    # NOTE: best_labels is {family_name: raw_label}. We need the actual
    # raw label strings here (e.g. "Tabla", "Sitar", "Moo"), not the dict
    # keys — using keys silently broke matching for any family whose name
    # differs from its label, and meant multi-instrument bursts always
    # fell through to whichever check appeared first in the code below.
    bl = [v.lower() for v in best_labels.values()]

    # IMPORTANT: check for a SPECIFIC species first. "Cattle, bovinae"
    # folds into the generic "animal" family (via the keyword list), same
    # as plain "Animal"/"Wild animals" do — so if the blanket "only animal
    # family present -> ambient fallback" check below ran first, a real,
    # confidently-detected cow would get swallowed into a generic ambient
    # sentence before we ever looked at what it actually was. Specific
    # species must always win over the generic fallback.
    if "moo" in bl or any("cattle" in b for b in bl):
        return "[गाय की मूँ मूँ सुनाई दे रही है]"

    if "dog" in bl or "bark" in bl or "bow-wow" in bl:
        return "[कुत्ते के भौंकने की आवाज़ सुनाई दे रही है]"

    if "cat" in bl or "meow" in bl or "purr" in bl:
        return "[बिल्ली की म्याऊं सुनाई दे रही है]"

    if "horse" in bl or any("neigh" in b for b in bl) or "clip-clop" in bl:
        return "[घोड़े की टापों की आवाज़ सुनाई दे रही है]"

    if "owl" in bl or "hoot" in bl:
        return "[उल्लू की आवाज़ सुनाई दे रही है]"

    # "Patter" was being freely interpreted by GPT as rain ("बारिश की हल्की
    # आवाज़") with no basis.
    if "patter" in bl:
        return "[हल्की थपथपाहट जैसी आवाज़ सुनाई दे रही है]"

    if "animal" in families and len(families) == 1:
        return ambient_fallback_caption(families, scene_text)

    if "laugh_soft" in bl or "laughter" in bl or "giggle" in bl:
        return "[हल्की हँसी सुनाई दे रही है]"

    # Instrument detection: pick whichever instrument family ranks highest instead of choosing tabla no matter what.
    INSTRUMENT_HINDI = {
        "tabla":   "तबला",
        "sitar":   "सितार",
        "dhol":    "ढोल",
        "shehnai": "शहनाई",
        "flute":   "बांसुरी",
        "harmonium": "हारमोनियम",
        "violin":  "वायलिन",
    }
    detected_instruments = [INSTRUMENT_HINDI[fam.lower()] for fam in families
                            if fam.lower() in INSTRUMENT_HINDI]

    # Mantra/chant: this was a real gap — it was being confidently detected
    # (e.g. combined score 0.35+) and correctly surviving all the way into
    # the burst's family list but none named it.
    has_mantra = any(f.lower() in ("mantra", "chant") for f in families)

    if has_mantra and detected_instruments:
        joined = " और ".join(detected_instruments[:2])
        return f"[मंत्रोच्चारण के साथ {joined} बज रहा है]"
    if has_mantra:
        return "[मंत्रों का उच्चारण सुनाई दे रहा है]"

    if len(detected_instruments) == 1:
        return f"[संगीत के साथ {detected_instruments[0]} बज रहा है]"
    elif len(detected_instruments) >= 2:
        # Multiple instruments genuinely detected together in the same
        # burst — name both instead of arbitrarily picking just one.
        joined = " और ".join(detected_instruments[:2])
        return f"[{joined} के साथ संगीत बज रहा है]"

    LARGE_DRUM_FAMILIES = {"drum", "bass drum", "drum kit", "percussion"}
    if any(fam.lower() in LARGE_DRUM_FAMILIES for fam in families):
        return "[ढोल-नगाड़े जैसी थाप सुनाई दे रही है]"

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
        return "[संगीत बज रहा है]"

    # Walking through undergrowth — audio+vision synthesis
    walking = any(x in scene_text.lower()
                  for x in ["walk","path","trail","moving","strolling","approaching"])
    if outdoor and walking and fset & {"rustling", "creak", "footstep"}:
        ambient = fset & {"bird","crow","cricket","insect","wind"}
        if "crow" in ambient:
            return "[कदमों की आहट के साथ कौआ काँव-काँव कर रहा है]"
        if "bird" in ambient:
            return "[कदमों की आहट के साथ पक्षियों की चहचहाहट सुनाई दे रही है]"
        return "[कदमों की आहट और पत्तों की सरसराहट सुनाई दे रही है]"

    # Multi-sound combos — all deterministic Hindi so these bypass GPT entirely and stay consistent, instead of being handed to GPT as English reference text that then gets freely reinterpreted.
    if "crow" in fset and "bird" in fset:
        if "wind" in fset:
            return "[कौआ काँव-काँव कर रहा है, पक्षी चहचहा रहे हैं, हवा भी चल रही है]"
        return "[कौआ काँव-काँव कर रहा है और पक्षियों की चहचहाहट है]"
    if "bird" in fset and "wind" in fset:
        return "[हवा के साथ पक्षियों की चहचहाहट सुनाई दे रही है]"
    if "bird" in fset and "cricket" in fset:
        return "[पक्षियों की चहचहाहट और झींगुरों की आवाज़ है]"
    if "rustling" in fset and "wind" in fset:
        return "[हवा से पत्तों की सरसराहट सुनाई दे रही है]"
    if "rustling" in fset and "creak" in fset:
        return "[पत्तों की सरसराहट और टहनियों की चरमराहट है]"
    if "crow" in fset and "wind" in fset:
        return "[कौआ काँव-काँव कर रहा है और हवा चल रही है]"
    if "water" in fset and "bird" in fset:
        return "[पानी बहने की आवाज़ और पक्षियों की चहचहाहट है]"


    if "crow" in fset:
        return "[कौआ काँव-काँव कर रहा है]"
    if "bird" in fset:
        return "[पक्षियों की चहचहाहट सुनाई दे रही है]"
    if "cricket" in fset:
        return "[झींगुरों की आवाज़ सुनाई दे रही है]"
    if "insect" in fset:
        return "[कीड़ों की आवाज़ सुनाई दे रही है]"
    if "wind" in fset:
        return "[हवा चलने की आवाज़ सुनाई दे रही है]"
    if "rustling" in fset:
        return "[पत्तों की सरसराहट सुनाई दे रही है]"
    if "creak" in fset:
        return "[टहनियों की चरमराहट सुनाई दे रही है]"

    # Single/unclear family with no confident, specific identification 
    return ambient_fallback_caption(families, scene_text)

# ====================== BURST CONSOLIDATION ======================
def build_timeline(events: list, scene_index: list, logger, output_dir: Path = None) -> list:    
    accepted = [ev for ev in events if ev.get("should_caption")]
    if not accepted:
        return []

    accepted.sort(key=lambda e: e["timestamp_sec"])

    # Require music-family detections to persist across at least 2
    # consecutive accepted frames within a short window. 
    MUSIC_PERSISTENCE_SEC = 3.0
    music_fams = {"music", "tabla", "sitar", "dhol", "shehnai",
                  "flute", "harmonium", "violin"}
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
    MIN_SIREN_SPAN_SEC = 4.0  
    for burst in bursts:
        start_sec = burst[0]["timestamp_sec"]
        end_sec   = burst[-1]["timestamp_sec"] + 1.8


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

        # Sort by score, but ensure high-value families are never squeezed out of the top-3 by a higher-scoring but
        # less important family
        PRIORITY_FAMILIES = {"laugh_soft", "laugh_full"}
        ranked   = sorted(fam_best, key=lambda f: fam_best[f]["combined_score"], reverse=True)
        priority = [f for f in ranked if f in PRIORITY_FAMILIES]
        rest     = [f for f in ranked if f not in PRIORITY_FAMILIES]
        top_fams = (priority + rest)[:3]
        best_labels = {f: fam_best[f]["label"] for f in top_fams}

        # If a SPECIFIC species (moo, cattle, dog, cat, horse) is present
        # in this burst, drop any co-occurring GENERIC animal.
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

        # Use scene from highest-scoring event
        best_ev     = max(burst, key=lambda e: e["combined_score"])
        scene_text  = best_ev.get("scene_text", "")
        expressions = best_ev.get("expressions", [])

        # 1. Rule-based caption
        raw_caption = _rule_caption(top_fams, best_labels, scene_text, expressions)

        # If _rule_caption already gave us a confident, deterministic Hindi
        # phrase (specific species, named instrument, owl/crow, laughter,
        # or a scene-grounded ambient fallback), use it directly instead of
        # sending it through GPT. GPT has repeatedly been shown to
        # paraphrase these into inconsistent variants
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
        if ("पृष्ठभूमि" in hindi_caption or 
            any(word in hindi_caption.lower() for word in ["buzzing", "music", "background"])):
            hindi_caption = _clean_hindi_caption(hindi_caption)

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
    timeline = build_timeline(deduped, scene_index, logger, output_dir=output_dir)
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
    parser.add_argument("--florence-log",   default=None,
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

    out = Path("results_v14") / video_path.stem
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