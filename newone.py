# -*- coding: utf-8 -*-
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
import panns_assets
panns_assets.ensure_labels()  # must run before panns_inference is imported
from panns_inference import AudioTagging
from dotenv import load_dotenv  
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Music2Emotion is a git submodule next to this file; in the PyInstaller build
# it is bundled under the app's _internal folder. MUSIC2EMO_DIR overrides both.
MUSIC2EMO_SRC_DIR = Path(os.environ.get("MUSIC2EMO_DIR") or
                         Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "Music2Emotion")
sys.path.insert(0, str(MUSIC2EMO_SRC_DIR))


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
FRAMES_PER_MINUTE    = 20
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
    (["vehicle", "car", "engine", "motor", "traffic"], 0.28),
    (["plucked string"],                    0.048),
    (["singing"],                           0.075),
    (["crowd", "cheering", "chatter"],      0.055),
    (["thunder", "lightning", "thunderstorm", "bolt"], 0.025),
    (["fire", "crackling"],                 0.032),
    (["cattle", "sheep", "cow"],            0.025),
    (["water", "stream", "splash", "rain", "waterfall"], 0.035),
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
    (["telephone", "ringtone", "phone"],     0.06),
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
        "झाड़ियों में पत्तों की सरसराहट है",
        "पेड़ों की टहनियाँ हिल रही हैं",
        "सूखी पत्तियों के चरमराने की आवाज़ है",
        "घास की सरसराहट सुनाई दे रही है",
    ],
    "path_walking": [
        "कदमों की आहट सुनाई दे रही है",
        "पैरों तले मिट्टी की आहट है",
        "कदमों की धीमी आहट आ रही है",
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

# ====================== SLC DICTIONARY (client-provided caption bank) ======================
# Source: Hindi - SLC - Dictionary.docx. These are the exact, client-approved caption
# strings. Final captions must be drawn from this bank wherever a match applies; GPT is
# only a fallback for events this bank does not cover (see generate_hindi_caption).
#
# Per explicit instruction: music captions must NEVER name an instrument, and no caption
# (music or otherwise) may name an animal/bird species. A few dictionary entries do name
# an instrument or a species (e.g. "गंभीर वाद्ययंत्र ध्वनि", "उल्लू के बोलने की आवाज़",
# "घोड़ों की टापों की आवाज़", "वाद्य संगीत पर नृत्य") — those are intentionally excluded
# from automatic selection below and are never picked by the rule engine.

# ---- Music captions, grouped by the emotional register they express ----
# Selection is driven by Florence's visual scene description (and any detected facial
# expressions), never by which instrument PANNs happened to detect.
SLC_MUSIC_MOOD_CAPTIONS = {
    "scary": ["[डरावना संगीत]", "[अशुभ संगीत]"],
    "tense": ["[संदिग्ध पार्श्व संगीत]", "[तनावपूर्ण पार्श्वसंगीत]", "[चिंताजनक पार्श्वसंगीत]",
              "[तनाव भरा संगीत]", "[संशय भरा संगीत]", "[संशय भरी धुन]"],
    "mysterious": ["[रहस्यमय पार्श्व संगीत]", "[रहस्यमयी संगीत जारी है]"],
    "sad": ["[भावनात्मक पार्श्व संगीत]", "[उदास भरी धुन]", "[दुखभरा संगीत]",
            "[मार्मिक संगीत]", "[भावपूर्ण पार्श्व संगीत]"],
    "romantic": ["[रूमानी संगीत]"],
    "victorious": ["[विजयी संगीत]", "[विजयी धुन]"],
    # NOTE: "तीव्र संगीत" (intense) used to sit in this same random pool as
    # "उत्साहित"/"जोशीला" (excited/energetic-passionate, both clearly read
    # as fun) — but "intense" doesn't communicate fun at all, and random
    # selection meant a genuinely upbeat, cheering-crowd circus passage had
    # a 1-in-3 chance of coming out as "तीव्र संगीत" instead, which reads as
    # mismatched even though the CATEGORY choice (energetic) was correct.
    # Real, confirmed case: a long continuous upbeat stretch right next to
    # applause/cheering kept flickering to "तीव्र संगीत" via this exact
    # random pool. Removed it from the active default pool — "उत्साहित"/
    # "जोशीला" alone are the unambiguously-fun words in this category.
    "energetic": ["[उत्साहित पार्श्वसंगीत]", "[जोशीला संगीत]"],
    "happy": ["[खुशनुमा संगीत]", "[मधुर धुन]"],
    "dramatic": ["[नाटकीय संगीत]", "[नाटकीय संगीत जारी है]", "[गंभीर पार्श्व संगीत]"],
    "neutral": ["[मधुर धुन]"],
}

# Checked in this priority order — first category whose keywords match the visual
# scene text / expressions wins. "neutral" (theme music) is the final catch-all.
#
# IMPORTANT: built from real Florence output across three circus episodes' worth
# of logs, not guessed. Florence almost never describes a scene using an emotion
# adjective ("happy", "scared") — it's a literal activity/composition description
# ("two people embracing in a warm hug", "man on a trapeze performing an aerial
# stunt", "group of people dancing in a tent"). A keyword list built only around
# emotion words matches almost nothing in practice and falls through to neutral
# every time — which is exactly the "मधुर धुन for everything, even a circus"
# problem. These lists lead with the ACTIVITY/CONTEXT Florence actually
# describes, with the original emotion-adjective keywords kept as a secondary
# catch for scenes that do happen to state one directly.
SLC_MUSIC_MOOD_KEYWORDS = [
    ("scary", ["scared", "fear", "frightened", "horror", "terrified", "monster",
               "ghost", "creepy", "screaming"]),
    ("tense", ["tense", "worried", "anxious", "nervous", "suspicious", "doubt",
               "threat", "danger", "uneasy", "wary", "concerned",
               "precarious", "risky", "interrogat", "questioning",
               "confrontation", "standoff", "weapon", "chasing"]),
    ("mysterious", ["mysterious", "mystery", "shadow", "secret", "unknown", "fog",
                     "mist", "dark figure", "hidden"]),
    ("sad", ["sad", "crying", "cry", "grief", "mourning", "funeral", "tears",
             "sorrow", "heartbroken", "loss"]),
    ("romantic", ["romantic", "romance", "love", "kiss",
                   "holding hands", "affection", "embracing", "embrace", "hug",
                   "hugging", "gazing at each other", "intimate"]),
    ("victorious", ["victory", "winning", "triumph", "champion", "celebration of",
                     "success", "achievement", "cheering crowd"]),
    ("energetic", ["running", "chase", "chasing", "action", "fight", "battle",
                    "excited", "energetic", "urgent", "hurry",
                    "trapeze", "acrobat", "acrobatics", "aerial stunt", "stunt",
                    "tightrope", "juggling", "somersault", "balancing",
                    "climbing", "jumping", "rope course",
                    "spider web", "bicycle stunt"]),
    ("happy", ["happy", "joyful", "smiling", "laughing", "cheerful", "festive",
               "playful", "fun", "dancing", "dance", "celebrating", "wedding"]),
    # NOTE: "intense" used to be a bare keyword here — the exact same
    # ambiguity as "तीव्र संगीत" (also meaning "intense"), which was just
    # removed from the "energetic" category's random pool for the same
    # reason: it doesn't reliably signal SERIOUS/dramatic on its own, and is
    # just as likely to describe an exciting, thrilling circus stunt as a
    # solemn scene. Real, confirmed case: this fired on what was actually a
    # continuous, clearly-fun energetic passage, producing "नाटकीय संगीत"
    # right after a stretch that should have stayed energetic/happy.
    ("dramatic", ["serious", "dramatic", "argument",
                  "solemn", "grave", "determined", "resolute", "stern"]),
]

MUSIC_LIKE_FAMILIES = INSTRUMENT_FAMILIES | {
    "music", "drum", "bass drum", "drum kit", "percussion", "plucked string",
    "singing", "musical instrument",
}

# Single source of truth for "is this visually a nature/outdoor setting" —
# used everywhere a burst decides whether an animal/bird/insect cue is
# plausible. Previously three separate copies of this list existed
# (filter_labels_for_caption, final_cleanup, _rule_caption) and only one of
# them (the "natural_context" list inside _rule_caption) had "woods",
# "jungle", "trees", "grass", "leaves", "bushes" — the other two didn't.
# Florence describes this show's forest content as "woods"/"trees"/"grass"
# far more often than literally "forest", so those two narrower copies were
# treating a large fraction of genuinely outdoor/forest scenes as "not
# nature", which caused final_cleanup to force-drop otherwise-correct animal/
# bird captions right after _rule_caption had produced them (visible as
# "SKIPPED vague/empty caption" in the logs even though the caption itself
# was fine). One shared list now backs all three call sites.
NATURE_SCENE_KEYWORDS = [
    "forest", "outdoor", "nature", "field", "village", "farm", "animal",
    "bird", "cattle", "cow", "dog", "cat", "horse", "woods", "jungle",
    "trees", "tree", "grass", "leaves", "bushes", "meadow", "vegetation",
    "wilderness", "greenery", "hut", "rural",
]
TRAFFIC_SCENE_KEYWORDS = [
    "road", "street", "traffic", "vehicle", "car", "auto", "bus", "truck",
    "motor", "engine", "driving", "travel", "motorcycle", "scooter", "highway",
]

def is_nature_scene(scene_text: str) -> bool:
    st = (scene_text or "").lower()
    return any(k in st for k in NATURE_SCENE_KEYWORDS)

def is_traffic_scene(scene_text: str) -> bool:
    st = (scene_text or "").lower()
    return any(k in st for k in TRAFFIC_SCENE_KEYWORDS)

RINGTONE_HIGH_CONFIDENCE_WITHOUT_SCENE = 0.35

def _has_phone_scene_support(scene_text: str) -> bool:
    """A bird tweet that rises in pitch/volume can acoustically resemble a
    phone ring's ascending trill closely enough that PANNs classifies it as
    Telephone/Ringtone with real, non-marginal confidence (0.16-0.25+) and
    NO bird/chirp candidate anywhere in the same window to fall back on —
    confirmed from a real log where "Telephone" won at 0.247 with zero scene
    text available and zero competing bird label. Text-matching can't correct
    a label PANNs never proposed, but it CAN require actual corroboration
    (either a phone visibly in shot, or overwhelming confidence) before
    trusting an isolated moderate-confidence ringtone read, the same way
    vehicle already requires either a traffic scene or a very high score.

    NOTE: also excludes a specific, confirmed watermark artifact — this
    show has a persistent "Waves Ott for..." promotional overlay burned
    into the frames, and Florence sometimes describes it as "a person
    holding a phone with the words 'Waves Ott for...'" — the word "phone"
    here is Florence trying to describe the watermark graphic, not a real
    phone in the story. Real, confirmed case: this exact phrase was the
    ONLY scene-text evidence behind a burst that traced back to a bright
    musical accent (Ringtone spiked to 0.716 for one frame, then Music
    dominated at 0.5-0.7 for the rest) — a real phone conversation
    wouldn't leave literally no other trace in the scene text across the
    whole burst. If this watermark phrase is present, phone-related
    keywords are not trusted as genuine scene support."""
    st = (scene_text or "").lower()
    if "waves ott" in st or "ott for" in st:
        return False
    return any(k in st for k in ["phone", "mobile", "smartphone", "cell phone", "telephone", "call"])

SOLO_MELANCHOLIC_INSTRUMENTS = {"violin", "sitar", "sarangi", "santoor", "flute", "shehnai", "tanpura"}
RHYTHMIC_INSTRUMENTS = {"tabla", "dhol", "mridangam", "manjira", "drum", "bass drum", "drum kit", "percussion"}

# ============================================================================
# MUSIC EMOTION RECOGNITION via Music2Emo (real audio, not instrument-guessing)
# ============================================================================
# The instrument-character heuristic that used to live here (SOLO_MELANCHOLIC_
# INSTRUMENTS -> "sad", RHYTHMIC_INSTRUMENTS -> "energetic") was always a
# workaround for not having a real emotion signal — it inferred mood from
# WHICH instrument PANNs guessed, and this whole session has repeatedly found
# PANNs' fine-grained instrument identification to be unreliable (trumpet read
# as an elephant, sitar/tabla mixups, xylophone read as a phone). Guessing
# mood from a shaky instrument ID just inherits that unreliability one layer
# up. Retired in favor of actually listening to the audio.
#
# Model: Music2Emo (amaai-lab/music2emo, https://github.com/AMAAI-Lab/Music2Emotion)
#   - Genuinely trained on real human emotion annotations (PMEmo, DEAM,
#     EmoMusic, MTG-Jamendo) via a MERT backbone — unlike using a general
#     audio-caption model (e.g. CLAP) zero-shot for a task it was never
#     actually trained on.
#   - Real published accuracy on the coarse valence/arousal quadrant this
#     code uses: ~80-91% depending on task (see the paper, arXiv:2502.03979).
#     Fine-grained categorical accuracy is much lower — which is exactly why
#     this only asks it for a QUADRANT (positive/negative x high/low arousal),
#     not a direct 9-way category, and leans on scene-text to pick a specific
#     word within whichever quadrant the audio confidently lands in.
#   - NOT pip-installable. Setup required: `git clone
#     https://github.com/AMAAI-Lab/Music2Emotion`, `pip install -r
#     requirements.txt` inside it (python 3.10, torch==2.3.1), then make sure
#     that cloned directory is on PYTHONPATH so `from music2emo import
#     Music2emo` resolves. If it's not set up, everything below fails soft —
#     the pipeline falls back to the scene-text-only mood classifier and logs
#     a warning, it does not crash.
#
# SETUP NOT VERIFIED END-TO-END: I confirmed Music2Emo's documented API
# against its model card (the predict() call, its input/output shape, the
# valence/arousal 1-9 scale) but could not actually run it myself against
# real footage in this environment — unlike every other fix this session,
# which was verified by direct execution. Test this against a few of your
# own clips with known-obvious moods before trusting it in production.

_MUSIC2EMO_MODEL = None
_MUSIC2EMO_LOAD_FAILED = False
_MUSIC2EMO_DIR = None  # cached location of the Music2Emotion folder, set once during load

def _music2emo_workdir() -> Path:
    """Music2Emo reads saved_models/ and inference/data/ relative to the cwd
    and writes ./temp_out and ./output there, so it has to run from a
    writable folder holding those. That is normally the Music2Emotion folder
    itself; if that is read-only (e.g. the app was installed under Program
    Files) the model files are copied once into the app's data folder."""
    if os.access(MUSIC2EMO_SRC_DIR, os.W_OK):
        return MUSIC2EMO_SRC_DIR
    work = Path(os.environ.get("PLANETREAD_DATA_DIR", Path.home() / ".planetread")) / "Music2Emotion"
    for sub in ("saved_models", "inference"):
        if not (work / sub).exists():
            shutil.copytree(MUSIC2EMO_SRC_DIR / sub, work / sub)
    return work

def _get_music2emo_model(logger=None):
    """Lazily load Music2Emo once per process. Returns None (never raises)
    if the library isn't set up — callers must treat None as "fall back to
    scene-text", not as an error to propagate."""
    global _MUSIC2EMO_MODEL, _MUSIC2EMO_LOAD_FAILED, _MUSIC2EMO_DIR
    log = logger or logging.getLogger()
    if _MUSIC2EMO_MODEL is not None:
        return _MUSIC2EMO_MODEL
    if _MUSIC2EMO_LOAD_FAILED:
        return None
    try:
        from music2emo import Music2emo
        # Music2Emo loads its own checkpoint (e.g. "saved_models/J_all.ckpt")
        # using a path RELATIVE to the current working directory, so switch
        # into its folder (see _music2emo_workdir) while it loads. Restored in
        # a finally block so a failure never leaves the pipeline in the wrong
        # directory.
        music2emo_dir = str(_music2emo_workdir())
        _MUSIC2EMO_DIR = music2emo_dir
        old_cwd = os.getcwd()
        try:
            os.chdir(music2emo_dir)
            _MUSIC2EMO_MODEL = Music2emo()
        finally:
            os.chdir(old_cwd)
        log.info("Music2Emo loaded successfully — music mood will use real audio valence/arousal.")
    except Exception as e:
        log.warning(f"Music2Emo not available ({e}) — music mood will use scene-text only. "
                    f"See setup instructions above SOLO_MELANCHOLIC_INSTRUMENTS in this file.")
        _MUSIC2EMO_LOAD_FAILED = True
        _MUSIC2EMO_MODEL = None
    return _MUSIC2EMO_MODEL

# The full-episode waveform is loaded once, well before per-burst captioning
# happens, but pick_music_caption() is several call layers away from that
# code (build_timeline -> _rule_caption -> pick_music_caption), and none of
# those functions currently pass audio data through. Threading a new
# parameter through all three signatures (and every existing call site) is a
# much larger, riskier change than this feature needs — a module-level
# handle set once, right where the waveform is already available, is the
# pragmatic choice here. Call set_audio_context(waveform, sample_rate) once
# after loading the audio (this is already wired into detect_audio_events
# below, right where it receives the waveform as a parameter).
_AUDIO_WAVEFORM = None
_AUDIO_SAMPLE_RATE = None

def set_audio_context(waveform, sample_rate: int):
    global _AUDIO_WAVEFORM, _AUDIO_SAMPLE_RATE
    _AUDIO_WAVEFORM = waveform
    _AUDIO_SAMPLE_RATE = sample_rate

def get_music_valence_arousal(start_sec: float, end_sec: float, logger=None):
    """Extract this burst's audio segment and run it through Music2Emo.
    Returns (valence, arousal, predicted_moods) on Music2Emo's native 1-9
    scale (5 = neutral midpoint), or (None, None, []) if the model isn't
    set up, no audio context was registered, or the segment is too short to
    analyze meaningfully. predicted_moods is Music2Emo's own separate
    multi-label mood tag list — used downstream to disambiguate WITHIN a
    quadrant when more than one dictionary category is plausible there
    (e.g. "happy" vs "romantic" both sit in positive+low-arousal; the tags
    themselves usually say which one it actually is).

    NOTE: every outcome here — success, skip, or failure — is logged via the
    SAME logger + file handler your per-episode .log files already use
    (logging.getLogger(name) + FileHandler, set up in setup_logger()), not
    the bare `logging` module's root logger. Root-logger messages don't
    propagate into that named logger's file — they'd only ever show up in
    your terminal (if at all), never in the .log file you'd actually share
    for debugging. Every line below is prefixed "M2E:" so you can grep for
    it directly: `grep "M2E:" your_episode.log`."""
    log = logger or logging.getLogger()
    model = _get_music2emo_model(log)
    if model is None:
        return None, None, []
    if _AUDIO_WAVEFORM is None or _AUDIO_SAMPLE_RATE is None:
        log.warning(f"M2E: no audio context registered for burst {start_sec:.1f}-{end_sec:.1f}s "
                    f"(set_audio_context() was never called) — falling back to scene-text.")
        return None, None, []
    sr = _AUDIO_SAMPLE_RATE
    start_sample = max(0, int(start_sec * sr))
    end_sample = min(len(_AUDIO_WAVEFORM), int(end_sec * sr))
    segment = _AUDIO_WAVEFORM[start_sample:end_sample]
    # Music2Emo (like most MER models) expects a real musical excerpt — a
    # burst under ~3 seconds doesn't give it enough to judge; let the
    # scene-text fallback handle short bursts instead of calling the model
    # on a fragment it was never evaluated on.
    if len(segment) < sr * 3:
        log.info(f"M2E: burst {start_sec:.1f}-{end_sec:.1f}s too short ({len(segment)/sr:.1f}s < 3s) "
                 f"— skipped, using scene-text fallback.")
        return None, None, []
    tmp_path = None
    old_cwd = os.getcwd()
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        sf.write(tmp_path, segment, sr)
        # Same relative-path concern as model loading — if predict() also
        # touches any of Music2Emo's own internal resources by relative
        # path, it needs the same cwd. tmp_path itself is already absolute
        # (from tempfile.mkstemp), so this doesn't affect finding the audio
        # file, only whatever Music2Emo does internally during inference.
        if _MUSIC2EMO_DIR:
            os.chdir(_MUSIC2EMO_DIR)
        result = model.predict(tmp_path)
        valence, arousal = result.get("valence"), result.get("arousal")
        predicted_moods = result.get("predicted_moods") or []
        log.info(f"M2E: burst {start_sec:.1f}-{end_sec:.1f}s -> valence={valence}, arousal={arousal}, "
                 f"moods={predicted_moods}")
        return valence, arousal, predicted_moods
    except Exception as e:
        log.warning(f"M2E: inference FAILED for burst {start_sec:.1f}-{end_sec:.1f}s: {e} "
                    f"— falling back to scene-text.")
        return None, None, []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        os.chdir(old_cwd)

def valence_arousal_to_quadrant(valence, arousal, center: float = 5.0, margin: float = 0.75):
    """Music2Emo's scale is 1-9, 5 = neutral. Require a real margin away
    from center before trusting a quadrant call — a score sitting right on
    5/5 isn't confidently anything, and falls through to the scene-text
    path rather than being forced into a quadrant it barely qualifies for."""
    if valence is None or arousal is None:
        return None
    if abs(valence - center) < margin and abs(arousal - center) < margin:
        return None
    positive = valence >= center
    high_arousal = arousal >= center
    if positive and high_arousal:
        return "positive_high_arousal"
    if not positive and high_arousal:
        return "negative_high_arousal"
    if not positive and not high_arousal:
        return "negative_low_arousal"
    return "positive_low_arousal"

# Which of the EXISTING SLC_MUSIC_MOOD_CAPTIONS category keys are plausible in
# each quadrant. Built word-by-word from the actual dictionary (not guessed):
# every phrase in "happy"/"energetic"/"victorious" is positive+high-arousal,
# every phrase in "scary"/"tense" is negative+high-arousal, "sad" is
# negative+low-arousal. NOTE: "happy" is ALSO reachable from positive+LOW
# arousal, not just high — confirmed real case: valence 5.8-6.2 with arousal
# just under the 5.0 midpoint (4.1-4.9, barely "low") kept landing here with
# ONLY "romantic" as an option, even though the model's own predicted_moods
# for these exact bursts were ['happy','fun','funny','upbeat','groovy',
# 'party'] — genuinely pleasant/fun music that just isn't hyper-energetic,
# not romantic at all. Calm-happy music is real and common; forcing every
# positive+calm reading into "romantic" was the bug, not the quadrant math.
VALENCE_AROUSAL_QUADRANTS = {
    "positive_high_arousal": ["happy", "energetic", "victorious"],
    "negative_high_arousal": ["scary", "tense"],
    "negative_low_arousal":  ["sad"],
    "positive_low_arousal":  ["romantic", "happy"],
}
# "dramatic" (नाटकीय/गंभीर) is only reachable from the NEGATIVE high-arousal
# quadrant. It used to straddle both polarities on the theory that a
# triumphant climax could be "dramatic" too — but "विजयी" (victorious)
# already exists specifically for that positive-triumphant case, and
# "उत्साहित"/"जोशीला" already cover happy-energetic. In practice, allowing
# "dramatic" into the positive quadrant just produced mismatched captions on
# genuinely upbeat/fun passages (real, confirmed case: नाटकीय संगीत showing
# up right after a long clearly-fun, cheering-crowd circus stretch). Reserve
# it for the negative/serious-intense reading only.
DRAMATIC_STRADDLES_QUADRANTS = {"negative_high_arousal"}
# "mysterious" doesn't sit on a valence-arousal axis at all — in music
# emotion research (e.g. the GEMS model) it's closer to its own "tension /
# wonder" dimension than a point on a 2D positive/negative-energetic/calm
# grid. Never assign it from the audio model's quadrant; it's decided purely
# by scene text, same as before this whole system existed — checked as an
# explicit bypass before the audio model even runs, in pick_music_caption.

# General principle: the audio itself has a character (a sustained solo
# string/wind instrument reads differently than a rhythmic drum), and that
# character should combine with whatever the scene shows — for many
# different settings, not one specific hardcoded case. This table says which
# moods each instrument character can plausibly support; it's used only to
# back up a WEAK visual cue into an actual mood pick, never to force a mood
# on its own (a tabla playing over an ordinary walking shot should still just
# be neutral, as already agreed) and never to override a scene that clearly
# states its own mood via SLC_MUSIC_MOOD_KEYWORDS above (that stays the
# primary signal).
INSTRUMENT_MOOD_LEAN = {
    "sad":       SOLO_MELANCHOLIC_INSTRUMENTS,
    "energetic": RHYTHMIC_INSTRUMENTS,
}

# Visual cues too weak or generic to assign a mood by themselves (unlike the
# stronger cues in SLC_MUSIC_MOOD_KEYWORDS) — but combined with an instrument
# whose character matches, they're a real signal. "A man sitting alone" is
# ambiguous on its own (could be a neutral establishing shot); "a man sitting
# alone" UNDER A SOLO VIOLIN is sad. "People clapping" is ambiguous alone;
# "people clapping" UNDER A DHOL/TABLA is energetic.
WEAK_SCENE_CUES = {
    "sad":       ["sitting alone", "standing alone", "by himself", "by herself",
                  "alone in", "all alone", "lonely", "solitary figure",
                  "staring", "lying down", "resting quietly"],
    "energetic": ["clapping", "cheering", "audience watching", "crowd watching",
                  "spotlight", "performing on stage", "applauding"],
}

_LAST_CAPTION_WAS_RANDOM_GUESS = False  # set by pick_music_caption, read by build_timeline
                                          # right after each _rule_caption call, for QC flagging
_LAST_BURST_SCORES = {"valence": None, "arousal": None, "predicted_moods": [], "candidates": [], "quadrant": None, "candidate_counts": {}, "decision_reason": None}
    # captures Music2Emo's raw output for the current burst, regardless of
    # whether it ended up confident or ambiguous — read by build_timeline
    # right after _rule_caption, same pattern as the guess flag above, so
    # the frontend review UI can show the actual numbers behind a caption,
    # not just whether it was flagged.

def pick_music_caption(scene_text: str, signal_text: str = "", families: set = None,
                        start_sec: float = None, end_sec: float = None, logger=None) -> str:
    """Pick a dictionary music caption. Primary signal is now Music2Emo
    listening to the actual burst audio (a real, trained emotion model,
    narrowed to a valence/arousal quadrant it's actually accurate at); scene
    text picks the specific word within whichever quadrant the audio
    confidently lands in, and remains the ONLY signal for "mysterious" (which
    doesn't fit a valence/arousal grid) and the sole fallback whenever the
    audio model is unavailable or its reading is too close to neutral to
    trust. No instrument name is ever produced by this function."""
    global _LAST_CAPTION_WAS_RANDOM_GUESS, _LAST_BURST_SCORES
    _LAST_CAPTION_WAS_RANDOM_GUESS = False  # reset at the start of every call
    _LAST_BURST_SCORES = {"valence": None, "arousal": None, "predicted_moods": [], "candidates": [], "quadrant": None, "candidate_counts": {}, "decision_reason": None}

    # A radio is a sound *source*, not a mood — special-case it to the
    # dictionary's own "radio music" entry regardless of visual emotion.
    if "radio" in (signal_text or "").lower():
        return "[रेडिओ संगीत]"

    text = (scene_text or "").lower()
    families = families or set()

    # The dictionary has "थीम संगीत" (theme music) as its own distinct entry,
    # separate from the generic neutral fallback "मधुर धुन". Reserve it
    # specifically for actual title-card/opening-credits moments. This has to
    # be a NARROW check: nearly every frame in this show's Florence output
    # mentions a watermark/on-screen text overlay (e.g. "...with the words
    # \"Waves Ott for SLC\" written across"), so matching on "text"/"words"/
    # "watermark" alone would wrongly fire on almost every ordinary scene. A
    # real title card is different — the ENTIRE frame is the text, e.g. "a
    # black background with the text \"The Sangam Period...\"" — there's no
    # actual scene content, unlike "a person standing in a forest ... with a
    # watermark". Only that specific pattern counts as a title card.
    if "black background" in text and "the text" in text:
        return "[थीम संगीत]"

    # "Mysterious" bypasses the audio model entirely — it structurally can't
    # be placed on a valence/arousal grid, so it's decided purely by scene
    # text, checked before anything audio-based runs.
    for mood, keywords in SLC_MUSIC_MOOD_KEYWORDS:
        if mood == "mysterious" and any(kw in text for kw in keywords):
            return random.choice(SLC_MUSIC_MOOD_CAPTIONS[mood])

    # "Bonfire" is Florence's literal, non-emotional word for what is very
    # often a funeral pyre in this kind of historical-drama content — it
    # never says "funeral" or "pyre" outright (checked exhaustively across a
    # whole episode, zero hits), so the existing "sad" keyword list (which
    # DOES include "funeral"/"mourning"/"grief") never matches. Real,
    # confirmed case: a sustained ~34-second passage over "a group of people
    # standing around a bonfire in a field" — static, no movement, no
    # celebration — came out as generic "[मधुर धुन]" throughout, when it was
    # very likely a solemn cremation scene. But "bonfire" alone is genuinely
    # ambiguous — a festive Holi/Lohri bonfire is equally real content — so
    # this only fires when there's no festive/celebratory language ALSO
    # present in the same scene text (which would flip the read entirely).
    if "bonfire" in text and not any(kw in text for kw in
            ["dancing", "dance", "celebrating", "festive", "colorful", "wedding", "playful"]):
        return random.choice(SLC_MUSIC_MOOD_CAPTIONS["sad"])

    # Same principle, two more common literal-Florence patterns for sad/
    # somber scenes that don't use any emotion word at all. NOT verified
    # against real footage the way the bonfire case above was — built from
    # the same reasoning (what would Florence's plain, activity-based
    # vocabulary actually say for this), but flagged honestly as unconfirmed
    # until seen in an actual log.
    #
    # 1) A deathbed/illness scene: someone lying down with others gathered
    # around them. Requires BOTH parts together — "lying in bed" alone is
    # just as likely a mundane resting scene, and "gathered around" alone
    # could be anything (a meal, a conversation). Together, it's a
    # specific, much less ambiguous pattern.
    LYING_DOWN_PHRASES = ["lying in bed", "lying on a bed", "lying on the bed", "lying down"]
    GATHERED_AROUND_PHRASES = ["gathered around", "surrounded by", "standing around", "sitting around"]
    if any(kw in text for kw in LYING_DOWN_PHRASES) and any(kw in text for kw in GATHERED_AROUND_PHRASES):
        return random.choice(SLC_MUSIC_MOOD_CAPTIONS["sad"])

    # 2) A lone, glum figure at night. "Sitting alone" by itself is already
    # handled much later as a weak last-resort cue (needs a matching solo
    # instrument too, since alone-in-frame isn't necessarily sad on its own
    # — could be a neutral establishing shot). Combined specifically with a
    # night/darkness cue, it's a stronger, more standalone signal worth
    # trusting earlier and without needing instrument corroboration.
    ALONE_PHRASES = ["sitting alone", "standing alone", "by himself", "by herself", "alone in"]
    NIGHT_PHRASES = ["at night", "late at night", "dark room", "dim light", "moonlight"]
    if any(kw in text for kw in ALONE_PHRASES) and any(kw in text for kw in NIGHT_PHRASES):
        return random.choice(SLC_MUSIC_MOOD_CAPTIONS["sad"])

    # Direct expressive words Florence occasionally does use, that describe
    # a downcast appearance without naming the emotion outright — safe to
    # trust standalone since they're specifically melancholy-coded, unlike
    # more generic words that need the combined checks above.
    for mood, keywords in [("sad", ["glum", "forlorn", "dejected", "downcast", "despondent"])]:
        if any(kw in text for kw in keywords):
            return random.choice(SLC_MUSIC_MOOD_CAPTIONS[mood])

    # 1) Ask the audio itself. If Music2Emo is set up and confidently places
    # this burst in a quadrant, narrow the candidate categories to that
    # quadrant (plus "dramatic" when the quadrant is one it can straddle),
    # then let scene text pick the specific word within that narrowed set —
    # the audio decides the rough mood, the scene breaks the tie.
    #
    # EXCEPTION: skip the audio model entirely when chant/mantra is also
    # present, even if it's too weak to fully commit to the mantra caption
    # itself. Real, confirmed case: a burst with genuine chanting present
    # (families=['music','chant','mantra']) got Music2Emo's own valence/
    # arousal reading of "sad" (explicit 'sad' tag in predicted_moods) —
    # very plausibly a real blind spot, not noise: Music2Emo was trained on
    # Western pop/film-score emotion datasets, and devotional chanting's
    # slow, repetitive, minor-key-adjacent qualities can superficially
    # resemble "sad" music acoustically to a model that's never seen this
    # style, even though the actual content is solemn/devotional, not sad.
    # Scene-text (or neutral) is a safer fallback than trusting a model on
    # exactly the content type it's least likely to have been trained on.
    if not (families and (families & {"chant", "mantra"})) and start_sec is not None and end_sec is not None:
        valence, arousal, predicted_moods = get_music_valence_arousal(start_sec, end_sec, logger=logger)
        _LAST_BURST_SCORES = {"valence": valence, "arousal": arousal, "predicted_moods": predicted_moods,
                               "candidates": [], "quadrant": None, "candidate_counts": {}, "decision_reason": None}
        quadrant = valence_arousal_to_quadrant(valence, arousal)
        log = logger or logging.getLogger()
        if quadrant:
            candidates = list(VALENCE_AROUSAL_QUADRANTS.get(quadrant, []))
            if quadrant in DRAMATIC_STRADDLES_QUADRANTS:
                candidates.append("dramatic")
            _LAST_BURST_SCORES["candidates"] = candidates
            _LAST_BURST_SCORES["quadrant"] = quadrant
            for mood, keywords in SLC_MUSIC_MOOD_KEYWORDS:
                if mood in candidates and any(kw in text for kw in keywords):
                    log.info(f"M2E: quadrant={quadrant}, scene-text picked '{mood}' from {candidates}")
                    _LAST_BURST_SCORES["decision_reason"] = "scene-text match"
                    return random.choice(SLC_MUSIC_MOOD_CAPTIONS[mood])
            # No scene-text tiebreak, but more than one category shares this
            # quadrant (e.g. "happy" and "romantic" both sit in positive+low
            # arousal) — Music2Emo's own separate mood tags usually say
            # which one it actually is. Real, confirmed case: valence/
            # arousal landed here with tags ['happy','fun','funny','upbeat',
            # 'groovy','party'] — genuinely pleasant/fun content, no
            # romantic tag anywhere — that used to get randomly assigned
            # "romantic" anyway purely because it was the only quadrant
            # member before "happy" was added here too. Checked before the
            # final blind random choice, and only used when it actually
            # narrows the field — never invents a category the quadrant
            # didn't already allow.
            if len(candidates) > 1 and predicted_moods:
                moods_lower = " ".join(predicted_moods).lower()
                MOOD_TAG_HINTS = {
                    "happy":    ["happy", "fun", "funny", "upbeat", "groovy", "party",
                                 "joy", "cheerful", "playful", "uplifting"],
                    "romantic": ["love", "romantic", "sexy", "sentimental", "tender"],
                    "sad":      ["sad", "melancholic", "depressive", "emotional"],
                    "energetic": ["energetic", "action", "powerful", "aggressive"],
                    "scary":    ["dark", "horror", "scary", "creepy"],
                    "tense":    ["suspense", "thriller", "tense", "anxious"],
                }
                # NOTE: this used to be "does ANY hint word match" per
                # candidate, requiring exactly one candidate to match at
                # all — but a single ambiguous tag sitting among a large,
                # overwhelmingly one-sided tag list could tie with the real
                # signal and cancel the whole disambiguation. Real, confirmed
                # case: 22 tags including happy/fun/funny/groovy/party/
                # upbeat/uplifting (7 clear happy hits) alongside a single
                # "sexy" tag (1 romantic hit) — both candidates "matched" at
                # all, so the exact-one-match rule bailed to random instead
                # of recognizing the obvious 7-to-1 majority. Counting actual
                # hits per candidate and requiring a genuine majority (not
                # just presence) fixes this without losing the original
                # safety: a real, close tie (e.g. 1-vs-1) still correctly
                # falls through to random rather than picking arbitrarily.
                candidate_counts = {
                    c: sum(1 for h in MOOD_TAG_HINTS.get(c, []) if h in moods_lower)
                    for c in candidates
                }
                _LAST_BURST_SCORES["candidate_counts"] = candidate_counts
                ranked = sorted(candidate_counts.items(), key=lambda kv: -kv[1])
                if ranked[0][1] > 0 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
                    chosen = ranked[0][0]
                    log.info(f"M2E: quadrant={quadrant}, mood-tag majority picked '{chosen}' "
                             f"{candidate_counts} (tags: {predicted_moods})")
                    _LAST_BURST_SCORES["decision_reason"] = "mood-tag majority"
                    return random.choice(SLC_MUSIC_MOOD_CAPTIONS[chosen])
            # Still no tiebreak — genuinely ambiguous within the quadrant,
            # but still meaningfully narrowed by real audio evidence, better
            # than a blind guess across all nine categories.
            if candidates:
                chosen = random.choice(candidates)
                log.info(f"M2E: quadrant={quadrant}, no tiebreak available, randomly picked '{chosen}' from {candidates}")
                _LAST_CAPTION_WAS_RANDOM_GUESS = True
                _LAST_BURST_SCORES["decision_reason"] = "random pick, no tiebreak"
                return random.choice(SLC_MUSIC_MOOD_CAPTIONS[chosen])
        elif valence is not None and arousal is not None:
            log.info(f"M2E: valence={valence}/arousal={arousal} too close to neutral to call a quadrant — using scene-text fallback.")

    # 2) Audio model unavailable, or its reading was too close to neutral to
    # trust — fall back to the full scene-text priority match across every
    # category, same as before this system existed.
    for mood, keywords in SLC_MUSIC_MOOD_KEYWORDS:
        if any(kw in text for kw in keywords):
            return random.choice(SLC_MUSIC_MOOD_CAPTIONS[mood])

    # 3) No strong keyword matched either. Before giving up to plain neutral,
    # check whether the scene has a WEAK cue that only means something once
    # combined with the instrument's character — the general principle being
    # that the audio of a setting and the setting itself should jointly
    # decide the caption. A tabla over an ordinary walking shot still stays
    # neutral here, since "walking" isn't a listed weak cue for any mood.
    for mood, cue_keywords in WEAK_SCENE_CUES.items():
        supporting_instruments = INSTRUMENT_MOOD_LEAN.get(mood, set())
        if families & supporting_instruments and any(kw in text for kw in cue_keywords):
            return random.choice(SLC_MUSIC_MOOD_CAPTIONS[mood])

    # Truly nothing matched anywhere — genuinely no signal to base a mood
    # on, not even a weak one. Deliberately NOT flagged as a guess: this is
    # the safe, conservative default when there's no evidence either way,
    # which is very often just correctly neutral (most background score
    # music isn't dramatically moody) — flagging every one of these would
    # flood QC review with noise rather than highlighting genuinely
    # uncertain moments. Only an actual random pick BETWEEN competing,
    # evidenced-but-tied options (above) counts as a real guess.
    return random.choice(SLC_MUSIC_MOOD_CAPTIONS["neutral"])

# ---- Non-music dictionary captions, keyed by the raw PANNs label / signal text
# that should trigger them. Checked before the generic ad hoc phrase table so the
# exact client-approved wording wins whenever it applies. Deliberately excludes
# owl/cricket/horse-hoof style species-specific entries per the no-animal rule.
def _kw_hit(text: str, keywords: list) -> bool:
    """Word-boundary-safe keyword match at the START only — not a plain
    substring check, and not a full \\bword\\b match either. Plain `keyword
    in text` silently matches inside unrelated words: "ice" matched inside
    "mice" (from the PANNs label "Rodents, rats, mice") and inside "police"
    (from "Police car (siren)"), which is exactly what caused every real
    "rodents" burst in a run to come out as "[बर्फ की आवाज़]" (ice/snow
    sound) — a real animal event, mislabeled purely by a text-matching bug,
    not a detection problem. A start-of-word boundary fixes that (there's no
    boundary between 'm' and 'i' inside "mice", so "ice" can no longer match
    there) while still allowing intentional prefix keywords like "vibrat"
    (-> vibrating/vibration) or "door clos" (-> closing/closed) to keep
    matching without needing an end boundary too."""
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw), text):
            return True
    return False

SLC_NON_MUSIC_MAP = [
    # --- doors / entry ---
    (["door open", "door opening"],                    "[दरवाजा खोलते हुए]"),
    (["door clos"],                                     "[दरवाज़ा बंद करते हुए]"),
    # NOTE: bare "knock" is deliberately NOT mapped here. "Knock" is a
    # generic AudioSet class for any sharp rap/tap sound — a cup set down on
    # a wooden table, a table rap, wood being tapped, or an actual door knock
    # can all plausibly produce this same label. Unconditionally assuming it
    # means "someone knocked on a door" is the same class of bug as "Wind
    # noise (microphone)" and "Trumpet" earlier — a generic/ambiguous class
    # being confidently narrowed to one specific real-world meaning. Handled
    # instead in dictionary_non_music_caption() below, gated on whether the
    # scene actually shows a door.
    # --- weapons / violence / impact (dictionary items 10-13,15,17,24,30,59-64) ---
    (["gunshot", "gun shot", "gun fire single"],         "[गोली की आवाज़]"),
    (["gunfire", "machine gun"],                         "[गोलियों की आवाज़]"),
    (["gun cock", "gun lock", "racking", "chamber"],     "[बंदूक लॉक की आवाज़]"),
    (["grenade"],                                        "[ग्रेनेड की आवाज़]"),
    (["explosion"],                                      "[धमाका]"),
    (["fight", "scuffle", "struggle", "punch", "brawl"], "[लड़ाई की आवाज़]"),
    (["fall", "falling", "thump", "thud"],               "[गिरने की आवाज़]"),
    (["groan", "grunt", "moan"],                         "[दर्द से करहाते हुए]"),
    (["cutting", "chop", "slice", "knife"],               "[काटने की आवाज़]"),
    (["choke", "choking", "gag"],                        "[गला घुटना]"),
    (["chain"],                                          "[चैन की आवाज़]"),
    (["shush", "hush", "quiet down", "silence gesture"], "[चुप करते हुए]"),
    # --- vehicles / transit ---
    (["helicopter"],                                     "[हेलीकाप्टर की आवाज़]"),
    (["train horn"],                                     "[ट्रेन का हॉर्न बजाना]"),
    (["train"],                                          "[ट्रेन की आवाज़]"),
    (["engine"],                                         "[गाड़ी इंजन की आवाज़]"),
    (["siren", "police car (siren)", "ambulance (siren)"], "[पुलिस का सायरन]"),
    (["ice", "snow", "sleet"],                           "[बर्फ की आवाज़]"),
    (["boat", "water vehicle", "rowboat", "canoe", "kayak",
      "sailboat", "sailing ship", "motorboat", "speedboat"], "[नाव की आवाज]"),
    # --- phone / devices ---
    (["telephone bell ringing", "ringtone", "telephone"], "[फोन बजा]"),
    (["cellphone", "cell phone"],                        "[सेलफ़ोन की घंटी]"),
    (["vibrat"],                                          "[फ़ोन वाइब्रेट करता है]"),
    (["hang up", "phone down", "receiver"],              "[फोन रखने की आवाज़]"),
    (["message", "notification", "sms", "text tone"],    "[मेसेज भेजने की आवाज़]"),
    (["beep"],                                            "[बीप बीप]"),
    (["microphone on", "mic on"],                        "[माइक चालू]"),
    (["microphone", "mic feedback", "mic noise"],        "[माइक साउंड]"),
    (["television"],                                     "[टीवी चल रही है]"),
    (["radio announcement", "pa announcement", "loudspeaker announcement"], "[रेडियो पर अनाउंसमेंट]"),
    (["radio"],                                           "[रेडिओ पर आवाज़]"),
    # --- bird / nature-adjacent, including PANNs' known confusions ---
    (["bird vocalization", "bird call", "bird song", "chirp",
      "printer", "sewing machine", "typewriter"],   "[पंछियों की चहचहाहट]"),
    (["hubbub", "chatter", "babble", "crowd"],       "[अस्पष्ट बातचीत]"),
    (["growl", "growling", "snarl"],                     "[गुर्रा रहा है]"),
    # --- movement ---
    (["run", "running", "jog", "jogging", "sprint"],      "[लोगों का भागना]"),
    (["walk", "footstep"],                                "[क़दमों की आवाज़]"),
    # --- human vocal / breath (dictionary items 28,38,48,49,51,54,81,82,91,92,108) ---
    (["cough"],                                           "[खाँसना]"),
    (["cry", "crying", "baby cry", "sobbing"],            "[रोना]"),
    (["sob", "whimper", "wheeze"],                          "[सिसकी]"),
    (["sniff"],                                           "[नाक सुड़कती है]"),
    (["pant", "panting"],                                 "[हाँफना]"),
    (["gasp", "startle"],                                 "[चौकना]"),
    (["sigh"],                                            "[आह भरता है]"),
    (["deep breath", "heavy breath", "inhale deeply", "exhale deeply"], "[गहरी सांस ली]"),
    (["breath", "breathing", "inhal", "exhal"],           "[सांस ली]"),
    (["angry voice", "angry tone", "irritated tone"],     "[गुस्से से भरा स्वर]"),
    (["heartbeat", "heart beat", "pulse"],                "[धड़कन की आवाज़]"),
    # --- misc ambience ---
    (["clock", "tick-tock", "ticking"],                   "[घड़ी की टिक-टिक लगातार जारी]"),
    (["cutlery", "silverware", "white noise"],            "[शांत आवाज़]"),
]
# Deliberately NOT wired in:
#  - item 3, 107: name a specific instrument -> excluded per the no-instrument rule
#  - item 103, 106: name a specific animal species (owl, horse) -> excluded per the no-animal rule
#  - item 66, 68, 70: speech/dialogue-related notes (foreign language, inaudible/
#    indistinct two-person dialogue) -> out of scope for this non-speech ambient
#    sound pipeline; speech is handled separately by the transcript/dubbing side
#  - item 5 (अस्पष्ट): a status marker ("unclear") used internally to detect and
#    drop bad captions, not itself an output caption
#  - item 102, 105: (cup thrown, beads) too narrow/rare to give a reliable audio
#    trigger keyword; left for GPT's dictionary-first fallback if it ever comes up

def dictionary_non_music_caption(best_labels: dict, scene_text: str = "") -> str:
    """Match raw detected label text against the SLC dictionary's non-music
    entries and return the exact client-approved phrase. Returns "" if no
    dictionary entry applies to this burst.

    IMPORTANT: only checks the DOMINANT (highest-priority/highest-scoring)
    family's own label — best_labels preserves top_fams order, so its first
    entry is the burst's actual lead signal. This used to pool every co-
    occurring family's label into one string and match a keyword found
    ANYWHERE in it, which let a weak secondary family hijack the whole
    burst's caption away from a much stronger one. Real evidence: a burst
    dominated by "Applause" at 0.309-0.329 across most of its frames (no
    dictionary entry of its own, meant to fall through to the later
    SOUND_PHRASE_MAP applause handling) got captioned "[लोगों का भागना]"
    (people running) instead, because "Run" only won a single anomalous
    frame at a lower 0.276 and had nothing to do with the burst's actual
    dominant sound — but pooled matching let it win anyway since "run" has
    a dictionary entry and "applause" doesn't. Checking only the lead
    family's own label means a non-dominant match can no longer override a
    dominant non-match; if the top family has no dictionary phrase, this
    correctly returns "" and lets the rest of _rule_caption's own
    score-ordered family loop (which does have an applause entry) decide."""
    if not isinstance(best_labels, dict) or not best_labels:
        return ""
    dominant_label = next(iter(best_labels.values()), "") or ""
    signal_text = dominant_label.lower()

    # Horse/clip-clop is a deliberate, narrow exception to "dominant family
    # only": it's specifically corroborated (see detect_audio_events) against
    # real footstep/walk/run evidence elsewhere in the SAME burst before it's
    # even allowed to become an event at all, so by the time it reaches here
    # it's already been vetted — checking the full pooled text for this one
    # specific pair is safe and intentional, not the general bug being fixed
    # below.
    full_signal_text = " ".join((v or "").lower() for v in best_labels.values())
    if _kw_hit(full_signal_text, ["horse", "clip-clop"]):
        return "[घोड़ों की टापों की आवाज़]"

    # Bare "knock" only becomes a door caption when the scene actually shows
    # a door — otherwise it's just as likely to be a cup set down, a table
    # rap, or wood being tapped. Falls through to the generic wood-sound
    # phrase in the ordinary SOUND_PHRASE_MAP handling below when there's no
    # door in view, rather than confidently naming a door that isn't there.
    if _kw_hit(signal_text, ["knock"]):
        if _kw_hit((scene_text or "").lower(), ["door", "doorway", "entrance"]):
            return "[दरवाजे पर दस्तक]"
        return ""

    # "Tick-tock"/"Tick" without a corroborating "Clock" label or an indoor/
    # room scene is the same class of confusion as "Printer"/"Sewing
    # machine" being mistaken for bird trilling, or "Wind noise
    # (microphone)" being mistaken for a phone — a real, passing PANNs
    # class (confirmed: Tick-tock 0.290, Tick 0.256) with essentially no
    # actual "Clock" corroboration (0.038, failing) and a scene that shows
    # a dramatic outdoor/ritual moment, not a room with a ticking clock.
    # Require either the Clock label itself or genuine indoor/room support
    # before committing to this specific claim.
    if _kw_hit(signal_text, ["tick-tock", "ticking"]) and not _kw_hit(signal_text, ["clock"]):
        st = (scene_text or "").lower()
        if not _kw_hit(st, ["room", "indoor", "inside", "house", "kitchen", "office"]):
            return ""

    # "Wind noise (microphone)" is a real PANNs/AudioSet class name — it means
    # wind buffeting the RECORDING microphone (a technical handling-noise
    # artifact), not an in-scene mic sound effect. It genuinely contains the
    # word "microphone", so it was matching the "मic sound" entries below on
    # a real word-boundary hit — this isn't a text-matching bug like ice/mice,
    # it's a wrong understanding of what that specific AudioSet class means.
    # The dictionary has no wind/ambient-noise entry at all, so the honest
    # answer here is "don't force it into an unrelated caption" — let it fall
    # through uncaptioned rather than mislabel it as a microphone sound.
    if _kw_hit(signal_text, ["wind noise"]):
        return ""

    for keywords, caption in SLC_NON_MUSIC_MAP:
        if _kw_hit(signal_text, keywords):
            return caption
    return ""

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
    "पेड़ों की पत्तियों की सरसराहट सुनाई दे रही है",
    "झाड़ियों में पत्तों की सरसराहट है",
    "घास और झाड़ियों की सरसराहट सुनाई दे रही है",
]
AMBIENT_FALLBACK_GENERIC = []
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
        return random.choice(AMBIENT_FALLBACK_NATURE)
    if AMBIENT_FALLBACK_GENERIC:
        return random.choice(AMBIENT_FALLBACK_GENERIC)
    return ""

def ambient_fallback_caption(families: list = None, scene_text: str = None) -> str:
    """Bracket-wrapped FINAL fallback — used only as a true last resort
    when GPT itself errors out and no adaptive help is possible at all."""
    hint = ambient_fallback_hint(families, scene_text)
    return f"[{hint}]" if hint else ""


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
        _panns_model = AudioTagging(checkpoint_path=str(panns_assets.ensure_checkpoint()), device=DEVICE)
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

2. MUSIC SPECIFICATIONS (CRITICAL — NO INSTRUMENTS, EVER):
   - NEVER name any musical instrument (no तबला, सितार, ढोल, शहनाई, बांसुरी, हारमोनियम, वायलिन, गिटार, पियानो, or any other instrument) under any circumstances, even if one is listed in "Detected sounds".
   - Describe music ONLY by its emotional register, inferred from the scene context: e.g. "भावनात्मक पार्श्व संगीत" (emotional), "रहस्यमय पार्श्व संगीत" (mysterious), "तनावपूर्ण पार्श्वसंगीत" (tense), "डरावना संगीत" (scary), "खुशनुमा संगीत" (happy), "विजयी संगीत" (victorious), "रूमानी संगीत" (romantic), "नाटकीय संगीत" (dramatic). If no clear mood is evident, use "मधुर धुन".

3. HINDI ORTHOGRAPHY & SYNTAX:
   - Bird sounds (crow, owl, or any other): Always use "पक्षियों की चहचहाहट सुनाई दे रही है" — do not name the species.
   - Laughter: Use "हल्की हँसी सुनाई दे रही है".

4. OBJECTIVITY & BANNED WORDS (CRITICAL OTT QC RULE):
   - BANNED vague filler words: "पृष्ठभूमि" (background), "वातावरण" (environment/atmosphere), "आस-पास" (nearby/around), "हलचल" (movement/stir), "कोलाहल" (commotion — not a client-approved term).
   - BANNED subjective/dramatic adjectives: "डरावनी", "भयावह", "सुरीली", "मनमोहक". ("मधुर" is allowed ONLY inside the fixed phrase "मधुर धुन" used for neutral/unclear music per rule 2 — never combine it with any other word.)
   - BANNED time-of-day references: Do NOT describe night/day (रात/दिन). You only know what was HEARD.

0. DICTIONARY-FIRST (READ BEFORE WRITING ANYTHING):
   - This client has a fixed, pre-approved bank of Hindi caption phrases (below). If ANY of these phrases plausibly matches what "Detected sounds" describes, output that exact phrase and nothing else — do not paraphrase it, reword it, or invent a more "natural-sounding" alternative.
   - Only write a new phrase from scratch if genuinely nothing in this bank fits. A new phrase must still follow every rule below (grounding, no instruments/species, banned words, length).
   - Approved bank: शांत आवाज़ | दरवाजा खोलते हुए | गन की आवाज़ | लड़ाई की आवाज़ | गोलियों की आवाज़ | गिरने की आवाज़ | गोली की आवाज़ | टीवी चल रही है | रेडिओ पर आवाज़ | हेलीकाप्टर की आवाज़ | गाड़ी की आवाज़ | क़दमों की आवाज़ | बर्फ की आवाज़ | खाँसना | ट्रेन की आवाज़ | ट्रेन का हॉर्न बजाना | रेडियो पर अनाउंसमेंट | पंछियों की चहचहाहट | गाड़ी इंजन की आवाज़ | मुँह से आवाज़ | दरवाज़ा बंद करते हुए | रोना | हँसते हुए | फोन बजा | फोन रखने की आवाज़ | चिल्लाना | धड़कन की आवाज़ | डरावनी आवाज़ | हँसी | हाँफना | सांस ली | सिसकी | चौकना | बीप बीप | गहरी सांस ली | मोबाईल बज रहा है | दरवाजे पर ठोकने की आवाज | लोगों का भागना | डर से चिल्लाना | गोली का चलना | ग्रेनेड की आवाज़ | दर्द से करहाते हुए | धमाका | काटने की आवाज़ | गला घुटना | चुप करते हुए | अश्रव्य संवाद | अस्पष्ट बातचीत | विदेशी भाषा में | फ़ोन वाइब्रेट करता है | घड़ी की टिक-टिक लगातार जारी | नाक सुड़कती है | मेसेज भेजने की आवाज़ | दरवाजे पर दस्तक | आह भरता है | दरवाजा बंद होता है | सेलफ़ोन की घंटी | गुर्रा रहा है | पुलिस का सायरन | मधुर धुन
   - Known model confusions to watch for: PANNs frequently mislabels rapid bird trilling as "Printer" or "Sewing machine" (both are high-frequency repetitive-pulse sounds that confuse the classifier) — if the visual scene has no office/sewing context, treat these as bird sound and use "पंछियों की चहचहाहट". Crowd murmur/hubbub/indistinct group chatter -> "अस्पष्ट बातचीत", never an invented word like "कोलाहल".
   - CRITICAL — do not default to a phone/ringtone entry when uncertain: "फोन बजा", "सेलफ़ोन की घंटी", "फ़ोन वाइब्रेट करता है", and "मोबाईल बज रहा है" may ONLY be used when "Detected sounds" itself explicitly says "Telephone", "Ringtone", "Phone", "Cellphone", or "Mobile". If the detected label is generic or says something like "Ambient/unclear sound", picking a phone-related phrase is not a safe guess — a rising bird call and a phone's ascending ring trill are acoustically similar enough that this is a known, real confusion, and defaulting to a phone caption on weak/uncertain input repeats that mistake instead of avoiding it. On genuinely uncertain input, prefer "शांत आवाज़" instead.

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
        return ""


@traceable(name="polish_hindi_caption")
def polish_hindi_caption(caption: str, detected_labels, logger) -> str:
    label_values = list(detected_labels.values()) if isinstance(detected_labels, dict) else detected_labels
    labels_str = ", ".join(label_values) if label_values else "None"

    system_prompt = f"""You are a Quality Control (QC) Inspector for Hindi SDH Timed-Text on major OTT platforms.

Your sole job is to audit and polish the input Hindi subtitle card to ensure strict OTT platform compliance:
1. FORMATTING: Wrap entirely in square brackets [...] with NO internal punctuation (no '।', '.', '!', '?').
2. CHARACTER LIMIT: Ensure word count does NOT exceed 6-8 words (max 42 characters) for CPS compliance.
3. ANIMAL/BIRD GENERALIZATION: If the caption names a specific animal or bird species (गाय, कुत्ता, बिल्ली, घोड़ा, कौआ, उल्लू, etc.), replace it with the generic "जानवर की आवाज़" (for animals) or "पक्षियों की चहचहाहट" (for birds) — do not name exact species.
4. NO INSTRUMENTS: If the caption names any musical instrument (तबला, सितार, ढोल, शहनाई, बांसुरी, हारमोनियम, वायलिन, गिटार, पियानो, etc.), replace it with a mood-based generic music phrase instead (e.g. "मधुर धुन", "भावनात्मक पार्श्व संगीत") — instruments must never be named.
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
    if any(word in caption for word in ["आस-पास", "आसपास", "हलचल", "इर्द-गिर्द"]):
        caption = re.sub(r"(?:आस-पास|आसपास|हलचल|इर्द-गिर्द)", "", caption)
        caption = re.sub(r"\s{2,}", " ", caption).strip()
        caption = caption.replace(" की आवाज़ सुनाई", " आवाज़ सुनाई")
        if caption in ("[]", "[ ]", ""):
            caption = ""

    # Fix incomplete captions (previously this patched them by appending
    # "सुनाई दे रहा/रही है" — no longer wanted, so just leave them as a
    # clean noun phrase instead).
    if caption.strip() == "[संगीत]":
        caption = "[संगीत बज रहा है]"

    # Remove all generic vague/ambiguous captions, including light/noise fillers.
    for bad in ["अस्पष्ट ध्वनि", "अस्पष्ट आवाज़"]:
        if bad in caption:
            return ""

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
        return ""

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

    # Register the raw waveform for pick_music_caption's Music2Emo call later
    # — this is the earliest point in the pipeline that has the waveform, and
    # captioning happens well after this function returns.
    set_audio_context(waveform, SAMPLE_RATE)

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
            scene_phone_hint = any(k in scene["scene_text"].lower() for k in ["phone", "telephone", "mobile", "cell phone"])
            scene_thunder_hint = any(k in scene["scene_text"].lower() for k in ["thunderstorm", "storm", "lightning", "thunder", "rain"])
            is_phone_label = _kw_hit(label.lower(), ["telephone", "ringtone", "phone"])
            is_thunder_label = _kw_hit(label.lower(), ["thunder", "lightning", "bolt", "storm"])

            if is_phone_label and (scene_phone_hint or raw >= 0.02):
                passed = True
            elif is_thunder_label and (scene_thunder_hint or raw >= 0.025):
                passed = True
            else:
                passed = combined >= ACCEPT_THRESHOLD

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
                    "scene_text":         scene["scene_text"][:220],
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
            # A winning "Horse"/"Clip-clop" needs the SAME scrutiny as a
            # rescued (non-winning) one gets a few lines below — otherwise
            # this exact gap lets it through unconditionally just because it
            # narrowly beat equally-weak competitors in one frame. Real,
            # confirmed case: Clip-clop (0.207) "won" over Horse (0.185) and
            # Animal (0.170) — all three within 0.04 of each other, i.e. not
            # a real separation, just noise — while the only genuine
            # corroborating evidence, "Walk, footsteps" (0.060), FAILED its
            # own threshold. Scene was a woman standing near a microphone;
            # nothing horse-related in sight. Require either real footstep
            # corroboration in the same frame, or a high enough score to
            # stand on its own regardless (a genuinely loud, unambiguous
            # clip-clop shouldn't need a footstep to also be detected).
            if _kw_hit(best["label"].lower(), ["horse", "clip-clop"]):
                frame_passing_labels = " ".join(c["label"].lower() for c in candidates if c["pass"])
                has_footstep = _kw_hit(frame_passing_labels, ["walk", "footstep", "run"])
                if not has_footstep and best["combined_score"] < 0.35:
                    logger.info(f"  → WINNER {best['label']} REJECTED "
                                f"(no footstep corroboration, score {best['combined_score']:.3f} < 0.35)")
                    best = None
        if best:
            events.append(best)

        # Keep near-miss passing detections as secondary events, even when they
        # lose the single-winner vote for this 0.2s window. Without this, a
        # genuinely-detected priority sound (laughter, a phone ring, thunder,
        # metal, mantra/chant, an instrument, a specific animal roar/bark/whip)
        # that scores a hair below the frame's top pick is thrown away entirely
        # and never becomes an "event" — so it can never appear in a burst's
        # family list, no matter how well-supported it was. This was the exact
        # cause of "Laughter"/"Chuckle, chortle"/"Snicker" (all ✓, 0.15-0.19)
        # vanishing behind a narrowly-higher "Domestic animals, pets" (0.197)
        # winner, which then got captioned as a plain animal sound.
        SECONDARY_PRESERVE_FAMILIES = INSTRUMENT_FAMILIES | {
            "laugh_soft", "laugh_full", "snicker", "human_reaction",
            "ringtone", "thunder", "metal", "mantra", "chant",
        }
        # Horse/clip-clop specifically: neither "exotic" (safe to always trust)
        # nor should they be silently dropped — real evidence shows both
        # sides. In a music-heavy trapeze scene, "Horse"/"Clip-clop" at
        # ~0.15 combined were a spurious co-detection alongside dominant
        # Music (rhythmic movement/backing track mistaken for hoofbeats). But
        # in a real walking scene, "Horse"/"Clip-clop" at a near-identical
        # ~0.16 score alongside dominant "Walk, footsteps" were a genuine,
        # corroborated horse — the dictionary has its own exact phrase for
        # this ("[घोड़ों की टापों की आवाज़]") that was being missed. The
        # distinguishing signal isn't the confidence score (they're nearly
        # identical in both cases) — it's what ELSE is passing in the same
        # frame: real footstep/walk/run evidence corroborates a real horse;
        # a music/instrument winner with no footstep evidence doesn't.
        frame_passing_labels = " ".join(c["label"].lower() for c in candidates if c["pass"])
        has_footstep_corroboration = _kw_hit(frame_passing_labels, ["walk", "footstep", "run"])

        for c in candidates:
            if not c["pass"]:
                continue
            if best and c["label"] == best["label"]:
                continue  # already recorded as the winner event above
            is_whip = "whip" in c["label"].lower()
            is_exotic_animal = _is_exotic_animal_label(c["label"])
            is_priority_family = _family(c["label"]) in SECONDARY_PRESERVE_FAMILIES
            is_corroborated_horse = (_kw_hit(c["label"].lower(), ["horse", "clip-clop"])
                                      and has_footstep_corroboration)
            if not (is_whip or is_exotic_animal or is_priority_family or is_corroborated_horse):
                continue
            events.append({
                "timestamp_sec":      round(start_sec, 2),
                "label":              c["label"],
                "raw_confidence":     c["raw"],
                "palette_score":      c["palette"],
                "combined_score":     c["combined"],
                "should_caption":     True,
                "scene_text":         scene["scene_text"][:220],
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
    "elephant", "roar", "roaring", "growl",
    "lion", "tiger", "owl", "hoot",
}

EXOTIC_ANIMAL_LABELS = {"elephant", "roar", "roaring", "growl",
                         "lion", "tiger", "roaring cats", "lions", "tigers"}

def _is_exotic_animal_label(label: str) -> bool:
    """Loud, distinctive animal sounds that are hard to confuse with anything
    else (roar, elephant, tiger, growl) — safe to trust on weak evidence and
    to always preserve even when they lose a per-frame vote to Music."""
    ll = (label or "").lower()
    return any(k in ll for k in EXOTIC_ANIMAL_LABELS)

def _is_specific_animal_label(label: str) -> bool:
    ll = (label or "").lower()
    if ll in SPECIFIC_ANIMAL_LABELS:
        return True
    # Only explicit sound descriptors or named animal cues should count as
    # a real roar/bark/whinny signal. Broad family labels like "wild animals"
    # are too generic and cause the caption to anchor to the wrong frame.
    return any(k in ll for k in [
        "elephant", "roar", "roaring", "growl",
        "lion", "tiger", "roaring cats", "lions", "tigers",
        "bark", "bow-wow", "meow", "purr", "moo", "neigh",
        "whinny", "clip-clop", "howl", "hoot", "cattle",
        "dog", "cat", "horse", "cow"
    ])

def filter_labels_for_caption(events: list, min_combined_score: float = 0.18) -> list:
    """
    Filter out weak detections before sending to LLM.
    This is the fix against random animal captions.
    """
    # A repeated pattern across many frames is itself real corroborating
    # evidence, even when each individual instance is too weak on its own —
    # real, confirmed case: "Chirp, tweet" genuinely passed its own class
    # threshold (0.14-0.18) repeatedly across many consecutive frames, but
    # never once reached the bird/crow-specific 0.50 bar below, and the
    # scene wasn't recognized as "nature" either — so EVERY instance got
    # discarded and the whole burst fell back to generic "[शांत ध्वनि]"
    # despite real, sustained, audible chirping throughout. Pre-count how
    # many bird/crow-family candidates exist anywhere in this whole event
    # list before filtering starts, so a genuinely repeated pattern can get
    # a lower bar than a single, possibly-spurious one-off reading.
    _bird_candidate_count = sum(1 for ev in events if _family(ev.get("label", "")) in ("bird", "crow"))

    filtered = []
    for ev in events:
        score = ev.get("combined_score", 0)
        label = ev.get("label", "")
        fam   = _family(label)
        scene_text = (ev.get("scene_text") or "").lower()
        thunder_scene = any(k in scene_text for k in ["thunderstorm", "storm", "lightning", "thunder", "rain"])
        nature_scene = is_nature_scene(scene_text)
        traffic_scene = is_traffic_scene(scene_text)

        if thunder_scene and fam in {"animal", "bird", "crow", "vehicle"}:
            continue

        # Loud, unambiguous animal sounds (roar/elephant/tiger/growl) should always
        # pass so they can anchor burst timing, regardless of scene context. Quiet,
        # easily-confused specific labels (meow, horse/clip-clop, dog, cow) are
        # deliberately NOT given this free pass — they fall through to the normal
        # animal/nature-scene scrutiny below, since these are exactly the labels
        # that PANNs confuses with sustained violin notes, rhythmic backing
        # tracks, and footstep-like textures during music-heavy segments.
        if _is_exotic_animal_label(label):
            if score >= 0.12:  # lower bar for specific sounds, just need something real
                filtered.append(ev)
            continue

        # Horse/clip-clop that reaches this point already passed a real
        # corroboration check upstream (detect_audio_events only creates
        # this event when genuine footstep/walk/run evidence exists in the
        # SAME frame — see is_corroborated_horse there) — it doesn't need a
        # second, independent gate re-litigating plausibility. NOTE: this
        # used to be `if score < 0.40 or not nature_scene: continue`, the
        # same OR-bug as the animal check below, and a 0.40 score floor that
        # would have rejected the confirmed real horse case entirely (0.159-
        # 0.168 combined, corroborated but well under 0.40). Given it's
        # already vetted, trust it the same way whip/ringtone/laugh get a
        # low, unconditional bar a few lines below.
        #
        # CRITICAL ORDERING: this check must run BEFORE the generic
        # "animal"/"snake" block, not after — "horse" (unlike "clip-clop")
        # is itself a literal keyword in SOUND_FAMILIES' "animal" bucket, so
        # _family("Horse") returns "animal", not some separate "horse"
        # family. That routed it straight into the stricter generic-animal
        # gate below, which requires 0.55 without scene support — and a
        # real, corroborated "Horse" at 0.159-0.352 doesn't clear that,
        # even though it's already been vetted by footstep corroboration
        # upstream. Confirmed real regression: this exact case (a screenshot
        # showing a horse mid-jump) went from wrongly captioned as generic
        # footsteps to producing no dictionary caption at all once the
        # stricter animal bar was added, because "Horse" got caught by a
        # gate meant for uncorroborated generic "Animal" readings.
        if _kw_hit(label.lower(), ["clip-clop", "neigh", "whinny", "horse"]):
            if score >= 0.10:
                filtered.append(ev)
            continue

        if fam in ("animal", "snake") and not _is_exotic_animal_label(label):
            if _kw_hit(label.lower(), ["elephant", "roar", "roaring", "growl","lion","tiger","roaring cats","lions","tigers"]):
                filtered.append(ev)
                continue
            # NOTE: this was `if score < 0.45 or not nature_scene: continue` —
            # an OR, meaning a candidate needed BOTH high confidence AND
            # visual nature support just to survive, discarding it if EITHER
            # was missing. That's backwards from the stated intent ("fix
            # against random animal captions" — i.e. filter out weak noise,
            # not filter out strong signals lacking Florence's confirmation).
            # Real, confirmed case: a generic "Animal" at 0.634 combined —
            # very strong, unambiguous evidence, corroborated by a specific
            # "Roaring cats (lions, tigers)" in the same burst — got
            # discarded here purely because the scene was "two people
            # sitting on a bench" (no nature keywords). Changed to AND: only
            # discard when BOTH the score is weak AND there's no scene
            # support — matching how vehicle's confidence gate already
            # works elsewhere in this file.
            #
            # BUT a bare generic "Animal" with no specific/exotic label
            # backing it up anywhere in the burst is weaker evidence than
            # one that's corroborated (like the roar case was, by "Roaring
            # cats" surviving independently via its own exotic carve-out
            # just above). Confirmed real case: "Animal" alone at 0.504,
            # with a scene showing two circus performers with no animal in
            # sight at all. Require a higher bar (0.55) for a bare,
            # uncorroborated "Animal" to stand entirely on its own without
            # any scene support or specific label backing it — the lower
            # 0.45 bar remains available whenever nature_scene genuinely
            # supports it.
            if not nature_scene and score < 0.55:
                continue
            if nature_scene and score < 0.45:
                continue

        if fam in ("bird", "crow") and score < 0.50:
            # Require either scene support OR a genuinely repeated pattern
            # (at least 4 bird/crow candidates somewhere in this burst) —
            # real, confirmed case above. A single weak reading still needs
            # nature_scene to survive; a sustained one doesn't, since the
            # repetition itself is the corroboration Florence's scene text
            # failed to provide. Explicitly appended here (not left to fall
            # through to the generic min_combined_score floor below) — the
            # actual confirmed scores (0.144-0.175) sit BELOW that generic
            # 0.18 floor too, so without this explicit append, the whole
            # point of the repetition exception would be silently undone by
            # an unrelated, later check.
            if nature_scene or _bird_candidate_count >= 4:
                filtered.append(ev)
                continue
            continue

        # Phone/ringtone cues can be brief and still meaningful, so they get a lower bar.
        if fam == "ringtone" and score >= 0.06:
            filtered.append(ev)
            continue

        # Whip-crack and animal roars are short but very specific sounds and should not be
        # discarded just because they are brief bursts rather than sustained ambience.
        if fam == "whip" and score >= 0.05:
            filtered.append(ev)
            continue

        # Real laugh cues are human-relevant and should not be stripped just because
        # they are brief or slightly lower-scoring than nearby music.
        if fam in {"laugh_soft", "laugh_full"} and score >= 0.05:
            filtered.append(ev)
            continue

        if score >= min_combined_score:
            filtered.append(ev)

    return filtered


def final_cleanup(caption: str, detected_labels, scene_text: str = None) -> str:
    """
    Safety net for thunder/rain/lightning scenes.
    In storm scenes, the only valid caption is thunder; all generic
    vehicle/animal/human fallbacks are rejected.
    """
    caption_lower = caption.lower()
    label_values = list(detected_labels.values()) if isinstance(detected_labels, dict) else detected_labels
    labels_str = " ".join(label_values).lower()
    scene_lower = (scene_text or "").lower()
    thunder_scene = any(k in scene_lower for k in ["thunderstorm", "storm", "lightning", "thunder", "rain", "rainfall", "cloudburst", "pouring rain"])

    human_distress_terms = ["scream", "screaming", "cry", "crying", "yell", "yelling",
                            "groan", "groaning", "grunt", "grunting", "moan", "moaning",
                            "sob", "sobbing", "wheeze", "gasp", "panic", "pain", "hurt",
                            "fell", "fall", "fallen", "injured", "distress", "help", "crowd"]

    def _drop_vague(caption: str) -> str:
        """Vague/generic cards are not allowed in final output."""
        if not caption:
            return ""
        vague = ["अस्पष्ट"]
        if any(v in caption for v in vague):
            return ""  # empty = skip this burst
        return caption
    # Hard rule: storm scenes may not caption animal/vehicle/human sounds.
    if thunder_scene:
        if any(k in caption_lower for k in ["बिजली", "गरज", "thunder", "lightning", "storm"]):
            return "[बिजली गरजने की आवाज़]"
        if any(k in caption_lower for k in ["जानवर", "पक्षियों", "वाहन", "कार", "गाड़ी", "ट्रेन", "रेल", "बस", "व्यक्ति", "human", "vehicle", "animal", "bird", "crow"]):
            return "[बिजली गरजने की आवाज़]"
        if any(k in labels_str for k in ["animal", "bird", "vehicle", "human_reaction", "crow"]):
            return "[बिजली गरजने की आवाज़]"
        return "[बिजली गरजने की आवाज़]"


    # Only keep animal/vehicle generic captions in scenes where they are actually plausible.
    traffic_scene = is_traffic_scene(scene_lower)
    nature_scene = is_nature_scene(scene_lower)

    explicit_animal_signal = bool(re.search(
        r"(roar|roaring|growl|trumpet|lion|tiger|elephant|cow|dog|cat|horse|clip-clop|bark|moo|neigh|howl|cattle|bovinae|meow|purr|owl|hoot|canidae|wolves)",
        labels_str,
        re.IGNORECASE,
    ))
    explicit_whip_signal = bool(re.search(
        r"(चाबुक|whip|whip crack|whip-crack)",
        labels_str,
        re.IGNORECASE,
    ))
    vehicle_like_signal = bool(re.search(
        r"(वाहन|vehicle|car|bus|truck|train|motor|engine|auto|scooter|motorcycle)",
        labels_str,
        re.IGNORECASE,
    ))

    # Real animal/whip cues should not be stripped just because the scene is not a forest/farm.
    # Generic 'animal' labels alone are not enough evidence and must still be rejected.
    if not nature_scene and not explicit_animal_signal and not explicit_whip_signal:
        if re.search(r"(पक्षियों|कौआ|उल्लू|गाय|कुत्ता|बिल्ली|घोड़ा|जानवर)", caption_lower):
            if vehicle_like_signal or "thunder" in labels_str or "metal" in labels_str:
                return ""
            return ""

        # Only force unclear when caption says vehicle but labels have NO vehicle signal at all
    if re.search(r"(वाहन|कार|गाड़ी|ट्रेन|रेल|बस|साइकिल|auto)", caption_lower):
        return ""

        # If vehicle label exists but scene is not traffic and score path was weak,
        # prefer dropping the misleading vehicle wording only when metal/thunder present
        if not traffic_scene and any(k in labels_str for k in ["metal", "clank", "clang", "thunder", "lightning"]):
            if "metal" in labels_str or "clank" in labels_str or "clang" in labels_str:
                return "[धातु की खनक]"
            if "thunder" in labels_str or "lightning" in labels_str:
                return "[बिजली गरजने की आवाज़]"

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
    caption = _drop_vague(caption)
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
                   "fowl", "rooster", "chicken", "owl", "hoot", "turkey",
                   "duck", "quack", "goose", "honk", "chirp", "tweet"],
    # Human reaction sounds were missing before, so they are included here.
    "human_reaction": ["wheeze", "groan", "grunt", "pant", "gasp", "sigh",
                        "battle cry", "whimper", "scream", "shout", "yell",
                        "sob", "sniff", "cough"],
    "whistle":    ["whistle", "wolf-whistle", "whistling"],
    "laugh_soft": ["chuckle", "chortle", "giggle", "snicker"],
    "laugh_full": ["laughter", "belly laugh"],
    "cricket":    ["cricket"],
    "insect":     ["insect", "buzz"],
    "wind":       ["wind"],
    "rustling":   ["rustl"],
    "creak":      ["creak"],
    "footstep":   ["footstep", "walk", "run", "jog"],
    "thunder":    ["thunder", "thunderstorm", "lightning", "bolt", "storm"],
    "music":      ["music", "musical instrument", "plucked string",
                   "bowed string", "wind instrument", "singing",
                   "trumpet", "brass instrument", "clarinet", "saxophone",
                   "trombone", "french horn"],
    "crowd":      ["crowd", "cheering", "chatter"],
    "applause":   ["applause", "clapping"],
    "vehicle":    ["vehicle", "car", "engine", "motor", "motorcycle",
                   "truck", "traffic", "power window", "electric window"],
    "animal":     ["cattle", "cow", "bull", "dog", "bark", "cat", "horse",
                   "neigh", "frog", "animal", "howl",
                   "wild animal", "domestic animal", "elephant",
                   "roar", "roaring", "growl", "lion", "tiger"],
    "whip":       ["whip", "whip crack", "whip-crack"],
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
        # "Power windows, electric windows" (a real, strong car-door/window
        # event — proven from a log where it peaked at 0.426, the strongest
        # signal in its whole burst) contains "wind" as a literal prefix of
        # "windows", which wrongly classified it as generic ambient wind
        # instead of vehicle. That hid the real vehicle signal entirely and
        # let a much weaker "tick-tock" (0.284) win the burst instead. A
        # real word-boundary exists there (fresh word starting with "wind"),
        # so the word-boundary-safe _kw_hit below can't catch this one on
        # its own — needs its own explicit exclusion.
        if fam == "wind" and "window" in ll:
            continue
        # NOTE: this used to use plain `k in ll` substring matching — the
        # SAME class of bug fixed elsewhere with _kw_hit, but never applied
        # to this foundational function itself. Real, confirmed case:
        # "Marimba, xylophone" contains "phone" as a literal substring
        # (xylo-PHONE), which classified a music/instrument detection as
        # "ringtone" family and produced a phone-ringing caption during
        # what was actually just a xylophone in the score. _family() is the
        # single most foundational label→family mapping in the whole
        # pipeline — every other word-boundary fix this session was
        # downstream of this function still using the unsafe version.
        if _kw_hit(ll, kws):
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
    "thunder":    "Bijee lightning and thunder can be heard in the distance.",
    "fire":       "A fire can be heard crackling.",
    "music":      "Music can be heard playing in the background.",
    "crowd":      "A crowd can be heard in the background.",
    "bell":       "Bells can be heard ringing in the distance.",
}

def _rule_caption(families: list, best_labels: dict,
                   scene_text: str, expressions: list, fam_scores: dict = None,
                   start_sec: float = None, end_sec: float = None, logger=None) -> str:
    """Generate a deterministic rule-based caption.

    fam_scores: optional mapping of family -> combined_score to allow
    gating low-confidence high-risk families (e.g. mantra, bird) so we
    don't repeatedly caption them on very weak evidence.
    """
    fset = set(families)
    fam_scores = fam_scores or {}
    scene_lower = (scene_text or "").lower()
    # Minimum combined_score required to treat certain high-risk families as deterministic
    MANTRA_MIN_SCORE = 0.40
    BIRD_MIN_SCORE = 0.40

    signal_text = " ".join((v or "").lower() for v in best_labels.values())


    thunder_scene = any(k in scene_lower for k in ["thunderstorm", "storm", "lightning", "thunder", "rain", "rainfall", "cloudburst", "pouring rain"])
    if thunder_scene and (fset & {"thunder", "vehicle", "animal", "bird", "crow", "human_reaction", "metal", "water", "wind"}):
        return "[बिजली गरजने की आवाज़]"

    if "ringtone" in fset:
        # "Telephone bell ringing"/"Ringtone"/"Alarm clock" all sound
        # acoustically similar to a real musical bell/chime — a genuinely
        # sustained cluster of these labels (not a one-frame blip like the
        # earlier xylophone case) still turned out to be a musical bell
        # during a song: confirmed real case, "bell" family ALSO present in
        # the same burst, ringtone's peak (0.373) cleared the relative-to-
        # music check but was still well below music's own peak (0.514) —
        # meaning it was corroborated by another instrument family the
        # whole time, not a real phone. When "bell" is present in the same
        # burst as "music", require ringtone to be unambiguously the
        # loudest thing in the burst (not just half of music's strength)
        # before trusting it over the much more likely musical-bell read.
        if "bell" in fset and "music" in fset and fam_scores.get("ringtone", 0.0) < fam_scores.get("music", 0.0):
            fset.discard("ringtone")
        # This session has now hit the SAME confusion four separate times —
        # bright, ringing, chime-like musical moments (bells, xylophone,
        # sharp accents) reading as "Ringtone"/"Telephone" at genuinely high
        # confidence, in two different shapes: sustained-but-never-winning
        # (0.41, never beats music which stays at 0.5-0.6 the whole burst)
        # and a single very sharp spike (0.72 for one frame, then music
        # resumes normally for the rest). Notably: every single high-
        # confidence ringtone-during-music case checked this whole session
        # has turned out to be a false positive — none have been a real
        # phone. That's strong enough, repeated evidence to change the rule
        # itself rather than add a fourth narrow special case: whenever
        # "music" is genuinely present in the same burst, the "confident
        # enough to skip scene support" bypass no longer applies at all —
        # require an actual phone in the scene regardless of how high the
        # audio confidence reads. The bypass remains available for ringtone
        # occurring WITHOUT music, where this specific instrument confusion
        # can't apply.
        elif "music" in fset:
            if not (fam_scores.get("ringtone", 0.0) >= 0.10 and _has_phone_scene_support(scene_lower)
                    and fam_scores.get("ringtone", 0.0) >= 0.5 * fam_scores.get("music", 0.0)):
                fset.discard("ringtone")
            else:
                return "[फ़ोन की घंटी बज रही है]"
        elif (
            (fam_scores.get("ringtone", 0.0) >= 0.10 and _has_phone_scene_support(scene_lower)
             and fam_scores.get("ringtone", 0.0) >= 0.5 * fam_scores.get("music", 0.0)) or
            fam_scores.get("ringtone", 0.0) >= RINGTONE_HIGH_CONFIDENCE_WITHOUT_SCENE
        ):
            return "[फ़ोन की घंटी बज रही है]"
        # Gate failed — this must not just skip the early return. "ringtone"
        # staying in fset meant a completely separate, unconditional mapping
        # later in this same function (the SOUND_PHRASE_MAP dict's own
        # "ringtone" entry) still matched it and produced the exact same
        # phone caption anyway, bypassing this whole gate and the GPT
        # backstop both — a real, confirmed case where the gate check here
        # succeeded but the caption still came out wrong because nothing
        # removed "ringtone" from fset for the rest of the function to see.
        fset.discard("ringtone")
        # Deliberately no caption committed here — not phone (rejected by
        # the gate), and not a guessed alternative either (tried "bird":
        # right often enough for this show, but still a guess dressed up as
        # an answer). Falls through to the rest of this function instead of
        # returning immediately, so if this burst ALSO has another real,
        # legitimate family (e.g. ringtone + bird together), that other
        # family still gets its own correct caption — only a burst where
        # ringtone was the sole signal ends up with nothing to say.

    if "thunder" in fset and fam_scores.get("thunder", 0.0) >= 0.06:
        return "[बिजली गरजने की आवाज़]"

    nature_scene = is_nature_scene(scene_lower)
    traffic_scene = is_traffic_scene(scene_lower)

    if "clip-clop" in fset and not nature_scene:
        fset.discard("clip-clop")
    # NOTE: this used to require 0.35 confidence without scene support — a
    # bar chosen assuming Florence's scene text is a reliable way to confirm
    # or deny a vehicle's presence. It isn't. Real evidence, repeatedly:
    # Florence describing a scene with a car plainly in frame as "a group of
    # men standing... in front of" with no mention of the car at all, and
    # scene text truncating mid-sentence before it ever gets to naming what's
    # actually there. Gating audio evidence behind a visual description that
    # keeps failing to describe the visual isn't a safety check, it's a coin
    # flip against Florence's completeness. Lowered the bar substantially —
    # trust the audio classifier's own confidence more, since PANNs scoring
    # "Vehicle"/"Power windows, electric windows" at 0.25+ is real, specific
    # evidence, not noise, and scene support is now a bonus that lowers the
    # bar further rather than a requirement that vehicle can't clear without.
    if "vehicle" in fset and not traffic_scene and fam_scores.get("vehicle", 0.0) < 0.20:
        fset.discard("vehicle")
    specific_animal_in_signal = _kw_hit(signal_text, ["elephant", "roar", "roaring", "growl", "lion", "tiger", "roaring cats", "lions", "tigers", "horse", "clip-clop"])
    vehicle_like_in_signal = _kw_hit(signal_text, ["vehicle", "car", "bus", "truck", "train", "motor", "engine", "auto", "scooter", "motorcycle"])
    if "animal" in fset and not nature_scene and not specific_animal_in_signal:
        if vehicle_like_in_signal or "thunder" in signal_text or "metal" in signal_text:
            fset.discard("animal")
        elif fam_scores.get("animal", 0.0) < 0.45:
            fset.discard("animal")
    if "bird" in fset and not nature_scene and fam_scores.get("bird", 0.0) < 0.50:
        fset.discard("bird")
    if "crow" in fset and not nature_scene and fam_scores.get("crow", 0.0) < 0.50:
        fset.discard("crow")
    if thunder_scene:
        for fam in ("vehicle", "animal", "bird", "crow"):
            fset.discard(fam)
    elif "animal" in fset and (
        fam_scores.get("animal", 0.0) < 0.18 and
        not nature_scene
    ):
        fset.discard("animal")
    elif "bird" in fset and (
        fam_scores.get("bird", 0.0) < BIRD_MIN_SCORE and
        not nature_scene
    ):
        fset.discard("bird")
    elif "vehicle" in fset and (
        fam_scores.get("vehicle", 0.0) < 0.18 and
        not traffic_scene
    ):
        fset.discard("vehicle")

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

    # Impact / fight / fall / clamor → human reaction, never vehicle/animal.
    # Scream/yell/shout get their own dictionary-backed captions below instead
    # of being silently discarded — they used to sit in the same "return
    # empty" bucket as generic impact noise (thump/crash/etc), which meant a
    # real, clearly-detected scream never produced a caption at all.
    SCREAM_KEYS = ("scream", "yell", "shout")
    if any(_kw_hit(b, SCREAM_KEYS) for b in bl):
        fset.discard("vehicle")
        fset.discard("animal")
        fset.discard("snake")
        fset.discard("bird")
        if any(_kw_hit(b, ["scream"]) for b in bl):
            return "[डर से चिल्लाना]"
        return "[चिल्लाना]"

    # Impact / fight / fall / clamor → human reaction, never vehicle/animal.
    # These now route through the dictionary map instead of being discarded
    # outright — thump/thud/fall -> "[गिरने की आवाज़]", groan/grunt -> "[दर्द
    # से करहाते हुए]", crowd/hubbub -> "[अस्पष्ट बातचीत]" all have real
    # client-approved phrases. "Crash"/"smash"/"slam" don't map to a clean
    # dictionary phrase on their own (too easily confused with a generic
    # loud bang), so those still fall through to GPT as a genuine last resort.
    ALL_IMPACT_KEYS = ("thump", "thud", "slam", "crash", "smash",
                       "groan", "grunt", "crowd", "battle cry")
    if any(_kw_hit(b, ALL_IMPACT_KEYS) for b in bl):
        fset.discard("vehicle")
        fset.discard("animal")
        fset.discard("snake")
        fset.discard("bird")
        # "Battle cry" is a genuinely ambiguous PANNs class — it fires on
        # both real combat vocalizations AND a crowd roaring/cheering, which
        # sound acoustically similar. Rather than try to disambiguate which
        # one it is, "चिल्लाना" (shouting) is accurate either way — a real
        # fight involves shouting, and a clamoring crowd is also shouting.
        # This is now a hard, unconditional rule for "battle cry" — it no
        # longer needs crowd corroboration to fire, and no longer falls
        # through to GPT (which was independently reaching for the
        # dictionary's "[लड़ाई की आवाज़]" (fight sound) as its own literal-
        # match guess, wrongly committing to "fight" specifically even
        # without a real fight on screen — confirmed with a screenshot of a
        # smiling man, not a fight, captioned "fight sound" anyway).
        if _kw_hit(" ".join(bl), ["battle cry"]):
            return "[चिल्लाना]"
        NO_DICT_IMPACT_KEYS = ("slam", "crash", "smash")
        if any(_kw_hit(b, NO_DICT_IMPACT_KEYS) for b in bl):
            return ""

    # Specific animal sounds win before the generic fallback, even when a nearby music
    # burst or a non-nature scene makes the raw score look less decisive. This is the
    # exact fix for roaring tiger/elephant cues being suppressed by generic Animal+Music.
    # NOTE: "horse" and "clip-clop" used to be in this set too, which meant this
    # generic caption fired and returned BEFORE dictionary_non_music_caption's
    # dedicated horse-specific check ever got a chance to run — a real,
    # confirmed case where a horse mid-jump (corroborated Horse/Clip-clop +
    # real footstep evidence) still came out as generic "animal sound"
    # instead of the dictionary's actual "[घोड़ों की टापों की आवाज़]". Removed
    # them here so the more specific handler downstream gets first say.
    ANIMAL_SOUND_LABELS = {"moo", "cattle, bovinae", "cow",
                           "dog", "bark", "bow-wow", "canidae, dogs, wolves",
                           "cat", "meow", "purr",
                           "elephant", "roar", "roaring", "growl",
                           "lion", "tiger"}
    if any(b in ANIMAL_SOUND_LABELS for b in bl):
        return "[जानवर की आवाज़ सुनाई दे रही है]"
    if _kw_hit(signal_text, ["elephant", "roar", "roaring", "growl", "lion", "tiger"]):
        return "[जानवर की आवाज़ सुनाई दे रही है]"
    if _kw_hit(signal_text, ["roaring cats", "lions", "tigers"]):
        return "[जानवर की आवाज़ सुनाई दे रही है]"

    if "whip" in fset or any(_kw_hit(b, ["whip"]) for b in bl):
        return "[चाबुक की आवाज़]"

    # "Patter" was being freely interpreted by GPT as rain with no basis.
    if "patter" in bl:
        return "[मृदु थपथपाहट की आवाज़]"

    # Very specific alert/human sounds should win over generic vehicle/background labels
    # when the same burst contains both.
    # Strong thunder wins even without scene keywords
    if "thunder" in fset and fam_scores.get("thunder", 0.0) >= 0.12:
        return "[बिजली गरजने की आवाज़]"
    # Weaker thunder still needs a storm/rain/dark hint
    if "thunder" in fset and fam_scores.get("thunder", 0.0) >= 0.06 and any(
        k in scene_lower for k in ["thunderstorm", "storm", "lightning", "thunder", "rain", "dark", "night"]
    ):
        return "[बिजली गरजने की आवाज़]"
    # Metal wins over weak/medium vehicle (tools, knives, clanks)
    if "metal" in fset and fam_scores.get("metal", 0.0) >= 0.08:
        if fam_scores.get("vehicle", 0.0) < 0.40:
            return "[धातु की खनक]"

    # NOTE: this used to check the raw `families` list, which bypassed every
    # nature_scene/fam_score safety check applied to `fset` above — so ANY
    # burst whose top family was ['animal'] got captioned as an animal sound
    # unconditionally, even a low-confidence one that lost a close call to
    # real competing evidence (e.g. actual laughter) in the same burst. Now
    # checks `fset`, so it only fires if "animal" survived all the plausibility
    # gating earlier in this function.
    if "animal" in fset and len(fset) == 1:
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
    if "snicker" in fset:
        return "[हल्की हँसी सुनाई दे रही है]"

    # Music/instrument cues: NEVER name the instrument. The client dictionary's
    # music captions are chosen by the emotional register of the visual scene
    # (Florence) plus the instrument's character used only as an internal mood
    # signal (e.g. a lone violin over an isolated figure reads as sad) — never
    # by literally naming which instrument PANNs detected. A tabla and a
    # violin in the same tense scene still get the same caption, because what
    # matters to the viewer is the mood, not which instrument was playing.
    # NOTE: this used to be "any music-like family anywhere in the top
    # families", regardless of whether music was actually the strongest
    # signal — meaning a weak, non-dominant music reading could still
    # unconditionally override a genuinely stronger competing sound. Real,
    # confirmed case: a quiet outdoor ambient scene where "bird" was listed
    # FIRST (meaning its own aggregated score was the highest of the three
    # families present, 0.434) and a weak "music" reading (0.396, ~91% of
    # bird's score) — some soft background tonal noise, not real music —
    # still won the whole caption purely because it existed somewhere in
    # the top-3 list at all. A first attempt at fixing this allowed music
    # to win when "close enough" (within 10% of the top score) — but 0.396
    # vs 0.434 IS within that margin (91%), so the same confirmed-bad case
    # still passed. Simplified to strict dominance instead of a percentage
    # margin: music must actually BE the top-ranked family (families[0],
    # since families is already sorted by score) to get this override — no
    # "close enough" allowance that a real case can slip through.
    has_music_signal = bool(families) and families[0].lower() in MUSIC_LIKE_FAMILIES

    # Resolve mantra ahead of music — chanting is a distinct sound, not "music".
    has_mantra = any(f.lower() in ("mantra", "chant") for f in families)

    if has_mantra:
        # Require a higher confidence for mantra to be considered definitive
        if fam_scores and fam_scores.get("mantra", 0.0) < MANTRA_MIN_SCORE and fam_scores.get("chant", 0.0) < MANTRA_MIN_SCORE:
            has_mantra = False
        else:
            return "[मंत्रों का उच्चारण सुनाई दे रहा है]"
    if has_music_signal:
        return pick_music_caption(scene_text, signal_text, set(f.lower() for f in families),
                                   start_sec=start_sec, end_sec=end_sec, logger=logger)

    # Audience reaction pairs are a narrow, readable exception to the normal single-sound rule.
    if "applause" in fset and "whistle" in fset:
        return "[तालियों और सीटियों की आवाज़]"
    if "applause" in fset and "wood" in fset:
        return "[तालियों की आवाज़]"

    # Outdoor walking scenes can use rustling or footsteps as a stronger visual match,
    # but not as a generic filler when the scene is empty or unrelated.
    walking = any(x in scene_text.lower()
                  for x in ["walk","path","trail","moving","strolling","approaching"])
    natural_context = outdoor or is_nature_scene(scene_lower)
    if natural_context and walking and fset & {"rustling", "creak", "footstep"}:
        ambient = fset & {"bird","crow","cricket","insect","wind"}
        if "crow" in ambient or "bird" in ambient:
            return "[पक्षियों की चहचहाहट सुनाई दे रही है]"
        return "[पत्तों की सरसराहट सुनाई दे रही है]"

        # Engine/vehicle beats ambient wind when both fire
    if "vehicle" in fset and fam_scores.get("vehicle", 0.0) >= 0.24:
        if fam_scores.get("wind", 0.0) < 0.50:
            fset.discard("wind")
            return "[गाड़ी की आवाज़]"

    # Client dictionary's exact non-music phrases (door, phone, siren, gunshot,
    # etc.) win over the generic ad hoc phrase table below whenever they apply.
    _dict_caption = dictionary_non_music_caption(best_labels, scene_text)
    if _dict_caption:
        # "लोगों का भागना" (people running) and "क़दमों की आवाज़" (footsteps)
        # fired completely unconditionally here — no scene-support check at
        # all — even though a separate mechanism a few lines above
        # (natural_context) was clearly built to gate exactly this kind of
        # footstep-family sound, and never got the chance since this path
        # returns first. Confirmed wrong with a real screenshot: "Run" won
        # at 0.466 during a shot of two men's faces pressed against a
        # circus safety net — nobody running on screen, no running-related
        # word anywhere in the scene text. Require actual outdoor/natural
        # context, or very high confidence, before committing to either
        # caption; otherwise let it fall through instead of guessing wrong.
        if _dict_caption in ("[लोगों का भागना]", "[क़दमों की आवाज़]"):
            if not natural_context and fam_scores.get("footstep", 0.0) < 0.55:
                _dict_caption = ""
        if _dict_caption:
            return _dict_caption

    # After the early rules, score order decides the final phrase, with music as a normal candidate.
    # NOTE: "ringtone" is deliberately NOT in this table. It has its own
    # gated check earlier in this function (confidence + scene-support
    # required) — this loop iterates the raw `families` list, not `fset`, so
    # discarding "ringtone" from fset when the gate fails does NOT stop it
    # from being picked up again here. A real, confirmed bug: the early gate
    # correctly blocked the deterministic shortcut, but "ringtone" still sat
    # in `families`, matched this unconditional table, and produced the same
    # ungated phone caption anyway — bypassing the gate, the sanitization,
    # AND the GPT backstop, since a match here returns a bracket-wrapped
    # Devanagari string that reads as fully deterministic.
    SOUND_PHRASE_MAP = {
        "crow":      "[पक्षियों की चहचहाहट]",
        "bird":      "[पक्षियों की चहचहाहट]",
        "animal":    "[जानवर की आवाज़]",
        "applause":  "[तालियों की आवाज़]",
        "whistle":   "[सीटी की आवाज़]",
        "bell":      "[घंटी की आवाज़]",
        "thunder":   "[बिजली गरजने की आवाज़]",
        "wood":      "[लकड़ी की आवाज़]",
        "whip":      "[चाबुक की आवाज़]",
        "metal":     "[धातु की खनक]",
        "glass":     "[काँच की आवाज़]",
        "fire":      "[आग की चटचटाहट]",
        "cricket":   "[झींगुरों की आवाज़]",
        "insect":    "[कीड़ों की आवाज़]",
        "music":     "[मधुर धुन]",
    }

    # Scene context can suppress implausible sounds but not real animal cases.
    _scene_category, _implausible = _scene_palette(scene_text)
    _implausible = _implausible - {"animal"}

    for fam_name in families:  # already score-sorted for this burst
        fl = fam_name.lower()
        if fl in _implausible:
            continue
        # Do not invent generic wind/leaf/footstep captions when the scene is empty or
        # unrelated. Those are ambient filler, not reliable detections.
        if fl in {"wind", "rustling", "creak", "footstep"} and not natural_context:
            continue
        # "Howl" is genuinely ambiguous in AudioSet's own class definition —
        # it covers both an animal howling AND wind howling, two acoustically
        # similar but semantically very different sounds. It was mapped into
        # the "animal" family specifically to stop it falling through as an
        # unrecognized label (which GPT then filled in with invented text,
        # e.g. "[शांत माहौल]" — confirmed real case). But that fix means a
        # bare "Howl" reading with no other corroborating animal evidence
        # now confidently asserts "[जानवर की आवाज़]" even when it's just as
        # likely to be wind — confirmed real case: sustained Howl at 0.4-0.58
        # in a forest scene with no animal visible and no storm/wind wording
        # either. Rather than guess which one it is, fall back to the
        # dictionary's existing "genuinely uncertain ambient sound" phrase
        # (already used elsewhere for this exact kind of ambiguity) instead
        # of confidently naming an animal — UNLESS something more specific
        # actually corroborates a real animal (a genuinely distinct label
        # like Moo, Roar, Bark, or explicit animal wording in the scene).
        if fl == "animal" and best_labels.get("animal", "").lower() == "howl":
            # NOTE: best_labels is keyed one-entry-per-family, so checking
            # its own values can't detect a DIFFERENT, more specific animal
            # family that won its own key in this same burst (e.g. "moo" is
            # its own family, not nested under "animal") — check the
            # families list itself for that, not best_labels.
            other_animal_families = {"moo", "roar", "growl", "bark"} & {f.lower() for f in families}
            has_specific_corroboration = bool(other_animal_families) or any(
                k in (scene_text or "").lower() for k in
                ["dog", "wolf", "cow", "cattle", "lion", "tiger", "elephant", "horse", "animal"])
            if not has_specific_corroboration:
                return "[शांत आवाज़]"
        if fl in SOUND_PHRASE_MAP:
            return SOUND_PHRASE_MAP[fl]

    # Ambiguous cases stay as a hint instead of a forced caption, so GPT can adapt to the real signal.
    return ""

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
        # If a specific rumble/roar/whip is inside a longer mixed burst, anchor the
        # subtitle to the actual onset of that sound rather than the beginning of the
        # surrounding music cluster. Use the earliest specific cue in the burst so the
        # caption lands at the real sound start instead of a few seconds late or on a
        # neighboring generic frame.
        specific_roar_ev = min(
            (ev for ev in burst if _is_exotic_animal_label(ev["label"])),
            key=lambda e: (e["timestamp_sec"], -e.get("combined_score", 0.0)),
            default=None,
        )
        specific_whip_ev = min(
            (ev for ev in burst if "whip" in (ev.get("label") or "").lower()),
            key=lambda e: (e["timestamp_sec"], -e.get("combined_score", 0.0)),
            default=None,
        )
        anchor_ev = specific_roar_ev or specific_whip_ev or burst[0]
        start_sec = anchor_ev["timestamp_sec"]
        end_sec   = max(anchor_ev["timestamp_sec"] + 1.8, burst[-1]["timestamp_sec"] + 1.8)
        # Thunder / storm should stay on screen longer so the caption persists
        # across the full burst instead of being replaced by a generic music label.
        is_thunder = any(_family(ev["label"]) == "thunder" for ev in burst)
        if is_thunder:
            end_sec = max(end_sec, start_sec + 7.0)  # hold ~7s minimum

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

        # Best label per family. Prefer specific animal cues (roar/elephant/trumpet,
        # or a footstep-corroborated horse/clip-clop — see detect_audio_events,
        # which only ever creates a horse/clip-clop event when real footstep
        # evidence backs it up in the same frame, so it's already safe to trust
        # here) over generic "Animal" labels in the same burst, so real sound
        # events are not lost. Confirmed real case: "Animal" won a frame at
        # 0.410 alongside a strongly corroborated "Clip-clop"/"Horse" (0.35+,
        # backed by real "Walk, footsteps" in the same frame) — without this
        # preference, the generic "Animal" stayed as the family's
        # representative, which then failed the nature-scene discard gate
        # (0.410 < 0.45, non-nature scene) and lost the horse signal entirely,
        # even though the dedicated horse caption check exists specifically
        # for this. Preferring the corroborated horse label here means the
        # rest of the pipeline sees a signal that's already been vetted.
        def _is_corroborated_horse_label(lbl):
            return _kw_hit((lbl or "").lower(), ["horse", "clip-clop"])
        fam_best = {}
        for ev in sorted(burst, key=lambda e: -e["combined_score"]):
            fam = _family(ev["label"])
            if fam not in fam_best:
                fam_best[fam] = ev
            elif fam == "animal" and fam_best[fam] and not _is_exotic_animal_label(fam_best[fam]["label"]) and _is_exotic_animal_label(ev["label"]):
                fam_best[fam] = ev
            elif (fam == "animal" and fam_best[fam]
                  and not _is_corroborated_horse_label(fam_best[fam]["label"])
                  and not _is_exotic_animal_label(fam_best[fam]["label"])
                  and _is_corroborated_horse_label(ev["label"])):
                fam_best[fam] = ev

        # Every family that had at least one surviving candidate in this
        # burst, BEFORE narrowing down to the capped top_fams below — this
        # is the actual "everything the model considered before deciding"
        # picture, not just the winner(s). Captured here specifically
        # because top_fams gets reordered/capped by several priority rules
        # right after this point; fam_best itself is only ever read from
        # there, never modified, so this is the one safe place to snapshot
        # the full candidate set intact.
        all_candidates = sorted(
            [{"label": ev["label"], "family": fam, "score": round(ev.get("combined_score", 0.0), 3)}
             for fam, ev in fam_best.items()],
            key=lambda c: -c["score"]
        )

        # Use the strongest event as the scene anchor before any scene-based rules.
        best_ev = max(burst, key=lambda e: e["combined_score"])
        scene_text = best_ev.get("scene_text", "")
        expressions = best_ev.get("expressions", [])

        # Promote real alert sounds ahead of generic music or ambience. NOTE:
        # plain "animal" is deliberately NOT in this set. It used to be, which
        # meant ANY animal detection — even a weak one failing its own
        # threshold — got sorted ahead of a much stronger, more specific
        # "bird" signal just because "animal" was hardcoded as priority. Real
        # evidence: a burst with Goose/Honk/Fowl all passing at 0.4-0.5+
        # combined score (an unambiguous, strong bird signal) still lost to a
        # weak "Animal" candidate at 0.099-0.151 (FAILING its own threshold)
        # purely because of this hardcoded priority, producing the wrong
        # generic "[जानवर की आवाज़]" instead of the correct bird caption.
        # Loud, unambiguous exotic animal cues (roar/elephant/tiger) already
        # get their own dedicated force-to-top treatment below via
        # `specific_animal_hit` — that's the correct place for animal to ever
        # jump the score order, not a blanket rule for every animal label.
        # NOTE: "whip" used to be in this blanket set with no score check at
        # all — real evidence: "Whip" at 0.045-0.055 (never even passing its
        # own class threshold, essentially noise-floor) still jumped ahead of
        # "Music" dominating the same burst at 0.2-0.44, because membership
        # here bypasses score entirely. Same bug class as the earlier
        # "animal" priority issue. Moved to its own score-gated branch below,
        # matching how thunder/metal/human_reaction already work.
        PRIORITY_FAMILIES = {"laugh_soft", "laugh_full", "human_reaction", "ringtone", "thunder", "metal"}
        ranked   = sorted(fam_best, key=lambda f: fam_best[f]["combined_score"], reverse=True)
        priority = [f for f in ranked if f in PRIORITY_FAMILIES]
        rest     = [f for f in ranked if f not in PRIORITY_FAMILIES]
        top_fams = (priority + rest)[:3]
        # NOTE: this used to unconditionally strip "vehicle" from top_fams
        # right here, no condition attached at all — a leftover blanket rule
        # that silently undid every vehicle confidence/scene-support gate
        # built into _rule_caption, since it ran upstream of all of them. It
        # only ever looked safe because a burst where vehicle was the SOLE
        # family hit the "if not top_fams" fallback a few lines down and got
        # restored — but when vehicle co-occurred with any other family
        # (e.g. tick-tock), that fallback never fired, and vehicle was gone
        # for good. Real evidence: a burst with a genuine, strong car/window
        # event (0.426 combined, the single highest score in its burst) lost
        # out to a far weaker tick-tock (0.284) purely because of this line.
        # Vehicle plausibility is already handled properly by the specific,
        # conditional "drop_vehicle" checks right below (thunder/metal/
        # human_reaction/storm-scene competition) and by _rule_caption's own
        # gates — this blanket removal was both redundant and wrong.

        # Real roar/elephant/tiger cues must beat surrounding music when they are the
        # actual signal in the burst. This deliberately only checks EXOTIC labels
        # (roar/elephant/tiger/growl) — quiet specific labels like meow/horse are
        # too easily confused with sustained violin notes and rhythmic music/
        # footstep textures to be trusted enough to force animal to the top and
        # override the burst's anchor timing.
        specific_animal_hit = any(_is_exotic_animal_label(ev["label"]) for ev in burst)
        specific_animal_ev = min(
            (ev for ev in burst if _is_exotic_animal_label(ev["label"])),
            key=lambda e: (e["timestamp_sec"], -e.get("combined_score", 0.0)),
            default=None,
        )
        # NOTE: this used to be `any("whip" in label for ev in burst)` with
        # ZERO score requirement — meaning a "Whip" candidate that never even
        # passed its own class threshold (confirmed real case: 0.045, 0.055,
        # both failing) still unconditionally forced whip to the front of
        # top_fams, completely bypassing the separate score-gated fix already
        # made elsewhere in this file for this exact reason. This is a
        # second, independent path to the same bug — this file has now hit
        # this "one gated check, one forgotten ungated duplicate" pattern
        # several times this session (ringtone, vehicle, and now whip).
        # Require the winning whip candidate to actually be a real, passing
        # detection before it's allowed to jump the queue.
        # Confirmed by direct user testimony + a screenshot: at this exact
        # kind of burst (Whip co-occurring with Music, no visible whip act
        # on screen), the real content is people screaming out of shock —
        # PANNs is picking up the sharp, sudden acoustic shape of a scream
        # and calling it "Whip"/"Whoosh, swoosh, swish" instead. There is no
        # scream/yell/shout candidate anywhere in these bursts to redirect
        # to (checked exhaustively — only weak, failing Sigh/Bellow appear),
        # so this can't be fixed by pointing at a better label; the model
        # itself never proposes the right one. Raised the bar well above the
        # confirmed-false case (0.168) so weak, ambiguous whip readings like
        # that no longer get trusted — a real, loud whip-crack should clear
        # this comfortably; this specific 0.168 case should not.
        WHIP_HIT_MIN_SCORE = 0.30
        whip_hit = any(
            "whip" in (ev.get("label") or "").lower() and ev.get("combined_score", 0.0) >= WHIP_HIT_MIN_SCORE
            for ev in burst
        )
        if specific_animal_hit and "animal" in fam_best:
            fam_best["animal"] = specific_animal_ev or fam_best["animal"]
            top_fams = ["animal"] + [f for f in top_fams if f != "animal"]
        if whip_hit:
            if "whip" in fam_best:
                whip_ev = min(
                    (ev for ev in burst if "whip" in (ev.get("label") or "").lower()),
                    key=lambda e: (e["timestamp_sec"], -e.get("combined_score", 0.0)),
                    default=None,
                )
                if whip_ev:
                    fam_best["whip"] = whip_ev
            top_fams = ["whip"] + [f for f in top_fams if f != "whip"]

        # Laughter is a real human cue and should survive even when nearby music
        # or ambience is scored slightly higher in the same burst.
        laugh_family = next((f for f in ranked if f in {"laugh_soft", "laugh_full"}), None)
        if laugh_family:
            top_fams = [laugh_family] + [f for f in top_fams if f != laugh_family][:2]

        # Keep short but real alert/human sounds even when they are lower-scoring than music or ambience.
        for force_family in ("laugh_soft", "laugh_full", "thunder", "human_reaction", "metal", "ringtone", "whip"):
            if force_family in fam_best:
                score = fam_best[force_family].get("combined_score", 0.0)
                scene_l = (scene_text or "").lower()
                if force_family in {"laugh_soft", "laugh_full"} and score >= 0.07:
                    top_fams = [force_family] + [f for f in top_fams if f != force_family][:2]
                elif force_family == "ringtone" and (
                    (score >= 0.10 and _has_phone_scene_support(scene_text)
                     # Real evidence: Ringtone at 0.144 got force-promoted
                     # ahead of Music at 0.590 — nearly 4x stronger — purely
                     # because the scene happened to mention "talking on
                     # [the phone]". Scene support alone shouldn't let a
                     # signal this much weaker hijack the caption from
                     # something this dominant. Require ringtone to be at
                     # least half the strength of a co-occurring music
                     # signal, not just clear its own absolute bar.
                     and score >= 0.5 * fam_best.get("music", {}).get("combined_score", 0.0)) or
                    score >= RINGTONE_HIGH_CONFIDENCE_WITHOUT_SCENE
                ):
                    top_fams = [force_family] + [f for f in top_fams if f != force_family][:2]
                elif force_family == "thunder" and (
                    score >= 0.12 or
                    (score >= 0.06 and any(k in scene_l for k in ["thunderstorm", "storm", "lightning", "thunder", "rain", "dark", "night"]))
                ):
                    top_fams = [force_family] + [f for f in top_fams if f != force_family][:2]
                elif force_family == "human_reaction" and score >= 0.08:
                    top_fams = [force_family] + [f for f in top_fams if f != force_family][:2]
                elif force_family == "metal" and score >= 0.08:
                    top_fams = [force_family] + [f for f in top_fams if f != force_family][:2]
                elif force_family == "whip" and score >= 0.30:
                    top_fams = [force_family] + [f for f in top_fams if f != force_family][:2]

        # Suppress generic vehicle labels when the burst clearly matches a thunderstorm or clanking metal cue.
        scene_lower = (scene_text or "").lower()
        # Drop vehicle when thunder, metal, or human impact is competing
        if "vehicle" in fam_best:
            v_sc = fam_best["vehicle"].get("combined_score", 0.0)
            drop_vehicle = False
            if "thunder" in fam_best:
                drop_vehicle = True
            if "metal" in fam_best and fam_best["metal"].get("combined_score", 0.0) >= 0.08 and v_sc < 0.40:
                drop_vehicle = True
            if "human_reaction" in fam_best and fam_best["human_reaction"].get("combined_score", 0.0) >= 0.08 and v_sc < 0.40:
                drop_vehicle = True
            if any(k in scene_lower for k in ["thunderstorm", "storm", "lightning", "thunder", "rain"]):
                drop_vehicle = True
            if drop_vehicle:
                top_fams = [f for f in top_fams if f != "vehicle"]

        
        # Remove weak generic false positives that should not dominate a burst.
        if "applause" in fam_best and "wood" in fam_best and (
            fam_best["applause"].get("combined_score", 0.0) >= fam_best["wood"].get("combined_score", 0.0) * 0.8
            or fam_best["wood"].get("combined_score", 0.0) < 0.18
        ):
            top_fams = ["applause"] + [f for f in top_fams if f not in {"applause", "wood"}]
        if not top_fams:
            top_fams = list(fam_best.keys())[:1]

        best_labels = {f: fam_best[f]["label"] for f in top_fams if f in fam_best}
        # Pass family-level combined scores so rule-based logic can gate
        # low-confidence high-risk families (mantra, bird, etc.).
        fam_scores = {f: fam_best[f].get("combined_score", 0.0) for f in top_fams}

        # Specific animal sounds override generic animal fallback in the same burst.
        specific_present = any(
            _is_specific_animal_label(lbl) for lbl in best_labels.values()
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
            if _family(lbl) == "animal" and not _is_specific_animal_label(lbl):
                best_labels[f] = "Ambient/unclear sound (do not name species)"

        # A ringtone/telephone read without phone-scene support and without
        # overwhelming confidence gets the same treatment as an ambiguous
        # animal label above — don't let GPT confidently assert "phone" from
        # a label PANNs itself wasn't fully sure about, when nothing on
        # screen supports it either.
        ringtone_was_suppressed = False
        for f, lbl in list(best_labels.items()):
            if _family(lbl) == "ringtone" and not _has_phone_scene_support(scene_text):
                if fam_scores.get(f, 0.0) < RINGTONE_HIGH_CONFIDENCE_WITHOUT_SCENE:
                    # NOTE: earlier this said "(do not name as phone)" — which
                    # put the literal word "phone" directly into the text GPT
                    # reads, undermining the whole point of sanitizing it.
                    # GPT only ever sees label VALUES, not the dict's family-
                    # name keys, so the fix has to avoid the trigger word
                    # entirely rather than mention-and-negate it.
                    best_labels[f] = "Ambient/unclear sound (uncertain origin)"
                    ringtone_was_suppressed = True

        # A "Run"/"Walk" read without outdoor/natural scene support and
        # without high confidence gets the same treatment — the dictionary-
        # level gate for this (in _rule_caption) only blocks the
        # deterministic "[लोगों का भागना]" shortcut, but GPT was still
        # independently reaching the same wrong conclusion on its own from
        # the raw "Run" label, just worded differently ("लोगों के दौड़ने
        # की आवाज़" instead of "लोगों का भागना") — confirmed with a
        # screenshot of two men's faces at a circus safety net, nobody
        # running. Sanitizing here closes that gap the same way it did for
        # ringtone/phone. natural_context is recomputed here since it's
        # otherwise only local to _rule_caption.
        _scene_l_fs = (scene_text or "").lower()
        _outdoor_fs = any(x in _scene_l_fs for x in ["forest", "outdoor", "trees", "jungle", "nature", "path"])
        natural_context = _outdoor_fs or is_nature_scene(_scene_l_fs)
        footstep_was_suppressed = False
        for f, lbl in list(best_labels.items()):
            if _family(lbl) == "footstep" and not natural_context:
                if fam_scores.get(f, 0.0) < 0.55:
                    best_labels[f] = "Ambient/unclear sound (uncertain origin)"
                    footstep_was_suppressed = True

        # Same treatment for "Tick-tock"/"Tick" without real "Clock"
        # corroboration or an indoor scene — the dictionary-level gate in
        # _rule_caption correctly blocks the deterministic "[घड़ी की टिक-टिक
        # लगातार जारी]", but GPT was still independently reconstructing the
        # same wrong idea in slightly different wording ("[घड़ी की टिक-
        # टिक]", missing "लगातार जारी") from the raw label alone — confirmed
        # from a real log, same burst, same family, right after the
        # dictionary gate was added. "tick-tock" is its own literal orphan
        # family name (never mapped into any SOUND_FAMILIES bucket).
        _has_clock_support = _kw_hit((scene_text or "").lower(), ["room", "indoor", "inside", "house", "kitchen", "office"])
        tick_tock_was_suppressed = False
        for f, lbl in list(best_labels.items()):
            if _family(lbl) in ("tick-tock", "tick") and "clock" not in (lbl or "").lower():
                if not _has_clock_support:
                    best_labels[f] = "Ambient/unclear sound (uncertain origin)"
                    tick_tock_was_suppressed = True

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

        # The SLC client dictionary is authoritative: _rule_caption tries every
        # deterministic dictionary match first (music mood, non-music phrase
        # bank, and the earlier hand-tuned rules). Only when NOTHING in the
        # dictionary applies do we fall through to GPT — grounded in the real
        # detected signal, and still bound by the same no-instrument/no-animal-
        # species constraints via the system prompt.
        raw_caption = _rule_caption(top_fams, best_labels, scene_text, expressions, fam_scores,
                                     start_sec=start_sec, end_sec=end_sec, logger=logger)

        # A rejected ringtone with nothing else in the burst to fall back on
        # should stay uncaptioned rather than let GPT take a guess at it —
        # _rule_caption already discarded "ringtone" internally and found
        # nothing else to say, and top_fams reducing to just ringtone (or
        # nothing) confirms there wasn't a second real signal underneath it.
        if ringtone_was_suppressed and not raw_caption and set(top_fams) <= {"ringtone"}:
            logger.info(f"  [{start_sec:.1f}→{end_sec:.1f}s] SKIPPED rejected ringtone, no other signal families={top_fams}")
            continue

        # Same clean-omission treatment for a rejected footstep/run burst
        # with nothing else in it.
        if footstep_was_suppressed and not raw_caption and set(top_fams) <= {"footstep"}:
            logger.info(f"  [{start_sec:.1f}→{end_sec:.1f}s] SKIPPED rejected footstep/run, no other signal families={top_fams}")
            continue

        # Same for a rejected tick-tock burst with nothing else in it.
        if tick_tock_was_suppressed and not raw_caption and set(top_fams) <= {"tick-tock", "tick"}:
            logger.info(f"  [{start_sec:.1f}→{end_sec:.1f}s] SKIPPED rejected tick-tock, no other signal families={top_fams}")
            continue

        is_deterministic = (
            raw_caption.startswith("[") and
            any("\u0900" <= ch <= "\u097F" for ch in raw_caption)
        )

        if is_deterministic:
            hindi_caption = raw_caption
        else:
            gpt_hint = ambient_fallback_hint(top_fams, scene_text)
            hindi_caption = generate_hindi_caption(
                raw_caption=gpt_hint,
                scene_text=scene_text,
                detected_labels=best_labels,
                expressions=expressions,
                logger=logger
            )

            if hindi_caption and len(hindi_caption) > 5:
                hindi_caption = polish_hindi_caption(
                    caption=hindi_caption,
                    detected_labels=best_labels,
                    logger=logger
                )

            if not hindi_caption or not hindi_caption.strip():
                hindi_caption = ambient_fallback_caption(top_fams, scene_text)

            # Hard backstop: even after the prompt instruction against it,
            # GPT can still default to a phone/ringtone phrase when given a
            # genuinely vague "Ambient/unclear sound" hint — a real, observed
            # case, and prompt compliance alone isn't a guarantee. If the
            # ringtone gate already decided this burst shouldn't be trusted
            # as a phone (no scene support, no overwhelming confidence), any
            # phone-sounding GPT output gets rejected here regardless of
            # phrasing, not just the exact strings the deterministic shortcut
            # would have produced.
            if ringtone_was_suppressed and hindi_caption:
                PHONE_WORDS = ["फोन", "फ़ोन", "मोबाईल", "मोबाइल", "सेलफ़ोन", "सेलफोन", "घंटी"]
                if any(w in hindi_caption for w in PHONE_WORDS):
                    hindi_caption = ambient_fallback_caption(top_fams, scene_text) or "[शांत आवाज़]"

            # Same hard backstop for footstep/run: GPT independently
            # reinvented "[लोगों के दौड़ने की आवाज़]" (different wording,
            # same wrong meaning as "[लोगों का भागना]") from the sanitized
            # hint alone — confirmed with a real log where the dictionary
            # gate correctly blocked the exact phrase, but GPT still landed
            # on running/fleeing from the vague "Ambient/unclear sound"
            # input. Catch any phrasing built from the "run"/"flee" root
            # words, not just the one exact string.
            if footstep_was_suppressed and hindi_caption:
                RUN_WORDS = ["भाग", "दौड़"]
                if any(w in hindi_caption for w in RUN_WORDS):
                    hindi_caption = ambient_fallback_caption(top_fams, scene_text) or "[शांत आवाज़]"

            # Same backstop for tick-tock: confirmed real case where GPT
            # reconstructed "[घड़ी की टिक-टिक]" (missing "लगातार जारी") from
            # the sanitized hint alone, immediately after the dictionary
            # gate correctly blocked the exact phrase — same mechanism as
            # ringtone/footstep, just a different root word this time.
            if tick_tock_was_suppressed and hindi_caption:
                CLOCK_WORDS = ["टिक", "घड़ी"]
                if any(w in hindi_caption for w in CLOCK_WORDS):
                    hindi_caption = ambient_fallback_caption(top_fams, scene_text) or "[शांत आवाज़]"

            if not hindi_caption:
                # Truly nothing to say about this burst — dictionary had no
                # match, GPT errored out, and there's no scene-grounded hint.
                continue

        hindi_caption = enforce_caption_format(hindi_caption)
        hindi_caption = final_cleanup(hindi_caption, best_labels, scene_text)
        hindi_caption = fix_hindi_issues(hindi_caption)

        # Final consistency pass catches mismatches between family signal and caption wording.
        # NOTE: the "पक्षियों" row used to hard-force "[जानवर की आवाज़]" whenever "animal"
        # was anywhere in top_fams — even when the bird caption was correct (e.g. produced
        # by the SLC dictionary's raw-label match, which can legitimately fire on a bird
        # label that _family() grouped under a different family name than "bird"). That
        # silently overwrote real bird captions with animal captions. It now re-derives via
        # _rule_caption like every other row here, instead of blindly assuming "animal".
        _fam_lower = set(f.lower() for f in top_fams)
        _consistency_checks = [
            ("पक्षियों", {"bird", "crow"} & _fam_lower, None),
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

        if not hindi_caption or not hindi_caption.strip() or "अस्पष्ट" in hindi_caption:
            logger.info(f"  [{start_sec:.1f}→{end_sec:.1f}s] SKIPPED vague/empty caption families={top_fams}")
            continue

        # QC flag: was this caption a genuine guess rather than a confident
        # rule match? Two distinct kinds of guessing, both real: (1) GPT had
        # to freely invent the wording because no dictionary rule matched at
        # all (is_deterministic was False), or (2) the dictionary path DID
        # produce valid, in-dictionary text, but only by randomly picking
        # between multiple tied, equally-plausible mood candidates with no
        # real tiebreak (pick_music_caption's "no tiebreak available"
        # branch) — this passes the is_deterministic bracket/Devanagari
        # check since it's real dictionary text, so it needs its own signal,
        # not just "was GPT involved". Re-read the module flag here (not
        # just at the _rule_caption call above) in case the consistency
        # check below re-derived the caption via a second _rule_caption call.
        is_guess = (not is_deterministic) or _LAST_CAPTION_WAS_RANDOM_GUESS

        timeline.append({
            "start_sec":   round(start_sec, 2),
            "end_sec":     round(end_sec, 2),
            "caption":     hindi_caption,
            "families":    top_fams,
            "scene_text":  scene_text[:100],
            "is_guess":    is_guess,
            "valence":            _LAST_BURST_SCORES["valence"],
            "arousal":            _LAST_BURST_SCORES["arousal"],
            "predicted_moods":    _LAST_BURST_SCORES["predicted_moods"],
            "mood_candidates":    _LAST_BURST_SCORES["candidates"],
            "quadrant":           _LAST_BURST_SCORES["quadrant"],
            "candidate_scores":   _LAST_BURST_SCORES["candidate_counts"],
            "decision_reason":    _LAST_BURST_SCORES["decision_reason"],
            "all_candidates":     all_candidates,
        })

        logger.info(
            f"  [{start_sec:.1f}→{end_sec:.1f}s] families={top_fams}"
            f"{' [GUESS]' if is_guess else ''}\n"
            f"    final='{hindi_caption}'"
        )

    # Fill genuine silence gaps: if there's a stretch of 3+ seconds between
    # two people speaking (or before the first / after the last line of
    # dialogue) with NO existing caption anywhere in it — nothing else this
    # pipeline detected was confident enough to fill it — insert a short,
    # 3-second "[शांत आवाज़]" caption at the start of that gap. This is the
    # SAME dictionary phrase already used elsewhere in this file as the
    # approved fallback for white noise / an uncertain quiet sound (see the
    # "cutlery, silverware, white noise" mapping and the GPT system prompt's
    # own guidance to default here on genuinely uncertain input) — reused
    # for consistency rather than inventing a new phrase. This only fires
    # when NOTHING else already covers any part of the gap; a real detected
    # sound anywhere in that stretch is left completely alone and takes
    # priority. Caption is capped at 3 seconds regardless of how long the
    # actual silent stretch runs — the point is to avoid a viewer staring
    # at a long silent gap with literally nothing on screen, not to caption
    # the entire silence.
    MIN_SILENCE_GAP_SEC = 3.0
    SILENCE_CAPTION_DURATION_SEC = 3.0
    if speech_segments:
        sorted_speech = sorted(speech_segments, key=lambda s: s[0])
        # scene_index samples Florence at a regular interval across the
        # WHOLE video, so its last timestamp is a much more reliable proxy
        # for "how long is this video" than the last detected audio event,
        # which could stop well short of the end if nothing happens there.
        audio_duration_sec = max((e["t"] for e in scene_index), default=None) if scene_index else None

        gap_bounds = []
        cursor = 0.0
        for seg_start, seg_end in sorted_speech:
            if seg_start - cursor >= MIN_SILENCE_GAP_SEC:
                gap_bounds.append((cursor, seg_start))
            cursor = max(cursor, seg_end)
        if audio_duration_sec is not None and audio_duration_sec - cursor >= MIN_SILENCE_GAP_SEC:
            gap_bounds.append((cursor, audio_duration_sec))

        for gap_start, gap_end in gap_bounds:
            # The 3-second window itself must be checked, not the whole
            # (possibly much longer) gap — a detection late in an 8-second
            # gap shouldn't suppress a caption at the start of it, since the
            # actual trigger here is absence of SPEECH (already established
            # by gap_bounds above), not absence of any sound whatsoever
            # across the entire stretch.
            cap_start = gap_start
            cap_end = min(gap_start + SILENCE_CAPTION_DURATION_SEC, gap_end)
            already_covered = any(
                seg["start_sec"] < cap_end and seg["end_sec"] > cap_start
                for seg in timeline
            )
            if already_covered:
                continue
            timeline.append({
                "start_sec": round(cap_start, 2),
                "end_sec": round(cap_end, 2),
                "caption": "[शांत आवाज़]",
                "families": ["silence_gap"],
                "scene_text": "",
                "is_guess": False,
                "valence": None, "arousal": None, "predicted_moods": [],
                "mood_candidates": [], "quadrant": None, "candidate_scores": {},
                "decision_reason": "no speech for 3+ seconds, nothing else detected in that window",
                "all_candidates": [],
            })
            logger.info(f"  [{cap_start:.1f}→{cap_end:.1f}s] SILENCE GAP FILLED (no speech, "
                        f"no other detection for {gap_end - gap_start:.1f}s) final='[शांत आवाज़]'")
        # New entries above were appended out of chronological order relative
        # to the main loop's output — every pass below this point assumes a
        # sorted timeline (they index adjacent entries directly), so this
        # must be re-sorted before any of them run.
        timeline.sort(key=lambda seg: seg["start_sec"])

    # Backward continuity pass for theme music: a title card is usually the
    # LAST few seconds of a much longer intro sequence, not the whole thing —
    # real evidence from BEK episodes 20-23 shows a single continuous ~60-70s
    # musical intro playing over non-text visuals (e.g. "a vibrant rose..."),
    # with the literal on-screen title card only appearing in the final 4-5
    # seconds. Per-burst scene-text matching can only ever catch that last
    # burst, mislabeling the other 90% of the same continuous theme song as
    # generic "[मधुर धुन]". Fix: once a real title-card burst is found, walk
    # backward through immediately-preceding music bursts (small gap, still
    # music-family, currently neutral) and relabel them as the same theme
    # song too, stopping at the first non-music or oddly-gapped burst. This
    # generalizes to any show with a similar intro structure instead of
    # guessing a fixed time window.
    # 15s was too tight — real evidence from BEK 21/22 shows a genuine ~21.5s
    # gap mid-intro (a quiet passage in the same continuous theme song, or a
    # stretch where PANNs simply didn't register anything), which fell just
    # outside the old threshold and stopped the backward walk before it ever
    # reached the earlier part of the intro. 30s matches SCENE_WINDOW_SEC
    # already used elsewhere in this file for "how far can we reasonably
    # assume continuity," while still being far short of the multi-minute
    # gaps that separate the intro from genuinely unrelated later content.
    MAX_THEME_GAP_SEC = 30.0
    for i, entry in enumerate(timeline):
        if entry["caption"] != "[थीम संगीत]":
            continue
        j = i - 1
        prev_end = entry["start_sec"]
        while j >= 0:
            prev = timeline[j]
            # NOTE: this used to check MUSIC_LIKE_FAMILIES only — chant/
            # mantra are deliberately kept separate from "music" elsewhere
            # in this file (chanting is treated as its own distinct sound),
            # but for THIS specific check — is this burst part of the same
            # continuous intro piece — that distinction is the wrong one to
            # apply. Real, confirmed case: a single 1.8s burst reading as
            # bare "chant" (likely a brief invocation blended into the
            # opening theme, common in this kind of devotional/historical
            # content) sat in the middle of an otherwise fully continuous
            # ~70s musical intro, and broke the ENTIRE backward walk right
            # there — leaving everything before it stuck on generic
            # "[मधुर धुन]" instead of correctly reading as the theme song.
            # A brief vocal/chant moment within a continuous title sequence
            # doesn't mean the theme music has ended.
            is_music_family = any(f.lower() in MUSIC_LIKE_FAMILIES | {"chant", "mantra"} for f in prev["families"])
            gap = prev_end - prev["end_sec"]
            if not is_music_family or gap > MAX_THEME_GAP_SEC:
                break
            # Overwrite regardless of which mood this sub-segment got — a
            # title sequence is one continuous, fixed piece of music, not
            # narrative content whose mood should vary shot-to-shot. Real
            # data showed a stray "[खुशनुमा संगीत]" mid-intro (Florence
            # picked up something upbeat-looking in one frame) that would
            # otherwise have blocked the backward walk and left everything
            # before it stuck on "[मधुर धुन]". The anchor here (a confirmed
            # title-card burst) is specific and rare enough that propagating
            # backward through the whole continuous run is safe.
            prev["caption"] = "[थीम संगीत]"
            prev_end = prev["start_sec"]
            j -= 1

    # Mood-persistence walk, generalizing the theme-continuity fix above
    # beyond just theme music. Once a burst gets a confident, specific mood
    # (e.g. "[गंभीर पार्श्व संगीत]", "[तनावपूर्ण पार्श्वसंगीत]"), it's often
    # the same continuous underlying score (a whole tense/dramatic cue, not
    # a single instant) playing across MULTIPLE nearby bursts — but each
    # burst is still classified independently, so several bursts in a row
    # can dip to generic "[मधुर धुन]" even while the surrounding stretch is
    # clearly one continuous confident passage. A single-gap patch isn't
    # enough for this — a real dramatic/tense cue can run long enough to
    # hit two or more consecutive uncertain bursts, not just one. This walks
    # FORWARD from every confident mood reading, extending it through any
    # number of immediately-following neutral music bursts (small gaps),
    # and only stops at a genuine interruption: a different confident mood
    # (a real transition — never overwritten), a non-music burst, or too
    # large a gap. This is what "persisting across the whole context it's
    # present" actually means: the mood holds until something real changes
    # it, not just for one adjacent burst.
    NEUTRAL_CAPTIONS = set(SLC_MUSIC_MOOD_CAPTIONS["neutral"])
    MAX_MOOD_GAP_SEC = 10.0
    i = 0
    while i < len(timeline):
        entry = timeline[i]
        is_confident_mood = (
            entry["caption"] not in NEUTRAL_CAPTIONS
            and entry["caption"] != "[थीम संगीत]"
            and any(f.lower() in MUSIC_LIKE_FAMILIES for f in entry["families"])
        )
        if not is_confident_mood:
            i += 1
            continue
        j = i + 1
        prev_end = entry["end_sec"]
        while j < len(timeline):
            nxt = timeline[j]
            gap = nxt["start_sec"] - prev_end
            is_music = any(f.lower() in MUSIC_LIKE_FAMILIES for f in nxt["families"])
            if not is_music or gap > MAX_MOOD_GAP_SEC:
                break
            if nxt["caption"] in NEUTRAL_CAPTIONS:
                nxt["caption"] = entry["caption"]
                prev_end = nxt["end_sec"]
                j += 1
                continue
            # A genuinely different confident mood (or theme music) — a real
            # transition, e.g. tense building into scary. Stop here, this is
            # not a flicker to overwrite.
            break
        i = j if j > i + 1 else i + 1

    # Outlier-flicker smoothing: Music2Emo classifies each burst independently
    # from a short, isolated audio clip — real film-score audio can have a
    # brief accent, a sound-effect bleed, or just per-clip noise that makes
    # ONE burst read as a genuinely different confident mood than the scene
    # around it, even though the scene itself hasn't actually changed. Real,
    # reported pattern: a continuous serious/tense passage (a person falling
    # and screaming) flickering to "happy"-quadrant readings for individual
    # bursts in the middle, only correctly reading as serious at the very end.
    # The walk above only fills NEUTRAL gaps, by design — it deliberately
    # never overwrites a genuinely different confident mood, since that
    # could be a real transition (tense building into scary). This second
    # pass is narrower and safer: it only replaces a SINGLE burst whose
    # mood disagrees with BOTH neighbors, when those two neighbors agree
    # WITH EACH OTHER. That specific shape — same mood, one different burst,
    # same mood again — is a strong signal of an isolated misread rather
    # than a real transition (a genuine transition wouldn't revert back).
    i = 1
    while i < len(timeline) - 1:
        entry, prev_entry, next_entry = timeline[i], timeline[i - 1], timeline[i + 1]
        this_confident = (entry["caption"] not in NEUTRAL_CAPTIONS and entry["caption"] != "[थीम संगीत]"
                          and any(f.lower() in MUSIC_LIKE_FAMILIES for f in entry["families"]))
        prev_confident = (prev_entry["caption"] not in NEUTRAL_CAPTIONS and prev_entry["caption"] != "[थीम संगीत]"
                          and any(f.lower() in MUSIC_LIKE_FAMILIES for f in prev_entry["families"]))
        next_confident = (next_entry["caption"] not in NEUTRAL_CAPTIONS and next_entry["caption"] != "[थीम संगीत]"
                          and any(f.lower() in MUSIC_LIKE_FAMILIES for f in next_entry["families"]))
        if not (this_confident and prev_confident and next_confident):
            i += 1
            continue
        gap_before = entry["start_sec"] - prev_entry["end_sec"]
        gap_after = next_entry["start_sec"] - entry["end_sec"]
        if gap_before > MAX_MOOD_GAP_SEC or gap_after > MAX_MOOD_GAP_SEC:
            i += 1
            continue
        if prev_entry["caption"] == next_entry["caption"] and entry["caption"] != prev_entry["caption"]:
            entry["caption"] = prev_entry["caption"]
        i += 1

    # Mantra continuity: a real, confirmed case from BEK_EP_016 — a single
    # continuous ~75-second chanting passage (890.8s-965.8s) where every
    # burst genuinely has "chant"/"mantra" in its family list, but ONLY the
    # very last burst (947.2-965.8s) individually cleared MANTRA_MIN_SCORE.
    # The five bursts before it fell through to generic "[मधुर धुन]" one at
    # a time, fragmenting one continuous chant into "silence...silence...
    # silence...finally mantra" instead of reading as one sustained chant
    # throughout — same underlying problem as the theme-song fragmentation
    # fixed above, just never extended to mantra specifically. Once a real
    # mantra burst is confirmed, walk in BOTH directions through immediately
    # adjacent bursts (small gap) that already have chant/mantra in their
    # own family list but are still sitting on the generic neutral default
    # (meaning they didn't individually clear the confidence bar) and
    # relabel them too — stops at the first burst that either isn't
    # chant/mantra-tagged at all, or already has some OTHER confident
    # caption (a real, different sound taking priority is never overwritten).
    MANTRA_CAPTION = "[मंत्रों का उच्चारण]"
    MAX_MANTRA_GAP_SEC = 15.0
    MAX_CONSECUTIVE_UNTAGGED = 2

    def _has_chant_tag(seg):
        return any(f.lower() in {"chant", "mantra"} for f in seg["families"])

    for i, entry in enumerate(timeline):
        if entry["caption"] != MANTRA_CAPTION:
            continue
        # backward — a burst lacking the chant/mantra tag is only bridged
        # if a tagged burst is confirmed to exist within the tolerance
        # window just beyond it (real interior gap, passage genuinely
        # continues) — never bridged based on local consecutive-count
        # alone, which can't tell a real mid-passage dip apart from the
        # passage's actual edge: a plain consecutive counter that resets
        # on every tagged hit would happily absorb a boundary burst one
        # step past where the real chanting evidence stops, since from a
        # purely local view a single untagged gap there looks identical
        # to a genuine brief dip in the middle — confirmed case: an
        # earlier version of this walk absorbed exactly such a boundary
        # burst sitting just outside the real ~75s passage.
        j = i - 1
        cursor_end = entry["start_sec"]
        while j >= 0:
            prev = timeline[j]
            gap = cursor_end - prev["end_sec"]
            if gap > MAX_MANTRA_GAP_SEC or prev["caption"] not in NEUTRAL_CAPTIONS:
                break
            if not _has_chant_tag(prev):
                lookback = timeline[max(0, j - MAX_CONSECUTIVE_UNTAGGED):j]
                if not any(_has_chant_tag(seg) for seg in lookback):
                    break
            prev["caption"] = MANTRA_CAPTION
            cursor_end = prev["start_sec"]
            j -= 1
        # forward
        k = i + 1
        cursor_start = entry["end_sec"]
        while k < len(timeline):
            nxt = timeline[k]
            gap = nxt["start_sec"] - cursor_start
            if gap > MAX_MANTRA_GAP_SEC or nxt["caption"] not in NEUTRAL_CAPTIONS:
                break
            if not _has_chant_tag(nxt):
                lookahead = timeline[k + 1:k + 1 + MAX_CONSECUTIVE_UNTAGGED]
                if not any(_has_chant_tag(seg) for seg in lookahead):
                    break
            nxt["caption"] = MANTRA_CAPTION
            cursor_start = nxt["end_sec"]
            k += 1

    logger.info("=" * 65)
    return timeline

# ====================== SRT ======================
def _srt_ts(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{int((sec%1)*1000):03d}"

def write_srt(timeline: list, path: Path, blank_guesses: bool = False):
    """Write an SRT file. When blank_guesses=True, any segment flagged
    is_guess=True gets an EMPTY caption line instead of the guessed text —
    the timestamp block stays (so a reviewer knows exactly which time range
    still needs a caption filled in), but no unverified/guessed text ships
    in the file meant to go out as-is. The actual guessed text isn't lost —
    it's still in guess_review.txt, results.json, and the full reference
    SRT (blank_guesses=False) written alongside this one."""
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(timeline, 1):
            caption = "" if (blank_guesses and seg.get("is_guess")) else seg['caption']
            f.write(f"{i}\n{_srt_ts(seg['start_sec'])} --> "
                    f"{_srt_ts(seg['end_sec'])}\n{caption}\n\n")

# ====================== FINAL VIDEO ======================
def generate_final_video(video_path: str, timeline: list,
                          output_path: Path, logger, guess_spans: list = None):
    """Renders BOTH things together: an empty bracket "[ ]" during the real
    analysis window the audio model needed BEFORE each caption (using the
    actual WINDOW_SEC constant, 0.96s — the genuine amount of audio PANNs/
    Music2Emo needs to process before it can commit to a classification for
    a given moment), AND the actual final caption during its own displayed
    time — matching exactly what's in the ship-ready captions.srt: real
    text for a confident caption, blank for one flagged is_guess. This
    means every caption in the video reads as: [ ] (model working) ->
    either real text or silence (the actual final result), so the
    processing-window visualization and the real shipped output are both
    visible together, not one replacing the other."""
    if not timeline:
        logger.info("Empty timeline — skipping video."); return
    logger.info(f"Generating video with {len(timeline)} caption segments "
                f"(processing-window markers + final captions)...")
    guess_spans = guess_spans or []

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        silent_path = tmp.name
    out = cv2.VideoWriter(silent_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    tl  = sorted(timeline, key=lambda s: s["start_sec"])

    # One processing window per caption: [caption_start - WINDOW_SEC, caption_start).
    # Multiple windows are allowed to overlap or sit back-to-back (e.g. two
    # captions closer together than WINDOW_SEC apart) — the per-frame check
    # below just asks "is this frame covered by ANY window", so overlapping
    # windows naturally render as one continuous bracket with no extra
    # merging logic needed.
    processing_windows = [
        {"start_sec": max(0.0, seg["start_sec"] - WINDOW_SEC), "end_sec": seg["start_sec"]}
        for seg in timeline
    ]
    processing_windows.sort(key=lambda w: w["start_sec"])

    for fi in range(total):
        ret, frame = cap.read()
        if not ret: break
        t = fi / fps
        in_processing_window = next((pw for pw in processing_windows if pw["start_sec"] <= t < pw["end_sec"]), None)
        if in_processing_window:
            frame = render_caption(frame, "[ ]")
        else:
            active = next((s for s in tl if s["start_sec"] <= t < s["end_sec"]), None)
            if active and not active.get("is_guess"):
                frame = render_caption(frame, active["caption"])
            # a guessed caption's own display window intentionally shows
            # nothing here, matching captions.srt's blank line exactly —
            # the [ ] bracket already appeared during its processing window
            # just before this point, so the uncertainty is still visible,
            # just not restated a second time over the caption's own span.
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
        # Chapter markers now cover every processing window (one per
        # caption), not just the flagged guess spans — many more chapters
        # than before, but that's the point: comparable, one-to-one against
        # every caption in the timeline, not just the uncertain subset.
        if processing_windows:
            _embed_processing_chapters(output_path, processing_windows, ffmpeg_bin, logger)
    except Exception as e:
        logger.error(f"Video merge failed: {e}")
    finally:
        try: os.unlink(silent_path)
        except: pass

def _embed_processing_chapters(video_path: Path, processing_windows: list, ffmpeg_bin: str, logger):
    """Embed standard MP4 chapter markers at each model-processing window —
    same mechanism as the earlier guess-span chapters (verified with
    ffprobe before that was wired in), just applied to every caption's
    analysis window instead of only the flagged/uncertain subset. MP4
    chapters are contiguous ranges (each one runs to the next chapter's
    start), so gaps between processing windows get their own plain
    "(normal)" chapter to keep the whole timeline covered."""
    # imageio-ffmpeg (used when ffmpeg is not on PATH, e.g. the Windows
    # build) ships no ffprobe, so fall back to OpenCV for the duration.
    try:
        duration_probe = subprocess.run(
            [ffmpeg_bin.replace("ffmpeg", "ffprobe"), "-i", str(video_path),
             "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"],
            capture_output=True, text=True
        )
        duration = float(duration_probe.stdout.strip())
    except (OSError, ValueError, TypeError):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0.0
        cap.release()
    if not duration:
        logger.warning("Could not determine video duration — skipping chapter markers.")
        return

    lines = [";FFMETADATA1", ""]
    cursor = 0.0
    for pw in sorted(processing_windows, key=lambda w: w["start_sec"]):
        if pw["start_sec"] < cursor:
            continue  # overlapping with the previous window — already covered
        if pw["start_sec"] > cursor:
            lines += ["[CHAPTER]", "TIMEBASE=1/1000",
                      f"START={int(cursor * 1000)}", f"END={int(pw['start_sec'] * 1000)}",
                      "title=(normal)", ""]
        lines += ["[CHAPTER]", "TIMEBASE=1/1000",
                  f"START={int(pw['start_sec'] * 1000)}", f"END={int(pw['end_sec'] * 1000)}",
                  "title=PROCESSING", ""]
        cursor = max(cursor, pw["end_sec"])
    if cursor < duration:
        lines += ["[CHAPTER]", "TIMEBASE=1/1000",
                  f"START={int(cursor * 1000)}", f"END={int(duration * 1000)}",
                  "title=(normal)", ""]

    chapters_path = video_path.with_suffix(".chapters.txt")
    chapters_path.write_text("\n".join(lines), encoding="utf-8")
    tmp_path = video_path.with_suffix(".chapters_tmp.mp4")
    try:
        subprocess.run([
            ffmpeg_bin, "-y", "-i", str(video_path), "-i", str(chapters_path),
            "-map_metadata", "1", "-codec", "copy", str(tmp_path)
        ], check=True, capture_output=True)
        tmp_path.replace(video_path)
        logger.info(f"Embedded {len(processing_windows)} processing-window chapter marker(s) into "
                    f"{video_path.name} — visible as navigable points on the seek bar in VLC "
                    f"and other chapter-aware players.")
    except Exception as e:

        logger.warning(f"Chapter marker embedding failed (video itself is still fine): {e}")
    finally:
        chapters_path.unlink(missing_ok=True)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

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
    # Two versions: the primary "captions.srt" is the one meant to actually
    # ship — guessed segments are left with a blank caption line (timing
    # block intact) rather than shipping unverified text that looks
    # identical to a confident caption. "captions_with_guesses.srt" is the
    # full reference version with every caption filled in, guesses
    # included, for comparison or for anyone who wants to see the system's
    # best-effort output even where it wasn't confident.
    srt_path = output_dir / "captions.srt"
    write_srt(timeline, srt_path, blank_guesses=True)
    logger.info(f"SRT (ship-ready, guesses left blank): {srt_path}")

    srt_full_path = output_dir / "captions_with_guesses.srt"
    write_srt(timeline, srt_full_path, blank_guesses=False)
    logger.info(f"SRT (full reference, includes guesses): {srt_full_path}")

    # ---- Guess spans: collapse the per-segment is_guess flags into
    # contiguous "starts guessing here, stops here" ranges, so a reviewer
    # can jump straight to the uncertain stretches instead of reading every
    # segment's flag individually. Adjacent guessed segments (small gap
    # between them) merge into one span; a confident segment in between
    # ends the span, same logic already used elsewhere in this file for
    # merging adjacent captions into continuous runs.
    def _format_timestamp(total_seconds: float) -> str:
        """MM:SS, or HH:MM:SS once the video is over an hour — directly
        usable in a video player's seek bar, unlike raw seconds."""
        td = timedelta(seconds=max(0, total_seconds))
        total = int(td.total_seconds())
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    guess_spans = []
    current_span = None
    for seg in timeline:
        if seg.get("is_guess"):
            if current_span is None:
                current_span = {"start_sec": seg["start_sec"], "end_sec": seg["end_sec"],
                                 "captions": [seg["caption"]]}
            else:
                current_span["end_sec"] = seg["end_sec"]
                if seg["caption"] not in current_span["captions"]:
                    current_span["captions"].append(seg["caption"])
        else:
            if current_span is not None:
                guess_spans.append(current_span)
                current_span = None
    if current_span is not None:
        guess_spans.append(current_span)

    # Add human-readable timestamps to every span for the JSON output too —
    # not just the dedicated review file below — so anyone reading
    # results.json directly still gets a player-seekable time, not just
    # raw seconds they'd have to convert by hand.
    for span in guess_spans:
        span["start_display"] = _format_timestamp(span["start_sec"])
        span["end_display"]   = _format_timestamp(span["end_sec"])

    if guess_spans:
        logger.info(f"Guess spans (low-confidence stretches for QC review): {len(guess_spans)}")
        for span in guess_spans:
            logger.info(f"  GUESS SPAN: {span['start_display']} → {span['end_display']} "
                        f"({span['start_sec']:.1f}s-{span['end_sec']:.1f}s)")

    # ---- Dedicated guess-review file: the actual point of this feature —
    # a short, clean, skimmable list a person can read in seconds and use
    # to jump straight to the exact moments worth checking, without
    # opening JSON or scrubbing through the whole video. Blank lines
    # deliberately separate each entry so the start/stop of every span is
    # visually obvious at a glance, not just another densely-packed log line.
    review_path = output_dir / "guess_review.txt"
    with open(review_path, "w", encoding="utf-8") as f:
        if not guess_spans:
            f.write("No low-confidence captions in this episode — nothing flagged for review.\n")
        else:
            total_guess_sec = sum(s["end_sec"] - s["start_sec"] for s in guess_spans)
            f.write(f"GUESS SPANS — {len(guess_spans)} stretch(es), "
                    f"{total_guess_sec:.0f}s total, worth a manual check\n")
            f.write("=" * 60 + "\n")
            for i, span in enumerate(guess_spans, 1):
                f.write("\n")
                f.write(f"[{i}] {span['start_display']} \u2192 {span['end_display']}\n")
                for cap in span["captions"]:
                    f.write(f"    {cap}\n")
                f.write("\n")
            f.write("=" * 60 + "\n")
    logger.info(f"Guess review: {review_path}")

    # ---- JSON ----
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "caption_segments": len(timeline),
            "timeline": timeline,
            "guess_spans": guess_spans,
            "raw_events": len(raw_events),
            "deduped_events": len(deduped),
            "speech_segments": len(speech_segments),
        }, f, indent=2, ensure_ascii=False)

    if GENERATE_FINAL_VIDEO and timeline:
        generate_final_video(video_path, timeline,
                             output_dir / "final_output.mp4", logger,
                             guess_spans=guess_spans)

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

    out = Path("results_v7") / video_path.stem
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