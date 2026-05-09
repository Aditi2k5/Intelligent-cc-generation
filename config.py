"""
config.py
=========
Central configuration for the Intelligent CC Suggestion Tool.
All tunable parameters live here — modify this file to adjust
pipeline behavior without touching module code.
"""

# ─────────────────────────────────────────────
# GENERAL
# ─────────────────────────────────────────────
OUTPUT_DIR = "demo_results"
FRAMES_DIR = "demo_results/frames"
LOG_LEVEL = "INFO"           # DEBUG | INFO | WARNING | ERROR

# ─────────────────────────────────────────────
# MODULE 1 — Sound Event Detection
# ─────────────────────────────────────────────
YAMNET_MODEL_PATH = "https://tfhub.dev/google/yamnet/1"

# Sliding window for audio analysis
AUDIO_WINDOW_SEC   = 0.96    # YAMNet native window (960 ms)
AUDIO_HOP_SEC      = 0.48    # 50 % overlap → smoother detection
AUDIO_SAMPLE_RATE  = 16000   # YAMNet requires 16 kHz mono

# Minimum raw YAMNet confidence to even consider a class
YAMNET_RAW_THRESHOLD = 0.10

# After priority boosting, minimum score to emit an event
AUDIO_EMIT_THRESHOLD = 0.25

# Merge nearby events of the same category within this window (seconds)
EVENT_MERGE_GAP_SEC = 1.5

# Maximum events per category across the whole clip (avoids flooding)
MAX_EVENTS_PER_CATEGORY = 8

# Default caption duration when no end-time is available (seconds)
DEFAULT_CAPTION_DURATION_SEC = 2.0

# ─────────────────────────────────────────────
# Sound Category Definitions
#
# Each entry:
#   "CATEGORY_KEY": {
#       "display":   human-readable CC label
#       "priority":  HIGH | MEDIUM | LOW
#       "boost":     multiplier applied to raw YAMNet score
#       "yamnet":    list of YAMNet class substrings to match
#   }
# ─────────────────────────────────────────────
SOUND_CATEGORIES = {
    # ── HIGH PRIORITY (scream / sudden impact / alarm) ──────────────────
    "SCREAM": {
        "display":  "[ Screaming ]",
        "priority": "HIGH",
        "boost":    1.8,
        "yamnet":   ["scream", "shout", "yell", "shriek", "wail", "cry"],
    },
    "EXPLOSION": {
        "display":  "[ Explosion ]",
        "priority": "HIGH",
        "boost":    1.9,
        "yamnet":   ["explosion", "bang", "burst", "blast", "boom"],
    },
    "GUNSHOT": {
        "display":  "[ Gunshot ]",
        "priority": "HIGH",
        "boost":    1.9,
        "yamnet":   ["gunshot", "gunfire", "shot", "ricochet", "firearm"],
    },
    "GLASS_BREAK": {
        "display":  "[ Glass Breaking ]",
        "priority": "HIGH",
        "boost":    1.7,
        "yamnet":   ["glass", "shatter", "breaking"],
    },
    "CRASH": {
        "display":  "[ Crash ]",
        "priority": "HIGH",
        "boost":    1.6,
        "yamnet":   ["crash", "collision", "impact", "smash"],
    },
    "ALARM": {
        "display":  "[ Alarm / Siren ]",
        "priority": "HIGH",
        "boost":    1.7,
        "yamnet":   ["alarm", "siren", "beep", "buzzer", "alert", "horn"],
    },

    # ── MEDIUM PRIORITY (human / social sounds) ─────────────────────────
    "LAUGHTER": {
        "display":  "[ Laughter ]",
        "priority": "MEDIUM",
        "boost":    1.4,
        "yamnet":   ["laugh", "giggle", "chuckle", "cackle"],
    },
    "APPLAUSE": {
        "display":  "[ Applause ]",
        "priority": "MEDIUM",
        "boost":    1.4,
        "yamnet":   ["applause", "clapping", "clap"],
    },
    "CRYING": {
        "display":  "[ Crying ]",
        "priority": "MEDIUM",
        "boost":    1.5,
        "yamnet":   ["crying", "sobbing", "weeping", "whimper"],
    },
    "KNOCK": {
        "display":  "[ Knocking ]",
        "priority": "MEDIUM",
        "boost":    1.3,
        "yamnet":   ["knock", "tap", "rap", "pound"],
    },
    "DOORBELL": {
        "display":  "[ Doorbell ]",
        "priority": "MEDIUM",
        "boost":    1.5,
        "yamnet":   ["doorbell", "ding dong", "bell"],
    },
    "PHONE": {
        "display":  "[ Phone Ringing ]",
        "priority": "MEDIUM",
        "boost":    1.4,
        "yamnet":   ["telephone", "ringtone", "phone", "mobile"],
    },

    # ── MEDIUM PRIORITY (animal sounds) ─────────────────────────────────
    "DOG": {
        "display":  "[ Dog Barking ]",
        "priority": "MEDIUM",
        "boost":    1.3,
        "yamnet":   ["dog", "bark", "howl", "growl", "whine"],
    },
    "CAT": {
        "display":  "[ Cat ]",
        "priority": "MEDIUM",
        "boost":    1.2,
        "yamnet":   ["cat", "meow", "purr", "hiss"],
    },
    "RAT_SQUEAK": {
        "display":  "[ Squeak / Rodent ]",
        "priority": "MEDIUM",
        "boost":    1.5,
        "yamnet":   ["squeak", "squeal", "rodent", "mouse", "rat"],
    },

    # ── MEDIUM PRIORITY (object / environment sounds) ────────────────────
    "CHAIR_CREAK": {
        "display":  "[ Creaking ]",
        "priority": "MEDIUM",
        "boost":    1.4,
        "yamnet":   ["creak", "squeak", "crunch", "groan", "grind"],
    },
    "FOOTSTEPS": {
        "display":  "[ Footsteps ]",
        "priority": "MEDIUM",
        "boost":    1.2,
        "yamnet":   ["footstep", "walk", "stomp", "running"],
    },
    "DOOR": {
        "display":  "[ Door ]",
        "priority": "MEDIUM",
        "boost":    1.3,
        "yamnet":   ["door", "slam", "close", "open"],
    },
    "THUNDER": {
        "display":  "[ Thunder ]",
        "priority": "MEDIUM",
        "boost":    1.5,
        "yamnet":   ["thunder", "lightning", "storm"],
    },
    "MUSIC": {
        "display":  "[ Music ]",
        "priority": "MEDIUM",
        "boost":    1.0,
        "yamnet":   [
            "music", "song", "melody", "beat", "drum", "guitar", "piano",
            "sitar", "tabla", "flute", "violin", "instrument",
        ],
    },

    # ── LOW PRIORITY (ambient / filler — filtered aggressively) ─────────
    "AMBIENT": {
        "display":  "[ Background Noise ]",
        "priority": "LOW",
        "boost":    0.5,
        "yamnet":   [
            "silence", "noise", "hum", "murmur", "ambient",
            "white noise", "static",
        ],
    },
}

# YAMNet class substrings that should ALWAYS be discarded
# (transport noise, common background that YAMNet over-predicts)
YAMNET_BLACKLIST = [
    "vehicle", "car", "truck", "motorcycle", "bus", "train", "aircraft",
    "engine", "bicycle", "traffic", "road", "rail", "boat", "ship",
    "rain", "drizzle", "thunder shower",           # handled separately above
    "wind", "rustle", "leaves",
    "television", "radio",                         # usually speech channel bleed
    "printer", "keyboard",                         # office ambient
    "air conditioning", "fan", "vacuum cleaner",
    "crowd", "hubbub",                             # too generic
    "speech", "conversation", "narration",         # speech — not CC
    "singing",                                     # usually part of lyrics track
]

# ─────────────────────────────────────────────
# MODULE 2 — Visual Reaction Detection
# ─────────────────────────────────────────────

# Seconds to sample around each audio event timestamp
VISUAL_WINDOW_BEFORE_SEC = 0.5
VISUAL_WINDOW_AFTER_SEC  = 1.2

# Max frames sampled per window (evenly spaced)
VISUAL_MAX_FRAMES_PER_WINDOW = 8

# Minimum number of valid (face-detected) frames to trust a score
VISUAL_MIN_VALID_FRAMES = 2

# MediaPipe Face Mesh confidence thresholds
MEDIAPIPE_DETECTION_CONFIDENCE  = 0.5
MEDIAPIPE_TRACKING_CONFIDENCE   = 0.5

# ── Facial Landmark Indices ──────────────────────────────────────────────────
# Eye Aspect Ratio (EAR) — wide eyes = surprise / fear
EYE_LANDMARKS = {
    "left":  {"top": 159, "bottom": 145, "inner": 133, "outer": 33},
    "right": {"top": 386, "bottom": 374, "inner": 362, "outer": 263},
}
# Mouth Aspect Ratio (MAR) — open mouth = surprise / reaction
MOUTH_LANDMARKS = {
    "top": 13, "bottom": 14, "left": 78, "right": 308,
    "top2": 12, "bottom2": 15,
}
# Brow Raise — upper brow vs eye-corner distance
BROW_LANDMARKS = {
    "left_brow":  [70, 63, 105, 66, 107],
    "right_brow": [300, 293, 334, 296, 336],
    "left_eye_top":  159,
    "right_eye_top": 386,
}

# Baseline EAR (neutral relaxed eye)
EAR_BASELINE = 0.25
# EAR delta that counts as "wide eyes" reaction
EAR_REACTION_DELTA = 0.05

# Baseline MAR (closed mouth)
MAR_BASELINE = 0.05
# MAR delta that counts as "open mouth" reaction
MAR_REACTION_DELTA = 0.06

# Brow raise threshold (normalized by face height)
BROW_RAISE_THRESHOLD = 0.02

# Weights for combining sub-scores into the final visual reaction score
VISUAL_WEIGHT_EAR  = 0.35
VISUAL_WEIGHT_MAR  = 0.40
VISUAL_WEIGHT_BROW = 0.25

# ─────────────────────────────────────────────
# MODULE 3 — Fusion Decision Engine
# ─────────────────────────────────────────────

# Weight given to audio confidence vs visual reaction score
FUSION_AUDIO_WEIGHT  = 0.65
FUSION_VISUAL_WEIGHT = 0.35

# Priority-tier overrides: lower threshold → easier to emit caption
FUSION_THRESHOLD = {
    "HIGH":   0.28,   # scream, explosion → lenient
    "MEDIUM": 0.40,   # laughter, animal → normal
    "LOW":    0.60,   # ambient → strict
}

# Suppress duplicate captions of the same category within N seconds
CAPTION_DEDUP_SEC = 3.0

# SRT subtitle display duration (seconds) per priority tier
SRT_DISPLAY_DURATION = {
    "HIGH":   2.5,
    "MEDIUM": 2.0,
    "LOW":    1.5,
}

# Minimum gap between any two SRT entries (seconds)
SRT_MIN_GAP_SEC = 0.3