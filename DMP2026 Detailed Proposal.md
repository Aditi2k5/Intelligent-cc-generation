# C4GT DMP 2026 — Project Proposal

## Intelligent Closed Caption (CC) Suggestion Tool
**Organization:** PlanetRead
**Program:** Code for GovTech (C4GT) — Digital Mentorship Programme 2026
**Project Category:** AI / Accessibility / Regional Language Media

---

## Table of Contents

1. [Contributor Information](#1-contributor-information)
2. [Executive Summary](#2-executive-summary)
3. [Problem Statement](#3-problem-statement)
4. [Proposed Solution](#4-proposed-solution)
5. [Multi-Language Support](#5-multi-language-support)
6. [Deployment Strategy](#6-deployment-strategy)
7. [Technical Implementation Detail](#7-technical-implementation-detail)
8. [Goals and Deliverables](#8-goals-and-deliverables)
9. [Project Timeline — 16 Weeks](#9-project-timeline--16-weeks)
10. [Mid-Point Milestone](#10-mid-point-milestone)
11. [Expected Outcomes](#11-expected-outcomes)
12. [Future Work (Post-DMP)](#12-future-work-post-dmp)
13. [Why This Contributor](#13-why-this-contributor)
14. [Setup and Installation](#14-setup-and-installation)
15. [References](#15-references)

---

## 1. Contributor Information

| Field | Details |
|-------|---------|
| **Project Title** | Intelligent Closed Caption (CC) Suggestion Tool |
| **Organization** | PlanetRead |
| **Programme** | C4GT Digital Mentorship Programme 2026 |
| **Primary Language** | Python 3.10+ |
| **Deployment Targets** | CLI Tool + Production Web Application |
| **Contributor** | Aditi Prabakaran, 3rd Year CSE Student |
| **Tech Stack** | Python, Tensorflow, Pytorchm OpenCV, Flask, SQL, Supabase, Firebase |
| **Repository** | *(https://github.com/Aditi2k5/Intelligent-cc-generation)* |

---

## 2. Executive Summary

Closed Captioning (CC) for non-speech audio events — `[ Glass Breaking ]`, `[ Laughter ]`, `[ Alarm ]` — is a critical accessibility requirement for deaf and hard-of-hearing audiences worldwide. For organizations like PlanetRead, which produce and distribute educational video content in Hindi and regional Indian languages at scale, manually adding these CC annotations is time-consuming, inconsistent, and expensive. Crucially, it has historically been done only in English — leaving Hindi, Tamil, Telugu, and other regional-language audiences without CC labels in their own language.

This project delivers an **AI-powered, fully automated, multilingual CC Suggestion Tool** that:

1. **Detects non-speech audio events** in any video file using YAMNet with a sliding-window approach, category-aware confidence boosting, and an aggressive blacklist tuned for Indian content.
2. **Assesses visual reactions** to those events by analysing facial expressions across a temporal window around each detected sound using MediaPipe Face Mesh.
3. **Makes an intelligent CC/no-CC decision** via a priority-aware weighted fusion engine that avoids over-captioning.
4. **Outputs subtitles in the user's chosen language** — including Hindi (Devanagari), Tamil, Telugu, Bengali, Marathi, Kannada, Malayalam, Gujarati, and Punjabi — making the tool genuinely useful for India's linguistic diversity.
5. **Ships in two deployment modes:** a full-featured production web application with a visual editor interface, and a standalone CLI tool for power users and automation pipelines.

The tool is designed to reduce manual CC annotation effort by **an estimated 60–80%** while keeping humans in the loop for final quality checks.

---

## 3. Problem Statement

### 3.1 The Accessibility Gap in Indian Regional Media

The vast majority of Hindi and regional-language video content — particularly educational and public-interest media — lacks non-speech CC annotations. The primary reason is operational: adding CC by hand requires a trained human editor to watch every second of video, decide whether a sound is narratively significant, and type a label at the correct timestamp. For a one-hour video, this typically takes 2–4 hours of editor time.

### 3.2 The Language Exclusion Problem

Even when non-speech CC exists in Indian media, it is almost universally written in English — `[ Screaming ]`, `[ Glass Breaking ]` — regardless of the language of the surrounding content. A hearing-impaired viewer watching a Tamil film or a Hindi educational video must read CC labels in a language that may not be their first. For a tool designed to improve literacy and accessibility, this is a significant shortcoming. A viewer whose primary language is Telugu should see `[ అరుపు ]` (screaming), not `[ Screaming ]`.

### 3.3 The Over-Captioning Problem

Existing automated tools tend to caption everything — including wind, background hum, or ambient traffic — producing CC files that are more distracting than helpful and that editors must heavily prune before use. The ideal tool should only suggest captions for sounds that genuinely affect the speaker or the scene, and should silently skip low-impact ambient noise.

### 3.4 The Indian Content Challenge

Standard sound classification models (trained primarily on English-language Western media) frequently misfire on Indian content:
- Tabla and dholak rhythms get classified as generic "drum" or "noise"
- Street sounds in Indian cities (autorickshaw horns, vendor calls) confuse vehicle/traffic classifiers
- Regional-language speech patterns affect voice activity detection around sound events
- Firecrackers during Diwali are misclassified as gunshots or explosions

A robust solution for PlanetRead's use case must explicitly account for and mitigate these failure modes.

### 3.5 The Accessibility Gap in Tooling

Currently, there is no open-source, freely available, production-quality tool that an Indian accessibility editor can simply open in a browser, upload a video, and receive a multilingual SRT file ready for review. The gap is not just in the ML models — it is in the complete workflow from upload to usable output.

---

## 4. Proposed Solution

### 4.1 Architecture Overview

The tool is a three-module Python pipeline that accepts any video file and produces a multilingual subtitle-ready output:

```
Input Video
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Module 1: Sound Event Detection                     │
│  YAMNet + sliding window + blacklist filter          │
│  + priority-aware boost + temporal merge + cap       │
└─────────────────────────┬────────────────────────────┘
                          │ [AudioEvent list]
                          ▼
┌──────────────────────────────────────────────────────┐
│  Module 2: Visual Reaction Detection                 │
│  MediaPipe Face Mesh + temporal window               │
│  + EAR / MAR / Brow Raise + weighted aggregation    │
└─────────────────────────┬────────────────────────────┘
                          │ [VisualScore per timestamp]
                          ▼
┌──────────────────────────────────────────────────────┐
│  Module 3: Fusion Decision Engine                    │
│  Weighted fusion + priority threshold                │
│  + deduplication + SRT gap enforcement               │
└─────────────────────────┬────────────────────────────┘
                          │ [CaptionEntry list]
                          ▼
┌──────────────────────────────────────────────────────┐
│  Module 4: Multilingual CC Renderer          [NEW]   │
│  Language selector → translated CC labels            │
│  Script-aware rendering (Devanagari, Tamil, etc.)    │
└─────────────────────────┬────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
     output.srt       report.json    annotated frames
  (chosen language)  (full report)   (with labels)
```

### 4.2 Module 1 — Sound Event Detection (Goal 1)

**Status: Implemented and tested**

The audio detection pipeline goes significantly beyond a naive single-pass YAMNet call:

**Sliding window with 50% overlap:**
A 0.96-second analysis window slides across the audio track with a 0.48-second hop (50% overlap), meaning any sound lasting as little as 0.5 seconds will appear in at least one window. This is critical for short sounds like rat squeaks, glass breaking, and chair creaks that a single-pass approach misses.

**21 semantic sound categories** spanning three priority tiers:

| Priority | Categories |
|----------|-----------|
| HIGH | Scream, Explosion, Gunshot, Glass Breaking, Crash, Alarm/Siren |
| MEDIUM | Laughter, Applause, Crying, Knock, Doorbell, Phone, Dog, Cat, Rodent Squeak, Chair Creak, Footsteps, Door, Thunder, Music |
| LOW | Ambient/Background Noise |

**Blacklist filter:** 28 YAMNet class substrings are pre-emptively discarded — including all vehicle/transport classes, rain, wind, crowd noise, and speech — which are the primary sources of false positives on Indian content.

**Priority-aware confidence boosting:** Each category applies a multiplier (1.0×–1.9×) to the raw YAMNet score before thresholding. This compensates for YAMNet's documented under-confidence on rare or unusual sounds.

**Temporal merging:** Events of the same category within 1.5 seconds are collapsed into a single event spanning the full detection window.

**Per-category cap:** At most 8 events per category per clip prevents any single sound type from flooding the output.

### 4.3 Module 2 — Visual Reaction Detection (Goal 2 / Mid-Point Milestone)

**Status: Implemented and tested**

For each detected audio event, the visual module samples up to 8 frames across a temporal window of [timestamp − 0.5 s, timestamp + 1.2 s] and runs MediaPipe Face Mesh on each frame.

**Three facial action features computed per frame:**

- **Eye Aspect Ratio (EAR):** Measures how wide-open the eyes are. Wide eyes (EAR significantly above a 0.25 neutral baseline) indicate surprise or fear.
- **Mouth Aspect Ratio (MAR):** Measures mouth opening. Open mouth (MAR above a 0.05 baseline) indicates shock or laughter.
- **Brow Raise:** Normalised distance from eyebrow landmarks to eye top, relative to face height. Raised brows combined with wide eyes strongly indicate startle or surprise.

**Temporal weighting:** Frames closer to the audio event receive exponentially higher weight, so a reaction occurring 0.2 s after the sound dominates over a neutral expression 1.0 s before it.

**Multi-face support:** Up to 4 faces are tracked simultaneously. The most reactive face is used as the representative score, handling group reaction scenarios common in Indian educational videos.

**Graceful degradation:** When no face is found, the module returns a zero visual score and the fusion engine automatically switches to audio-only mode.

### 4.4 Module 3 — Fusion Decision Engine (Goal 3)

**Status: Implemented and tested**

**Weighted fusion formula:**
```
fusion_score = 0.65 × audio_confidence + 0.35 × visual_reaction_score
```

**Priority-aware thresholds:**

| Priority | Threshold | Rationale |
|----------|-----------|-----------|
| HIGH | 0.28 | Screams and explosions must not be missed |
| MEDIUM | 0.40 | Standard signal-to-noise balance |
| LOW | 0.60 | Ambient sounds require very strong evidence |

**Audio-only fallback:** When no face is detected, all thresholds are reduced by 20% to maintain reasonable coverage.

**Temporal deduplication:** Two accepted captions of the same category within 3 seconds are compared by fusion score; the lower-scoring one is suppressed.

**SRT gap enforcement:** Guarantees no two subtitle entries overlap and maintains a minimum 0.3-second gap between any two entries.

---

## 5. Multi-Language Support

### 5.1 Overview

This is a new, dedicated module (Module 4) that transforms the English-language CC labels produced by the fusion engine into the user's chosen output language. It runs as the final step of the pipeline, after all ML processing is complete, and adds zero latency to the detection phase.

### 5.2 Supported Languages

The tool will support **10 Indian languages** at launch, covering over 90% of India's population:

| Code | Language | Script | Script Name | Sample CC |
|------|----------|--------|-------------|-----------|
| `en` | English | Latin | — | `[ Screaming ]` |
| `hi` | Hindi | Devanagari | देवनागरी | `[ चीख ]` |
| `ta` | Tamil | Tamil | தமிழ் | `[ கத்துகிறார்கள் ]` |
| `te` | Telugu | Telugu | తెలుగు | `[ అరుపు ]` |
| `bn` | Bengali | Bengali | বাংলা | `[ চিৎকার ]` |
| `mr` | Marathi | Devanagari | देवनागरी | `[ ओरडणे ]` |
| `kn` | Kannada | Kannada | ಕನ್ನಡ | `[ ಕಿರುಚಾಡು ]` |
| `ml` | Malayalam | Malayalam | മലയാളം | `[ നിലവിളി ]` |
| `gu` | Gujarati | Gujarati | ગુજરાતી | `[ ચીખ ]` |
| `pa` | Punjabi | Gurmukhi | ਗੁਰਮੁਖੀ | `[ ਚੀਕ ]` |

### 5.3 Translation Architecture

**Static translation dictionary (primary method):**
The CC label set is finite and controlled — there are exactly 21 sound categories, each with one display string. Rather than using a live translation API (which introduces latency, cost, and network dependency), all translations are stored as a static dictionary in `translations.py`:

```python
# translations.py  (excerpt)
CC_LABELS = {
    "SCREAM": {
        "en": "[ Screaming ]",
        "hi": "[ चीख ]",
        "ta": "[ கத்துகிறார்கள் ]",
        "te": "[ అరుపు ]",
        "bn": "[ চিৎকার ]",
        "mr": "[ ओरडणे ]",
        "kn": "[ ಕಿರುಚಾಡು ]",
        "ml": "[ നിലവിളി ]",
        "gu": "[ ચીખ ]",
        "pa": "[ ਚੀਕ ]",
    },
    "LAUGHTER": {
        "en": "[ Laughter ]",
        "hi": "[ हँसी ]",
        "ta": "[ சிரிப்பு ]",
        "te": "[ నవ్వు ]",
        "bn": "[ হাসি ]",
        "mr": "[ हास्य ]",
        "kn": "[ ನಗು ]",
        "ml": "[ ചിരി ]",
        "gu": "[ હાસ્ય ]",
        "pa": "[ ਹਾਸਾ ]",
    },
    # ... all 21 categories translated into all 10 languages
}
```
**Fallback chain:** If a translation for a specific category is missing in the chosen language, the tool falls back to English, logs a warning, and adds a note to the JSON report flagging the missing translation.

### 5.4 CLI Usage

```bash
# English (default)
python main.py --video clip.mp4

# Hindi subtitles
python main.py --video clip.mp4 --lang hi

# Tamil subtitles
python main.py --video clip.mp4 --lang ta

# Telugu subtitles
python main.py --video clip.mp4 --lang te

# List all supported languages
python main.py --list-languages
```

### 5.5 Web App Usage

In the web application, the language selector is a prominent dropdown on the upload page:

```
[ Upload Video ]  ─────────────────────────────────────────────
                  Select output language:  [ Hindi ▾ ]
                  ┌──────────────────────┐
                  │ ✓ Hindi              │
                  │   Tamil              │
                  │   Telugu             │
                  │   Bengali            │
                  │   Marathi            │
                  │   Kannada            │
                  │   Malayalam          │
                  │   Gujarati           │
                  │   Punjabi            │
                  │   English            │
                  └──────────────────────┘
                  [ Generate CC ]
```

The selected language is passed through the entire pipeline and applied at the rendering step. The preview panel in the editor interface shows labels in the chosen script. The downloaded SRT file contains the correct Unicode characters for the selected language.

---

## 6. Deployment Strategy

The tool ships in two distinct deployment modes targeting different user types. Both share the same underlying Python pipeline — the deployment layer is a thin wrapper over the core modules.

### 6.1 Deployment Mode A — Production Web Application

#### 6.1.1 Target User

Accessibility editors, content teams, and non-technical users at PlanetRead, broadcasters, EdTech companies, and government accessibility departments who need a zero-install, browser-based workflow.

#### 6.1.2 Technology Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| Backend API | **FastAPI** (Python) | Async, fast, auto-generates OpenAPI docs, native Python so pipeline imports work directly |
| Task queue | **Celery + Redis** | Video processing is long-running (30 s – 5 min); tasks must be async and non-blocking |
| Frontend | **React + Tailwind CSS** | Component-based, responsive, handles Indian script rendering out of the box |
| File storage | **Local filesystem** (dev) / **AWS S3** (prod) | S3 for scalable video upload/download; pre-signed URLs for security |
| Database | **PostgreSQL** | Stores job history, user feedback, and review decisions |
| Deployment | **Docker Compose** | Single `docker-compose up` starts API + worker + Redis + DB + frontend |
| Hosting | **Railway / Render / AWS EC2** | PaaS options for zero-downtime deployment |

#### 6.1.3 User Workflow

```
User opens browser
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  UPLOAD PAGE                                        │
│  ┌──────────────────────────────────────────────┐  │
│  │  Drop video file here (MP4, AVI, MKV, MOV)  │  │
│  └──────────────────────────────────────────────┘  │
│  Output language:  [ Hindi ▾ ]                      │
│  Processing mode:  ● Full (Audio + Visual)          │
│                    ○ Audio-only (faster)            │
│  [ Generate Captions ]                              │
└─────────────────────────────────────────────────────┘
        │  POST /api/jobs  (upload + params)
        ▼
┌─────────────────────────────────────────────────────┐
│  PROCESSING PAGE                                    │
│  Job ID: cc-2026-05-09-001                         │
│                                                     │
│  ████████████░░░░░░░░ 63%                           │
│  Stage: Visual Reaction Detection …                 │
│  Detected so far: 3 events                          │
└─────────────────────────────────────────────────────┘
        │  GET /api/jobs/{id}/status  (polling)
        ▼
┌─────────────────────────────────────────────────────┐
│  REVIEW PAGE                                        │
│  ┌─────────────────────────────────────────────┐   │
│  │  0:02.4  [ चीख ]             fusion=0.70  ✓│   │
│  │  0:05.7  [ हँसी ]            fusion=0.61  ✓│   │
│  │  0:08.1  [ शीशा टूटना ]      fusion=0.54  ✓│   │
│  └─────────────────────────────────────────────┘   │
│  [ Accept All ]  [ Edit Selected ]  [ Download SRT ]│
└─────────────────────────────────────────────────────┘
```

#### 6.1.4 API Design

```
POST   /api/jobs              Upload video + start processing job
GET    /api/jobs/{id}/status  Poll job progress (0–100%) + stage name
GET    /api/jobs/{id}/result  Retrieve completed caption entries
PATCH  /api/jobs/{id}/review  Submit editor Accept/Reject/Edit decisions
GET    /api/jobs/{id}/srt     Download the final SRT file
GET    /api/languages         List all supported output languages
GET    /api/health            Health check for load balancer
```

All endpoints return JSON. The SRT download endpoint returns `Content-Type: text/plain; charset=utf-8` with `Content-Disposition: attachment; filename="output.srt"`.

#### 6.1.5 Job Processing Flow

```
Upload → FastAPI endpoint
    → Save video to /uploads/{job_id}/input.mp4
    → Create job record in PostgreSQL (status: QUEUED)
    → Push task to Celery queue via Redis
    → Return {job_id, status: "queued"}

Celery worker picks up task:
    → Update status: PROCESSING, stage: "audio_detection"
    → Run Module 1 (sound_detector.py)
    → Update status: PROCESSING, stage: "visual_analysis", progress: 40%
    → Run Module 2 (visual_detector.py)
    → Update status: PROCESSING, stage: "fusion", progress: 80%
    → Run Module 3 (fusion_engine.py)
    → Apply language translation (translations.py)
    → Write output.srt + report.json
    → Update status: COMPLETE, progress: 100%

Client polls GET /status until complete, then redirects to review page.
```

#### 6.1.6 Editor Review Interface

The review page presents each CC suggestion as a card with:
- Timestamp and duration (synced to a video player embed)
- CC label in the chosen language
- Audio/visual/fusion score breakdown (expandable)
- Three action buttons: **✓ Accept**, **✗ Reject**, **✎ Edit**

The "Edit" button opens an inline text field where the editor can correct or rephrase the CC text in the chosen language, including typing in Devanagari or other scripts via the OS input method.

All decisions are saved to PostgreSQL and used as future training data for threshold calibration.

---

### 6.2 Deployment Mode B — CLI Tool

#### 6.2.1 Target User

Developers, researchers, pipeline integrators, and power users who need to process videos programmatically, in batch, or as part of a larger automation workflow.

#### 6.2.2 Installation

```bash
pip install cc-suggestion-tool
```

The package is published to PyPI with all dependencies declared in `setup.cfg`. ffmpeg is documented as a system dependency in the README.

#### 6.2.3 Full CLI Reference

```
usage: cctools [-h] COMMAND [OPTIONS]

Commands:
  run       Process a single video file
  batch     Process all videos in a folder
  review    Open interactive review session for a completed job
  languages List all supported output languages
  version   Print version and dependency info

────────────────────────────────────────────────────
cctools run [OPTIONS]

  --video PATH          Input video file (required)
  --lang CODE           Output language [default: en]
                        Choices: en hi ta te bn mr kn ml gu pa
  --output DIR          Output directory [default: demo_results/]
  --no-visual           Skip visual reaction detection (faster)
  --no-frames           Skip annotated frame export
  --debug               Enable DEBUG-level logs
  --format {srt,sls}    Output subtitle format [default: srt]
  --threshold FLOAT     Override fusion threshold (0.0–1.0)

Examples:
  cctools run --video lecture.mp4 --lang hi
  cctools run --video clip.mp4 --lang ta --no-visual --output /tmp/out/
  cctools run --video movie.mp4 --debug

────────────────────────────────────────────────────
cctools batch [OPTIONS]

  --input-dir DIR       Folder containing video files (required)
  --output-dir DIR      Root folder for results [default: batch_results/]
  --lang CODE           Output language [default: en]
  --workers INT         Parallel worker processes [default: CPU count]
  --no-visual           Skip visual detection for all videos
  --format {srt,sls}    Output subtitle format [default: srt]

Examples:
  cctools batch --input-dir /videos/ --lang hi --workers 4
  cctools batch --input-dir /videos/ --lang te --output-dir /out/

────────────────────────────────────────────────────
cctools languages

  Prints a table of all supported language codes and scripts:
  en  English     Latin
  hi  Hindi       Devanagari
  ta  Tamil       Tamil
  te  Telugu      Telugu
  ... (all 10)
```

#### 6.2.4 Terminal Output Example

```
╔══════════════════════════════════════════════════════════╗
║   Intelligent CC Suggestion Tool  •  PlanetRead / C4GT  ║
║   Language: Hindi (हिन्दी)  •  Format: SRT              ║
╚══════════════════════════════════════════════════════════╝

[1/3] Running sound event detection …
[2/3] Running visual reaction detection …
[3/3] Running fusion engine …

════════════════════════════════════════════════════════════
  Pipeline complete in 18.3 s
  Audio events detected : 4
  Captions emitted      : 3
  Language              : Hindi (hi)

  Caption preview:
  ──────────────────────────────────────────────────────
    2.40s →  4.40s   [ चीख ]                fusion=0.703
    5.76s →  7.76s   [ हँसी ]               fusion=0.612
    9.12s → 11.12s   [ शीशा टूटना ]         fusion=0.541
  ──────────────────────────────────────────────────────
  SRT  → demo_results/output.srt
  JSON → demo_results/report.json
════════════════════════════════════════════════════════════
```

#### 6.2.5 Batch Processing Output

```
cctools batch --input-dir /videos/ --lang hi --workers 4

Processing 12 videos with 4 workers …

  [1/12] lecture_01.mp4         ✓  3 captions  (22.1 s)
  [2/12] classroom_demo.mp4     ✓  1 caption   (14.6 s)
  [3/12] interview_segment.mp4  ✓  5 captions  (31.4 s)
  ...
  [12/12] community_video.mp4   ✓  2 captions  (19.8 s)

Batch complete in 3m 41s
Total videos processed : 12
Total captions emitted : 38
Output directory       : batch_results/
Batch summary report   : batch_results/batch_summary.json
```

#### 6.2.6 Programmatic Python API

The CLI is a thin wrapper over a Python API that can be imported directly:

```python
from cc_suggestion_tool import CCPipeline

pipeline = CCPipeline(lang="hi", output_dir="results/")
result = pipeline.run("video.mp4")

print(result.captions)
# [
#   CaptionEntry(start=2.4, end=4.4, text="[ चीख ]", fusion=0.703),
#   CaptionEntry(start=5.76, end=7.76, text="[ हँसी ]", fusion=0.612),
# ]

pipeline.write_srt(result, "output.srt")
pipeline.write_json(result, "report.json")
```

---

### 6.3 Shared Infrastructure

Both deployment modes share the same underlying pipeline. The only difference is the layer above it:

```
┌─────────────────────────┐    ┌─────────────────────────┐
│   Web Application       │    │   CLI Tool              │
│   FastAPI + React       │    │   cctools run / batch   │
│   Celery + Redis        │    │   Python API            │
└────────────┬────────────┘    └────────────┬────────────┘
             │                              │
             └──────────┬───────────────────┘
                        ▼
         ┌──────────────────────────────┐
         │  Core Pipeline (shared)      │
         │  sound_detector.py           │
         │  visual_detector.py          │
         │  fusion_engine.py            │
         │  translations.py             │
         │  srt_writer.py               │
         └──────────────────────────────┘
```

This means every bug fix, model improvement, or new language added to the core pipeline automatically benefits both deployment modes.

---

## 7. Technical Implementation Detail

### 7.1 Complete Tech Stack

| Component | Technology | Mode |
|-----------|-----------|------|
| Audio classification | YAMNet (TF Hub) | Both |
| Audio extraction | ffmpeg + soundfile + librosa | Both |
| Face mesh | MediaPipe Face Mesh | Both |
| Video decoding | OpenCV | Both |
| Language translation | Static dictionary (translations.py) | Both |
| Output format | SRT + JSON | Both |
| Testing | pytest (100+ tests) | Both |
| Backend API | FastAPI | Web only |
| Task queue | Celery + Redis | Web only |
| Frontend | React + Tailwind CSS | Web only |
| Database | PostgreSQL | Web only |
| Containerisation | Docker + Docker Compose | Web only |
| CLI packaging | setuptools + PyPI | CLI only |
| Batch processing | multiprocessing.Pool | CLI only |

### 7.2 Complete Project File Structure

```
cc_suggestion_tool/
│
├── config.py                      # All tunable parameters
├── main.py                        # CLI entry point
├── requirements.txt               # Python dependencies
├── setup.cfg                      # PyPI package config
│
├── modules/
│   ├── sound_detector.py          # Module 1: YAMNet audio detection
│   ├── visual_detector.py         # Module 2: MediaPipe face reaction
│   ├── fusion_engine.py           # Module 3: Weighted decision engine
│   └── translations.py            # Module 4: Multilingual CC labels  ← NEW
│
├── utils/
│   ├── logger.py                  # Colour-coded structured logging
│   └── srt_writer.py              # SRT + JSON output writers
│
├── tests/
│   ├── conftest.py                # Shared fixtures and markers
│   ├── test_sound_detector.py     # 34 unit tests
│   ├── test_visual_detector.py    # 19 unit tests
│   ├── test_fusion_engine.py      # 47 unit tests
│   └── test_translations.py       # 20 unit tests             ← NEW
│
├── webapp/                                                     ← NEW
│   ├── api/
│   │   ├── main.py                # FastAPI application
│   │   ├── routes/
│   │   │   ├── jobs.py            # Job CRUD endpoints
│   │   │   ├── review.py          # Editor review endpoints
│   │   │   └── languages.py      # Language listing endpoint
│   │   ├── models.py              # SQLAlchemy ORM models
│   │   ├── tasks.py               # Celery task definitions
│   │   └── schemas.py             # Pydantic request/response schemas
│   │
│   └── frontend/
│       ├── src/
│       │   ├── App.jsx
│       │   ├── pages/
│       │   │   ├── Upload.jsx     # Video upload + language picker
│       │   │   ├── Processing.jsx # Progress bar + live stage updates
│       │   │   └── Review.jsx     # Caption cards + Accept/Reject/Edit
│       │   └── components/
│       │       ├── LanguagePicker.jsx
│       │       ├── CaptionCard.jsx
│       │       └── ScoreBreakdown.jsx
│       └── package.json
│
├── Dockerfile                     # Pipeline + API container
├── docker-compose.yml             # Full stack: API + worker + Redis + DB + frontend
└── demo_results/                  # Auto-created per pipeline run
    ├── output.srt
    ├── report.json
    ├── pipeline.log
    └── frames/
```

### 7.3 Key Design Decisions

**Single config file:** Every threshold, weight, window size, and category definition lives in `config.py`. A reviewer, mentor, or future contributor can change the pipeline's entire behaviour without opening any module file. New sound categories with translations in all 10 languages can be added in one block with no code changes.

**Static translation dictionary:** All 21 × 10 = 210 CC translations are stored as a plain Python dictionary. This is offline, zero-latency, auditable, and correctable by a PlanetRead editor with no coding knowledge.

**Shared core pipeline:** The web app and CLI are both thin wrappers over the same `modules/` directory. There is no code duplication between deployment modes.

**Dataclass-driven data flow:** `AudioEvent`, `VisualScore`, `FaceFrameScore`, and `CaptionEntry` are Python dataclasses with explicit types. This makes inter-module contracts clear and enables `asdict()` serialisation for JSON.

**Mock-safe test design:** All 100+ tests stub TensorFlow, MediaPipe, and OpenCV before importing modules. The test suite runs in approximately 1 second with no internet connection and no ML model downloads required.

**Graceful degradation chain:** If ffmpeg absent → librosa fallback. If Module 2 fails → audio-only mode. If language missing → English fallback. If no events → empty SRT written. The pipeline always produces output.

---

## 8. Goals and Deliverables

### Goal 1 — Sound Event Detection Module ✅ Completed

| Deliverable | Status |
|------------|--------|
| Audio extraction via ffmpeg with librosa fallback | Done |
| YAMNet sliding-window inference (50% overlap) | Done |
| 28-entry blacklist for Indian content false positives | Done |
| 21 semantic sound categories across 3 priority tiers | Done |
| Priority-aware confidence boosting (1.0×–1.9×) | Done |
| Temporal merging and per-category capping | Done |
| 34 unit tests with full mock coverage | Done |

### Goal 2 — Visual Reaction Detection Module ✅ Completed (Mid-Point Milestone)

| Deliverable | Status |
|------------|--------|
| Temporal sampling window around each audio event | Done |
| MediaPipe Face Mesh landmark extraction | Done |
| EAR (Eye Aspect Ratio) computation | Done |
| MAR (Mouth Aspect Ratio) computation | Done |
| Brow Raise computation | Done |
| Temporal weighting | Done |
| Multi-face support (up to 4 faces) | Done |
| 19 unit tests with synthetic geometry | Done |

### Goal 3 — CC Decision Engine & SRT/SLS Output ✅ Completed

| Deliverable | Status |
|------------|--------|
| Weighted fusion formula (65/35 audio/visual split) | Done |
| Priority-aware thresholds (HIGH/MEDIUM/LOW) | Done |
| Audio-only fallback mode | Done |
| Temporal deduplication | Done |
| SRT gap enforcement | Done |
| Annotated frame export | Done |
| SRT + JSON output writers | Done |
| 47 unit tests including integration tests | Done |
| CLI entry point with all flags | Done |

### Goal 4 — Multi-Language Support 🔲 Planned (DMP Phase 2)

| Deliverable | Status |
|------------|--------|
| `translations.py` with all 21 categories × 10 languages | Planned |
| `--lang CODE` CLI flag | Planned |
| Language listing command (`cctools languages`) | Planned |
| UTF-8 SRT output for all Indian scripts | Planned |
| Language picker in web UI | Planned |
| 20 unit tests for translation module | Planned |
| Fallback to English when translation missing | Planned |

### Goal 5 — Production Web Application 🔲 Planned (DMP Phase 3–4)

| Deliverable | Status |
|------------|--------|
| FastAPI backend with async job processing | Planned |
| Celery + Redis task queue | Planned |
| PostgreSQL job database | Planned |
| React + Tailwind CSS frontend | Planned |
| Upload / Processing / Review page flow | Planned |
| Editor Accept/Reject/Edit interface | Planned |
| SRT download endpoint | Planned |
| Docker Compose deployment config | Planned |

### Goal 6 — CLI Package & Batch Processing 🔲 Planned (DMP Phase 3)

| Deliverable | Status |
|------------|--------|
| `cctools` CLI command via PyPI install | Planned |
| `cctools batch` with parallel workers | Planned |
| Python API for programmatic use | Planned |
| Progress bars via tqdm | Planned |
| Batch summary JSON report | Planned |

---

## 9. Project Timeline — 16 Weeks

### Phase 0 — Pre-DMP Work Already Completed

| Area | Completed Work |
|------|---------------|
| Module 1 | YAMNet pipeline with blacklist, boost, merge, cap |
| Module 2 | MediaPipe face mesh with EAR/MAR/Brow + temporal aggregation |
| Module 3 | Fusion engine with priority thresholds, dedup, SRT enforcement |
| Testing | 100 unit tests, all passing, run in ~1 second |
| CLI | `main.py` with `--video`, `--output`, `--no-visual`, `--debug` flags |
| Config | `config.py` with all tunable parameters |

---

### Phase 1 — Validation & Baseline (Weeks 1–2)

**Objective:** Establish a quantitative performance baseline on real content.

| Week | Tasks |
|------|-------|
| **Week 1** | Run the pipeline on 15 sample Hindi/regional video clips (Creative Commons or PlanetRead-provided). Document all false positives and false negatives from Module 1. Build a simple annotation spreadsheet: video, timestamp, ground-truth CC, predicted CC, correct Y/N. |
| **Week 2** | Analyse Module 1 errors. Extend the blacklist with observed Indian false-positive classes. Validate Module 2 on clips with varying lighting and face angles. Compute baseline precision, recall, and F1 score. Write the baseline evaluation report. |

**Milestone:** Baseline evaluation report with precision/recall/F1 on ≥ 15 test clips.

---

### Phase 2 — Multi-Language Support (Weeks 3–5)

**Objective:** Deliver complete multilingual CC output as a production-ready feature.

| Week | Tasks |
|------|-------|
| **Week 3** | Create `modules/translations.py`. Populate all 21 sound category labels for English, Hindi, and Tamil (highest priority languages). Write the 20-unit test suite for the translation module. Add `--lang` flag to `main.py`. Verify SRT files render correctly in VLC for all three languages. |
| **Week 4** | Add translations for Telugu, Bengali, Marathi. Test SRT files in Subtitle Edit and YouTube's caption upload flow. Verify Devanagari and Bengali scripts render correctly on Windows, macOS, and Android. Add `cctools languages` command. |
| **Week 5** | Add translations for Kannada, Malayalam, Gujarati, Punjabi. Test all 10 languages end-to-end. Write the fallback logic (missing translation → English + warning). Integrate the language selector into the CLI's `--help` output. |

**Milestone:** All 10 languages producing correct, UTF-8 SRT output verified on 3 platforms.

---

### Phase 3 — CLI Package & Batch Processing (Weeks 6–8)

**Objective:** Ship a proper installable CLI package with batch support.

| Week | Tasks |
|------|-------|
| **Week 6** | Refactor `main.py` into a proper `CCPipeline` class with a clean Python API. Write `setup.cfg` and `pyproject.toml` for PyPI packaging. Publish a test release to TestPyPI. Verify `pip install cc-suggestion-tool && cctools run --video test.mp4` works on a clean virtual environment. |
| **Week 7** | Implement `cctools batch` with `multiprocessing.Pool` for parallel video processing. Add `tqdm` progress bars for both single-video and batch modes. Write the batch summary JSON report (total videos, total captions, per-video stats, duration). |
| **Week 8** | Performance profiling on a 10-minute video. Implement optional spectrogram-based silence skipping in Module 1 to reduce processing time by 20–40% on sparse audio. Document processing speed benchmarks in the README. Target: < 0.5× real-time on CPU. |

**Milestone:** `pip install cc-suggestion-tool` works cleanly; a 12-video batch completes in under 5 minutes on CPU.

---

### Phase 4 — Web Application (Weeks 9–13)

**Objective:** Build and deploy the production web application.

| Week | Tasks |
|------|-------|
| **Week 9** | Set up FastAPI project structure. Implement `POST /api/jobs` (upload + enqueue) and `GET /api/jobs/{id}/status` (polling). Set up Celery + Redis. Write the Celery task that wraps the CCPipeline. Test the async job flow with curl. |
| **Week 10** | Implement `GET /api/jobs/{id}/result`, `PATCH /api/jobs/{id}/review`, and `GET /api/jobs/{id}/srt` endpoints. Set up PostgreSQL with SQLAlchemy. Write Pydantic schemas for all request/response bodies. Add OpenAPI documentation. |
| **Week 11** | Build the React frontend: Upload page with drag-and-drop, language picker dropdown, and processing mode selector. Build the Processing page with a polling progress bar showing the current stage name and live event count. |
| **Week 12** | Build the Review page: caption cards in the chosen language script, video player embed synced to timestamps, Accept/Reject/Edit buttons, SRT download button. Build the `ScoreBreakdown` expandable component showing audio/visual/fusion scores. |
| **Week 13** | Write `Dockerfile` and `docker-compose.yml` for the full stack. Deploy to Railway or Render. End-to-end test: upload a Hindi video on the live URL, select Hindi, download the SRT, verify content. Run a load test with 5 concurrent jobs. |

**Milestone:** Live deployment at a public URL; 5 concurrent jobs processing correctly; Hindi SRT downloads working.

---

### Phase 5 — Indian Content Adaptation & Editor Feedback (Weeks 14–15)

**Objective:** Tune the tool specifically for Indian content and collect real editor feedback.

| Week | Tasks |
|------|-------|
| **Week 14** | Add 8 India-specific sound categories: Dhol/Dholak, Tabla, Shehnai, Firecrackers (Diwali), Autorickshaw Horn, Conch Shell (Shankha), Crowd Chanting, Train Whistle. Add translations for all new categories in all 10 languages. Test on public-domain Indian media clips. Curate a 50-clip Hindi/regional benchmark dataset with manual CC ground truth. |
| **Week 15** | Conduct a structured usability session with 2–3 PlanetRead editors using the web application. Collect 100+ Accept/Reject/Edit decisions. Analyse patterns: which categories over-caption, which under-caption. Adjust thresholds in `config.py`. Measure editor acceptance rate (target: ≥ 80%). |

**Milestone:** 50-clip benchmark dataset published; editor acceptance rate ≥ 80% measured on review session data.

---

### Phase 6 — Final Hardening & Submission (Week 16)

**Objective:** Polish, document, and submit all deliverables.

| Week | Tasks |
|------|-------|
| **Week 16** | Write complete README covering both deployment modes, all CLI flags, and web app workflow. Write contributor guide for future DMP participants. Record a 5-minute video walkthrough. Final precision/recall report comparing Phase 1 baseline to final numbers. Tag a v1.0.0 release on GitHub. Submit all deliverables to the C4GT portal. |

**Final Milestone:** v1.0.0 tagged; all deliverables submitted; live web app running; CLI installable from PyPI.

---

## 10. Mid-Point Milestone

As specified in the project brief, the mid-point milestone is the completion of **Goal 1 (Sound Event Detection)** and **Goal 2 (Visual Reaction Detection)**.

**Both goals are already fully implemented** as of programme start, with verifiable evidence:

| Evidence | Detail |
|----------|--------|
| `modules/sound_detector.py` | 453 lines — full YAMNet pipeline |
| `modules/visual_detector.py` | 425 lines — MediaPipe + EAR/MAR/Brow |
| `tests/test_sound_detector.py` | 34 passing tests |
| `tests/test_visual_detector.py` | 19 passing tests |
| `modules/fusion_engine.py` | 543 lines — full fusion + SRT output |
| `tests/test_fusion_engine.py` | 47 passing tests |
| `main.py` | Complete CLI, 367 lines |

This means the DMP period is used entirely for **multi-language support, web deployment, batch processing, Indian content adaptation, editor feedback, and production polish** — rather than catching up on core functionality.

---

## 11. Expected Outcomes

### 11.1 Primary Deliverables

| # | Deliverable | Description |
|---|------------|-------------|
| 1 | **Production pipeline** | Tested 4-module Python pipeline (3 implemented + translations module) |
| 2 | **10-language SRT output** | CC labels in English, Hindi, Tamil, Telugu, Bengali, Marathi, Kannada, Malayalam, Gujarati, Punjabi |
| 3 | **Production web app** | FastAPI + React web application, Docker-deployed, publicly accessible |
| 4 | **CLI tool on PyPI** | `pip install cc-suggestion-tool` → `cctools run --video clip.mp4 --lang hi` |
| 5 | **Batch processing** | `cctools batch` processes a folder with parallel workers |
| 6 | **120+ unit tests** | pytest suite running in < 10 seconds, 100% pass rate |
| 7 | **50-clip benchmark dataset** | Hindi/regional content with manual CC ground truth — public release |
| 8 | **JSON event report** | Full structured report for analytics and audit |
| 9 | **Editor review interface** | Web UI for Accept/Reject/Edit with decision storage |
| 10 | **Full documentation** | README, contributor guide, architecture diagram, 5-minute demo video |

### 11.2 Quantitative Targets

| Metric | Target |
|--------|--------|
| Precision (valid CC / total suggested) | ≥ 75% |
| Recall (valid CC / total ground-truth CCs) | ≥ 70% |
| Editor acceptance rate | ≥ 80% |
| Processing speed (CPU laptop) | < 0.5× real-time |
| Language coverage | 10 languages, 21 categories each |
| Translation accuracy (reviewed by native speakers) | ≥ 95% |
| Test suite pass rate | 100% |
| Test suite run time | < 10 seconds |
| Web app concurrent job capacity | ≥ 5 simultaneous |
| Batch processing speed (CPU, 4 workers) | ≥ 3 videos/minute for 30-second clips |

### 11.3 Acceptance Criteria Mapping

**Criterion 1: Detect non-speech audio events**
→ Module 1 detects 21 categories (expandable to 29 with Indian sounds) with sliding-window YAMNet, boost, blacklist, and merge. Validated with 34 unit tests.

**Criterion 2: Assess speaker/scene reaction**
→ Module 2 computes EAR + MAR + Brow Raise across a temporal window. Validated with 19 unit tests.

**Criterion 3: Produce CC-annotated SRT avoiding over-captioning**
→ Module 3 applies priority-aware fusion thresholds with deduplication. Module 4 applies the chosen language. Validated with 47 unit tests. Output importable into any standard subtitle tool.

---

## 12. Future Work (Post-DMP)

| Enhancement | Description | Complexity |
|-------------|------------|------------|
| **Real-time streaming mode** | Process live video stream; output CC events via WebSocket | High |
| **YAMNet fine-tuning** | Fine-tune on curated Indian sound dataset (dholak, shehnai, firecrackers) | High |
| **LLM-enhanced CC text** | Generate contextually richer captions: `[ Chair creaking as host shifts nervously ]` | Medium |
| **SLS integration** | Direct API integration with PlanetRead's Same Language Subtitling pipeline | Medium |
| **More regional languages** | Add Odia, Assamese, Sindhi, Kashmiri, Urdu (Nastaliq script) | Medium |
| **Confidence calibration** | Platt scaling on editor decision data for calibrated probability output | Medium |
| **Mobile application** | React Native app for on-device processing of short clips | High |
| **Speaker diarisation** | Detect who is reacting (speaker A vs speaker B) in multi-person content | High |
| **Emotion classification** | Classify the type of reaction (fear, joy, surprise, disgust) for richer CC text | Medium |
| **Community translation portal** | Web interface for native speakers to review and correct translations | Low |

---

## 13. Why This Contributor

### 13.1 Work Already Done

Before submitting my proposal, I have already:

- Implemented all three core detection modules from scratch (5 Python files, ~1,600 lines of production-quality, well-commented code)
- Written 100 unit tests that all pass with mocked dependencies in under 1 second
- Designed a category system with 21 sound classes, 3 priority tiers, and 28 blacklist entries specifically tuned for Indian content failure modes
- Implemented a temporal visual reaction window, EAR/MAR/Brow facial scoring, and weighted temporal aggregation across up to 4 simultaneous faces
- Built a full CLI with graceful degradation (Module 2 failure → audio-only; no face detected → auto-fallback; missing library → clear error message)
- Produced annotated output frames, a JSON event report, and a valid SRT file on real test clips

### 13.2 Architectural Thinking

The deployment strategy (shared core pipeline, thin wrapper for web and CLI) is not an afterthought — it is a deliberate architectural decision that prevents code duplication and ensures that every improvement benefits both deployment modes simultaneously. The static translation dictionary design was chosen over a live translation API specifically to avoid per-request cost and network dependency, making the tool viable for PlanetRead's offline and low-bandwidth use cases.

### 13.3 Understanding of PlanetRead's Mission

PlanetRead's core mission is Same Language Subtitling (SLS) for literacy improvement in India — a domain where every on-screen text element must be accurate enough to not mislead a learner reading along, sparse enough to not distract from the primary speech subtitle, and culturally calibrated to Indian audio environments. This proposal addresses all three requirements through the blacklist system, priority-tier thresholds, multi-language output, and the planned Indian content adaptation phase.

### 13.4 Commitment

I am available for the full 16-week DMP 2026 programme and commit to:
- Weekly progress updates via the C4GT platform
- Bi-weekly mentor sync calls
- All code submitted via pull request with passing CI checks
- A public GitHub repository with Issues tracking progress against this timeline
- Final submission including working demo URL, PyPI package, documentation, and benchmark dataset

---

## 14. Setup and Installation

### CLI

```bash
# Install system dependency
sudo apt install ffmpeg           # Ubuntu/Debian
brew install ffmpeg               # macOS

# Install the tool
pip install cc-suggestion-tool

# Run on a video in Hindi
cctools run --video lecture.mp4 --lang hi

# Run on a video in Tamil (audio-only mode)
cctools run --video clip.mp4 --lang ta --no-visual

# Batch process a folder in Telugu
cctools batch --input-dir /videos/ --lang te --workers 4

# List all supported languages
cctools languages
```

### Web Application (Docker)

```bash
git clone https://github.com/<your-username>/cc-suggestion-tool.git
cd cc-suggestion-tool

# Start the full stack (API + worker + Redis + DB + frontend)
docker-compose up --build

# Open in browser
open http://localhost:3000
```

### Development Setup

```bash
git clone https://github.com/<your-username>/cc-suggestion-tool.git
cd cc-suggestion-tool
pip install -r requirements.txt

# Run on a video
python main.py --video clip.mp4 --lang hi

# Run the test suite (no model downloads required — all mocked)
python -m pytest tests/ -v
```

---

## 15. References

- PlanetRead — Same Language Subtitling: https://planetread.org
- YAMNet (Yet Another Mobile Network): https://tfhub.dev/google/yamnet/1
- AudioSet (Google): Gemmeke et al., 2017 — https://research.google.com/audioset/
- MediaPipe Face Mesh: Kartynnik et al., 2019 — https://google.github.io/mediapipe/solutions/face_mesh
- PANNs (Pretrained Audio Neural Networks): Kong et al., 2020 — https://github.com/qiuqiangkong/audioset_tagging_cnn
- C4GT DMP 2026: https://codeforgovtech.in

---

*Proposal submitted for C4GT DMP 2026 | PlanetRead | Intelligent CC Suggestion Tool*
*Version: 2.0 | Date: May 2026*
