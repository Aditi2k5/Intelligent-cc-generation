# Module 2: Speaker/Scene Reaction Detection

**Part of Intelligent Closed Caption (CC) Suggestion Tool**  
**DMP 2026 - PlanetRead**
**Demo Video: https://youtu.be/J8Rm3TNLO6A**

## Goal

This module detects **visible reactions** from people on screen when a non-speech audio event occurs (e.g., door slam, explosion, laughter, sudden sound).  

It solves the core problem:  
**Audio alone cannot decide importance.** A loud sound with strong visual reaction (flinch, surprise, head turn) should get a caption, while background noise with no reaction should be ignored.

## Features

- Takes timestamps from Module 1 (audio events) and analyzes corresponding video frames.
- Uses facial landmarks to detect eye openness, mouth movement, and brow raise.
- Returns a **reaction score** (0.0 to 1.0) + expression label for each timestamp.
- Saves visualized frames and JSON results for easy review.
- Robust handling for vertical videos (YouTube Shorts) and varying lighting.

## Tech Stack

- **MediaPipe Face Mesh** — Primary model for 468 facial landmarks
- **OpenCV** — Video frame extraction and visualization
- **NumPy** — Landmark calculations

**Why this stack?** Lightweight, runs on CPU, good accuracy on Indian faces, and no heavy dependencies.

## How to Run

pip install -r requirements.txt
python module2.py

## Limitations

Works best with clearly visible faces (may struggle in very dark scenes or extreme angles).
Currently uses only facial landmarks (no body pose yet).
Scoring is heuristic-based (can be improved with more data).

## Future Improvements (Next Steps)

Add MediaPipe Pose for body movement / flinch detection.
Add temporal analysis (check multiple frames around each timestamp).
Fine-tune scoring using Indian video dataset.
Integrate smoothly with Module 1 (automatic timestamp passing) and Module 3 (fusion engine).
