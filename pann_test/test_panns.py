import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from panns_inference import AudioTagging

# ====================== CONFIG ======================
VIDEO_FOLDER = "data"
OUTPUT_FILE = "panns_results.txt"
SAMPLE_RATE = 32000
# ===================================================

print("Loading PANNs model...")
model = AudioTagging(checkpoint_path=None, device='cpu')

results = []

def extract_audio_ffmpeg(video_path: str) -> np.ndarray:
    """Robust audio extraction using ffmpeg"""
    video_path = str(Path(video_path).resolve())

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            try:
                import imageio_ffmpeg
                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception as e:
                raise RuntimeError("ffmpeg not found.") from e

        cmd = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", video_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "wav",
            tmp_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        data, _ = sf.read(tmp_path, dtype="float32")
        return data
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


for filename in os.listdir(VIDEO_FOLDER):
    if filename.endswith(('.mp4', '.mkv', '.avi', '.mov')):
        video_path = os.path.join(VIDEO_FOLDER, filename)
        print(f"\nProcessing: {filename}")

        try:
            # Step 1: Extract audio
            waveform = extract_audio_ffmpeg(video_path)

            if waveform is None or len(waveform) == 0:
                print(f"  ❌ Could not extract audio")
                continue

            # Step 2: Fix shape for PANNs
            if waveform.ndim == 1:
                waveform = waveform.reshape(1, -1)

            # Step 3: Run PANNs (ROBUST version)
            output = model.inference(waveform)

            # Handle both dict and tuple return types
            if isinstance(output, dict):
                clipwise_output = output['clipwise_output']
            elif isinstance(output, (list, tuple)):
                clipwise_output = output[0]
            else:
                clipwise_output = output

            # Make sure we have a 1D array of scores
            if hasattr(clipwise_output, 'ndim') and clipwise_output.ndim > 1:
                clipwise_output = clipwise_output[0]

            # Get top 5 predictions
            top5_idx = np.argsort(clipwise_output)[::-1][:5]
            top5 = [(model.labels[i], float(clipwise_output[i])) for i in top5_idx]

            results.append({
                "video": filename,
                "top_predictions": top5
            })

            print(f"Top predictions for {filename}:")
            for label, score in top5:
                print(f"  {label:<45} → {score:.4f}")

        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")

# Save results
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"\n=== {r['video']} ===\n")
        for label, score in r['top_predictions']:
            f.write(f"{label}: {score:.4f}\n")

print(f"\n✅ Results saved to {OUTPUT_FILE}")