import cv2
import mediapipe as mp
import numpy as np
import json
import os
from typing import List, Dict

class VisualReactionAnalyzer:
    """Module 2: Speaker/Scene Reaction Detection"""
    
    def __init__(self):
        print("Module 2 - Visual Reaction Analyzer")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )

    def analyze_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return {"reaction_score": 0.0, "expression": "no_face", "confidence": 0.0}
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        #metrics
        eye_open = abs(landmarks[159].y - landmarks[145].y) + abs(landmarks[386].y - landmarks[374].y)
        mouth_open = abs(landmarks[13].y - landmarks[14].y)
        brow_raise = abs(landmarks[70].y - landmarks[300].y)
        
        #scoring formula
        score = (eye_open * 8 + mouth_open * 12 + brow_raise * 6)
        reaction_score = min(1.0, score)
        
        expression = "strong_surprise" if reaction_score > 0.65 else \
                    "moderate_reaction" if reaction_score > 0.35 else "neutral"
        
        return {
            "reaction_score": round(reaction_score, 3),
            "expression": expression,
            "eye_openness": round(float(eye_open), 3),
            "mouth_openness": round(float(mouth_open), 3),
            "confidence": round(reaction_score, 3)
        }

    def analyze_video_at_timestamps(self, video_path: str, timestamps: List[float] = None):
        if timestamps is None:
            timestamps = [1, 2, 3, 4, 5, 6]
        
        os.makedirs("module2_results", exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        
        print(f"Video: {video_path}")
        print(f"FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}\n")
        
        results = {}
        
        for ts in timestamps:
            frame_no = int(ts * cap.get(cv2.CAP_PROP_FPS))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print(f"❌ Could not read frame at {ts}s")
                continue
            
            data = self.analyze_frame(frame)
            results[ts] = data
            
            #save frames
            annotated = frame.copy()
            text = f"{ts}s | {data['reaction_score']} | {data['expression']}"
            cv2.putText(annotated, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            cv2.imwrite(f"module2_results/frame_{ts}s.jpg", annotated)
            print(f"✅ {ts}s → Score: {data['reaction_score']} | {data['expression']}")
        
        cap.release()
        
        with open("module2_results/results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\nModule 2 Completed")
        print("Results saved in 'module2_results' folder")
        return results


if __name__ == "__main__":
    analyzer = VisualReactionAnalyzer()
    analyzer.analyze_video_at_timestamps("sample_video.mp4")