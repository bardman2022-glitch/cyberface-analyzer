import os
import base64
import math
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="CyberFace Telegram Mini App Backend")

# Enable CORS for Telegram WebApp access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reference placeholders (injected from gui.py)
analyzer = None
predictor = None

class FrameData(BaseModel):
    image_base64: str
    target_group: str = "Universal"
    draw_hud: bool = True

class DeepAnalysisData(BaseModel):
    face_crop_base64: str
    target_group: str = "Universal"
    is_webcam: bool = False

class SlotData(BaseModel):
    score: float
    geom_score: float
    symmetry: float
    golden_ratio: float
    details: list

class CombinedData(BaseModel):
    slots: dict
    target_group: str = "Universal"

def decode_base64_image(b64_str: str) -> np.ndarray:
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

def encode_image_base64(img: np.ndarray) -> str:
    try:
        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_bytes = base64.b64encode(buffer)
        return "data:image/jpeg;base64," + b64_bytes.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image encoding error: {e}")

@app.post("/analyze-frame")
def analyze_frame_endpoint(data: FrameData):
    if analyzer is None or predictor is None:
        raise HTTPException(status_code=503, detail="Engines not loaded on PC.")
        
    img = decode_base64_image(data.image_base64)
    hud_frame, face_crop, metrics = analyzer.analyze_frame(img, target_group=data.target_group, draw_hud=data.draw_hud)
    
    hud_b64 = encode_image_base64(hud_frame)
    crop_b64 = ""
    
    if metrics["detected"] and face_crop is not None:
        crop_b64 = encode_image_base64(face_crop)
        geom_data = {
            "symmetry": metrics.get("symmetry", 0.0),
            "golden_ratio": metrics.get("golden_ratio", 0.0),
            "overall_geom": metrics.get("overall_geom", 0.0)
        }
        score_10, raw_score = predictor.predict(face_crop, target_group=data.target_group, is_webcam=False, geom_data=geom_data)
        metrics["ai_score"] = score_10 if score_10 is not None else 0.0
        metrics["raw_score"] = raw_score if raw_score is not None else 0.0
        
    return {
        "hud_image": hud_b64,
        "face_crop": crop_b64,
        "metrics": metrics
    }

@app.post("/predict-deep")
def predict_deep_endpoint(data: DeepAnalysisData):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor engine not loaded on PC.")
        
    img = decode_base64_image(data.face_crop_base64)
    # Run deep TTA prediction
    score_10, raw_score = predictor.predict_deep(img, target_group=data.target_group, is_webcam=data.is_webcam)
    return {
        "ai_score": score_10 if score_10 is not None else 0.0,
        "raw_score": raw_score if raw_score is not None else 0.0
    }

@app.post("/calculate-combined")
def calculate_combined_endpoint(data: CombinedData):
    active_slots = {k: v for k, v in data.slots.items() if v is not None}
    
    if "Frontal" not in active_slots:
        raise HTTPException(status_code=400, detail="Frontal slot is required for combined calculation.")
        
    # Weights allocation
    weights = {"Frontal": 0.40}
    side_slots = [k for k in active_slots if k != "Frontal"]
    
    if side_slots:
        remaining_weight = 0.60
        w_each = remaining_weight / len(side_slots)
        for k in side_slots:
            weights[k] = w_each
    else:
        weights["Frontal"] = 1.0
        
    ai_score = sum(active_slots[k]["score"] * weights[k] for k in active_slots)
    geom_score = sum((active_slots[k]["geom_score"] or 0.0) * weights[k] for k in active_slots)
    symmetry = active_slots["Frontal"]["symmetry"] or 0.0
    golden_ratio = sum((active_slots[k]["golden_ratio"] or 0.0) * weights[k] for k in active_slots)
    
    # Potential calculation
    geom_factor = geom_score / 100.0
    potential_gain = (10.0 - ai_score) * (0.20 + 0.25 * geom_factor)
    potential_score = min(10.0, ai_score + potential_gain)
    
    # Tier text & color
    def get_tier_info(score):
        is_female = "Woman" in data.target_group
        if score < 3.0: return "SUB-3", "#ff3333"
        elif score < 4.0: return "SUB", "#ff6666"
        elif score < 5.0: return "LTB" if is_female else "LTN", "#ffcc00"
        elif score < 6.0: return "MTB" if is_female else "MTN", "#00f0ff"
        elif score < 7.0: return "HTB" if is_female else "HTN", "#00ff64"
        elif score < 8.0: return "STACYLITE" if is_female else "CHADLITE", "#bf00ff"
        else: return "STACY" if is_female else "CHAD", "#ff007f"
        
    tier_text, tier_color = get_tier_info(ai_score)
    
    # Percentile
    mean = 5.0
    std = 1.15
    z = (ai_score - mean) / std
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    top_pct = (1.0 - cdf) * 100.0
    top_pct = max(0.01, min(99.9, top_pct))
    
    return {
        "ai_score": ai_score,
        "geom_score": geom_score,
        "symmetry": symmetry,
        "golden_ratio": golden_ratio,
        "potential_score": potential_score,
        "tier_text": tier_text,
        "tier_color": tier_color,
        "top_pct": top_pct
    }

# Mount static webapp directory
tma_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tma")
if os.path.exists(tma_dir):
    app.mount("/", StaticFiles(directory=tma_dir, html=True), name="static")
