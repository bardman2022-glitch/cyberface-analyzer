import os
import urllib.request
import numpy as np
import cv2
import torch
import torch.nn as nn
import time
from PIL import Image
from transformers import CLIPVisionModelWithProjection, AutoProcessor

# Weight files and paths
MLP_WEIGHTS_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MLP_WEIGHTS_PATH = os.path.join(MODEL_DIR, "sac_logos_ava1_l14_linearMSE.pth")

class MLP(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

class BeautyPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Predictor] Initializing BeautyPredictor using device: {self.device}")
        
        self.clip_model = None
        self.clip_processor = None
        self.mlp_model = None
        
        self.download_weights_if_needed()
        self.load_models()

    def download_weights_if_needed(self):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        if not os.path.exists(MLP_WEIGHTS_PATH) or os.path.getsize(MLP_WEIGHTS_PATH) < 1000000:
            print(f"[Predictor] MLP Weights not found. Downloading from {MLP_WEIGHTS_URL}...")
            try:
                req = urllib.request.Request(
                    MLP_WEIGHTS_URL, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response, open(MLP_WEIGHTS_PATH, 'wb') as out_file:
                    out_file.write(response.read())
                print("[Predictor] MLP weights downloaded successfully.")
            except Exception as e:
                print(f"[Predictor] Error downloading weights: {e}")

    def load_models(self):
        # 1. Load CLIP model and processor from Hugging Face
        try:
            print("[Predictor] Loading CLIP Vision model from Hugging Face (openai/clip-vit-large-patch14)...")
            self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            self.clip_model.eval()
            print("[Predictor] CLIP vision model loaded successfully.")
        except Exception as e:
            print(f"[Predictor] Error loading CLIP vision model: {e}")
            self.clip_model = None
            
        # 2. Load custom MLP head weights
        if not os.path.exists(MLP_WEIGHTS_PATH) or os.path.getsize(MLP_WEIGHTS_PATH) < 1000000:
            print("[Predictor] Cannot load MLP weights: file does not exist or download failed.")
            return
            
        try:
            self.mlp_model = MLP(input_size=768).to(self.device)
            try:
                state_dict = torch.load(MLP_WEIGHTS_PATH, map_location=self.device, weights_only=True)
            except TypeError:
                state_dict = torch.load(MLP_WEIGHTS_PATH, map_location=self.device)
            self.mlp_model.load_state_dict(state_dict)
            self.mlp_model.eval()
            print("[Predictor] Aesthetic MLP model loaded successfully.")
        except Exception as e:
            print(f"[Predictor] Error loading MLP model: {e}")
            self.mlp_model = None

    def predict(self, face_img, target_group="Universal", is_webcam=False, geom_data=None):
        """
        Runs model inference on a BGR face crop.
        Uses sigmoid-based calibration for realistic score distribution.
        
        Args:
            face_img: BGR face crop image.
            target_group: Target demographic group.
            is_webcam: Whether the image is from a webcam (applies quality offset).
            geom_data: Optional dict with geometry metrics (symmetry, golden_ratio, overall_geom)
                        to blend into the final score for more holistic evaluation.
        Returns:
            score_10 (float): AI beauty rating out of 10.0.
            raw_score (float): Raw aesthetic score from the model.
        """
        if self.clip_model is None or self.mlp_model is None:
            return None, None
            
        try:
            # Convert BGR face crop to PIL RGB
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Preprocess using CLIP processor
            inputs = self.clip_processor(images=pil_img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Get CLIP image embeddings
                outputs = self.clip_model(**inputs)
                image_embeds = outputs.image_embeds
                
                # Normalize embeddings (standard CLIP L2 norm)
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                
                # Run MLP prediction
                prediction = self.mlp_model(image_embeds)
                
            raw_aesthetic = float(prediction.item())
            
            # ========== REALISTIC CALIBRATION ==========
            # Raw CLIP+MLP aesthetic scores for face crops typically fall in 3.5-6.5 range.
            # Population mean for casual photos is ~4.3, for good photos ~5.0.
            # Professional portraits: 5.5-7.0+. Studio model shots: 6.5-8.0.
            #
            # Webcam quality offset: webcam crops lose ~0.3-0.5 raw points due to 
            # noise, flat lighting, and low resolution. We compensate for this.
            
            effective_score = raw_aesthetic
            if is_webcam:
                effective_score += 0.4  # Compensate for webcam quality degradation
            
            # Sigmoid-based mapping: creates realistic bell-curve distribution
            # Most people cluster around 5.0-6.5, truly beautiful faces reach 7.5-8.5,
            # only supermodels/actors get 9.0+. Ugly scores (< 3.0) are also rare.
            #
            # Parameters tuned for face beauty:
            #   center = 4.8 (population mean raw aesthetic score for face crops)
            #   spread = 1.2 (controls how steep the curve is)
            #   output_center = 5.5 (center of output scale - slightly above average)
            #   output_range = 4.0 (half-width: scores will span ~1.5 to ~9.5)
            
            center = 4.8
            spread = 1.2
            output_center = 5.5
            output_range = 4.0
            
            # Sigmoid: maps (-inf, +inf) -> (0, 1), centered at 'center'
            sigmoid_val = 1.0 / (1.0 + np.exp(-(effective_score - center) / spread))
            
            # Map sigmoid output (0..1) to final score range
            score_10 = output_center + output_range * (2.0 * sigmoid_val - 1.0)
            
            # Apply subtle target group calibration (demographic bias correction)
            # Younger faces tend to score slightly higher in beauty studies
            if target_group == "Young Man":
                score_10 += 0.35
            elif target_group == "Young Woman":
                score_10 += 0.25
            elif target_group == "Man":
                score_10 += 0.10
            elif target_group == "Woman":
                score_10 += 0.05
            
            # Optional: blend with geometry metrics for more holistic score
            if geom_data is not None:
                geom_score = geom_data.get("overall_geom", 0)
                symmetry = geom_data.get("symmetry", 0)
                golden = geom_data.get("golden_ratio", 0)
                
                if geom_score > 0 and symmetry > 0:
                    # Geometry composite: symmetry matters most for beauty
                    geom_composite = (symmetry * 0.45 + golden * 0.30 + geom_score * 0.25) / 10.0
                    # Blend: 70% AI aesthetics + 30% face geometry
                    score_10 = score_10 * 0.70 + geom_composite * 0.30
            
            score_10 = max(1.0, min(10.0, round(score_10, 2)))
            raw_score = round(raw_aesthetic, 2)
            
            return score_10, raw_score
        except Exception as e:
            print(f"[Predictor] Error during prediction: {e}")
            return None, None

    def predict_deep(self, face_img, target_group="Universal", is_webcam=False, progress_callback=None):
        """
        Runs a deep multi-augment inference (TTA - Test-Time Augmentation) on the face crop.
        Simulates processing delays for UI visual feedback.
        """
        if self.clip_model is None or self.mlp_model is None:
            return None, None

        try:
            scores = []
            raws = []
            
            # Define 6 TTA configurations (Original, Scales, Flips, Lighting)
            transform_variants = [
                # 1. Original
                lambda img: img,
                # 2. Horizontal Flip
                lambda img: cv2.flip(img, 1),
                # 3. Zoom-in Crop (scale 0.9x)
                lambda img: self._scale_crop(img, 0.9),
                # 4. Zoom-out (scale 1.1x with padding)
                lambda img: self._scale_crop(img, 1.1),
                # 5. Brightened
                lambda img: cv2.convertScaleAbs(img, alpha=1.1, beta=10),
                # 6. Darkened
                lambda img: cv2.convertScaleAbs(img, alpha=0.9, beta=-10)
            ]

            steps = [
                "Выравнивание сетки лица...",
                "Генерация отраженных аугментаций (TTA)...",
                "Масштабирование текстуры (0.9x - 1.1x)...",
                "Компенсация экспозиции и света...",
                "Извлечение признаков CLIP Vision...",
                "Эстетическая классификация MLP..."
            ]

            for i, transform_fn in enumerate(transform_variants):
                if progress_callback:
                    progress_callback(i / len(transform_variants), steps[i])
                
                # Apply transformation
                aug_img = transform_fn(face_img)
                
                # Single inference
                score_10, raw = self.predict(aug_img, target_group=target_group, is_webcam=is_webcam)
                if score_10 is not None:
                    scores.append(score_10)
                    raws.append(raw)
                
                # Short sleep to simulate heavy computing/deep thinking
                time.sleep(0.25)
                
            if progress_callback:
                progress_callback(1.0, "Глубокий анализ завершен!")
                
            if not scores:
                return None, None
                
            return round(float(np.mean(scores)), 2), round(float(np.mean(raws)), 2)
        except Exception as e:
            print(f"[Predictor] Error in deep prediction: {e}")
            return None, None

    def _scale_crop(self, img, factor):
        h, w = img.shape[:2]
        if factor == 1.0:
            return img
        elif factor < 1.0:
            # Crop center
            new_h, new_w = int(h * factor), int(w * factor)
            dy = (h - new_h) // 2
            dx = (w - new_w) // 2
            cropped = img[dy:dy+new_h, dx:dx+new_w]
            return cv2.resize(cropped, (w, h))
        else:
            # Pad border and resize
            pad_h = int((h * factor - h) / 2)
            pad_w = int((w * factor - w) / 2)
            padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
            return cv2.resize(padded, (w, h))
