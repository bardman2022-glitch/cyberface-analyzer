import os
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

class FaceGeometryAnalyzer:
    def __init__(self):
        self.download_model_if_needed()
        
        # Initialize MediaPipe Tasks FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Central axis landmarks (Forehead center, nose bridge, nose tip, chin center)
        self.axis_indices = [10, 6, 4, 152]
        
        # Symmetric landmark pairs
        self.symmetric_pairs = [
            (33, 263),    # Eye outer corners
            (133, 362),   # Eye inner corners
            (70, 300),    # Eyebrow outer corners
            (107, 336),   # Eyebrow inner corners
            (61, 291),    # Mouth corners
            (234, 454),   # Cheek outer edges
            (172, 397),   # Jaw outer edges
            (102, 329),   # Nose outer wings
        ]

    def download_model_if_needed(self):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
            print(f"[Analyzer] Face Landmarker task model not found. Downloading from {MODEL_URL}...")
            try:
                req = urllib.request.Request(
                    MODEL_URL, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
                    out_file.write(response.read())
                print("[Analyzer] Model downloaded successfully.")
            except Exception as e:
                print(f"[Analyzer] Error downloading face landmarker model: {e}")

    def _point_to_line_dist(self, p, a, b):
        """Calculates distance of point p from line defined by points a and b in 2D."""
        x0, y0 = p
        x1, y1 = a
        x2, y2 = b
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        if den == 0:
            return 0
        return num / den

    def _calculate_angle(self, p1, p2, p3):
        """Calculates the angle in degrees between vectors p2->p1 and p2->p3."""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        cos_angle = dot_product / (norm_v1 * norm_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return float(np.degrees(angle))

    def analyze_frame(self, frame, target_group="Universal"):
        """
        Analyzes a single frame for facial geometry and prepares a face crop.
        Returns:
            processed_frame (ndarray): Frame with neon HUD drawn.
            face_crop (ndarray): BGR cropped face image (or None if no face detected).
            metrics (dict): Calculated scores and logs.
        """
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)
        
        # Default empty output
        processed_frame = frame.copy()
        face_crop = None
        metrics = {
            "detected": False,
            "yaw": 0.0,
            "pose": "Unknown",
            "symmetry": 0.0,
            "golden_ratio": 0.0,
            "overall_geom": 0.0,
            "details": []
        }
        
        if not results.face_landmarks:
            return processed_frame, face_crop, metrics
            
        landmarks = results.face_landmarks[0]
        metrics["detected"] = True
        
        # Convert landmarks to pixel coordinates
        pts = []
        for lm in landmarks:
            pts.append((int(lm.x * w), int(lm.y * h)))
            
        # Get face bounding box with padding
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        fw, fh = max_x - min_x, max_y - min_y
        pad_x = int(fw * 0.20)
        pad_y = int(fh * 0.25)
        
        x1 = max(0, min_x - pad_x)
        y1 = max(0, min_y - pad_y)
        x2 = min(w, max_x + pad_x)
        y2 = min(h, max_y + pad_y)
        
        if x2 > x1 and y2 > y1:
            face_crop = frame[y1:y2, x1:x2].copy()

        # ------------------- 0. ESTIMATE YAW (HEAD ROTATION) -------------------
        # Nose tip (4), left cheek (234), right cheek (454)
        dx_left = abs(pts[4][0] - pts[234][0])
        dx_right = abs(pts[4][0] - pts[454][0])
        total_dx = dx_left + dx_right
        
        if total_dx > 0:
            ratio = dx_left / total_dx
            # Map ratio [0, 1] to yaw degrees [-80, 80]
            yaw_deg = (ratio - 0.5) * -160.0
        else:
            yaw_deg = 0.0
            
        metrics["yaw"] = round(yaw_deg, 1)
        
        abs_yaw = abs(yaw_deg)
        if abs_yaw < 15:
            metrics["pose"] = "Frontal"   # Анфас
            pose_name = "АНФАС"
        elif abs_yaw < 50:                # Полупрофиль
            if yaw_deg > 0:
                metrics["pose"] = "Left Semi-profile"
                pose_name = "ЛЕВЫЙ ПОЛУПРОФИЛЬ"
            else:
                metrics["pose"] = "Right Semi-profile"
                pose_name = "ПРАВЫЙ ПОЛУПРОФИЛЬ"
        else:                             # Профиль
            if yaw_deg > 0:
                metrics["pose"] = "Left Profile"
                pose_name = "ЛЕВЫЙ ПРОФИЛЬ"
            else:
                metrics["pose"] = "Right Profile"
                pose_name = "ПРАВЫЙ ПРОФИЛЬ"

        # ------------------- 1. SET IDEAL GEOMETRIC RATIOS BASED ON GROUP -------------------
        ideal_hw = 1.618
        ideal_nose = 1.618
        ideal_mouth_nose = 1.618
        ideal_eye_ratio = 1.0
        ideal_jaw_width = 0.80
        
        if target_group == "Young Man":    # Юноша (14-20)
            ideal_hw = 1.58
            ideal_jaw_width = 0.83
            lenient_mode = True
        elif target_group == "Man":        # Мужчина (21+)
            ideal_hw = 1.618
            ideal_jaw_width = 0.85
            lenient_mode = False
        elif target_group == "Young Woman":  # Девушка (14-20)
            ideal_hw = 1.60
            ideal_jaw_width = 0.75
            lenient_mode = True
        elif target_group == "Woman":      # Женщина (21+)
            ideal_hw = 1.63
            ideal_jaw_width = 0.75
            lenient_mode = False
        else:
            lenient_mode = False

        # ------------------- 2. GEOMETRIC ANALYSIS -------------------
        details = [f"Целевая группа: {target_group}", f"Определен ракурс: {pose_name} ({abs_yaw:.1f}°)"]

        if metrics["pose"] == "Frontal":
            # Symmetry
            axis_a = pts[10]
            axis_b = pts[152]
            sym_scores = []
            for left_idx, right_idx in self.symmetric_pairs:
                p_left = pts[left_idx]
                p_right = pts[right_idx]
                d_left = self._point_to_line_dist(p_left, axis_a, axis_b)
                d_right = self._point_to_line_dist(p_right, axis_a, axis_b)
                max_d = max(d_left, d_right)
                pair_sym = 1.0 - abs(d_left - d_right) / max_d if max_d > 0 else 1.0
                sym_scores.append(pair_sym)
                
            avg_symmetry = np.mean(sym_scores) * 100.0
            if lenient_mode:
                avg_symmetry = min(100.0, avg_symmetry * 1.03 + 1.0)
                
            metrics["symmetry"] = round(avg_symmetry, 1)
            details.append(f"Общая симметрия лица: {avg_symmetry:.1f}%")
            
            # Proportions
            face_height = math.dist(pts[10], pts[152])
            face_width = math.dist(pts[234], pts[454])
            ratio_hw = face_height / face_width if face_width > 0 else 0
            match_hw = 1.0 - abs(ratio_hw - ideal_hw) / ideal_hw
            details.append(f"Соотношение В/Ш: {ratio_hw:.2f} (Идеал: {ideal_hw:.2f}, Совп: {max(0.0, match_hw)*100:.1f}%)")
            
            # Nose
            nose_len = math.dist(pts[6], pts[4])
            nose_width = math.dist(pts[102], pts[329])
            ratio_nose = nose_len / nose_width if nose_width > 0 else 0
            match_nose = 1.0 - abs(ratio_nose - ideal_nose) / ideal_nose
            details.append(f"Пропорции носа L/W: {ratio_nose:.2f} (Идеал: {ideal_nose:.2f}, Совп: {max(0.0, match_nose)*100:.1f}%)")

            # Jaw Width
            jaw_w = math.dist(pts[172], pts[397])
            ratio_jaw = jaw_w / face_width if face_width > 0 else 0
            match_jaw = 1.0 - abs(ratio_jaw - ideal_jaw_width) / ideal_jaw_width
            details.append(f"Индекс челюсти: {ratio_jaw:.2f} (Идеал: {ideal_jaw_width:.2f}, Совп: {max(0.0, match_jaw)*100:.1f}%)")

            # Eye ratio
            eye_dist = math.dist(pts[133], pts[362])
            eye_w = math.dist(pts[33], pts[133])
            ratio_eye = eye_dist / eye_w if eye_w > 0 else 0
            match_eye = 1.0 - abs(ratio_eye - ideal_eye_ratio) / ideal_eye_ratio
            
            gr_matches = [max(0.0, match_hw), max(0.0, match_nose), max(0.0, match_jaw), max(0.0, match_eye)]
            avg_gr = np.mean(gr_matches) * 100.0
            
            if lenient_mode:
                avg_gr = min(100.0, avg_gr * 1.03 + 1.0)
                
            metrics["golden_ratio"] = round(avg_gr, 1)
            metrics["overall_geom"] = round((avg_symmetry * 0.5) + (avg_gr * 0.5), 1)

        elif "Semi-profile" in metrics["pose"]:
            # Semi-profile
            metrics["symmetry"] = 0.0
            nl_angle = self._calculate_angle(pts[6], pts[4], pts[2])
            ideal_angle = 93.0 if "Man" in target_group else 100.0
            match_nl = 1.0 - abs(nl_angle - ideal_angle) / ideal_angle
            details.append(f"Носогубный угол: {nl_angle:.1f}° (Идеал: {ideal_angle:.1f}°, Совп: {max(0.0, match_nl)*100:.1f}%)")
            
            avg_gr = match_nl * 100.0
            if lenient_mode:
                avg_gr = min(100.0, avg_gr * 1.05 + 2.0)
                
            metrics["golden_ratio"] = round(avg_gr, 1)
            metrics["overall_geom"] = round(avg_gr, 1)
            
        else: # Profile
            metrics["symmetry"] = 0.0
            nl_angle = self._calculate_angle(pts[6], pts[4], pts[2])
            ideal_angle = 93.0 if "Man" in target_group else 100.0
            match_nl = 1.0 - abs(nl_angle - ideal_angle) / ideal_angle
            details.append(f"Носогубный угол: {nl_angle:.1f}° (Идеал: {ideal_angle:.1f}°, Совп: {max(0.0, match_nl)*100:.1f}%)")
            
            profile_angle = self._calculate_angle(pts[10], pts[6], pts[152])
            match_profile = 1.0 - abs(profile_angle - 168.0) / 168.0
            details.append(f"Профиль лица (угол): {profile_angle:.1f}° (Идеал: 168.0°, Совп: {max(0.0, match_profile)*100:.1f}%)")
            
            avg_gr = ((match_nl + match_profile) / 2.0) * 100.0
            if lenient_mode:
                avg_gr = min(100.0, avg_gr * 1.05 + 2.0)
                
            metrics["golden_ratio"] = round(avg_gr, 1)
            metrics["overall_geom"] = round(avg_gr, 1)

        metrics["details"] = details

        # ------------------- 3. DRAW NEON HUD OVERLAY -------------------
        # All HUD drawing overlays removed by user request for a completely clean camera/photo view.
        
        return processed_frame, face_crop, metrics
