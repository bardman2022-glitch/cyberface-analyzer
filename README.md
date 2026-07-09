# CyberFace Analyzer 🤖✨

**CyberFace Analyzer** is a hybrid desktop application that combines state-of-the-art **deep learning visual aesthetics assessment** with **geometric facial proportion analysis**. It uses a dual-engine architecture to evaluate facial symmetry, golden ratio alignment, and overall aesthetic appeal in real-time or from uploaded photos.

Designed for local execution, it automatically downloads and optimizes its models on launch without sending any data to the cloud.

---

## Key Features 🚀

- **Hybrid Analysis Engine**: Combines **neural aesthetics estimation** (CLIP + MLP head) with **facial landmark geometry** (symmetry and proportion analysis).
- **Clean Camera / Photo Feed**: No distracting bounding boxes or line grids overlaid on your face — see yourself exactly as you look, while all math runs seamlessly in the background.
- **Percentile Scoring (TOP %)**: Calculates your exact looksmaxxing percentile ranking relative to the general population using a normal distribution bell curve (CDF).
- **Looksmaxxing Tiers Display**: Classifies ratings into standard classification tiers with dynamic color highlights:
  - `0.1 - 2.9`: **Sub-3** (Red)
  - `3.0 - 3.9`: **Sub** (Light Red)
  - `4.0 - 4.9`: **LTN** (Low Tier Normal - Orange)
  - `5.0 - 5.9`: **MTN** (Mid Tier Normal - Cyan)
  - `6.0 - 6.9`: **HTN** (High Tier Normal - Neon Green)
  - `7.0 - 7.9`: **Chadlite** (Purple)
  - `8.0+`: **Chad** (Neon Magenta)
- **Deep Multi-Augment Inference (TTA)**: Test-Time Augmentation (6 configurations: mirror flips, scaling, lighting adjustments) for maximum stability and accuracy.
- **Multiple Demographic Settings**: Includes youth and adult adjustments, as well as a **Strict Accuracy** mode for raw, uncompromised calibration.
- **Cyberpunk User Interface**: Beautiful dark-mode cyberpunk UI styled in English using CustomTkinter.

---

## How It Works 🛠️

### 1. Neural Aesthetics (CLIP + MLP)
The predictor uses the **CLIP-ViT-L/14** vision transformer from OpenAI to extract a high-dimensional (768d) feature embedding from the cropped face. This normalized embedding is fed into a 4-layer custom Multi-Layer Perceptron (MLP) trained on photographic aesthetics. Raw scores are calibrated using a sigmoid mapping to represent realistic population distribution scores.

### 2. Facial Geometry (MediaPipe FaceMesh)
The analyzer tracks 468 landmark points using **MediaPipe Face Landmarker**:
- **Symmetry**: Computes horizontal distance symmetry across the vertical central axis for 8 key facial features.
- **Golden Ratio (1.618)**: Evaluates face width-to-height ratio, nose proportions, eye spacing, and jaw metrics.
- **Profile Angles**: Measures nasolabial and facial projection angles for side profiles.

---

## Installation & Setup 📦

### Prerequisites
- Python 3.8 to 3.12 (Python 3.10+ recommended)
- A webcam (optional, for real-time scanner tab)

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/bardman2022-glitch/cyberface-analyzer.git
   cd cyberface-analyzer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you have an NVIDIA GPU, you may want to install PyTorch with CUDA support for faster processing, though CPU-only PyTorch works perfectly out of the box.*

3. **Launch the application**:
   ```bash
   python main.py
   ```

On first startup, the application will automatically download:
- The MediaPipe `face_landmarker.task` model (~3.7MB) into `models/`.
- Christoph Schuhmann's trained aesthetic MLP weights `sac_logos_ava1_l14_linearMSE.pth` (~3.7MB) into `models/`.
- Hugging Face will cache the `openai/clip-vit-large-patch14` vision encoder model.

---

## License 📄
This project is open-source and available under the MIT License.
