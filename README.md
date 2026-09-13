# DeepFake Image Detection System (Fine-Tuned ResNet-50)

A deep learning web application that detects whether an uploaded face image is **real** or **AI-generated (deepfake)**, powered by a **ResNet-50** model fine-tuned on ~252,000 face images.

---

## Project Structure

```
minor_project/
|
|-- README.md
|-- train_resnet.py                  # Training script (local version)
|
|-- dataset-combined/
|   |-- fake/                        # 121,860 AI-generated / deepfake images
|   |-- real/                        # 127,438 authentic real images
|   `-- train_resnet.py              # Colab-adapted training script
|
|-- saved_model/
|   |-- resnet50_finetuned_custom.pth  # Trained model weights (~94 MB)
|   `-- finetuning_curves_custom.png   # Loss, Accuracy & LR plots
|
`-- webapp/
    |-- app.py                       # Flask backend (prediction API)
    `-- templates/
        `-- index.html               # Frontend UI (drag-and-drop, animated results)
```

---

## Dataset

| Category        | Count     |
|-----------------|-----------|
| Fake images     | 121,860   |
| Real images     | 127,438   |
| **Total**       | **249,298** |

**Split:** 75% train / 15% validation / 10% test

---

## Model Architecture — ResNet-50

| Property         | Value                                  |
|------------------|----------------------------------------|
| Architecture     | ResNet-50                              |
| Training mode    | Fine-tuning (transfer learning from pretrained weights) |
| Weight init      | Pretrained ImageNet weights            |
| Task             | Binary Classification (Real vs Fake)   |
| Input size       | 224 x 224 px                           |
| Output classes   | `FAKE` (index 0), `REAL` (index 1)     |
| Total parameters | ~23.5 million                          |
| Loss Function    | Cross Entropy Loss                     |
| Optimizer        | SGD + Nesterov momentum                |
| LR Scheduler     | Cosine Annealing (LR: 0.01 -> 1e-6)   |
| Test Accuracy    | 90.67% (after fine-tuning, saved at epoch 10) |

### Skip Connection (Core ResNet Idea)

```
Input (x)
  |
  |----------------------|  (identity skip connection)
  v                      |
[Conv -> BN -> ReLU]     |
  v                      |
[Conv -> BN]             |
  v                      |
  (+) <------------------+
  v
[ReLU]
  v
Output H(x) = F(x) + x
```

---

## Training Configuration

| Hyperparameter | Value  |
|----------------|--------|
| Image Size     | 224    |
| Batch Size     | 64     |
| Epochs         | 20     |
| Learning Rate  | 0.01   |
| Momentum       | 0.9    |
| Weight Decay   | 1e-4   |
| Validation     | 15%    |
| Test Split     | 10%    |
| Random Seed    | 42     |

### Data Augmentation (Training only)

- Resize to 256x256, then RandomCrop to 224x224
- Random Horizontal Flip
- Color Jitter (brightness, contrast, saturation)
- ImageNet Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

---

## Getting Started

### Prerequisites

```bash
pip install flask torch torchvision pillow matplotlib numpy
```

### 1. Train the Model (Optional)

A pre-trained model is already included in `saved_model/`.

**For Google Colab (recommended — free GPU):**

```python
# Cell 1 — Install
!pip install torch torchvision matplotlib numpy pillow -q

# Cell 2 — Mount Drive + Extract Dataset
from google.colab import drive
import zipfile, os
drive.mount('/content/drive')
if not os.path.exists('/content/dataset-combined'):
    with zipfile.ZipFile('/content/drive/MyDrive/dataset-combined.zip', 'r') as z:
        z.extractall('/content/')

# Cell 3 — Run Training
# (paste dataset-combined/train_resnet.py contents here)
# Make sure: BASE_DIR = "/content/dataset-combined"
```

**For local training (slow without GPU):**
```bash
cd dataset-combined
python train_resnet.py
```

### 2. Run the Web App

```bash
cd webapp
python app.py
```

Open in browser: **http://localhost:5000**

---

## Web Application

The Flask web app provides a premium dark-themed UI with:

- **Drag & Drop** image upload (JPG, PNG, WEBP — max 10 MB)
- **Real-time prediction** via ResNet-50 inference
- **Animated probability bars** for FAKE % and REAL %
- **Confidence score** and verdict explanation
- **Automatic GPU/CPU** detection

### API Reference

| Method | Endpoint   | Description             |
|--------|------------|-------------------------|
| GET    | `/`        | Serve the web UI        |
| POST   | `/predict` | Classify uploaded image |

**POST `/predict` — Request:**
```
Content-Type: multipart/form-data
Body: image (file field)
```

**POST `/predict` — Response:**
```json
{
  "label":      "FAKE",
  "confidence": 97.43,
  "fake_prob":  97.43,
  "real_prob":  2.57
}
```

**Error Response:**
```json
{
  "error": "No image uploaded"
}
```

---

## Training Results

| Metric             | Value  |
|--------------------|--------|
| Test Accuracy      | 90.67% (after fine-tuning) |
| Best Epoch         | 10     |
| Training Platform  | Google Colab (Tesla T4 GPU, 15.6 GB VRAM) |
| Training Time      | ~5-8 hours (20 epochs, batch=64)           |

Training curves (loss, accuracy, learning rate) are saved at:
```
saved_model/finetuning_curves_custom.png
```

---

## Tech Stack

| Layer        | Technology                        |
|--------------|-----------------------------------|
| Model        | PyTorch + torchvision (ResNet-50) |
| Web Backend  | Flask 3.x (Python)                |
| Frontend     | HTML5, CSS3, Vanilla JavaScript   |
| Image I/O    | Pillow (PIL)                      |
| Training     | Google Colab T4 GPU               |
| Data         | ~249K face images (fake + real)   |

---

## Notes

- The model was **fine-tuned** from ImageNet pretrained weights and achieved 90.67% test accuracy.
- Inference runs on **GPU (CUDA)** if available, otherwise falls back to **CPU**.
- Model file: `saved_model/resnet50_finetuned_custom.pth` (~94 MB)
- The model performs best on face images similar to the training set.
- For production use, retrain with more epochs or use pretrained weights (transfer learning).

---
