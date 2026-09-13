from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io, os

app = Flask(__name__)

BASE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_model", "resnet50_finetuned_custom.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):
    checkpoint  = torch.load(path, map_location=device, weights_only=True)
    class_names = [c.upper() for c in checkpoint["class_names"]]
    img_size    = checkpoint.get("img_size", 224)
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(class_names))
    m.load_state_dict(checkpoint["model_state_dict"])
    m.eval()
    return m.to(device), class_names, img_size

print("Loading Model...")
try:
    model, CLASS_NAMES, _ = load_model(BASE_MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        def infer(m, cnames):
            if m is None:
                return {"error": "Model not found on disk"}
            with torch.no_grad():
                probs = torch.softmax(m(tensor), dim=1)[0]
                pred_idx = int(torch.argmax(probs).item())
            return {
                "fake_prob": round(float(probs[0]) * 100, 2),
                "real_prob": round(float(probs[1]) * 100, 2),
                "label": cnames[pred_idx],
                "confidence": round(float(probs[pred_idx]) * 100, 2)
            }

        return jsonify(infer(model, CLASS_NAMES))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
