import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import joblib
import os
from PIL import Image
from flask import Flask, render_template, request, jsonify

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, 'models')
TRAIN_DIR = os.path.join(BASE_DIR, 'Dataset_new', 'train')  # check if this is correct

MODEL_PATHS = {
    'VGG19': os.path.join(MODELS_DIR, 'vgg19_tomato.pth'),
    'DenseNet121': os.path.join(MODELS_DIR, 'best_densenet121.pth'),
    'ResNet50': os.path.join(MODELS_DIR, 'best_resnet50.pth'),
}

META_LEARNER_PATH = os.path.join(MODELS_DIR, 'stacking_meta_learner.joblib')
CLASS_FILE = os.path.join(MODELS_DIR, 'classes.txt')  # <-- your saved label file

NUM_CLASSES = 10
IMAGE_SIZE = 256
DEVICE = torch.device("cpu")  # CPU for deployment

# Global variables
BASE_MODELS = {}
META_LEARNER = None
CLASS_NAMES = []
DATA_TRANSFORMS = None

# -----------------------------
# Helper Functions
# -----------------------------

def get_class_names():
    """Loads class names from a text file, fallback to dataset if not found."""
    if os.path.exists(CLASS_FILE):
        with open(CLASS_FILE, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(class_names)} class names from {CLASS_FILE}")
        return class_names
    else:
        print(f"WARNING: {CLASS_FILE} not found, attempting to load from dataset...")
        try:
            dummy_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])
            temp_dataset = datasets.ImageFolder(TRAIN_DIR, dummy_transforms)
            print(f"Loaded {len(temp_dataset.classes)} class names from dataset directory.")
            return temp_dataset.classes
        except Exception as e:
            print(f"ERROR: Could not load class names: {e}")
            return [f"Class_{i}" for i in range(NUM_CLASSES)]

def load_base_model(model_name, num_classes, path):
    """Loads model architecture and trained weights."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found: {path}")

    if model_name == 'VGG19':
        model = models.vgg19(weights=None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'DenseNet121':
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'ResNet50':
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    state_dict = torch.load(path, map_location=DEVICE)
    new_state_dict = {k.replace("model.", "").replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model

def get_model_predictions(model, image_tensor):
    """Generates softmax probabilities for a single image."""
    with torch.no_grad():
        inputs = image_tensor.to(DEVICE)
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs.cpu().numpy()

# -----------------------------
# Model Initialization
# -----------------------------
def initialize_models():
    global BASE_MODELS, META_LEARNER, CLASS_NAMES, DATA_TRANSFORMS

    print("\n--- Initializing Ensemble ---")
    print(f"BASE_DIR: {BASE_DIR}")

    # Data transforms (must match training setup)
    DATA_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load class names
    CLASS_NAMES = get_class_names()
    if CLASS_NAMES[0].startswith("Class_"):
        print("WARNING: Placeholder class names used. Check dataset or class file.")

    # Load base models
    print("\n--- Loading Base Models ---")
    for name, path in MODEL_PATHS.items():
        try:
            BASE_MODELS[name] = load_base_model(name, NUM_CLASSES, path)
            print(f"{name} loaded successfully.")
        except Exception as e:
            print(f"Failed to load {name}: {e}")

    # Load meta-learner
    print("\n--- Loading Meta-Learner ---")
    try:
        META_LEARNER = joblib.load(META_LEARNER_PATH)
        print("Meta-Learner loaded successfully.")
    except Exception as e:
        print(f"Failed to load Meta-Learner: {e}")

    print(f"\nEnsemble Initialization Complete. Classes found: {len(CLASS_NAMES)}")

import subprocess

def get_disease_remedy_ollama(disease_name):
    """
    Get remedy suggestions from a local Ollama model (Phi-3, Llama3, etc.)
    """
    try:
        prompt = f"Suggest organic and chemical remedies for {disease_name} disease in tomato plants. Keep it short, clear, and practical."
        result = subprocess.run(["ollama", "run", "phi3", prompt],
                                capture_output=True, text=True, timeout=60)
        return result.stdout.strip() if result.stdout else "No remedy generated. Please try again."
    except Exception as e:
        return f"Error while generating remedy: {e}"

# -----------------------------
# Flask Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/app')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not META_LEARNER or not BASE_MODELS or not CLASS_NAMES:
        return jsonify({'error': 'Model not properly initialized'}), 500

    try:
        img = Image.open(file.stream).convert('RGB')
        image_tensor = DATA_TRANSFORMS(img).unsqueeze(0)

        # Generate meta-features
        S_new = np.zeros((1, len(MODEL_PATHS) * NUM_CLASSES))
        for i, (name, model) in enumerate(BASE_MODELS.items()):
            preds = get_model_predictions(model, image_tensor)
            S_new[:, i * NUM_CLASSES:(i + 1) * NUM_CLASSES] = preds

        # Meta-learner final prediction
        final_idx = META_LEARNER.predict(S_new)[0]
        final_probs = META_LEARNER.predict_proba(S_new)[0]
        confidence = float(np.max(final_probs))
        if 0 <= final_idx < len(CLASS_NAMES):
            raw_label = CLASS_NAMES[final_idx]
            display_name = raw_label.split('___')[-1].replace('_', ' ').strip()
        else:
            display_name = f"Unknown (Index {final_idx})"
        
        if "healthy" in display_name.lower():
            remedies = "The leaf appears healthy. No remedies are required. Keep monitoring your plants regularly."
        else:
            remedies = get_disease_remedy_ollama(display_name)

        return jsonify({
            'success': True,
            'prediction': display_name,
            'confidence': round(confidence * 100, 2),
            'remedy': remedies,
            'filename': file.filename
        })

    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({'error': f'Internal error: {e}'}), 500

# -----------------------------
# Run Server
# -----------------------------
if __name__ == '__main__':
    initialize_models()
    app.run(debug=True)
