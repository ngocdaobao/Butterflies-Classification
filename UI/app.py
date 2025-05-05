from flask import Flask, request, jsonify, render_template
import os
import sys
import numpy as np
from PIL import Image
import io
import base64
import torch
from torchvision import transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import alexnet_model

app = Flask(__name__)
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global model
    model = alexnet_model(pretrained=False, num_classes=len(butterfly_classes), full_finetune=False)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()

butterfly_classes = {
    0: {"commonName": "Common Yellow Swallowtail"},
    1: {"commonName": "Monarch Butterfly"},
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/classify', methods=['POST'])
def classify_butterfly():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')  # Chuyển ảnh sang RGB nếu không phải

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(img).unsqueeze(0).to(device)  

        if model is None:
            load_model()

        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            class_id = torch.argmax(probs).item()
            confidence = probs[class_id].item()

        butterfly_info = butterfly_classes.get(class_id, {"common-name": "Không xác định"})

        result = {"common-name": butterfly_info["common-name"],}

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)