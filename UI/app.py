from flask import Flask, request, jsonify, render_template
import os
import sys
import numpy as np
from PIL import Image
import io
import base64
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import ButterflyClassifier  

app = Flask(__name__)
model = None

def load_model():
    global model
    model = ButterflyClassifier()  
    model.load('path/to/model/weights.h5')  

butterfly_classes = {
    0: {
        "species": "Papilio machaon",
        "commonName": "Bướm đuôi én vàng (Common Yellow Swallowtail)",
        "family": "Papilionidae",
    },
    1: {
        "species": "Danaus plexippus",
        "commonName": "Bướm chúa (Monarch Butterfly)",
        "family": "Nymphalidae",
    },
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
        img = Image.open(io.BytesIO(file.read()))
        
        img = img.resize((224, 224))  
        img_array = np.array(img) / 255.0  
        
        if model is None:
            load_model()
        
        predictions = model.predict(np.expand_dims(img_array, axis=0))
        class_id = np.argmax(predictions)
        confidence = float(predictions[0][class_id] * 100)
        
        butterfly_info = butterfly_classes.get(class_id, {
            "species": "Unknown",
            "commonName": "Không xác định",
            "family": "Không xác định",
        })
        
        result = {
            "species": butterfly_info["species"],
            "commonName": butterfly_info["commonName"],
            "confidence": confidence
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)