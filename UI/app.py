from flask import Flask, request, jsonify, render_template
import os
import sys
import numpy as np
from PIL import Image
import io
import base64

# Thêm thư mục gốc vào sys.path để import các module từ dự án
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import mô hình từ code đã có
from model import ButterflyClassifier  # Điều chỉnh tên class theo mô hình của bạn

app = Flask(__name__)

# Khởi tạo mô hình
model = None

def load_model():
    global model
    model = ButterflyClassifier()  # Điều chỉnh theo cách khởi tạo mô hình của bạn
    model.load('path/to/model/weights.h5')  # Điều chỉnh đường dẫn tới trọng số mô hình

# Danh sách tên các loài bướm (điều chỉnh theo mô hình của bạn)
butterfly_classes = {
    0: {
        "species": "Papilio machaon",
        "commonName": "Bướm đuôi én vàng (Common Yellow Swallowtail)",
        "family": "Papilionidae",
        "habitat": "Đồng cỏ, vườn, rừng thưa",
        "distribution": "Châu Âu, Châu Á, Bắc Mỹ"
    },
    1: {
        "species": "Danaus plexippus",
        "commonName": "Bướm chúa (Monarch Butterfly)",
        "family": "Nymphalidae",
        "habitat": "Đồng cỏ, vườn hoa",
        "distribution": "Châu Mỹ, Úc, New Zealand"
    },
    # Thêm các loài khác theo mô hình của bạn
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/classify', methods=['POST'])
def classify_butterfly():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Đọc file ảnh từ request
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
        
        # Tiền xử lý ảnh (điều chỉnh theo yêu cầu của mô hình)
        img = img.resize((224, 224))  # Điều chỉnh kích thước theo mô hình
        img_array = np.array(img) / 255.0  # Chuẩn hóa, điều chỉnh theo tiền xử lý của bạn
        
        # Dự đoán (điều chỉnh theo API của mô hình)
        if model is None:
            load_model()
        
        predictions = model.predict(np.expand_dims(img_array, axis=0))
        class_id = np.argmax(predictions)
        confidence = float(predictions[0][class_id] * 100)
        
        # Lấy thông tin về loài bướm
        butterfly_info = butterfly_classes.get(class_id, {
            "species": "Unknown",
            "commonName": "Không xác định",
            "family": "Không xác định",
            "habitat": "Không xác định",
            "distribution": "Không xác định"
        })
        
        # Kết quả trả về
        result = {
            "species": butterfly_info["species"],
            "commonName": butterfly_info["commonName"],
            "confidence": confidence,
            "additionalInfo": {
                "family": butterfly_info["family"],
                "habitat": butterfly_info["habitat"],
                "distribution": butterfly_info["distribution"]
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Tải mô hình khi khởi động
    load_model()
    # Chạy app
    app.run(debug=True, port=5000)