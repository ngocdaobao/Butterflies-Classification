import os
import numpy as np
from skimage import color, io, transform
from skimage.feature import hog
from joblib import load

# ───── CẤU HÌNH ─────
# Đường dẫn tới model đã train (svm_hog_with_valid.joblib)
MODEL_PATH = r"D:\Github\Butterflies-Classification\src\hog_svm\svm_hog_pycharm.joblib"
# Đường dẫn tới ảnh cần phân loại (thay bằng đường dẫn của bạn)
IMAGE_PATH = r"D:\Github\Butterflies-Classification\dataset\test\AMERICAN SNOOT\5.jpg"

# Tham số HOG phải giống lúc train
IMAGE_SIZE = (128, 128)
HOG_PARAMS = {
    'orientations':    9,
    'pixels_per_cell': (16, 16),
    'cells_per_block': (2, 2),
    'block_norm':      'L2-Hys',
    'transform_sqrt':  True,
}

def extract_hog(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = transform.resize(img, IMAGE_SIZE, anti_aliasing=True)
    return hog(img, **HOG_PARAMS)


def main():
    # Load model, scaler và danh sách classes
    obj = load(MODEL_PATH)
    scaler = obj['scaler']
    clf = obj['model']
    classes = obj['classes']

    # Kiểm tra file ảnh
    if not os.path.isfile(IMAGE_PATH):
        print(f"Không tìm thấy file ảnh: {IMAGE_PATH}")
        return

    # Trích feature HOG
    feat = extract_hog(IMAGE_PATH).reshape(1, -1)
    feat = scaler.transform(feat)

    # Dự đoán
    pred = clf.predict(feat)[0]
    print(f"Ảnh {os.path.basename(IMAGE_PATH)} được phân loại là: {pred}")

if __name__ == "__main__":
    main()
