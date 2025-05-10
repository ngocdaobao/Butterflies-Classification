# train_hog_svm_pycharm.py

import os
import numpy as np
from skimage import color, io, transform
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from joblib import dump
from collections import Counter


# ───── THIẾT LẬP (hard-code) ─────
# Thay đường dẫn này cho đúng nơi bạn để folder dataset trong project
DATASET_DIR = r"D:\Github\Butterflies-Classification\dataset"

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

def load_split(split_dir):
    X, y = [], []
    classes = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])
    for label in classes:
        folder = os.path.join(split_dir, label)
        for fn in os.listdir(folder):
            if fn.lower().endswith(('.jpg','jpeg','png')):
                X.append(extract_hog(os.path.join(folder, fn)))
                y.append(label)
    if not X:
        raise RuntimeError(f"No images found in {split_dir}")
    return np.vstack(X), np.array(y), classes

def main():
    # Load data
    X_train, y_train, classes = load_split(os.path.join(DATASET_DIR, 'train'))
    X_val,   y_val,   _       = load_split(os.path.join(DATASET_DIR, 'valid'))
    X_test,  y_test,  _       = load_split(os.path.join(DATASET_DIR, 'test'))

    # Chuẩn hóa
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train SVM
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", verbose=False, decision_function_shape='ovr')
    clf.fit(X_train, y_train)

    # Đánh giá trên validation
    print("\n=== Classification report on VALIDATION ===")
    y_pred_val = clf.predict(X_val)
    print(classification_report(y_val, y_pred_val, target_names=classes))

    # Đánh giá trên test
    print("\n=== Classification report on TEST ===")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=1))

    # 5) Lưu model - khi nào cần lưu lại hãng bật lên
    # out_file = 'svm_hog_pycharm.joblib'
    # dump({'model': clf, 'scaler': scaler, 'classes': classes}, out_file)
    # print(f"\nModel saved to {out_file}")

if __name__ == "__main__":
    main()

