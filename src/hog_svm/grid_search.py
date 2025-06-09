import os
import numpy as np
from skimage import color, io, transform
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from joblib import dump
from collections import Counter
from sklearn.model_selection import GridSearchCV

# ───── THIẾT LẬP (hard-code) ─────
# Thay đường dẫn này cho đúng nơi bạn để folder dataset trong project
DATASET_DIR = r"Butterflies-Classification\dataset"

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
    print("=== Load Data ===")
    print("Loading data...")
    X_train, y_train, classes = load_split(os.path.join(DATASET_DIR, 'train'))
    X_val,   y_val,   _       = load_split(os.path.join(DATASET_DIR, 'valid'))
    X_test,  y_test,  _       = load_split(os.path.join(DATASET_DIR, 'test'))

    # Chuẩn hóa
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    #grid search
    # Tạo một danh sách các tham số để thử nghiệm
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    }

    # Khởi tạo biến để lưu trữ kết quả tốt nhất
    best_param = {'C': [], 
                  'kernel': [],}
    best_score = 0
    print("=== Grid Search ===")
    
    for C in param_grid['C']:
        for kernel in param_grid['kernel']:
            print(f"Training SVM with C={C}, kernel={kernel}...")
            clf = SVC(C=C, kernel=kernel, gamma='scale', verbose=False, decision_function_shape='ovr')
            clf.fit(X_train, y_train)
            score = clf.score(X_val, y_val)
            print(f"Validation score for C={C}, kernel={kernel}: {score:.4f}")
            # Lưu tham số nếu score tốt hơn
            if score > best_score:
                best_score = score
                best_param['C'] = C
                best_param['kernel'] = kernel

    print(f"Best parameters: {best_param}, Best score: {best_score:.4f}")

if __name__ == "__main__":
    main()