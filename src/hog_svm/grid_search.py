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
        'C': np.logspace(-2, 10, 13),
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': np.logspace(-9, 3, 13)
    }

    # Khởi tạo biến để lưu trữ kết quả tốt nhất
    best_param = {'C': [], 
                  'kernel': [],
                  'gamma': [],}
    best_score = 0
    # Biến lưu trữ kết quả với từng setting
    results = {}
    
    print("=== Grid Search ===")
    # Duyệt qua tất cả các tham số trong danh sách
    for C in param_grid['C']:
        for kernel in param_grid['kernel']:
            for gamma in param_grid['gamma']:
                # Khởi tạo SVM với các tham số hiện tại
                clf = SVC(kernel=kernel, C=C, gamma=gamma, verbose=False, decision_function_shape='ovr')
                
                # Huấn luyện mô hình
                clf.fit(X_train, y_train)
                
                # Dự đoán trên tập validation
                y_pred_val = clf.predict(X_val)
                
                # Tính toán độ chính xác
                score = np.mean(y_pred_val == y_val)
                results[(C, kernel, gamma)] = score
                print(f"Tham số: C={C}, kernel={kernel}, gamma={gamma} => Độ chính xác: {score:.4f}")
                
                # Nếu độ chính xác cao hơn độ chính xác tốt nhất hiện tại, lưu lại tham số và độ chính xác
                if score > best_score:
                    best_score = score
                    best_param['C'] = C
                    best_param['kernel'] = kernel
                    best_param['gamma'] = gamma
    
    # In ra tham số tốt nhất và độ chính xác tương ứng
    print(f"\nTham số tốt nhất: C={best_param['C']}, kernel={best_param['kernel']}, gamma={best_param['gamma']} => Độ chính xác: {best_score:.4f}")

if __name__ == "__main__":
    main()