# improved_hog_svm_with_valid.py
# File này làm với GRID SEARCH để tiện thử các kernel, C, Gamma

import os
import numpy as np
from skimage import color, io, transform
from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import classification_report
from joblib import dump

# path dataset
DATASET_DIR = r"D:\Github\Butterflies-Classification\dataset"

# HOG parameters
IMAGE_SIZE = (64, 64)
HOG_PARAMS = {
    'orientations':    9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm':      'L2-Hys',
    'transform_sqrt':  True,
}

# Grid search
PARAM_GRID = {
    # có thể tinh chỉnh thêm C, Gamma
    'svc__kernel':['rbf','linear']
}

def extract_hog(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = transform.resize(img, IMAGE_SIZE, anti_aliasing=True)
    return hog(img, **HOG_PARAMS)


def load_split(folder):
    X, y = [], []
    classes = sorted(d for d in os.listdir(folder)
                     if os.path.isdir(os.path.join(folder, d)))
    for label in classes:
        sub = os.path.join(folder, label)
        for fn in os.listdir(sub):
            if fn.lower().endswith(('.jpg','jpeg','png')):
                X.append(extract_hog(os.path.join(sub, fn)))
                y.append(label)
    if not X:
        raise RuntimeError(f"No images found in {folder}")
    return np.vstack(X), np.array(y), classes


def main():
    # Load train, valid, test
    X_train, y_train, classes = load_split(os.path.join(DATASET_DIR, 'train'))
    X_valid, y_valid, classes2 = load_split(os.path.join(DATASET_DIR, 'valid'))
    X_test,  y_test,  _        = load_split(os.path.join(DATASET_DIR, 'test'))

    # sanity check
    assert classes == classes2, "Train/Valid must share same class folders"

    # Build PredefinedSplit: train → fold = -1, valid → fold = 0
    X_comb = np.vstack([X_train, X_valid])
    y_comb = np.concatenate([y_train, y_valid])
    test_fold = np.concatenate([
        np.full(len(X_train), -1, dtype=int),
        np.zeros(len(X_valid), dtype=int)
    ])
    ps = PredefinedSplit(test_fold)

    # Pipeline: scaler + SVC
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc',    SVC(decision_function_shape='ovr', class_weight='balanced'))
    ])

    # Grid search over train/valid
    grid = GridSearchCV(pipe, PARAM_GRID,
                        cv=ps, verbose=2, n_jobs=-1)
    grid.fit(X_comb, y_comb)
    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    # Report on valid & test
    print("\n=== VALIDATION REPORT ===")
    yv = best.predict(X_valid)
    print(classification_report(y_valid, yv, target_names=classes))

    print("\n=== TEST REPORT ===")
    yt = best.predict(X_test)
    print(classification_report(y_test, yt, target_names=classes))

    # 6) Save pipeline
    dump(best, 'hog_svm_with_valid.joblib')
    print("\nSaved model to hog_svm_with_valid.joblib")


if __name__ == "__main__":
    main()
