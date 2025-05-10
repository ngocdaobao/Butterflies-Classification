# train_svm_cascade_with_report.py

import os
import numpy as np
from skimage import color, io, transform
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from joblib import dump
from tqdm import tqdm
import argparse

# Cấu hình
IMAGE_SIZE  = (128, 128)
HOG_PARAMS  = {
    'orientations':    9,
    'pixels_per_cell': (16, 16),
    'cells_per_block': (2, 2),
    'block_norm':      'L2-Hys',
    'transform_sqrt':  True,
}
SVM_KERNEL  = 'rbf'
SVM_C       = 1.0
SVM_GAMMA   = 'scale'
CHUNK_SIZE  = 1000    # ảnh/chunk để gom SV

def extract_hog(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = transform.resize(img, IMAGE_SIZE, anti_aliasing=True)
    return hog(img, **HOG_PARAMS)

def get_file_label_pairs(split_dir):
    pairs = []
    for cls in sorted(os.listdir(split_dir)):
        d = os.path.join(split_dir, cls)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith(('.jpg','jpeg','png')):
                pairs.append((os.path.join(d, fn), cls))
    return pairs

def train_on_chunk(pairs, scaler):
    feats, labels = [], []
    for path, label in pairs:
        feats.append(extract_hog(path))
        labels.append(label)
    X = np.vstack(feats)
    X = scaler.transform(X)
    y = np.array(labels)
    clf = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA)
    clf.fit(X, y)
    return clf.support_vectors_, y[clf.support_]

def cascade_train(dataset_dir):
    train_pairs = get_file_label_pairs(os.path.join(dataset_dir, 'train'))
    classes     = sorted({lbl for _, lbl in train_pairs})

    # Fit scaler online
    scaler = StandardScaler()
    for i in range(0, len(train_pairs), CHUNK_SIZE):
        chunk = train_pairs[i:i+CHUNK_SIZE]
        Xc = np.vstack([extract_hog(p) for p,_ in chunk])
        scaler.partial_fit(Xc)

    # Train chunks + gom support vectors
    all_sv_feats, all_sv_labels = [], []
    for i in range(0, len(train_pairs), CHUNK_SIZE):
        chunk = train_pairs[i:i+CHUNK_SIZE]
        svf, svl = train_on_chunk(chunk, scaler)
        all_sv_feats.append(svf)
        all_sv_labels.append(svl)
    all_sv_feats  = np.vstack(all_sv_feats)
    all_sv_labels = np.concatenate(all_sv_labels)
    print(f"→ Collected {all_sv_feats.shape[0]} support vectors")

    # Final SVM trên support vectors
    final_clf = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA)
    final_clf.fit(all_sv_feats, all_sv_labels)
    return final_clf, scaler, classes

def evaluate_split(clf, scaler, classes, dataset_dir, split):
    pairs = get_file_label_pairs(os.path.join(dataset_dir, split))
    feats, labels = [], []
    for path, label in pairs:
        feats.append(extract_hog(path))
        labels.append(label)
    X = scaler.transform(np.vstack(feats))
    y = np.array(labels)
    y_pred = clf.predict(X)
    print(f"\n=== Classification report on {split.upper()} ===")
    print(classification_report(y, y_pred, target_names=classes))

def main():
    dataset_dir = "/dataset"

    print("→ Running cascade training …")
    clf, scaler, classes = cascade_train(dataset_dir)

    # Nếu không debug thì xóa train và valid của vòng for đi vì vô nghĩa
    for split in ('train', 'valid', 'test'):
        evaluate_split(clf, scaler, classes, dataset_dir, split)

    # Save model
    dump({'model': clf, 'scaler': scaler, 'classes': classes},
         'svm_hog_cascade_with_report.joblib')
    print("\n→ Model saved to svm_hog_cascade_with_report.joblib")

if __name__ == "__main__":
    main()
