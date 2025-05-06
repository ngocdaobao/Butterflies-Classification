from dataloader import get_data_svm, get_data_loader
from model import alexnet_model
from sklearn.svm import SVC
from torchvision.models import alexnet
from train import train
from test import evaluate
import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from loguru import logger
import argparse as arg

parser = arg.ArgumentParser(description="Train SVM on the Butterflies dataset")
parser.add_argument('--kernel', type=str, default='linear', help='Kernel type for SVM')
args = parser.parse_args()

kernel = args.kernel

#SVM with pretrained alexnet model
#extract features from the model
if kernel == 'cnn':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor_model = alexnet(pretrained=True)
    feature_extractor = extractor_model.features.to(device).eval()
    train_loader, valid_loader, test_loader = get_data_loader(batch_size=64)

    train_features = []
    train_labels = []
    for data, labels in tqdm.tqdm(train_loader):
        data = data.to(device)
        features = feature_extractor(data).view(data.size(0), -1).cpu().numpy()
        train_features.append(features)
        train_labels.append(labels.cpu().numpy())
    
    test_features = []
    test_labels = []
    for data, labels in tqdm.tqdm(test_loader):
        data = data.to(device)
        features = feature_extractor(data).view(data.size(0), -1).cpu().numpy()
        test_features.append(features)
        test_labels.append(labels.cpu().numpy())
    
    model = SVC()
    model.fit(train_features, train_labels)
    pred = model.predict(test_features)
    acc = (pred == test_labels).sum() / len(test_labels)
    logger.info(f"SVM with {kernel} kernel accuracy: {acc:.4f}")

else:
    X_train, y_train, X_test, y_test, X_valid, y_valid = get_data_svm()
    model = SVC()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = (pred == y_test).sum() / len(y_test)
    logger.info(f"SVM model {kernel} kernel accuracy: {acc:.4f}")