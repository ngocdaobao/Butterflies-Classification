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
import numpy as np

parser = arg.ArgumentParser(description="Train SVM on the Butterflies dataset")
parser.add_argument('--kernel', type=str, default='linear', help='Kernel type for SVM')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

kernel = args.kernel

#SVM with pretrained alexnet model
#extract features from the model
if kernel == 'cnn':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor_model = alexnet(pretrained=True)
    for param in extractor_model.parameters():
        param.requires_grad = False
    feature_extractor = extractor_model.features.to(device).eval()
    train_loader, valid_loader, test_loader = get_data_loader(batch_size=64)
    torch.manual_seed(args.seed)  # Set the random seed for reproducibility
    train_features = []
    train_labels = []
    for data, labels in tqdm.tqdm(train_loader):
        data = data.to(device)
        features = feature_extractor(data).view(data.size(0), -1).detach().cpu().numpy()
        train_features.append(features)
        train_labels.append(labels.cpu().numpy())
    
    test_features = []
    test_labels = []
    for data, labels in tqdm.tqdm(test_loader):
        data = data.to(device)
        features = feature_extractor(data).view(data.size(0), -1).detach().cpu().numpy()
        test_features.append(features)
        test_labels.append(labels.cpu().numpy())
    
    X_train = np.vstack(train_features)    
    y_train = np.hstack(train_labels)

    X_test = np.vstack(test_features)
    y_test = np.hstack(test_labels)
    
    model = SVC()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = (pred == y_test).sum() / len(y_test)
    logger.info(f"SVM with {kernel} kernel accuracy: {acc:.4f}")

else:
    X_train, y_train, X_test, y_test, X_valid, y_valid = get_data_svm()
    model = SVC()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = (pred == y_test).sum() / len(y_test)
    logger.info(f"SVM model {kernel} kernel accuracy: {acc:.4f}")