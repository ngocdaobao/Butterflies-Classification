from dataset.dataloader import get_data_svm
from model import alexnet_model, SVM
from train import train
from test import evaluate
import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from loguru import logger

X_train, y_train, X_test, y_test, X_valid, y_valid = get_data_svm()

model = SVM()
acc = model.predict(X_train, y_train, X_test, y_test)
logger.info(f"SVM model accuracy: {acc:.4f}")
