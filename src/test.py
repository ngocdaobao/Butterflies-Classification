import torch
import torch.nn 
from torchmetrics.classification import Accuracy, MulticlassAccuracy, MulticlassF1Score, MulticlassRecall, MulticlassPrecision 
import tqdm
from loguru import logger

def evaluate(model, device, test_loader, average='macro'):
    model.eval()
    precision = MulticlassPrecision(num_classes=100, average=average).to(device)
    recall = MulticlassRecall(num_classes=100, average=average).to(device)
    f1 = MulticlassF1Score(num_classes=100, average=average).to(device)
    accuracy = MulticlassAccuracy(num_classes=100, average=average).to(device)

    acc_list = []
    with torch.no_grad():
        for input, label in test_loader:
            input, label = input.to(device), label.to(device)
            output = model(input)
            _, pred = torch.max(output, 1)
            """
            acc = (pred == label).sum().item()
            acc_list.append(acc)
    acc_score = sum(acc_list)/len(test_loader.dataset)
    """
            precision.update(pred, label)
            recall.update(pred, label)
            f1.update(pred,label)
            accuracy.update(pred, label)
    acc_score = accuracy.compute()
    precision_score = precision.compute()
    recall_score = recall.compute()
    f1_score = f1.compute()
    logger.info(f'TEST PRECISION: {precision_score:.4f}')
    logger.info(f'TEST RECALL: {recall_score:.4f}')
    logger.info(f'TEST F1: {f1_score:.4f}')
    logger.info(f'TEST ACCURACY: {acc_score:.4f}')
    return acc_score, precision_score, recall_score, f1_score
