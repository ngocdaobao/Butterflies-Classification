import torch
import torch.nn 
import tqdm
from loguru import logger

def evaluate(model, device, test_loader):
    model.eval()
    acc_list = []
    with torch.no_grad():
        for input, label in test_loader:
            input, label = input.to(device), label.to(device)
            output = model(input)
            _, pred = torch.max(output, 1)
            acc = (pred == label).sum().item()
            acc_list.append(acc)
    acc_score = sum(acc_list)/len(test_loader.dataset)
    logger.info(f'TEST ACCURACY: {acc_score:.4f}')
    return acc_score
