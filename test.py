import torch
import torch.nn 
import tqdm

def evaluate(model, device, test_loader):
    model.eval()
    acc_list = []
    with torch.no_grad():
        for input, label in tqdm(test_loader):
            input, label = input.to(device), label.to(device)
            output = model(input)
            _, pred = torch.max(output, 1)
            acc = (pred == label).sum().item()
            acc_list.append(acc)
    acc_score = sum(acc_list)/len(test_loader.dataset)
    print(f'Test Accuracy: {acc_score:.4f}')
    return acc_score
