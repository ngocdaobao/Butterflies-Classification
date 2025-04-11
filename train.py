import torch
import torch.nn 
import torch.optim as optim
import torch.nn.functional as F
from model import alexnet_model
import tqdm

def train(model, device, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    loss_list = []
    acc_list = []
    loss_val_list = []
    acc_val_list = []
    print('Training started...')
    for epoch in tqdm.tqdm(range(num_epochs)):
        correct = 0
        loss_epoch = []
        model.train()
        for i, (input, label) in tqdm.tqdm(enumerate(train_loader)):
            input, label = input.to(device), label.to(device)

            #forward pass
            output = model(input)
            loss = criterion(output, label)
            loss_epoch.append(loss.item())

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #compute accuracy
            _, pred = torch.max(output, 1)
            correct += (pred == label).sum().item()
        
        loss_list.append(sum(loss_epoch)/len(loss_epoch))
        acc = correct/len(train_loader.dataset)
        acc_list.append(acc)
        print(f'Epoch {epoch+1}/{num_epochs} Loss: {loss_list[-1]:.4f}, Accuracy: {acc:.4f}')

        #validation
        print('Validating...')
        model.eval()
        correct = 0
        lost_epoch_val = []
        with torch.no_grad():
            for input, label in tqdm.tqdm(valid_loader):
                input, label = input.to(device), label.to(device)
                output = model(input)
                loss = criterion(output, label)
                lost_epoch_val.append(loss.item())
                _, pred = torch.max(output, 1)
                correct += (pred == label).sum().item()
        acc_val = correct/len(valid_loader.dataset)
        loss_val = sum(lost_epoch_val)/len(lost_epoch_val)
        print(f'Validation Loss: {loss_val:.4f}, Validation Accuracy: {acc_val:.4f}')
    return loss_list, acc_list, loss_val_list, acc_val_list