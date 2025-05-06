import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from torch.utils.data import DataLoader
import os

# Load the dataset

train_set = ImageFolder(root=os.path.join('dataset/train'), 
                        transform=Compose([Resize((224, 224)), 
                                           ToTensor(), 
                                           Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
valid_set = ImageFolder(root=os.path.join('dataset/valid'), 
                        transform=Compose([Resize((224, 224)), 
                                           ToTensor(), 
                                           Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
test_set = ImageFolder(root=os.path.join('dataset/test'),
                        transform=Compose([Resize((224, 224)), 
                                           ToTensor(), 
                                           Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

# Create the dataloaders
def get_data_loader(batch_size=64):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader

def get_data_svm():
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    X_valid = []
    y_valid = []

    # Load the training data
    for data, label in train_set:
        data = data[0].unsqueeze(0).view(-1).numpy()
        X_train.append(data)
        y_train.append(label)
    
    # Load the test data
    for data, label in test_set:
        data = data[0].unsqueeze(0).view(-1).numpy()
        X_test.append(data)
        y_test.append(label)
    
    # Load the validation data
    for data, label in valid_set:
        data = data[0].unsqueeze(0).view(-1).numpy()
        X_valid.append(data)
        y_valid.append(label)
        
    return X_train, y_train, X_test, y_test, X_valid, y_valid




