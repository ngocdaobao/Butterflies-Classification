import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from torch.utils.data import DataLoader


# Load the dataset
train_set = ImageFolder(root='train', 
                        transform=Compose([Resize((224, 224)), 
                                           ToTensor(), 
                                           Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
valid_set = ImageFolder(root='valid', 
                        transform=Compose([Resize((224, 224)), 
                                           ToTensor(), 
                                           Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
test_set = ImageFolder(root='test',
                        transform=Compose([Resize((224, 224)), 
                                           ToTensor(), 
                                           Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

# Create the dataloaders
def get_data_loader(batch_size=64):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader

print(train_set[0][0].size())

