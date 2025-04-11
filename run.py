from dataset.dataloader import get_data_loader
from model import alexnet_model
from train import train
from test import evaluate
import tqdm
import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = alexnet_model().to(device)
batch_size = 32

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader, valid_loader, test_loader = get_data_loader(batch_size=batch_size)

# Train the model
train(model, device, train_loader, valid_loader, criterion, optimizer, num_epochs=10)

# Save the model
model.state_dict(torch.save(model.state_dict(), 'model.pth'))

# Evaluate the model
evaluate(model, device, test_loader)