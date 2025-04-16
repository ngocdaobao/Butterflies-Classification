from dataset.dataloader import get_data_loader
from model import alexnet_model
from train import train
from test import evaluate
import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from loguru import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = alexnet_model()
model.to(device)
batch_size = 64

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

train_loader, valid_loader, test_loader = get_data_loader(batch_size=batch_size)

# Train the model
logger.info("Starting training...")
train(model, device, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=10)

# Save the model
logger.info("Saving model...")
model.state_dict(torch.save(model.state_dict(), 'model.pth'))

# Evaluate the model
logger.info("Evaluating model...")
evaluate(model, device, test_loader)