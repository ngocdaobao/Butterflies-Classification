from dataloader import get_data_loader
from model import alexnet_model
from train import train
from test import evaluate
import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from loguru import logger
import argparse as arg

parser = arg.ArgumentParser(description="Train a neural model on the Butterflies dataset")
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epoch training')
parser.add_argument('--num_classes', type=int, default=100, help='Number of classes in the dataset')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--full_finetune', type=bool, default=True)  
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
parser.add_argument('--momentum', type=float, default=0.8, help='Momentum for the optimizer')
parser.add_argument('--step_size', type=int, default=4, help='Step size for the learning rate scheduler')  
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for the learning rate scheduler')

args = parser.parse_args()
batch_size = args.batch_size
num_epochs = args.num_epochs
num_classes = args.num_classes
pretrained = args.pretrained
full_finetune = args.full_finetune
learning_rate = args.learning_rate
momentum = args.momentum
step_size = args.step_size
gamma = args.gamma

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = alexnet_model(pretrained=pretrained, num_classes=num_classes, full_finetune=full_finetune).to(device)
model.to(device)
batch_size = 64

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

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