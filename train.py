import torch
import torch.nn 
import torch.optim as optim
import torch.nn.functional as F
from model import alexnet_model
import tqdm
from loguru import logger

def train(model, device, train_loader, valid_loader,
          criterion, optimizer, scheduler=None, num_epochs=10):
    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []

    for epoch in range(1, num_epochs+1):
        logger.info(f"EPOCH {epoch}/{num_epochs}")
        
        # ——— TRAINING ———
        model.train()
        running_loss, running_correct = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss   += loss.item() * inputs.size(0)
            running_correct+= (outputs.argmax(1) == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc  = running_correct / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        logger.info(f" TRAIN   loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

        # ——— VALIDATION ———
        model.eval()
        val_running_loss, val_running_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

                val_running_loss   += loss.item() * inputs.size(0)
                val_running_correct+= (outputs.argmax(1) == labels).sum().item()

        val_loss = val_running_loss / len(valid_loader.dataset)
        val_acc  = val_running_correct / len(valid_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        logger.info(f" VALID   loss={val_loss:.4f}  acc={val_acc:.4f}\n")

        # ——— SCHEDULER ———
        if scheduler is not None:
            scheduler.step()

    return train_losses, train_accs, val_losses, val_accs
