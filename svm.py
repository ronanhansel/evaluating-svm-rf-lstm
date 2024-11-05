import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import logging
from datetime import datetime

# Create output directories for plots and logs
os.makedirs('plots', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--k_folds', type=int, default=3, help='number of folds')
args = parser.parse_args()

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'logs/svm_training_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # This will print to console too
    ]
)

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

class SVMModel(nn.Module):
    def __init__(self):
        super(SVMModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28*28, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)

def plot_accuracy(accuracies, fold, save_path='plots'):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, 'b-')
    plt.title(f'Training Accuracy over Epochs - Fold {fold+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(f'{save_path}/svm_accuracy_fold_{fold+1}.png')
    plt.close()

def plot_loss(losses, fold, save_path='plots'):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(losses) + 1), losses, 'orange')
    plt.title(f'Training Loss over Epochs - Fold {fold+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'{save_path}/svm_loss_fold_{fold+1}.png')
    plt.close()

def plot_confusion_matrix(cm, fold, save_path='plots'):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Fold {fold+1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{save_path}/svm_confusion_matrix_fold_{fold+1}.png')
    plt.close()

def train_model(model, train_loader, optimizer, epoch, fold, accuracies, losses):
    model.train()
    epoch_correct = 0
    epoch_total = 0
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        # Hinge loss
        one_hot_target = torch.zeros_like(output).scatter_(1, target.unsqueeze(1), 1.0)
        margins = output - output.gather(1, target.unsqueeze(1)).expand_as(output) + 1.0
        margins[one_hot_target.bool()] = 0
        loss = margins.clamp(min=0).mean()
        
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
        
        # Calculate batch accuracy
        _, predicted = torch.max(output.data, 1)
        batch_total = target.size(0)
        batch_correct = (predicted == target).sum().item()
        
        # Update epoch totals
        epoch_total += batch_total
        epoch_correct += batch_correct
        
        # Log progress
        if batch_idx % 100 == 0:
            current_accuracy = epoch_correct / epoch_total
            avg_loss = running_loss / (batch_idx + 1)
            logging.info(f'Fold: {fold+1}, Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                         f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {avg_loss:.6f}\tAccuracy: {current_accuracy:.4f}')
    
    # Calculate epoch-level metrics
    epoch_accuracy = epoch_correct / epoch_total
    epoch_loss = running_loss / len(train_loader)
    
    # Store epoch-level metrics
    accuracies.append(epoch_accuracy)
    losses.append(epoch_loss)
    
    return epoch_accuracy

# K-fold cross validation
kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
cross_val_scores = []

for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
    logging.info(f'\nFold {fold + 1}')
    
    train_sampler = SubsetRandomSampler(train_ids)
    val_sampler = SubsetRandomSampler(val_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=64, sampler=val_sampler)
    
    model = SVMModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Training with accuracy tracking
    accuracies = []
    losses = []
    for epoch in range(1, args.epochs + 1):
        train_accuracy = train_model(model, train_loader, optimizer, epoch, fold, accuracies, losses)
    
    # Plot training accuracy
    plot_accuracy(accuracies, fold)

    # Plot training loss
    plot_loss(losses, fold)
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate and plot confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plot_confusion_matrix(cm, fold)
    
    # Calculate metrics
    accuracy = correct / total
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    logging.info(f'\nFold {fold + 1} Results:')
    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info(f'Precision: {precision:.4f}')
    logging.info(f'Recall: {recall:.4f}')
    logging.info(f'F1 Score: {f1:.4f}')
    
    cross_val_scores.append({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# Print average results
avg_accuracy = np.mean([score['accuracy'] for score in cross_val_scores])
avg_precision = np.mean([score['precision'] for score in cross_val_scores])
avg_recall = np.mean([score['recall'] for score in cross_val_scores])
avg_f1 = np.mean([score['f1'] for score in cross_val_scores])

logging.info('\nAverage Results across all folds:')
logging.info(f'Accuracy: {avg_accuracy:.4f}')
logging.info(f'Precision: {avg_precision:.4f}')
logging.info(f'Recall: {avg_recall:.4f}')
logging.info(f'F1 Score: {avg_f1:.4f}')