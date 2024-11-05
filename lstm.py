import torch
import torch.nn as nn
import torch.optim as optim
import keras
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import logging
from datetime import datetime

# Create output directories
os.makedirs('plots', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/lstm_training_{timestamp}.log'),
        logging.StreamHandler()
    ]
)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.batch_norm(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        return out

def plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses, fold, save_path='plots'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracies
    ax1.plot(train_accuracies, label='Training')
    ax1.plot(val_accuracies, label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Fold {fold + 1} Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot losses
    ax2.plot(train_losses, label='Training')
    ax2.plot(val_losses, label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Fold {fold + 1} Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/lstm_metrics_fold_{fold + 1}.png')
    plt.close()

def plot_confusion_matrix(cm, fold, save_path='plots'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Fold {fold + 1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{save_path}/lstm_confusion_matrix_fold_{fold + 1}.png')
    plt.close()

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
    return correct / total, running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
    return (correct / total, running_loss / len(val_loader), 
            all_targets, all_predictions)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--k_folds', type=int, default=5, help='number of folds')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    args = parser.parse_args()
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_indices = torch.from_numpy(y_train).long()
    
    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Model parameters
    input_size = 28
    hidden_size = 128
    num_layers = 2
    num_classes = 10
    
    # Cross-validation setup
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    cross_val_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_tensor)):
        logging.info(f'\nFold {fold + 1}/{args.k_folds}')
        
        # Data loaders
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_indices), 
                                batch_size=64, sampler=train_sampler)
        val_loader = DataLoader(TensorDataset(x_train_tensor, y_train_indices), 
                              batch_size=64, sampler=val_sampler)
        
        # Initialize model and training components
        model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                                                epochs=args.epochs,
                                                steps_per_epoch=len(train_loader))
        
        # Training tracking
        train_accuracies = []
        val_accuracies = []
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(args.epochs):
            # Train
            train_acc, train_loss = train_epoch(model, train_loader, criterion, 
                                              optimizer, device, scheduler)
            train_accuracies.append(train_acc)
            train_losses.append(train_loss)
            
            # Validate
            val_acc, val_loss, targets, predictions = validate(model, val_loader, 
                                                            criterion, device)
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)
            
            # Log progress
            logging.info(f'Epoch {epoch+1}/{args.epochs}:')
            logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'models/best_model_fold_{fold+1}.pt')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logging.info(f'Early stopping triggered after epoch {epoch+1}')
                    break
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(f'models/best_model_fold_{fold+1}.pt'))
        
        # Final validation and metrics
        val_acc, val_loss, targets, predictions = validate(model, val_loader, 
                                                         criterion, device)
        
        # Calculate metrics
        precision = precision_score(targets, predictions, average='weighted')
        recall = recall_score(targets, predictions, average='weighted')
        f1 = f1_score(targets, predictions, average='weighted')
        
        # Plot metrics
        plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses, fold)
        plot_confusion_matrix(confusion_matrix(targets, predictions), fold)
        
        # Log results
        logging.info(f'\nFold {fold + 1} Final Results:')
        logging.info(f'Accuracy: {val_acc:.4f}')
        logging.info(f'Precision: {precision:.4f}')
        logging.info(f'Recall: {recall:.4f}')
        logging.info(f'F1 Score: {f1:.4f}')
        
        cross_val_scores.append({
            'accuracy': val_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Print average results
    avg_metrics = {
        metric: np.mean([score[metric] for score in cross_val_scores])
        for metric in ['accuracy', 'precision', 'recall', 'f1']
    }
    
    logging.info('\nAverage Results across all folds:')
    for metric, value in avg_metrics.items():
        logging.info(f'{metric.capitalize()}: {value:.4f}')

if __name__ == '__main__':
    main()