"""
Jewelry Image Classification Model
This module contains the main classifier for jewelry images.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import timm


class JewelryClassifier:
    """
    A classifier for jewelry images using transfer learning with PyTorch.
    """
    
    def __init__(self, img_size=(224, 224), num_classes=None, model_type='efficientnet'):
        """
        Initialize the jewelry classifier.
        
        Args:
            img_size: Tuple of (height, width) for input images
            num_classes: Number of jewelry categories
            model_type: Base model to use ('efficientnet', 'resnet50', 'mobilenet', 'vit')
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        print(f"Using device: {self.device}")
        
    def build_model(self, num_classes):
        """
        Build the classification model using transfer learning.
        
        Args:
            num_classes: Number of output classes
        """
        self.num_classes = num_classes
        
        # Select base model
        if self.model_type == 'efficientnet':
            # Using timm library for better EfficientNet implementation
            base_model = timm.create_model('efficientnet_b0', pretrained=True)
            num_features = base_model.classifier.in_features
            base_model.classifier = nn.Identity()
        elif self.model_type == 'resnet50':
            base_model = models.resnet50(pretrained=True)
            num_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
        elif self.model_type == 'mobilenet':
            base_model = models.mobilenet_v2(pretrained=True)
            num_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
        elif self.model_type == 'vit':
            # Vision Transformer
            base_model = timm.create_model('vit_base_patch16_224', pretrained=True)
            num_features = base_model.head.in_features
            base_model.head = nn.Identity()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Freeze base model initially
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Build classification head
        self.model = nn.Sequential(
            base_model,
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer and criterion
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        print(f"Model built with {sum(p.numel() for p in self.model.parameters())} parameters")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
        return self.model
    
    def unfreeze_base_model(self, unfreeze_from_layer=None):
        """
        Unfreeze the base model for fine-tuning.
        
        Args:
            unfreeze_from_layer: Layer index from which to start unfreezing (None = unfreeze all)
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")
        
        # Get the base model (first module in Sequential)
        base_model = self.model[0]
        
        # Unfreeze specified layers
        all_params = list(base_model.parameters())
        if unfreeze_from_layer is None:
            # Unfreeze all
            for param in base_model.parameters():
                param.requires_grad = True
            print(f"Unfroze all layers of base model")
        else:
            # Unfreeze from specific layer
            for param in all_params[unfreeze_from_layer:]:
                param.requires_grad = True
            print(f"Unfroze layers from index {unfreeze_from_layer} onwards")
        
        # Update optimizer with lower learning rate for fine-tuning
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=20, save_best=True, model_path='best_model.pth'):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_best: Whether to save the best model
            model_path: Path to save the best model
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch results
            print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if save_best:
                    self.save_model(model_path)
                    print(f'✓ Best model saved with accuracy: {val_acc*100:.2f}%')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break
        
        print(f'\nTraining completed! Best validation accuracy: {best_val_acc*100:.2f}%')
        return self.history
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
        """
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        top3_correct = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Testing'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                # Top-1 accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Top-3 accuracy
                _, top3_pred = outputs.topk(3, 1, True, True)
                top3_correct += top3_pred.eq(labels.view(-1, 1).expand_as(top3_pred)).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = {
            'test_loss': running_loss / total,
            'accuracy': correct / total,
            'top_3_accuracy': top3_correct / total
        }
        
        return metrics, np.array(all_predictions), np.array(all_labels)
    
    def predict(self, data_loader):
        """
        Make predictions on images.
        
        Args:
            data_loader: DataLoader containing images
        """
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        self.model.eval()
        all_predictions = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(data_loader, desc='Predicting'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                all_predictions.extend(outputs.argmax(1).cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probs)
    
    def save_model(self, filepath):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'img_size': self.img_size
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a model from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Rebuild model architecture
        self.model_type = checkpoint['model_type']
        self.num_classes = checkpoint['num_classes']
        self.img_size = checkpoint['img_size']
        self.build_model(self.num_classes)
        
        # Load state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Model loaded from {filepath}")
        return self.model
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.history or len(self.history['train_loss']) == 0:
            raise ValueError("No training history available.")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Plot accuracy
        axes[0].plot(epochs, self.history['train_acc'], label='Train Accuracy', marker='o')
        axes[0].plot(epochs, self.history['val_acc'], label='Val Accuracy', marker='s')
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
        axes[1].plot(epochs, self.history['val_loss'], label='Val Loss', marker='s')
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()


def create_data_loaders(data_dir, img_size=(224, 224), batch_size=32, 
                        num_workers=4, augment=True):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Parent directory containing 'train', 'val', 'test' subdirectories
        img_size: Target image size
        batch_size: Batch size for training
        num_workers: Number of worker threads for data loading
        augment: Whether to apply data augmentation
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    from pathlib import Path
    
    # Define transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size[0] + 32, img_size[1] + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    data_path = Path(data_dir)
    
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path / 'train',
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=data_path / 'val',
        transform=val_test_transform
    )
    
    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path / 'test',
        transform=val_test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Get class names
    class_names = train_dataset.classes
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {class_names}")
    
    return train_loader, val_loader, test_loader, class_names


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
