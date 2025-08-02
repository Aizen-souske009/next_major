#!/usr/bin/env python3
"""
Deepfake Detection Model Training Script using PyTorch
Trains a CNN model to detect AI-generated vs Real images
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from datetime import datetime
from PIL import Image
import random
from tqdm import tqdm
import glob

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class DeepfakeDataset(Dataset):
    """Custom dataset for deepfake detection"""
    
    def __init__(self, data_dir, transform=None, split='train', val_split=0.2):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load AI images (label 0)
        ai_dir = os.path.join(data_dir, 'AI')
        if os.path.exists(ai_dir):
            ai_images = glob.glob(os.path.join(ai_dir, '*.jpg')) + glob.glob(os.path.join(ai_dir, '*.png'))
            self.images.extend(ai_images)
            self.labels.extend([0] * len(ai_images))
        
        # Load Real images (label 1)
        real_dir = os.path.join(data_dir, 'Real')
        if os.path.exists(real_dir):
            real_images = glob.glob(os.path.join(real_dir, '*.jpg')) + glob.glob(os.path.join(real_dir, '*.png'))
            self.images.extend(real_images)
            self.labels.extend([1] * len(real_images))
        
        # Shuffle data
        combined = list(zip(self.images, self.labels))
        random.shuffle(combined)
        self.images, self.labels = zip(*combined)
        
        # Split data
        split_idx = int(len(self.images) * (1 - val_split))
        if split == 'train':
            self.images = self.images[:split_idx]
            self.labels = self.labels[:split_idx]
        else:  # validation
            self.images = self.images[split_idx:]
            self.labels = self.labels[split_idx:]
        
        print(f"{split.capitalize()} dataset: {len(self.images)} images")
        print(f"AI images: {sum(1 for l in self.labels if l == 0)}")
        print(f"Real images: {sum(1 for l in self.labels if l == 1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224), (0, 0, 0))), label
            return Image.new('RGB', (224, 224), (0, 0, 0)), label

class ImprovedCNN(nn.Module):
    """Improved CNN architecture for deepfake detection"""
    
    def __init__(self, num_classes=1):
        super(ImprovedCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResNetDeepfake(nn.Module):
    """ResNet-based model for deepfake detection"""
    
    def __init__(self, num_classes=1, pretrained=True):
        super(ResNetDeepfake, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class EfficientNetDeepfake(nn.Module):
    """EfficientNet-based model for deepfake detection"""
    
    def __init__(self, num_classes=1, pretrained=True):
        super(EfficientNetDeepfake, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        try:
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        except ImportError:
            print("EfficientNet not available, falling back to ResNet")
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)

class DeepfakeTrainer:
    """Main trainer class for deepfake detection"""
    
    def __init__(self, model, device, save_dir='training_output'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Create save directories
        os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device).float()
            
            optimizer.zero_grad()
            output = self.model(data).squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(output) > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device).float()
                
                output = self.model(data).squeeze()
                loss = criterion(output, target)
                
                running_loss += loss.item()
                predicted = (torch.sigmoid(output) > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Store for detailed metrics
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc, all_preds, all_targets
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
        """Main training loop"""
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7
        )
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc, val_preds, val_targets = self.validate_epoch(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, os.path.join(self.save_dir, 'models', 'best_model.pth'))
                print(f'New best model saved! Val Acc: {val_acc:.2f}%')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after {patience} epochs without improvement')
                break
        
        print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%')
        return best_val_acc
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.train_accs, label='Training Accuracy', color='blue')
        ax2.plot(self.val_accs, label='Validation Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', 'training_history.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate(self, test_loader):
        """Evaluate the model"""
        print("Evaluating model...")
        
        # Load best model
        checkpoint = torch.load(os.path.join(self.save_dir, 'models', 'best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data).squeeze()
                probs = torch.sigmoid(output)
                predicted = (probs > 0.5).float()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
        auc_score = roc_auc_score(all_targets, all_probs)
        
        # Classification report
        class_names = ['AI', 'Real']
        report = classification_report(all_targets, all_preds, 
                                     target_names=class_names, output_dict=True)
        
        print("Classification Report:")
        print(classification_report(all_targets, all_preds, target_names=class_names))
        print(f"AUC Score: {auc_score:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.save_dir, 'results', 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results
        results = {
            'accuracy': float(accuracy),
            'auc_score': float(auc_score),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.save_dir, 'results', 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def get_transforms(img_size=224):
    """Get data transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def main():
    parser = argparse.ArgumentParser(description='Train deepfake detection model with PyTorch')
    parser.add_argument('--dataset_path', default='Datasets_Balanced', 
                       help='Path to balanced dataset folder')
    parser.add_argument('--model_type', choices=['custom', 'resnet', 'efficientnet'], 
                       default='resnet', help='Type of model to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, default=224, help='Image size (square)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--output_dir', default='training_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' does not exist!")
        print("Please run the balance_dataset.py script first.")
        return
    
    print("=== PyTorch Deepfake Detection Training ===")
    print(f"Dataset: {args.dataset_path}")
    print(f"Model type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {device}")
    
    # Get transforms
    train_transform, val_transform = get_transforms(args.img_size)
    
    # Create datasets
    train_dataset = DeepfakeDataset(args.dataset_path, transform=train_transform, split='train')
    val_dataset = DeepfakeDataset(args.dataset_path, transform=val_transform, split='val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    if args.model_type == 'custom':
        model = ImprovedCNN(num_classes=1)
    elif args.model_type == 'resnet':
        model = ResNetDeepfake(num_classes=1, pretrained=True)
    elif args.model_type == 'efficientnet':
        model = EfficientNetDeepfake(num_classes=1, pretrained=True)
    
    # Create trainer
    trainer = DeepfakeTrainer(model, device, args.output_dir)
    
    # Train model
    best_acc = trainer.train(train_loader, val_loader, 
                           epochs=args.epochs, lr=args.learning_rate, 
                           patience=args.patience)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    results = trainer.evaluate(val_loader)
    
    # Save final model
    torch.save(model.state_dict(), 
              os.path.join(args.output_dir, 'models', 'final_model.pth'))
    
    print("\n=== Training Complete! ===")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Final AUC score: {results['auc_score']:.4f}")
    print(f"All outputs saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()