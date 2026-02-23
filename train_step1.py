# ===============================
# STEP-1: Train Classifier Head
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np

# -------------------------------
# 1. DEVICE SETUP
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 2. DATASET PATH (EDIT ONLY THIS)
# -------------------------------
dataset_path = r"D:\Projects\Final_Year_Project\Final Year Project Code\rare_disease"

# -------------------------------
# 3. TRANSFORMS
# -------------------------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# 4. LOAD DATASET
# -------------------------------
dataset = ImageFolder(root=dataset_path, transform=train_transforms)

print("Class to index mapping:")
print(dataset.class_to_idx)
print("Total images:", len(dataset))

# -------------------------------
# 5. TRAIN / VALIDATION SPLIT
# -------------------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Apply validation transforms
val_dataset.dataset.transform = val_transforms

# -------------------------------
# 6. CLASS WEIGHTS (IMBALANCE)
# -------------------------------
# Order must match dataset.class_to_idx
class_counts = np.array([229, 118, 140, 45, 114, 45, 43, 41])
weights = class_counts.sum() / (len(class_counts) * class_counts)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

# -------------------------------
# 7. BALANCED SAMPLER
# -------------------------------
train_labels = [dataset.targets[i] for i in train_dataset.indices]
sample_weights = [weights[label] for label in train_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# -------------------------------
# 8. DATALOADERS
# -------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# -------------------------------
# 9. MODEL (RESNET50)
# -------------------------------
model = models.resnet50(pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Custom classifier head
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 8)   # 8 classes
)

model = model.to(device)

# -------------------------------
# 10. LOSS & OPTIMIZER
# -------------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# -------------------------------
# 11. TRAINING LOOP (STEP-1)
# -------------------------------
num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {running_loss:.4f} "
          f"Train Acc: {train_acc:.4f} "
          f"Val Acc: {val_acc:.4f}")

# -------------------------------
# 12. SAVE MODEL
# -------------------------------
torch.save(model.state_dict(), "step1_classifier_head.pth")
print("STEP-1 training completed. Model saved as step1_classifier_head.pth")