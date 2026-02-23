import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os

# ======================
# CONFIGURATION
# ======================
dataset_path = r"D:\Projects\Final_Year_Project\Final Year Project Code\rare_disease"
num_classes = 8
batch_size = 16
num_epochs = 15   # keep smaller per fold
learning_rate = 1e-4
num_folds = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# TRANSFORMS
# ======================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======================
# DATASET
# ======================
full_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)
targets = np.array(full_dataset.targets)

print("Class mapping:", full_dataset.class_to_idx)
print("Total images:", len(full_dataset))

# ======================
# CLASS WEIGHTS
# ======================
class_counts = np.bincount(targets)
class_weights = 1.0 / class_counts
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# ======================
# MODEL FUNCTION
# ======================
def get_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )

    return model.to(device)

# ======================
# STRATIFIED K-FOLD
# ======================
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
    print(f"\n========== Fold {fold+1}/{num_folds} ==========")

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(
        datasets.ImageFolder(root=dataset_path, transform=val_transform),
        val_idx
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = get_model()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    # -------- TRAIN HEAD (Step-1 style inside each fold) --------
    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total

        # -------- VALIDATION --------
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")

    fold_accuracies.append(val_acc)

# ======================
# FINAL RESULTS
# ======================
fold_accuracies = np.array(fold_accuracies)
print("\n========== FINAL STRATIFIED 5-FOLD RESULTS ==========")
print("Fold Accuracies:", fold_accuracies)
print(f"Mean Accuracy: {fold_accuracies.mean():.4f}")
print(f"Std Deviation: {fold_accuracies.std():.4f}")