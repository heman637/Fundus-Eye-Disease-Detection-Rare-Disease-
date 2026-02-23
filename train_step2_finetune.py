# =========================================
# STEP-2: Fine-Tuning Top Backbone Layers
# =========================================

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # CUDA crash debugging (IMPORTANT)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np

# -------------------------------
# 1. DEVICE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 2. DATASET PATH
# -------------------------------
dataset_path = r"D:\Projects\Final_Year_Project\Final Year Project Code\rare_disease"

# -------------------------------
# 3. TRANSFORMS
# -------------------------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# 4. DATASET
# -------------------------------
dataset = ImageFolder(root=dataset_path, transform=train_transforms)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

val_dataset.dataset.transform = val_transforms

# -------------------------------
# 5. CLASS COUNTS (ORDER MUST MATCH FOLDERS)
# -------------------------------
class_counts = np.array([
    140,  # BRVO
    114,  # Blur fundus with suspected PDR
    45,   # Blur fundus without PDR
    45,   # CRVO
    229,  # DN
    41,   # MH
    43,   # MYA
    118   # ODE
])

weights = class_counts.sum() / (len(class_counts) * class_counts)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

# -------------------------------
# 6. BALANCED SAMPLER
# -------------------------------
train_labels = [dataset.targets[i] for i in train_dataset.indices]
sample_weights = [weights[label] for label in train_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    sampler=sampler,
    num_workers=0,      # WINDOWS SAFE
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,      # WINDOWS SAFE
    pin_memory=True
)

# -------------------------------
# 7. LOAD MODEL FROM STEP-1
# -------------------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 8)
)

model.load_state_dict(torch.load("step1_classifier_head.pth", map_location=device))
model = model.to(device)

# -------------------------------
# 8. UNFREEZE TOP LAYERS
# -------------------------------
for param in model.parameters():
    param.requires_grad = False

for layer in [model.layer3, model.layer4, model.fc]:
    for param in layer.parameters():
        param.requires_grad = True

# -------------------------------
# 9. LOSS & OPTIMIZER
# -------------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

# -------------------------------
# 10. TRAINING LOOP
# -------------------------------
epochs = 60
best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    train_correct, train_total = 0, 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # NaN protection
        if torch.isnan(loss):
            print("⚠️ NaN loss detected. Skipping batch.")
            continue

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    # ---------------- Validation ----------------
    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "step2_best_finetuned.pth")

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Acc: {train_acc:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

# -------------------------------
# 11. SAVE FINAL MODEL
# -------------------------------
torch.save(model.state_dict(), "final_model.pth")
print("✅ STEP-2 completed successfully.")
print("✅ Final model saved as final_model.pth")