import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
dataset_path = r"D:\Projects\Final_Year_Project\Final Year Project Code\rare_disease"
model_path = "final_model.pth"   # change if needed
num_classes = 8
batch_size = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# TRANSFORMS (NO AUGMENTATION)
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======================
# DATASET & LOADER
# ======================
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
class_names = dataset.classes

# ======================
# MODEL
# ======================
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, num_classes)
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ======================
# INFERENCE
# ======================
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ======================
# METRICS
# ======================
cm = confusion_matrix(y_true, y_pred)

accuracy = np.trace(cm) / np.sum(cm)
error_rate = 1 - accuracy

print("\nAccuracy:", accuracy)
print("Error Rate:", error_rate)

# ======================
# PER-CLASS METRICS
# ======================
TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP
TN = np.sum(cm) - (TP + FP + FN)

precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)          # Sensitivity
specificity = TN / (TN + FP + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

print("\nClass-wise Metrics:")
for i, cls in enumerate(class_names):
    print(f"{cls}:")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall (Sensitivity): {recall[i]:.4f}")
    print(f"  Specificity: {specificity[i]:.4f}")
    print(f"  F1-score: {f1[i]:.4f}")

# ======================
# CONFUSION MATRIX PLOT
# ======================
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ======================
# CLASSIFICATION REPORT
# ======================
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))