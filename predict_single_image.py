import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# -------------------------------
# 1. DEVICE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 2. CLASS NAMES
# -------------------------------
class_names = [
    "BRVO",
    "Blur fundus with suspected PDR",
    "Blur fundus without PDR",
    "CRVO",
    "DN",
    "MH_Macular_Pathology",
    "MYA_Myopia",
    "ODE"
]

# -------------------------------
# 3. IMAGE TRANSFORM
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# 4. LOAD MODEL
# -------------------------------
model = models.resnet50(weights=None)

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 8)
)

model.load_state_dict(torch.load("final_model.pth", map_location=device))
model = model.to(device)
model.eval()

print("Model loaded successfully")

# -------------------------------
# 5. ASK USER FOR IMAGE PATH
# -------------------------------
image_path = input("\nEnter full image path: ").strip()

if not os.path.exists(image_path):
    print("❌ Error: Image path does not exist")
    exit()

# -------------------------------
# 6. LOAD & PREPROCESS IMAGE
# -------------------------------
image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0).to(device)

# -------------------------------
# 7. PREDICTION
# -------------------------------
with torch.no_grad():
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, 1)

predicted_class = class_names[predicted.item()]
confidence_score = confidence.item() * 100

# -------------------------------
# 8. OUTPUT
# -------------------------------
print("\n===== PREDICTION RESULT =====")
print(f"Disease Detected : {predicted_class}")
print(f"Confidence       : {confidence_score:.2f}%")