# Fundus-Eye-Disease-Detection-Rare-Disease-

## Project Overview

This project focuses on the automatic detection of retinal diseases from fundus eye images using Deep Learning and Transfer Learning. The system classifies retinal images into multiple eye disease categories to assist in early diagnosis and clinical decision-making.

The model is trained using ImageNet-pretrained Convolutional Neural Networks (CNNs) with class-balanced learning, strong data augmentation, and stratified cross-validation to ensure robust performance on imbalanced medical datasets.

---

## Diseases Covered

The model classifies the following retinal disease classes:

1. DN (Diabetic Neuropathy)
   Retinal damage caused due to diabetes.

2. ODE (Optic Disc Edema)
   Swelling of the optic nerve head.

3. BRVO (Branch Retinal Vein Occlusion)
   Blockage of a branch of the retinal vein.

4. CRVO (Central Retinal Vein Occlusion)
   Blockage of the main retinal vein.

5. Blur fundus with suspected PDR
   Blurred retinal image with proliferative diabetic retinopathy.

6. Blur fundus without PDR
   Blurred retinal image without severe diabetic changes.

7. MYA (Myopia)
   Near-sightedness affecting retinal structure.

8. MH (Macular Pathology)
   Damage in the macula region affecting central vision.

---

## Dataset Distribution

| Class                          | Number of Samples |
| ------------------------------ | ----------------- |
| DN                             | 229               |
| ODE                            | 118               |
| BRVO                           | 140               |
| CRVO                           | 45                |
| Blur fundus with suspected PDR | 45                |
| Blur fundus without PDR        | 114               |
| MYA (Myopia)                   | 43                |
| MH (Macular Pathology)         | 41                |

The dataset is highly imbalanced, which is addressed using class-balanced training techniques.

---

## Methodology

### Transfer Learning

Instead of training a CNN from scratch, ImageNet-pretrained models are used. These models already understand basic visual features such as edges, shapes, and textures, which helps in faster and more effective learning on medical images.

---

## Backbone Models

* ResNet50
* EfficientNet-B0 (optional alternative)

---

## Model Architecture

Input Image (224 × 224)
→ Pretrained CNN Backbone (ResNet50)
→ Global Average Pooling
→ Fully Connected Layer (512 units)
→ Dropout (0.4)
→ Fully Connected Layer (8 classes)
→ Softmax Activation

---

## Handling Class Imbalance

### Weighted Cross-Entropy Loss

* Assigns higher importance to under-represented disease classes.
* Reduces bias toward majority classes.

### Balanced Sampling

* Ensures each training batch contains samples from all classes.
* Improves learning for rare diseases.

---

## Training Strategy

### Step 1: Train Classifier Head

* Backbone layers are frozen.
* Only the classifier layers are trained.
* Training duration: 10–15 epochs.
* Purpose: Learn disease-specific classification features.

### Step 2: Fine-Tuning

* Top 30–40% layers of the backbone are unfrozen.
* Entire network is trained.
* Training duration: 50–80 epochs.
* Low learning rate is used to prevent overfitting.
* Purpose: Adapt pretrained features to retinal images.

---

## Data Augmentation (Applied During Training Only)

To increase data diversity and reduce overfitting, the following techniques are applied:

* Horizontal Flip
* Rotation (±15 degrees)
* Brightness and Contrast Adjustment
* Random Resized Crop
* Slight Zoom

Data augmentation is not applied during validation or testing.

---

## Stratified 5-Fold Cross-Validation

* Dataset is divided into 5 folds.
* Each fold preserves the original class distribution.
* Model is trained and evaluated 5 times.
* Final performance is averaged across all folds.

This ensures reliable and unbiased evaluation.

---

## Evaluation Metrics

The model is evaluated using the following metrics:

* Accuracy
* Error Rate
* Precision
* Recall (Sensitivity)
* Specificity
* F1-Score
* Confusion Matrix

Sensitivity and specificity are emphasized as they are critical in medical diagnosis.

---

## Final Performance

* Overall Accuracy: Approximately 95%
* High sensitivity and specificity across disease classes.
* Strong performance on rare disease categories.

---

## Image Prediction (Inference)

The trained model can predict disease from:

* A single image path
* A folder path containing images

Example input:
D:\Projects\Final_Year_Project\Final Year Project Code\rare_disease\BRVO\01_BRVO.JPG

Example output:
Disease Detected: BRVO
Confidence: 94%

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Scikit-learn
* CUDA (GPU acceleration)

---

## Project Structure

rare_disease/
├── BRVO/
├── CRVO/
├── DN/
├── ODE/
├── MYA/
├── MH/
├── train_step1.py
├── train_step2_finetune.py
├── cross_validation.py
├── evaluate_model.py
├── predict_image.py
├── final_model.pth
└── README.md

---

## Learning Outcomes

This project helped in understanding:

* Transfer Learning
* CNN fine-tuning
* Medical image classification
* Class imbalance handling
* Model evaluation for healthcare applications

---

## Future Improvements

* Grad-CAM visualization for model explainability
* Web-based interface using Streamlit
* Multi-disease detection per image
* Integration with clinical systems

---

## Author

Final Year B.Tech Project
Biomedical Engineering / AI-ML Domain

