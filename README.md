# Breast Cancer Detection using Hybrid CNN & ViT

## Overview
This project implements a **Breast Cancer Detection** model using a **Hybrid Convolutional Neural Network (CNN) and Vision Transformer (ViT)**. The model is trained on a **Kaggle dataset** and achieves **99% accuracy** in classifying breast cancer images as **Benign or Malignant**.

## Dataset
The dataset used is from Kaggle and consists of **training and testing** images categorized into **Benign and Malignant** classes.

- **Dataset Source**: [Kaggle - Breast Cancer Diagnosis](https://www.kaggle.com/datasets/faysalmiah1721758/breast-cancer-diagnosis)

## Model Architecture
The model integrates:
- **ResNet-18** for feature extraction from images.
- **ViT (Vision Transformer)** for global context learning.
- **Focal Loss** for handling class imbalance.
- **AdamW Optimizer** with a **CosineAnnealingLR** scheduler.
- **Early Stopping** to prevent overfitting.

## Installation
### Prerequisites
Ensure you have Python installed. Install the required dependencies using:
```bash
pip install torch torchvision matplotlib scikit-learn
```

### Clone Repository
```bash
git clone https://github.com/PrangyaParmitaJena/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection
```

## Training the Model
Run the training script using:
```bash
python train.py
```
This script will:
1. Load the dataset.
2. Apply **data augmentation** (flipping, rotation, affine transformations, etc.).
3. Train the **CNN + ViT hybrid model**.
4. Save the best model (`best_hybrid_model.pth`).

## Model Performance
- **Train Loss & Test Loss Tracking**
- **Final Test Accuracy**: **99%**
- **Loss Curve Visualization**

## Results
The model's performance is visualized with a **loss curve** showing training and validation loss over epochs.

## Future Enhancements
- Implement Grad-CAM for explainability.
- Optimize hyperparameters for even better accuracy.
- Experiment with different transformers (Swin Transformer, DeiT, etc.).

## Acknowledgments
- Kaggle dataset contributors.
- PyTorch & TorchVision developers.

## License
MIT License. Feel free to use and improve upon this project.


### Contact
For any queries, feel free to reach out:
- **GitHub**: [PrangyaParmitaJena](https://github.com/PrangyaParmitaJena)
