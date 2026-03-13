# IT549: Deep Learning – Lab 3 Assignment  
## Image-Based AQI Classification using CNN and Pretrained Models

**Student Name:** Brinda Sorathiya
**Student ID:** 202301182
**Course:** IT549 - Deep Learning  

---

## 📌 Project Overview
This project demonstrates the application of deep learning for image classification by predicting the **Air Quality Index (AQI) category (AQI_Class)** purely from images of outdoor locations.

The primary objective is to build a complete **image classification pipeline in PyTorch** and compare the performance of:

- A custom **Convolutional Neural Network (CNN)** trained from scratch.  
- A **Pretrained CNN (ResNet18)** utilizing transfer learning.  

---

## 🗂️ Dataset

Dataset Link: **https://drive.google.com/drive/folders/1u-sBxgNB67GfhCQ2f7xRkDlF6fgIZZrP?usp=sharing**

The dataset consists of images and a corresponding metadata file:

- **sampled_images/**: Contains the raw image files of various locations.  
- **data.csv**: Contains the mapping of image filenames (`image_path/Filename`) to their target categories (`AQI_Class`).  

---

## ⚙️ Methodology & Pipeline

### Data Preparation
- Images were resized to **224 × 224 pixels**.  
- Pixel values were converted to tensors and normalized using **standard ImageNet mean and standard deviation**.  
- The dataset was split into **Train (70%)**, **Validation (15%)**, and **Test (15%)** sets using **stratified sampling** to maintain class distribution.  

### Basic CNN Model
- A custom CNN with **3 convolutional blocks**  
  `(Conv2d → ReLU → MaxPool)`  
- Followed by **fully connected layers and dropout**.  
- The model was trained **from scratch**.

### Pretrained Model (Transfer Learning)
- Used the **ResNet18** architecture pretrained on **ImageNet**.  
- The weights of the **first ~10 layers were frozen** to retain basic feature-extraction capabilities.  
- The final **fully connected classification layer was replaced** to output the specific number of `AQI_Class` categories and trained on the dataset.

---

## 📊 Results Summary

Both models were evaluated on the **unseen Test Set** using **Accuracy, Precision, Recall, and F1-score**.

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
|------|----------|----------------------|-------------------|---------------------|
| Basic CNN | 87.00% | 0.8743 | 0.8700 | 0.8694 |
| ResNet18 (Transfer Learning) | 95.11% | 0.9514 | 0.9511 | 0.9509 |

### Key Findings on Transfer Learning
The pretrained **ResNet18** model significantly outperformed the custom CNN trained from scratch. Because ResNet18 has already learned universal visual features *(edges, textures, gradients)* from millions of images on ImageNet, it converges much faster and extracts more meaningful representations of atmospheric scattering/smog than a model starting with randomized weights.

---

## 🔍 Misclassification Analysis

Despite the strong performance of the transfer learning model, predicting chemical air composition from RGB images has inherent limitations. An analysis of the test set misclassifications revealed the following primary causes for error:

### The "Invisible Pollutant" Trap
High AQI driven by invisible gases like **Ozone ($O_3$)** often presents with clear blue skies, deceiving the model into predicting **"Good"** or **"Moderate"** air quality.

### Weather vs. Pollution
Natural phenomena like **morning mist, fog, or heavy rain clouds** are frequently misclassified as **"Unhealthy"** or **"Hazardous"** industrial smog due to the similar reduction in visibility.

### Missing Horizons
Images taken at **tight angles dominated by concrete buildings** lack the **sky/horizon visibility** required for the model to accurately judge atmospheric haziness.

---

## 🚀 How to Run the Code

1. Clone this repository.

2. Ensure you have the required dependencies installed:

```bash
pip install torch torchvision pandas scikit-learn matplotlib seaborn pillow
```

3. Place data.csv and the zipped dataset in the root directory.

4. Open the Jupyter Notebook (.ipynb file) and run the cells sequentially.
The notebook handles data extraction, preprocessing, model training, evaluation and visualization.