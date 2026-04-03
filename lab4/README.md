# 🚀 Object Detection Evolution: From R-CNN to YOLOv8  

This project provides a comprehensive technical walkthrough of the evolution of Object Detection architectures. It spans from the foundational **Region-based Convolutional Neural Networks (R-CNN)** to the modern, high-performance **YOLOv8 (You Only Look Once)**.

The primary focus was to bridge the gap between **theoretical computer vision concepts and practical deployment**, specifically optimizing for **inference latency** and **mean Average Precision (mAP)**.

---

## 🚀 Key Features & Implementations  

### 🧠 Mathematical Foundations  
- From-scratch implementation of:
  - **Intersection over Union (IoU)**
  - **Non-Maximum Suppression (NMS)**  

---

### ⚡ Bottleneck Analysis  
- Quantitative comparison between:
  - **R-CNN (per-proposal inference)**
  - **Fast R-CNN (RoI Pooling)**  

- Demonstrated an **~87x speedup**

---

### 🔄 End-to-End Pipeline  
- Data engineering:
  - Pascal VOC → YOLO format conversion  
- Handling corrupted metadata  
- Fine-tuning state-of-the-art detection models  

---

### 🏗️ Modern Frameworks  
- PyTorch  
- Torchvision  
- Ultralytics (YOLOv8)  

---

## 🏗️ Technical Deep Dive  

### 1️⃣ The Geometry of Detection: IoU & NMS  

- **IoU**: Measures overlap between predicted and ground-truth bounding boxes  
- **NMS**: Removes duplicate detections based on confidence + IoU threshold  

#### 🔍 Insight  
- High threshold (0.9) → clutter (duplicate boxes)  
- Low threshold (0.1) → under-detection (missed objects in dense scenes)  

---

### 2️⃣ Solving the R-CNN Compute Bottleneck  

Original R-CNN models are slow because they perform a full forward pass for **every region proposal**.

- **R-CNN Approach**  
  - 100 proposals → 100 CNN passes  

- **Fast R-CNN Approach**  
  - 1 CNN pass + RoI Pooling  

#### 📊 Performance Comparison  

| Metric                      | R-CNN (ResNet18) | Fast R-CNN (RoI Pool) |
|---------------------------|------------------|------------------------|
| Inference Time (100 boxes)| 0.3981s          | 0.0045s                |
| Efficiency Gain           | Baseline         | **87.86x Faster** 🚀   |

---

### 3️⃣ Faster R-CNN & the RPN  

- Used **Faster R-CNN with Region Proposal Network (RPN)**  

#### 💡 Key Idea  
RPN replaces Selective Search by:
- Learning region proposals directly  
- Sharing convolutional features  

➡️ Eliminates external proposal generation  

---

### 4️⃣ Real-Time Detection with YOLOv8  

#### 📦 Data Engineering  
- Converted Pascal VOC XML → YOLO `.txt` format  

#### 🧹 Data Cleaning  
- Handled corrupted images (0×0 dimensions)  
- Dynamically read shapes using OpenCV  

#### 🏋️ Fine-Tuning  
- Model: **YOLOv8 Nano (yolov8n.pt)**  
- Training: **10 epochs on Tesla T4 GPU**  

---

## 📊 Performance Comparison  

| Model                      | Inference Time (ms) | Precision | Recall |
|---------------------------|--------------------|----------|--------|
| Faster R-CNN (Pre-trained)| 117.22             | 0.420*   | 0.838* |
| YOLO (Pre-trained)        | 34.76              | 0.676    | 0.624  |
| YOLO (Fine-tuned)         | **11.17**          | **0.886**| **0.799** |

> *Pre-trained models evaluated via COCO-to-Local class mapping  

---

### 📈 Fine-Tuned YOLO Results (Test Set)

- **mAP@50:** 0.8986  
- **mAP@50-95:** 0.6569  

---

## 🛠️ Tools & Technologies  

- **Languages:** Python 3.12  
- **Deep Learning:** PyTorch, Torchvision, Ultralytics (YOLOv8)  
- **Computer Vision:** OpenCV (cv2), PIL  
- **Data Analysis:** Pandas, NumPy, Matplotlib  
- **Environment:** Google Colab (Linux / Tesla T4)  

---

## 📂 Project Structure  
├── fruit_data/
│ ├── train/ # Images + XML/TXT annotations
│ └── test/ # Images + XML/TXT annotations
├── fruit_data.yaml # YOLO configuration file
├── lab4_analysis.ipynb # Main development notebook
└── README.md # Project documentation

---

## 🧠 Key Insights & Learnings  

- **mAP@50 can be misleading**  
  A model with high mAP@50 but low mAP@50–95 often has poor localization. Robust models maintain performance across stricter IoU thresholds.

- **Localization > Detection Confidence in real-world systems**  
  Accurate bounding boxes (high mAP@50–95) are more critical than just detecting object presence.

- **Feature sharing is the core optimization breakthrough**  
  The transition from R-CNN → Fast R-CNN → Faster R-CNN shows that eliminating redundant convolution operations is the key to scalability.

- **Region Proposal vs Regression-based detection**  
  Two-stage detectors (R-CNN family) optimize for accuracy, while single-shot detectors (YOLO) optimize for real-time performance.

- **Inference speed is architecture-dependent, not just hardware-dependent**  
  The ~87x speedup from Fast R-CNN came from architectural design, not better hardware.

- **Data quality directly impacts model performance**  
  Handling corrupted annotations and inconsistent image metadata was essential for stable YOLO training.

- **NMS threshold tuning is context-dependent**  
  A high IoU threshold retains duplicates, while a low threshold risks missing objects — especially in dense scenes.

- **Fine-tuning > Pre-trained models for domain-specific tasks**  
  Significant gains in precision and recall were achieved after adapting YOLO to the custom dataset.

---

## ▶️ How to Run  

```bash
# Clone the repository
git clone <your-repo-link>

# Navigate into project
cd <repo-name>

# Run the notebook
lab4_analysis.ipynb