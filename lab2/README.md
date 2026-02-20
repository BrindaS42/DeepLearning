# IT549 – Deep Learning Lab 2  
## GloVe Pretrained Embeddings for Movie Text Prediction

**Name:** Brinda Sorathiya  
**ID:** 202301182

---

##  Objective
This project demonstrates the use of pretrained **GloVe word embeddings** for:
1. **Regression** – Predicting movie `voting_average`
2. **Multi-Label Classification** – Predicting movie genres

Each experiment uses **only one text column at a time**.

---

##  Dataset
**Source:** Kaggle Movie Dataset  
**Used Columns:**
- overview
- tagline
- keywords
- genres (multi-label)
- voting_average (regression target)

---

##  Experimental Setup
- **Embedding:** GloVe WikiGiga (100D )
- **Document Embedding:** TF-IDF weighted averaging
- **Split:** 70% Train / 15% Validation / 15% Test
- **Framework:** PyTorch

---

##  Models

### Model A — Rating Prediction (Regression)
- Neural network with ReLU
- Metrics:
  - MSE
  - RMSE
- Baseline:
  - Global mean rating

---

### Model B — Genre Prediction (Multi-Label)
- Sigmoid outputs
- Loss: BCEWithLogitsLoss
- Metrics:
  - Micro-F1
  - Macro-F1
  - Hamming Loss
  - Jaccard Score

---

## Text Analysis
- Top 10 most frequent words per genre
- Bottom 10 least frequent words per genre
- TF-IDF based indicative words using linear analysis

---

