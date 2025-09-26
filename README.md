# News Classification with K-Nearest Neighbors (KNN)

This repository implements a complete pipeline to classify news articles into categories using a K-Nearest Neighbors (KNN) classifier. The codebase includes: Persian text preprocessing (normalization, tokenization, lemmatization), POS-tag filtering, document–term matrix construction, dimensionality reduction with PCA, model training, evaluation, and inference on new data. The pipeline relies on the **Hazm** NLP toolkit (including a POS tagger model file) and plain NumPy/Pandas implementations of KNN.

---

## 1) Project Goals

- Build a reproducible **text classification** pipeline for news articles.
- Use a **rule-based POS filter** + **lemmatization** to reduce noise.
- Represent documents with a **DTM (bag-of-words)** and reduce dimensionality via **PCA**.
- Train and evaluate a **KNN** classifier; export artifacts needed for later inference.

---

## 2) Data

Two CSVs are expected for a typical workflow:

- **Training**: `nlp_train.csv` with columns:
  - `Text`: the article body (Persian),
  - `Category`: the class label (e.g., «ورزشی», «سیاسی», …).
- **Testing / Inference**: `nlp_test.csv` (or any CSV with the same schema).

**Optional crawling utilities** are provided to collect raw data:
- A sports crawler that downloads content from a news sitemap and writes `news_data_3000.csv` with fields `Index`, `Content`, `Category`, `URL`.  
- A politics crawler that paginates an archive, fetches article content, and writes `c.csv` with fields `index`, `category`, `content`, `url`.  

> You can merge/normalize crawled CSVs into the `nlp_train.csv`/`nlp_test.csv` schema before training/testing.

---

## 3) Dependencies

- Python 3.9+  
- `hazm`, `numpy`, `pandas`, `scikit-learn`, `beautifulsoup4`, `requests`

Install:

```bash
pip install hazm numpy pandas scikit-learn beautifulsoup4 requests
```

---


## 4) Pipeline Overview

### 4.1 Preprocessing & POS Filtering

For each article:

1. **Normalize** the text.  
2. **Tokenize** into words.  
3. **POS-tag** tokens.  
4. **Filter** tokens by excluding closed-class and non-informative tags (e.g., NUM, ADP, PRON, DET, CCONJ, SCONJ, PUNCT, VERB, etc.).  
5. **Lemmatize** remaining tokens and join back to a processed string.

This logic is implemented identically in training and testing code, ensuring consistent features across stages.

### 4.2 Document–Term Matrix (DTM)

- **Training** builds a vocabulary and a **sparse→dense DTM** where each column is a unique term and each row is a document’s term counts. The script returns a dense NumPy matrix and saves the vocabulary (`unique.sav`).  
- **Testing** loads `unique.sav` and constructs a DTM **using the training vocabulary only** to ensure feature compatibility.

### 4.3 Dimensionality Reduction (PCA)

- On training, the pipeline fits **PCA** (`pcafit.sav`) and transforms `X_train` to `X_train_pca` (also saved as `X_train_pca.sav`).  
- On test/inference, the same PCA transform is applied to the test DTM.  

### 4.4 Classifier: KNN

Two KNN implementations appear across scripts:

- A **NumPy-based KNN** class with `fit/predict/score` and Euclidean distance, returning the majority label among k nearest neighbors. It also shows `LabelEncoder` usage for encoded labels and reports macro-F1/accuracy.  
- A simplified **myKNN** variant tailored for two classes ("Sport" vs. "Politics") and used for **test-time inference** after loading the PCA-transformed training matrix (`X_train_pca.sav`).  

---

## 5) Files & Artifacts

- **Training code**:
  - `train.py`: builds vocabulary, creates DTM, fits PCA, saves `unique.sav`, `pcafit.sav`, `X_train_pca.sav`.  
  - `main.py`: complete train/validation example with scaling, PCA, KNN, metrics; also saves artifacts.  
- **Inference / Test**:
  - `test.py`: loads `X_train_pca.sav`, `unique.sav`, `pcafit.sav`; preprocesses `nlp_test.csv`, builds DTM using training vocabulary, transforms via PCA, predicts with KNN, and prints metrics.  
- **Crawlers (optional)**:
  - `crawler.py`: sports news sitemap crawler (writes `news_data_3000.csv`).  
  - `crawler-politics.py`: politics archive crawler (writes `c.csv`).  
- **Model assets** (created by training):
  - `unique.sav`: list of unique vocabulary terms (order matters; used to align test features).  
  - `pcafit.sav`: fitted PCA transformer for projecting DTM to lower dimensions.  
  - `X_train_pca.sav`: PCA-transformed training matrix stored for KNN at inference time.  

**POS model (required):**  
- `pos_tagger.model` must be present to run token tagging in both training and testing.  
 

---

## 6) Implementation Notes

- **Vocabulary freeze:** the **training vocabulary** (in `unique.sav`) is reused at test time to avoid feature drift; test tokens not in the training list are ignored in vectorization.  
- **Dimensionality reduction:** PCA is fit **once** on training DTM and reused for all later projections. Choose `n_components` according to data scale; scripts default to 1000.  
- **Distance metric:** Euclidean distance is used in KNN. k is configurable (examples use k=3 or k=9 depending on the script).  
- **Evaluation:** `main.py` demonstrates accuracy, macro-F1, and confusion matrix; `test.py` prints a custom F1 for a binary setting.  

---
