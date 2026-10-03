# Modeling Pipeline Overview

This document explains the overall modeling flow for the Fake News Detection project, so every team member builds their notebook consistently.

## Shared Rules (apply to every model, every notebook)

1. **Dataset:** Always load `Dataset/fake_news_dataset_cleaned.csv` (missing values already filled, `label_encoded` already created: `fake=1, real=0`).
2. **Shared preprocessing file:** Always import from `preprocessing.py` at the project root — do not rewrite your own version of text cleaning, TF-IDF settings, or train/test split. This guarantees every model is trained/compared fairly on identical inputs.
```python
   from preprocessing import (
       preprocess_text, get_tfidf_features,
       get_train_test_split, RANDOM_STATE
   )
```
3. **Random seed:** Always use `RANDOM_STATE` (imported from `preprocessing.py`, currently `42`) for every model and split. Never hardcode a different value.
4. **Train/test split:** Always use `get_train_test_split()` from `preprocessing.py` — same `test_size=0.2`, `stratify=y`, `random_state=RANDOM_STATE` every time.
5. **Primary metric:** Recall on the **Fake** class (`label_encoded = 1`). Report using `recall_score(y_test, y_pred, pos_label=1)` 
6. **Text preprocessing:** Use `preprocess_text()` from `preprocessing.py` — lowercase → remove punctuation/numbers → tokenize → remove stopwords → lemmatize → TF-IDF (unigrams + bigrams).

---

## Baseline Model (Logistic Regression)

**Notebook:** `baseline_model.ipynb`

**Features:** TF-IDF only (title + text combined). No metadata.

**Why:** Logistic regression is inherently interpretable — coefficients directly explain predictions. No post-hoc explanation methods needed (no LIME/SHAP/PDP/ICE).

**Steps:**
1. Load cleaned data
2. Preprocess text (`preprocess_text()` from `preprocessing.py`)
3. TF-IDF vectorize (`get_tfidf_features()` from `preprocessing.py`)
4. Train/test split (`get_train_test_split()` from `preprocessing.py`)
5. Train Logistic Regression
6. Evaluate (classification report, confusion matrix, Fake-class recall)
7. Interpret coefficients → odds ratios (top words pushing toward fake/real)
8. Sanity check: look for suspicious/leaked terms (e.g., outlet names) in top coefficients