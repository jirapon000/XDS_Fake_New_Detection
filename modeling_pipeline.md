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

---

## Common Models (Random Forest, XGBoost, LightGBM, Decision Trees, SVM)

**Notebook:** one per model, e.g. `model_random_forest.ipynb`, `model_xgboost.ipynb`, etc.

**Features:** Same TF-IDF matrix as baseline — use `preprocessing.py` functions so the vectorizer settings and split are identical across every model.

**Why:** These are black-box models — need post-hoc explanation methods to interpret them.

**Steps (same for every model in this tier):**
1. Load cleaned data
2. Preprocess text (`preprocess_text()` — identical to baseline)
3. TF-IDF vectorize (`get_tfidf_features()` — identical settings to baseline)
4. Train/test split (`get_train_test_split()` — identical split to baseline)
5. Train the model
6. Evaluate (classification report, confusion matrix, Fake-class recall)
7. Select top-K important terms (chi-squared or mutual information ranking — only needs to be done once, can be reused across all Tier 2 models)
8. Apply explanation methods:
   - **LIME** — sanity-check individual predictions by removing words, observe probability shift
   - **PDP/ICE** — on top-K terms only (not full TF-IDF vocabulary)
   - **PFI** — confirm term importance via performance degradation (especially important for SVM, which has no native importance)
   - **TreeSHAP** — for tree-based models only (Random Forest, XGBoost, LightGBM, Decision Trees)
   - **Correlation heatmap** — on top-K terms, check multicollinearity
9. Compare recall against baseline — does the added complexity justify the drop in transparency?

**Best-performing model (by Fake-class recall) gets carried forward for the deepest interpretability analysis.**

---

## Tier 3: Unique Models (EBM, CatBoost)

*To be added — feature engineering approach still under research.*

---

## Final Comparison

Once all models are trained, compare across a single table: model name, Fake-class recall, accuracy, and interpretability approach used. This becomes the basis for the project's final discussion of the accuracy–interpretability trade-off.