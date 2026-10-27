# Modeling Pipeline Overview

This document explains the overall modeling flow for the Fake News Detection project, so every notebook is built consistently.

## Data Preparation (run once, in this order)

1. **`combine_dataset.ipynb`:** merges `Dataset/True.csv` (real, Reuters) and `Dataset/Fake.csv` (fake) into `Dataset/fake_news_combined.csv.gz`, adding `label` and `label_encoded` (`fake = 1`, `real = 0`).
2. **`dataset_eda.ipynb`:** explores the combined data and reports problems. It changes nothing and saves nothing.
3. **`clean_dataset.ipynb`:** fixes what the EDA found and saves `Dataset/fake_news_dataset_cleaned.csv.gz`:
   - strips extra spaces
   - removes the "CITY (Reuters) -" dateline from real articles
   - drops rows with an invalid date, empty texts and duplicate texts

Files are gzip-compressed to stay under GitHub's 100 MB limit; `pd.read_csv` reads them directly.

## Shared Rules (apply to every model notebook)

1. **Dataset:** always load `Dataset/fake_news_dataset_cleaned.csv.gz`.
2. **Features:** title + text only. `subject` and `date` each give the label away (see the EDA), so they are never used as features.
3. **Shared preprocessing file:** always import from `preprocessing.py`. Do not write your own text cleaning, TF-IDF settings or train/test split.
```python
   from preprocessing import (
       preprocess_text, get_split_tfidf,
       select_top_k_terms, RANDOM_STATE
   )
```
4. **Text preprocessing:** `preprocess_text()`: lowercase → replace all punctuation and non-ASCII characters (including curly quotes) with spaces → convert accented letters → remove numbers → tokenize → remove stopwords → lemmatize.
5. **TF-IDF and split:** `get_split_tfidf()`: stratified 80/20 split first, then TF-IDF (unigrams, up to 5,000 terms) fitted on the training text only, so no test information leaks into the vocabulary.
6. **Random seed:** always use `RANDOM_STATE` (currently `42`). Never hardcode a different value.
7. **Primary metric:** recall on the **Fake** class (`label_encoded = 1`), reported with `recall_score(y_test, y_pred, pos_label=1)`. Always report it with precision and accuracy, since recall alone can be maxed out by flagging everything.

---

## Baseline Model (Logistic Regression)

**Notebook:** `logistic_regression_TF-IDF.ipynb`

**Why:** logistic regression is interpretable by design. Its coefficients explain predictions directly, so no post-hoc methods are needed.

**Steps:**
- Load the cleaned data, preprocess, TF-IDF and split (shared functions)
- Train logistic regression
- Evaluate (classification report, confusion matrix, Fake-class recall)
- Sanity check: dummy classifier and permutation test (shuffled labels)
- Interpret coefficients (top terms pushing toward fake and real)
- Fingerprint check: retrain without source-fingerprint terms to see how much accuracy depends on them

---

## Readable Model (Decision Tree)

**Notebook:** `decision_tree.ipynb`

**Why:** a depth-3 tree can be read and checked in full. Because its true rules are known, it is also used to test whether the explanation methods report them correctly.

**Steps:** train (depth 3, at least 50 articles per leaf) → read the tree → compare depths with cross-validation → evaluate → PFI and TreeSHAP → LIME and SHAP waterfall on single articles. PDP, ICE and SP-LIME are skipped, because the tree itself already shows what they would.

---

## Black-Box Models (Random Forest, LightGBM, XGBoost, SVM)

**Notebooks:** `random_forest.ipynb`, `LightGBM.ipynb`, `XGBoost.ipynb`, `SVM.ipynb`

**Why:** these models can't be read directly, so they are explained with post-hoc methods.

**Model-specific notes:**
- **XGBoost** treats sparse zeros as missing values, so it is trained and explained on dense arrays.
- **SVM** is trained on the sparse matrix (much faster with 5,000 terms). It uses `decision_function` instead of probabilities, and PFI and ALE run on a 1,000-article sample because it is slow to score.

**Steps (same for every model):**
- Load, preprocess, TF-IDF and split (shared functions)
- Train the model
- Evaluate (classification report, confusion matrix, Fake-class recall, ROC AUC)
- **Top-K terms:** the 30 terms most linked to the label by chi-squared on the training data (`select_top_k_terms()`), shared across all models
- **Correlation check:** heatmap of the top-K terms, flagging pairs with |correlation| > 0.5. Several terms come as packages (photo-credit words, tweet links), which rules out methods that assume independent words.
- **Explanation methods:**
  - **Grouped PFI:** drop in Fake-class recall when terms are shuffled; correlated groups are shuffled together, other terms one at a time. A null check with 30 random terms separates real effects from noise.
  - **SHAP:** TreeSHAP for tree models; KernelSHAP for the SVM (top 30 terms only)
  - **ALE:** feature effects for the top 6 terms (used instead of PDP because of the correlation)
  - **ICE and centered ICE:** only for top terms outside the correlated groups, since ICE has no ALE version
  - **SP-LIME:** representative articles, chosen automatically
  - **LIME:** the most confident caught fake, the most clearly missed fake and the most confident false alarm
  - **SHAP waterfall:** the missed fake article, to compare with LIME on the same case
- Model summary

---

## Tier 3: Unique Models (EBM, CatBoost)

*To be added — feature engineering approach still under research.*

---

## Final Comparison

All models are compared in one table: Fake-class recall, accuracy, missed fakes, false alarms and the interpretability approach used. The final discussion covers the accuracy–interpretability trade-off and, using the explanations, what the models actually rely on: writing style versus source fingerprints.
