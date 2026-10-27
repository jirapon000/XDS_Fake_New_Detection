## Overview

This project applies explainable data science (XDS) techniques to fake news detection. The goal is to understand *why* a model classifies an article as fake or real, rather than treating it as a black box, and to check whether those reasons hold up.

## Dataset

We use the ISOT Fake and Real News dataset: real articles from Reuters (`Dataset/True.csv`) and fake articles from websites flagged as unreliable (`Dataset/Fake.csv`). The data is prepared in three steps:

1. `combine_dataset.ipynb` merges the two files and adds the label (`fake = 1`, `real = 0`).
2. `dataset_eda.ipynb` explores the data and looks for shortcuts that would let a model spot the label without reading the article.
3. `clean_dataset.ipynb` removes empty texts, duplicates and badly scraped rows, and strips the "CITY (Reuters) -" dateline that appears in almost every real article, leaving 38,633 articles (45.2% fake).

The EDA found that `subject` and `date` each give the label away, so they are never used as features. Other source fingerprints stay in the text on purpose, such as photo credits on fake sites ("Featured image via Getty Images") and "Reuters" in article bodies, so the explanation methods can show whether the models rely on them.

## Models

Each article's title and text are converted into TF-IDF word features (`preprocessing.py`). We train an interpretable baseline (logistic regression), a shallow decision tree that can be read in full, and four more complex models: Random Forest, LightGBM, XGBoost and an RBF-kernel SVM. All models share the same features and train/test split and are evaluated mainly on Fake-class recall, the share of fake articles caught.

## Explanation pipeline

Each complex model is explained with the same pipeline:

- **Global importance:** grouped Permutation Feature Importance (with a random-term check to separate real effects from noise) and SHAP (TreeSHAP for tree models, KernelSHAP for the SVM)
- **Feature effects:** ALE
- **Heterogeneity:** ICE and centered ICE
- **Individual decisions:** SP-LIME, LIME and SHAP waterfalls, compared on the same articles

A correlation check on the top terms found that several of them come as packages: the words of the photo credit (image, featured, getty, via) and of embedded tweet links (com, pic, twitter). This rules out two methods that assume the words are independent. So PFI shuffles each correlated group together instead of one term at a time, ALE replaces PDP, and ICE, which has no ALE version, only uses terms outside these groups.

The readable decision tree also serves as a test case: because its true rules are known, we can check whether the explanation methods report them correctly before trusting them on models we cannot read.

## Main findings

| Model | Fake recall | Accuracy | Missed fakes | False alarms |
|---|---|---|---|---|
| Logistic regression (baseline) | 0.969 | 0.980 | 108 | 44 |
| Decision tree (depth 3) | 0.923 | 0.883 | 269 | 639 |
| Random Forest | 0.935 | 0.966 | 228 | 33 |
| LightGBM | 0.966 | 0.980 | 119 | 35 |
| XGBoost | 0.965 | 0.980 | 122 | 33 |
| SVM (RBF) | 0.975 | 0.984 | 87 | 39 |

All models separate fake from real well, and the SVM is the most accurate. The explanations show what that accuracy rests on. Every model relies on Reuters' writing style (above all "said", plus weekdays and formal titles) and on the fake sites' page layout. The decision tree's very first question is whether an article contains "via" from a photo credit. LightGBM and XGBoost lean on the photo-credit words three times as much as the Random Forest does, while the SVM spreads its decision over thousands of words and barely depends on any single one.

The models' worst mistakes show the limit. A fake site republishing a wire-style report gets through, and a real article written casually gets flagged. The models have learned to tell Reuters from these particular fake sites, which works very well within this dataset, but they judge how an article is written and laid out, not whether what it says is true.