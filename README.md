# Fake News Detection: Explaining What the Models Look At

## Overview

This project was developed by Insight Lab, a class project group, and applies explainable data science (XDS) techniques to fake news detection. The goal is to understand *why* a model classifies an article as fake or real, rather than treating it as a black box, and to check whether those reasons hold up. A black-box model is typically judged only by performance metrics such as accuracy, precision, and recall, which can lead us to trust it without knowing how it reaches its decisions. But high accuracy alone doesn't mean a model can be trusted: it could be right for the wrong reasons, such as recognising a website's layout instead of reading what the article says. So we look past these metrics and ask two more questions: what is each model actually relying on, and can we trust the explanations that tell us?

## Dataset

We use the ISOT Fake and Real News dataset: real articles from Reuters (`Dataset/True.csv`) and fake articles from websites flagged as unreliable (`Dataset/Fake.csv`). The data is prepared in three steps:

1. `combine_dataset.ipynb` merges the two files and adds the label (`fake = 1`, `real = 0`).
2. `dataset_eda.ipynb` explores the data and looks for shortcuts that would let a model spot the label without reading the article.
3. `clean_dataset.ipynb` removes empty texts, duplicates and badly scraped rows, and strips the "CITY (Reuters) -" dateline that appears in almost every real article, leaving 38,633 articles (45.2% fake).

The EDA found that `subject` and `date` each give the label away, so they are never used as features. Other source fingerprints stay in the text on purpose, such as photo credits on fake sites ("Featured image via Getty Images") and "Reuters" in article bodies, so the explanation methods can show whether the models rely on them.

## Models

Each article's title and text are converted into TF-IDF word features (`preprocessing.py`). We train an interpretable baseline (logistic regression), a shallow decision tree (depth = 3) that can be read in full, and four black box models: Random Forest, LightGBM, XGBoost and an RBF-kernel SVM. All models share the same features and train/test split and are evaluated mainly on Fake-class recall, the share of fake articles caught.

## Explanation pipeline

Each bkack box model is explained with the same pipeline:

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

## Which Model We Explain and Which We Recommend

**The model we explain and test is the SVM, but the model we recommend deploying is logistic regression.**

The SVM is the most accurate model, and its reasoning can't be read directly: each decision compares an article with 7,865 stored training articles across 5,000 words, so no single number belongs to a word. That makes it the model whose explanations need testing, which `xai_evaluation.ipynb` does (faithfulness, stability, counterfactuals and ethics).

Its lead over logistic regression is real but small. `xai_evaluation.ipynb` shows the gap is not luck (the SVM is right on 33 of the 40 test articles where the two disagree, p = 0.00004), but it amounts to 21 fewer missed fakes out of 3,489. Logistic regression explains itself exactly through its coefficients, and it relies least of all six models on source fingerprints (14% of its LIME weight, against 20% for the SVM). For a gap that small, being able to see exactly why each article was flagged is worth more. The SVM also takes minutes to train instead of seconds, and its explanations are slow, approximate estimates, while logistic regression's are exact. A gain of 21 articles isn't worth a black box that is slower to train, run and explain.

For intended use, limitations, ethics and the full evaluation, see `MODEL_CARD.md`.

## Project Structure

| Step | File | What it does |
|---|---|---|
| 1 | `combine_dataset.ipynb` | Merges `True.csv` and `Fake.csv` and adds the label |
| 2 | `dataset_eda.ipynb` | Explores the data and lists problems (changes nothing) |
| 3 | `clean_dataset.ipynb` | Fixes those problems and saves the cleaned dataset |
| 4 | `preprocessing.py` | Shared text cleaning, TF-IDF and train/test split |
| 5 | `Model/Baseline Model/` | Logistic regression baseline |
| 6 | `Model/Common Models/` | Decision tree, Random Forest, LightGBM, XGBoost, SVM |
| 7 | `Model/XAI Synthesis/` | Compares the explanations of all six models side by side and recommends a model |
| 8 | `Model/XAI Evaluation/` | Checks whether the SVM's lead over logistic regression is real, and whether LIME and SHAP can be trusted (faithfulness, stability, counterfactuals, ethics) |
| 9 | `Model/Human Study/` | Study with real people: does seeing the LIME highlights help them predict the model? |
| – | `run_all.py` | Re-runs every notebook in order and saves all figures to `figures/` |
| – | `MODEL_CARD.md` | Intended use, training data, evaluation results and known limitations |

## Word Groups

To compare explanations, the project sorts important words into four groups, used the same way in every notebook:

| Group | Words | What they are |
|---|---|---|
| Source fingerprints | reuters, via, image, images, featured, getty, com, http, www, pic, twitter, wire, read, video, watch | Traces of how each site lays out its pages: photo credits ("Featured image via Getty Images"), links, site sign-offs ("21st Century Wire", "READ MORE"), embedded videos ("WATCH", "(VIDEO)") and the publisher's name |
| News-report words | said, told, minister, government, spokesman, presidential, representative, nov, monday to sunday | How Reuters writes: every claim attributed ("said on Tuesday"), formal titles, abbreviated months |
| Opinion words | just, like, know, don, right, fact, gop, hillary, mr, breaking, even | How the fake sites write: conversational, partisan and headline-style language |
| Other words | every other word | Topic and content |

## How to Run

Install the pinned dependencies (Python 3.14.2):

    pip install -r requirements.txt

**Everything in one command.** From the project root, run:

    python run_all.py

This re-runs every notebook in order (combine → EDA → clean → models → synthesis → evaluation) and saves every plot as a PNG in `figure/`, one folder per notebook. To re-run a single notebook, use `python run_all.py --only SVM`. To save figures without re-running, use `python run_all.py --figures-only`.

**By hand.** Run steps 1–3 once, in order: they create the cleaned dataset that every later notebook uses. Then run steps 5–8 in any order; each one loads the cleaned dataset and trains its own models. Step 9 (the human study) is run separately, because it needs answers from real participants.

All splits, models and explanation methods use the fixed seed `RANDOM_STATE = 42`, set in `preprocessing.py`. The SVM takes several minutes to train, so the two XAI notebooks save it to a `.joblib` file on the first run and reload it after that. Delete that file to retrain from scratch.