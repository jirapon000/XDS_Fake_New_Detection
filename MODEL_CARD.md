# Model Card: Fake News Screening Models (ISOT)

Insight Lab · Explainable Data Science (XDS 2026) · Asian Institute of Technology
Format follows Mitchell et al. (2019), *Model Cards for Model Reporting*.

## Model Details

We trained six text classifiers on the same features and the same train/test split. Two of them matter most:

| | Logistic regression (recommended) | SVM, RBF kernel (explained and tested) |
|---|---|---|
| Role | Baseline, and the model we recommend deploying | Most accurate model; the black box whose explanations we test |
| Settings | scikit-learn `LogisticRegression`, default L2 penalty (C = 1), `max_iter = 1000` | scikit-learn `SVC(kernel="rbf", C=1.0, gamma="scale")`, gamma = 1.014 |
| Why it is or isn't readable | One coefficient per word; each prediction is the sum of coefficient × TF-IDF weight | Each decision compares the article with 7,865 stored training articles (support vectors, 25% of the training set) across all 5,000 terms; no weight belongs to a single word |
| Output | Probability of fake | Decision value (above 0 = fake) |

The other four models (decision tree of depth 3, Random Forest, LightGBM and XGBoost) are described in their own notebooks in `Model/Common Models/`.

- **Input:** an article's title and body text, combined and cleaned (`preprocessing.py`).
- **Features:** TF-IDF over the 5,000 most frequent single words. Words in fewer than 5 articles or more than 90% of articles are dropped. The vectoriser is fitted on the training articles only.
- **Output label:** fake = 1, real = 0.
- **Random seed:** `RANDOM_STATE = 42` for the split, every model and every explanation method.

## Intended Use

- **Primary user:** a trust-and-safety reviewer at a news-sharing platform.
- **Decision supported:** for each article the screen flags as fake, whether to demote it in the feed straight away or send it to manual fact-checking first.
- **How it should be used:** as a first screen that ranks articles for human review, always shown together with its explanation (the words behind the flag), never as the final decision.

### Out of Scope

- **Deciding whether a claim is true.** The models judge how an article is written and laid out, not whether what it says is correct.
- **Use on other sources without retesting.** The models learned to tell Reuters from a particular set of fake sites (see Limitations).
- **Automatic removal or demotion without a human check.**
- **Use as evidence against an individual writer or publisher.**

## Training Data

- **Source:** the ISOT Fake News Dataset (Ahmed et al., 2017, 2018), from the Kaggle mirror "Fake and Real News Dataset" (Bisaillon, 2020).
- **Licence:** CC BY-NC-SA 4.0 (non-commercial; commercial deployment would need permission or different training data)
- **Raw size:** 44,898 articles. 21,417 real articles come from Reuters and 23,481 fake articles come from websites flagged as unreliable. Most cover politics and world news, 2015–2018.
- **Labels:** taken from which file an article is in. Labels describe the *source*, not a fact-check of each article.

### Cleaning

The cleaning steps are in `clean_dataset.ipynb`:

- The "CITY (Reuters) –" dateline, which opened 99.2% of real articles, was removed.
- 10 badly scraped rows, 631 empty texts and 5,624 duplicate texts were dropped.
- **Result:** 38,633 articles (45.2% fake).

### Features Excluded

- `subject` and `date` each give the label away, so neither is ever used. Every "politicsNews" article is real, and every 2015 article is fake.
- Curly quotes and other non-ASCII characters, which appear only in real articles, are replaced with spaces.
- Photo credits, links and "Reuters" inside article bodies were kept on purpose, so the explanations could show whether the models rely on them.

### Split

- A stratified 80/20 split gives 30,906 training articles and 7,727 test articles (3,489 fake).
- Duplicates were removed before splitting.
- 93 test articles (1.2%, all real) have a near-duplicate in training. Removing them changes accuracy by 0.0002.

## Evaluation Results

All results are on the same test set of 7,727 articles. The primary metric is Fake-class recall, because a missed fake is the costlier error for the reviewer. A model that labels every article real would score 54.8% accuracy.

| Model | Fake recall | Precision | Accuracy | ROC AUC | Missed fakes | False alarms |
|---|---|---|---|---|---|---|
| Logistic regression (recommended) | 0.969 | 0.987 | 0.980 | 0.998 | 108 | 44 |
| Decision tree, depth 3 | 0.923 | 0.834 | 0.883 | 0.936 | 269 | 639 |
| Random Forest | 0.935 | 0.990 | 0.966 | – | 228 | 33 |
| LightGBM | 0.966 | 0.990 | 0.980 | 0.998 | 119 | 35 |
| XGBoost | 0.965 | 0.990 | 0.980 | 0.998 | 122 | 33 |
| SVM, RBF (explained) | 0.975 | 0.989 | 0.984 | 0.998 | 87 | 39 |

- **The signal is real.** Logistic regression scores 0.980 in 5-fold cross-validation, against 0.518 on shuffled labels (permutation test p = 0.010, the minimum possible with 100 permutations).
- **The SVM's lead is real but small.**
  - Fake recall is 0.6 points higher (bootstrap 95% interval 0.3 to 0.9).
  - The two models disagree on 40 test articles, and the SVM is right on 33 of them (exact McNemar p = 4.2 × 10⁻⁵).
- **Why we recommend logistic regression anyway.** 21 extra catches out of 3,489 fakes do not outweigh exact, readable explanations. Logistic regression also puts the least weight of any model on source fingerprints: 14% of its LIME weight, against 20% for the SVM.
- **Source fingerprints are not the whole story.** Retraining logistic regression without 14 fingerprint words lowers accuracy only from 0.980 to 0.974.

### What the Models Rely On

- In every model, "said" pushes toward real, and "via", "image" and "video" push toward fake.
- The tree-based models put 33–38% of their LIME weight on source fingerprints, such as photo credits ("Featured image via Getty Images") and links.

## Explanation Quality

These checks were run on the SVM's explanations unless stated otherwise (`Model/XAI Evaluation/xai_evaluation.ipynb`).

| Check | Result |
|---|---|
| Faithfulness: LIME on logistic regression, where the true answer is known (50 articles) | Rank agreement 0.81; 84% of LIME's top 5 words in the true top 5; 99% correct direction |
| Faithfulness: KernelSHAP setup on logistic regression (50 articles) | Within 0.01 of the exact answer for 92% of articles |
| Faithfulness: LIME and TreeSHAP on the depth-3 tree (100 articles) | Median 99.4% of LIME's weight on the tree's 5 terms; TreeSHAP exactly 0 on the other 4,995 |
| Faithfulness: deleting words on the SVM (100 articles) | Deleting LIME's top words moves the score 19.5× as far as random words (0.85 vs 0.04); 31% vs 1% of predictions flip after 10 deletions |
| Stability: LIME across seeds (20 articles, 5 seeds) | Top-5 overlap 0.78 at 500 samples and 0.88 at 2,000; same top word 90% / 94% |
| Stability: KernelSHAP across seeds | Top-5 overlap 0.99 at 500 samples and 1.00 at 2,000 |
| Stability: small edits (5% of words deleted, SVM score moved < 0.05) | Top-5 overlap 0.80 for LIME and 0.95 for KernelSHAP; same top word 83% / 95% |
| Stability: grouped PFI across 5 seeds | Rankings vary (Spearman 0.51, top-5 overlap 0.45), because almost all SVM importances are near zero |
| Human evidence (Model/Human Study/human_study.ipynb) | 10 participants (5 per form), 100 answers: predicted the SVM correctly 68% of the time with LIME highlights vs 52% without; 90% vs 63% when the SVM was right, 35% vs 35% when it was wrong; no significance claimed |

**For the reviewer:** the top one or two words of a LIME explanation can be trusted. Words further down the list are a rough guide only. Any explanation shown to a reviewer should use at least 2,000 samples.

### Actionability

- Only editorial wording counts as actionable. Date and subject are not used, and deleting source fingerprints would only trick the model.
- **Wrong flags are fragile:** 64% of wrongly flagged real articles are cleared by deleting three editorial words or fewer.
- **Correct flags are not:** only 2.5% of correctly flagged fakes are cleared this way.

## Ethical Considerations

- **No personal data is used as features.** The dataset has no author, age, gender or location fields.
- **Personal names and handles are present.** Articles name public figures, and about a quarter of raw fake articles contain Twitter "@" handles. None were removed; we recommend masking handles before reuse.
- **Political opinion can act as a proxy.** Partisan words ("hillary", "gop") are among the top words that push toward fake. Political opinion is a special category under the GDPR (Article 9).
- **Errors on protected-group topics.** These counts are small, so we claim no significance.
  - Real articles that mention race are wrongly flagged 3.6% of the time (4 of 110), against 0.85% for other real articles.
  - Fake articles about immigration are missed 4.1% of the time (17 of 415), against 2.3%.
  - The photo-credit words that most signal "fake" appear together with race-related terms (phi up to 0.26).
- **Recommendation:** flags on articles about protected groups or partisan topics should get a human check before any action.

## Known Limitations

- **Labels describe sources, not truth.** The real class is a single wire service (Reuters), so the models learn to tell Reuters' style and layout from these fake sites' style and layout.
- **Known failure cases:**
  - A fake site republishing a wire-style report is missed.
  - A casually written real article can be flagged.
- **Narrow scope:**
  - Articles span 2015–2018 and are mostly about US politics.
  - Models trained on ISOT are known to lose much of their accuracy on other datasets.
- **Small evidence base:**
  - All results come from one train/test split.
  - The comparisons across models use 200 articles and one LIME setting.
  - The word groups cannot tell a word's uses apart (for example "image" in a photo credit and in normal text).
- **The fake sites' layout is a weakness.** Removing a photo credit is the quickest way for a fake article to get past the SVM.

## Reproducibility

- **Running order:** `combine_dataset.ipynb` → `clean_dataset.ipynb` → the model notebooks → `Model/XAI Synthesis/xai_synthesis.ipynb` → `Model/XAI Evaluation/xai_evaluation.ipynb`. `python run_all.py` runs them all in this order and saves every figure to `figure/`. The human study (`Model/Human Study/human_study.ipynb`) is run separately, because it needs real participants.
- **Environment:** Python 3.14.2. Dependencies are pinned in `requirements.txt`, including scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1, shap 0.53.0, lime 0.2.0.1 and nltk 3.10.3.
- **Seed:** `RANDOM_STATE = 42` everywhere.
- **SVM caching:** the SVM takes several minutes to train. It is saved to a `.joblib` file on first run; delete that file to retrain from scratch.

## Reference

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. In *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229).
