## Overview

This project applies explainable data science (XDS) techniques to fake news detection. The goal is to understand *why* a model classifies an article as fake or real, rather than treating it as a black box, and to check whether those reasons hold up.

Each article's title and text are converted into TF-IDF word features. Author, source and category are deliberately left out, so the models judge what an article says rather than who published it. We train an interpretable baseline (logistic regression), a shallow decision tree that can be read in full, and four more complex models: Random Forest, LightGBM, XGBoost and an RBF-kernel SVM. All models share the same features and train/test split and are evaluated on Fake-class recall, the share of fake articles caught.

Each model is explained with the same pipeline:
- **Global importance:** Grouped Permutation Feature Importance (with a random-term check to separate real effects from noise) and SHAP (TreeSHAP for tree models, KernelSHAP for the SVM)
- **Feature effects:** ALE, chosen over ALE after a correlation check showed the top terms are independent
- **Heterogeneity:** ICE and centered ICE
- **Individual decisions:** LIME, SP-LIME and SHAP waterfalls, compared on the same articles

The readable decision tree also serves as a test case: because its true rules are known, we can check whether the explanation methods report them correctly before trusting them on models we cannot read.

The main finding is that no model gets beyond about 52% accuracy, and the explanations show why. Every model relies on small pushes from ordinary words, the methods agree on the direction of the main terms across models, and no word acts as a reliable marker of fake reporting. A check of the excluded metadata found every source and category close to 50/50 fake. The project therefore shows explainability doing a different job from the one we expected: instead of explaining a working classifier, it reveals that the dataset carries almost no learnable signal, which a reviewer looking only at accuracy might miss.