## Overview

This project applies explainable data science (XDS) techniques to a fake news detection model, with the goal of understanding *why* an article gets classified as fake or real rather than treating the classifier as a black box.

News text is first converted into numerical features using TF-IDF, then used to train both an interpretable baseline (logistic regression) and more complex models (e.g., Random Forest, XGBoost, CatBoost, EBM). We apply interpretability techniques including LIME, SHAP/TreeSHAP, PDP/ICE, and Permutation Feature Importance; to identify which words, phrases, and stylistic patterns drive each prediction, check whether these explanations are faithful and stable, and surface potential shortcuts or biases (e.g., topic or category acting as a proxy) the model may have learned instead of genuine markers of unreliable reporting.

The end goal is not just an accurate fake news classifier, but one whose decisions a human reviewer can trace, trust, and act on.