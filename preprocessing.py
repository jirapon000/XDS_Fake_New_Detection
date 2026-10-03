"""
Shared preprocessing pipeline for Fake News Detection project.
All model notebooks (baseline + Tier 2) must import from here to ensure
identical preprocessing, TF-IDF settings, and train/test split across models.
"""

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Fixed seed used across the ENTIRE project — never change this
RANDOM_STATE = 42

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """Lowercase, remove punctuation/numbers, tokenize, remove stopwords, lemmatize."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]

    return " ".join(tokens)


def get_tfidf_features(df, max_features=5000, min_df=5, max_df=0.9, ngram_range=(1, 2)):
    """Fit TF-IDF on processed text. Returns X, y, and the fitted vectorizer."""
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )
    X = tfidf.fit_transform(df["processed_text"])
    y = df["label_encoded"]
    return X, y, tfidf


def get_train_test_split(X, y, test_size=0.2):
    """Standard split used by every model in this project."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )