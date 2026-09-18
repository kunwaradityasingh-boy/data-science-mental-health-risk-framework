import re

from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    """
    Perform basic text normalization.

    This is intentionally lightweight so that the original
    semantic information is largely preserved.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Replace URLs
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)

    # Replace Reddit-style user mentions
    text = re.sub(r"\bu/\w+\b", " USER ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Create the TF-IDF vectorizer used by the baseline model.
    """

    return TfidfVectorizer(
        preprocessor=clean_text,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def fit_tfidf(
    train_texts,
):
    """
    Fit TF-IDF only on training text.
    """

    vectorizer = build_tfidf_vectorizer()

    X_train = vectorizer.fit_transform(train_texts)

    return vectorizer, X_train


def transform_tfidf(
    vectorizer: TfidfVectorizer,
    texts,
):
    """
    Transform new text using an already-fitted vectorizer.
    """

    return vectorizer.transform(texts)