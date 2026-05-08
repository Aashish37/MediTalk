from sklearn.feature_extraction.text import TfidfVectorizer

from app.nlp.preprocessing import sentence_split


def extract_key_topics(text: str, max_topics: int = 8) -> list[str]:
    documents = sentence_split(text)
    if not documents:
        return []
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=60,
    )
    matrix = vectorizer.fit_transform(documents)
    scores = matrix.sum(axis=0).A1
    features = vectorizer.get_feature_names_out()
    ranked = sorted(zip(features, scores), key=lambda item: item[1], reverse=True)
    return [term for term, _ in ranked[:max_topics]]
