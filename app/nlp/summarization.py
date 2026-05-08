from __future__ import annotations

from collections import Counter
import math

from app.nlp.preprocessing import sentence_split, tokenize


def summarize(text: str, summarizer: object | None = None, max_sentences: int = 4) -> str:
    if summarizer and len(text.split()) > 80:
        generated = _transformer_summary(text, summarizer)
        if generated:
            return generated
    return _extractive_summary(text, max_sentences=max_sentences)


def _transformer_summary(text: str, summarizer: object) -> str | None:
    try:
        result = summarizer(text[:4000], max_length=130, min_length=35, do_sample=False)
        return result[0]["summary_text"].strip()
    except Exception:
        return None


def _extractive_summary(text: str, max_sentences: int) -> str:
    sentences = sentence_split(text.replace("\n", " "))
    if not sentences:
        return ""
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    words = [word for word in tokenize(text) if len(word) > 3]
    frequencies = Counter(words)
    if not frequencies:
        return " ".join(sentences[:max_sentences])

    scored: list[tuple[int, float, str]] = []
    for index, sentence in enumerate(sentences):
        tokens = tokenize(sentence)
        score = sum(frequencies[token] for token in tokens) / math.sqrt(len(tokens) or 1)
        clinical_bonus = 1.2 if any(word in sentence.lower() for word in ("pain", "fever", "medication", "doctor", "symptom")) else 1
        scored.append((index, score * clinical_bonus, sentence))

    selected = sorted(scored, key=lambda item: item[1], reverse=True)[:max_sentences]
    return " ".join(sentence for _, _, sentence in sorted(selected, key=lambda item: item[0]))
