"""Lightweight language detection (C-087): script detection + stopword scoring.

Covers the languages most common in the corpus (en, fr, es, pt, de, it, ru, ar, zh).
For broader coverage set LANGUAGE_DETECTION=azure (Azure AI Language) — TODO hook.
"""

import re

_STOPWORDS = {
    "en": "the and of to in is that for it with as on are this be by from or",
    "fr": "le la les et des du un une est pour que dans en sur pas au par",
    "es": "el la los las y de que en un una es por para con del se",
    "pt": "o a os as e de que em um uma para com do da não por",
    "de": "der die das und ist nicht ein eine zu den mit von für auf im",
    "it": "il la le e di che un una per con del della non sono è",
}
_STOP = {lang: set(words.split()) for lang, words in _STOPWORDS.items()}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(text: str) -> str:
    sample = text[:5000]
    letters = [c for c in sample if c.isalpha()]
    if not letters:
        return ""
    total = len(letters)
    if sum("؀" <= c <= "ۿ" for c in letters) / total > 0.3:
        return "ar"
    if sum("一" <= c <= "鿿" for c in letters) / total > 0.3:
        return "zh"
    if sum("Ѐ" <= c <= "ӿ" for c in letters) / total > 0.3:
        return "ru"
    words = [w.lower() for w in _WORD.findall(sample)]
    if not words:
        return ""
    scores = {lang: sum(w in stop for w in words) for lang, stop in _STOP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else ""
