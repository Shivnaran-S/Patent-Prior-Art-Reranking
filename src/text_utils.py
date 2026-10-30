"""
TEXT PREPROCESSING  --  the very first box in any IR pipeline.

Raw text cannot be matched directly: "Comprising", "comprises" and "comprised"
are the same concept, and words like "the" carry no information.  The classic
pipeline is:

    raw text -> tokenise -> lowercase -> remove stopwords -> stem -> terms

Every lexical retriever in this project (TF-IDF, BM25) calls `analyze()`, so
query and document are guaranteed to be processed identically.  (If they were
not, a query term could never match its own document term -- the single most
common bug in hand-written IR systems.)
"""

from __future__ import annotations

import re
from functools import lru_cache

# --------------------------------------------------------------------------- #
# stopwords
# --------------------------------------------------------------------------- #
# Standard English function words.
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "s", "same", "she", "should", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves",
}

# DOMAIN stopwords.  This matters a lot for patents: legal boilerplate appears
# in almost every claim, so it has near-zero discriminative power but inflates
# similarity between unrelated patents.  Removing it is a real, measurable win
# and is the kind of domain adaptation generic search engines do not do.
PATENT_STOPWORDS = {
    "claim", "claims", "claimed", "said", "wherein", "whereby", "comprising",
    "comprises", "comprise", "comprised", "consisting", "consists", "according",
    "preceding", "characterized", "characterised", "thereof", "therein",
    "thereby", "therefore", "thereto", "herein", "hereof", "respectively",
    "substantially", "plurality", "least", "one", "first", "second", "third",
    "fourth", "means", "device", "apparatus", "method", "system", "invention",
    "embodiment", "embodiments", "present", "preferably", "optionally", "fig",
    "figs", "figure", "figures", "example", "shown", "described", "provided",
    "having", "includes", "including", "included", "wherefrom", "also", "may",
}

STOPWORDS = ENGLISH_STOPWORDS | PATENT_STOPWORDS

# Keep letters, digits and internal hyphens/dots so chemical and technical
# tokens survive: "poly-ethylene", "3.5", "co2".
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")

MIN_TOKEN_LEN = 2
MAX_TOKEN_LEN = 30


# --------------------------------------------------------------------------- #
# stemming
# --------------------------------------------------------------------------- #
try:                                            # nltk ships Porter with no data download
    from nltk.stem import PorterStemmer

    _porter = PorterStemmer()

    @lru_cache(maxsize=200_000)
    def stem(token: str) -> str:
        """Porter stemmer: 'amplification' -> 'amplif', so it matches 'amplifier'."""
        return _porter.stem(token)

except Exception:                               # pragma: no cover - offline fallback
    _SUFFIXES = ("ational", "ization", "iveness", "fulness", "ousness", "ability",
                 "tional", "alize", "icate", "ative", "ical", "ness", "ment",
                 "ing", "ers", "ed", "es", "ly", "s")

    @lru_cache(maxsize=200_000)
    def stem(token: str) -> str:
        for suf in _SUFFIXES:
            if len(token) - len(suf) >= 4 and token.endswith(suf):
                return token[: -len(suf)]
        return token


# --------------------------------------------------------------------------- #
# the public API
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    """Lowercase + split into candidate tokens. No stopwords/stemming yet."""
    return [
        t for t in TOKEN_RE.findall(text.lower())
        if MIN_TOKEN_LEN <= len(t) <= MAX_TOKEN_LEN
    ]


def analyze(text: str, remove_stopwords: bool = True, do_stem: bool = True) -> list[str]:
    """The full pipeline: text -> list of index terms.

    The two flags exist so the Ablation page can switch each step off and show
    what it was actually contributing.
    """
    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    # Purely numeric tokens are noise in patents ("1", "22", "175c" is kept
    # because it is alphanumeric, but bare reference numerals are dropped).
    tokens = [t for t in tokens if not t.isdigit()]
    if do_stem:
        tokens = [stem(t) for t in tokens]
    return tokens


def analyze_steps(text: str) -> dict[str, list[str]]:
    """Return the intermediate state after each stage.

    Used by the 'Concepts' page to show the preprocessing pipeline running
    live on whatever text the user types.
    """
    raw = tokenize(text)
    no_stop = [t for t in raw if t not in STOPWORDS]
    no_num = [t for t in no_stop if not t.isdigit()]
    stemmed = [stem(t) for t in no_num]
    return {
        "tokenized": raw,
        "stopwords_removed": no_stop,
        "numerals_removed": no_num,
        "stemmed": stemmed,
        "removed_as_stopword": [t for t in raw if t in STOPWORDS],
    }


def snippet(text: str, max_words: int = 45) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
