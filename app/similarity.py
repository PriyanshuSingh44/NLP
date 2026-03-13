# similarity.py
# Core computation logic combining TWO deep learning models:
#
#  STEP 1 → SBERT encodes both sentences into vectors.
#  STEP 2 → Cosine similarity gives the TOPICAL similarity score.
#  STEP 3 → DistilBERT classifies the SENTIMENT of each sentence.
#  STEP 4 → If sentiments are OPPOSITE, we apply a penalty.
#  STEP 5 → Final score = semantic_score × sentiment_agreement_factor
#
# This way, "I like chocolate" vs "I hate chocolate" will correctly
# get a LOW score despite sharing the same topic.

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.model import get_sbert_model, get_sentiment_pipeline


# ------------------------------------------------------------------
# Sentiment Analysis Helper
# ------------------------------------------------------------------

def get_sentiment(text: str) -> dict:
    """
    Runs the DistilBERT sentiment classifier on a single sentence.

    Returns a dict like:
        { "label": "POSITIVE", "score": 0.9987 }
    where 'score' is the model's confidence (0 to 1).

    Labels produced by this model:
        "POSITIVE" → the sentence expresses a positive sentiment
        "NEGATIVE" → the sentence expresses a negative sentiment
    """
    pipe = get_sentiment_pipeline()
    # The pipeline returns a list; we take the first (and only) result
    result = pipe(text, truncation=True, max_length=512)[0]
    return result   # e.g. {"label": "POSITIVE", "score": 0.9987}


# ------------------------------------------------------------------
# Main Similarity Function
# ------------------------------------------------------------------

def compute_similarity(sentence1: str, sentence2: str) -> dict:
    """
    Computes a SENTIMENT-AWARE semantic similarity score.

    Algorithm:
    ──────────────────────────────────────────────────────────────
    1. Encode both sentences with SBERT → 384-dim vectors
    2. Compute cosine similarity → topical_score ∈ [0, 1]
    3. Classify sentiment of each sentence with DistilBERT
    4. Compute sentiment_agreement:
         • Same label  (POS-POS or NEG-NEG) → agreement = 1.0 (no penalty)
         • Opposite    (POS-NEG or NEG-POS) → agreement = 1 − (conf1 × conf2)
           This means: the MORE confident both models are about opposite
           sentiments, the STRONGER the penalty.
    5. adjusted_score = topical_score × sentiment_agreement
    ──────────────────────────────────────────────────────────────

    Example:
        "I like chocolate" vs "I hate chocolate"
        topical_score     ≈ 0.79  (same topic)
        sentiment1        = POSITIVE (conf 0.99)
        sentiment2        = NEGATIVE (conf 0.99)
        sentiment_agreement = 1 − (0.99 × 0.99) ≈ 0.02
        adjusted_score    ≈ 0.79 × 0.02 ≈ 0.016  ← correctly LOW ✅

    Args:
        sentence1 (str): First input sentence.
        sentence2 (str): Second input sentence.

    Returns:
        dict with keys:
            topical_score (float)       – raw SBERT cosine similarity
            adjusted_score (float)      – sentiment-penalised final score
            sentiment1 (dict)           – label + confidence for sentence1
            sentiment2 (dict)           – label + confidence for sentence2
            sentiment_agreement (float) – how aligned the sentiments are
    """
    # ── STEP 1 & 2: SBERT semantic similarity ──────────────────────
    sbert = get_sbert_model()
    embeddings = sbert.encode([sentence1, sentence2])

    vec1 = embeddings[0].reshape(1, -1)
    vec2 = embeddings[1].reshape(1, -1)

    topical_score = float(np.clip(cosine_similarity(vec1, vec2)[0][0], 0.0, 1.0))

    # ── STEP 3: Sentiment classification ───────────────────────────
    sentiment1 = get_sentiment(sentence1)   # {"label": "POSITIVE", "score": 0.99}
    sentiment2 = get_sentiment(sentence2)   # {"label": "NEGATIVE", "score": 0.99}

    # ── STEP 4: Sentiment agreement factor ─────────────────────────
    if sentiment1["label"] == sentiment2["label"]:
        # Same sentiment direction → no penalty at all
        sentiment_agreement = 1.0
    else:
        # Opposite sentiments → penalise based on how CONFIDENT both models are
        # If conf1=0.99 and conf2=0.99 → penalty = 0.98 → agreement ≈ 0.02
        # If conf1=0.55 and conf2=0.55 → penalty = 0.30 → agreement ≈ 0.70 (mild)
        opposition_strength = sentiment1["score"] * sentiment2["score"]
        sentiment_agreement = float(np.clip(1.0 - opposition_strength, 0.0, 1.0))

    # ── STEP 5: Final adjusted score ────────────────────────────────
    adjusted_score = float(np.clip(topical_score * sentiment_agreement, 0.0, 1.0))

    return {
        "topical_score":       round(topical_score, 4),
        "adjusted_score":      round(adjusted_score, 4),
        "sentiment1":          sentiment1,
        "sentiment2":          sentiment2,
        "sentiment_agreement": round(sentiment_agreement, 4),
    }


# ------------------------------------------------------------------
# Interpretation Helper
# ------------------------------------------------------------------

def interpret_score(score: float) -> str:
    """
    Returns a human-readable label for a similarity score.

    Args:
        score (float): Final similarity score in [0, 1].

    Returns:
        str: One of 'Highly Similar', 'Moderately Similar', 'Not Similar'.
    """
    if score >= 0.8:
        return "Highly Similar"
    elif score >= 0.5:
        return "Moderately Similar"
    else:
        return "Not Similar"
