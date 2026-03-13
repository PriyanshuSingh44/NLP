# routes.py
# API route definitions for the FastAPI application.
# Updated to return detailed results from the two-model pipeline:
#   - topical_score   (SBERT cosine similarity)
#   - adjusted_score  (penalised for opposite sentiments)
#   - sentiment info  (label + confidence for each sentence)

from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.similarity import compute_similarity, interpret_score

router = APIRouter()


# ------------------------------------------------------------------
# Request Schema
# ------------------------------------------------------------------

class SimilarityRequest(BaseModel):
    """Incoming POST body: two sentences to compare."""
    sentence1: str = Field(..., min_length=1, description="First sentence.")
    sentence2: str = Field(..., min_length=1, description="Second sentence.")


# ------------------------------------------------------------------
# Response Schema
# ------------------------------------------------------------------

class SentimentInfo(BaseModel):
    """Sentiment classification result for one sentence."""
    label: str  = Field(..., description="'POSITIVE' or 'NEGATIVE'")
    score: float = Field(..., description="Model confidence (0–1)")


class SimilarityResponse(BaseModel):
    """Full API response with raw + adjusted scores and sentiment info."""

    # ── Primary result ──────────────────────────────────────────
    adjusted_score:      float  = Field(..., description="Sentiment-aware final score (0–1).")
    interpretation:      str    = Field(..., description="Human-readable label for adjusted score.")

    # ── Breakdown ───────────────────────────────────────────────
    topical_score:       float  = Field(..., description="Raw SBERT cosine similarity (topic only).")
    sentiment_agreement: float  = Field(..., description="Sentiment alignment factor (0–1).")

    # ── Per-sentence sentiment ───────────────────────────────────
    sentiment1: SentimentInfo = Field(..., description="Sentiment of sentence 1.")
    sentiment2: SentimentInfo = Field(..., description="Sentiment of sentence 2.")

    # ── Echo inputs ─────────────────────────────────────────────
    sentence1: str = Field(..., description="Echo of input sentence 1.")
    sentence2: str = Field(..., description="Echo of input sentence 2.")


# ------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------

@router.post("/similarity", response_model=SimilarityResponse)
async def get_similarity(request: SimilarityRequest):
    """
    POST /similarity

    Computes a sentiment-aware semantic similarity score using two models:
      1. SBERT (all-MiniLM-L6-v2) for topical similarity.
      2. DistilBERT (sst-2) for sentiment polarity.

    The final score penalises sentences with opposite sentiments,
    so "I love chocolate" vs "I hate chocolate" correctly scores LOW.

    Example response:
    {
        "adjusted_score": 0.02,
        "interpretation": "Not Similar",
        "topical_score": 0.7972,
        "sentiment_agreement": 0.02,
        "sentiment1": {"label": "POSITIVE", "score": 0.9998},
        "sentiment2": {"label": "NEGATIVE", "score": 0.9997},
        "sentence1": "I like chocolate",
        "sentence2": "I hate chocolate"
    }
    """
    # Run the two-model pipeline
    result = compute_similarity(request.sentence1, request.sentence2)

    # Interpret the ADJUSTED (sentiment-aware) score
    interpretation = interpret_score(result["adjusted_score"])

    return SimilarityResponse(
        adjusted_score      = result["adjusted_score"],
        interpretation      = interpretation,
        topical_score       = result["topical_score"],
        sentiment_agreement = result["sentiment_agreement"],
        sentiment1          = SentimentInfo(**result["sentiment1"]),
        sentiment2          = SentimentInfo(**result["sentiment2"]),
        sentence1           = request.sentence1,
        sentence2           = request.sentence2,
    )
