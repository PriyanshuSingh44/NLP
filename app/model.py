# model.py
# Loads TWO deep learning models at startup (only once):
#
#  1. SentenceTransformer (all-MiniLM-L6-v2)
#     → Converts sentences into 384-dim embedding vectors.
#     → Used to measure TOPICAL / SEMANTIC similarity.
#
#  2. HuggingFace Sentiment Pipeline (distilbert-base-uncased-finetuned-sst-2-english)
#     → Classifies each sentence as POSITIVE or NEGATIVE with a confidence score.
#     → Used to detect OPPOSITE SENTIMENTS (e.g. "I love it" vs "I hate it").
#
# By combining both models we get a score that is sensitive to
# both topic AND sentiment direction.

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ------------------------------------------------------------------
# Model 1: Sentence-BERT for semantic / topical similarity
# ------------------------------------------------------------------
SBERT_MODEL = "all-MiniLM-L6-v2"
print(f"[model.py] Loading SBERT model: {SBERT_MODEL} ...")
sbert_model = SentenceTransformer(SBERT_MODEL)
print(f"[model.py] SBERT model loaded ✅")

# ------------------------------------------------------------------
# Model 2: DistilBERT fine-tuned on SST-2 (sentiment classification)
# This small but powerful model classifies text as POSITIVE / NEGATIVE.
# It was trained on 67,000+ movie reviews and generalises well.
# ------------------------------------------------------------------
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
print(f"[model.py] Loading Sentiment model: {SENTIMENT_MODEL} ...")
sentiment_pipeline = pipeline(
    task="text-classification",
    model=SENTIMENT_MODEL,
    # Run on CPU (-1) ; change to 0 for GPU if available
    device=-1,
)
print(f"[model.py] Sentiment model loaded ✅")


def get_sbert_model() -> SentenceTransformer:
    """Returns the globally loaded SentenceTransformer instance."""
    return sbert_model


def get_sentiment_pipeline():
    """Returns the globally loaded HuggingFace sentiment pipeline."""
    return sentiment_pipeline
