# 🧠 Semantic Similarity Measurement Using Pretrained Word Embeddings

A full-stack NLP web application that measures the **semantic similarity** between two sentences using **Sentence-BERT** (all-MiniLM-L6-v2) and **cosine similarity**. Built with **FastAPI** (backend) and **Vanilla HTML/CSS/JS** (frontend).

![Python](https://img.shields.io/badge/🐍_Python-3.12-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/⚡_FastAPI-Framework-009688?style=flat-square)
![Sentence Transformers](https://img.shields.io/badge/🤗_Sentence--BERT-all--MiniLM--L6--v2-orange?style=flat-square)
![scikit-learn](https://img.shields.io/badge/🔬_scikit--learn-ML-F7931E?style=flat-square)
![Docker](https://img.shields.io/badge/🐳_Docker-Containerized-2496ED?style=flat-square)
![NLP](https://img.shields.io/badge/🧠-NLP-purple?style=flat-square)

---

## 📌 Table of Contents

1. [What is Semantic Similarity?](#what-is-semantic-similarity)
2. [How Pretrained Embeddings Work](#how-pretrained-embeddings-work)
3. [How the Model is Used](#how-the-model-is-used)
4. [Project Structure](#project-structure)
5. [How to Run Locally](#how-to-run-locally)
6. [🐳 Run with Docker](#-run-with-docker)
7. [API Reference](#api-reference)
8. [Score Interpretation](#score-interpretation)
9. [Technologies Used](#technologies-used)

---

## 📖 What is Semantic Similarity?

**Semantic similarity** is the measure of how alike two pieces of text are in **meaning**, not just in the words they use.

For example:
| Sentence 1 | Sentence 2 | Similarity |
|---|---|---|
| "The dog barked loudly." | "The canine made a loud sound." | **High** – same meaning, different words |
| "I love pizza." | "The stock market crashed." | **Low** – completely unrelated topics |

Traditional keyword-based methods (like counting shared words) fail to capture this nuance. Deep learning models solve this by learning rich **semantic representations** of language.

---

## 🔡 How Pretrained Embeddings Work

A **word/sentence embedding** is a dense numerical vector (array of numbers) that represents the meaning of text in a mathematical space. The key idea is:

> **"Similar meanings → Similar vectors"**

### Training Process (simplified)
1. A large language model (like BERT) is trained on billions of sentences from the Internet.
2. During training, the model learns to place semantically related sentences **close together** in a high-dimensional vector space.
3. After training, we can **encode** any new sentence into this space without retraining.

### Sentence-BERT (SBERT)
- **SBERT** is a modification of the original BERT model, fine-tuned using **Siamese networks** to produce meaningful sentence-level embeddings.
- The model used here — `all-MiniLM-L6-v2` — maps sentences to **384-dimensional vectors**.
- It is compact, fast, and achieves excellent performance on semantic textual similarity (STS) benchmarks.

---

## 🤖 How the Model is Used

```
User Input (2 sentences)
        │
        ▼
  [FastAPI Backend]
        │
        ▼
  SentenceTransformer.encode(sentence1)  →  Vector1 (384 dims)
  SentenceTransformer.encode(sentence2)  →  Vector2 (384 dims)
        │
        ▼
  cosine_similarity(Vector1, Vector2)
        │
        ▼
  Score ∈ [0.0, 1.0]
        │
        ▼
  Interpretation + JSON Response
        │
        ▼
  [Frontend displays Result]
```

### Cosine Similarity Formula

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

- A value of **1.0** means the vectors point in the exact same direction → **identical meaning**.
- A value of **0.0** means the vectors are perpendicular → **completely unrelated**.

---

## 📁 Project Structure

```
nlp project/
│
├── app/
│   ├── __init__.py        # Marks 'app' as a Python package
│   ├── main.py            # FastAPI app setup, static files, template serving
│   ├── model.py           # Loads the SentenceTransformer model (once)
│   ├── similarity.py      # Core logic: encode sentences, compute cosine similarity
│   └── routes.py          # API route definitions (/similarity endpoint)
│
├── static/
│   └── style.css          # CSS: dark glassmorphism theme, animations
│
├── templates/
│   └── index.html         # Frontend: HTML + Vanilla JS (uses fetch API)
│
├── Dockerfile             # Docker image definition
├── .dockerignore          # Files excluded from the Docker build context
├── requirements.txt       # Python package dependencies
├── run.bat                # One-click startup script (Windows)
└── README.md              # This file
```

---

## 🚀 How to Run Locally

### Prerequisites
- Python **3.9** or higher
- `pip` package manager

### Step 1 – Clone / Download the Project

If you cloned from a repository:
```bash
git clone <your-repo-url>
cd "nlp project"
```

Or navigate to the project directory directly.

### Step 2 – Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 – Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** The first install will download the `all-MiniLM-L6-v2` model (~90 MB) automatically from Hugging Face. Ensure you have an internet connection.

> 💡 **For CPU-only PyTorch** (smaller download):
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### Step 4 – Run the Application

**Windows (one-click script):**
```bat
.\run.bat
```

**Windows / Linux / macOS (manual):**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 5 – Open in Browser

Navigate to: **[http://localhost:8000](http://localhost:8000)**

The interactive API documentation (Swagger UI) is also available at:  
**[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🐳 Run with Docker

Docker lets you run the app in an isolated container — **no Python installation or virtual environment needed**.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### Step 1 – Build the Docker Image

```bash
docker build -t nlp-similarity-app .
```

> ⚠️ **Note:** The first build will download the `all-MiniLM-L6-v2` model (~90 MB) from Hugging Face inside the container. Ensure you have an internet connection.

### Step 2 – Run the Container

```bash
docker run -p 8000:8000 nlp-similarity-app
```

This maps port **8000** on your machine to port **8000** inside the container.

### Step 3 – Open in Browser

Navigate to: **[http://localhost:8000](http://localhost:8000)**

> 💡 **Tip:** To run the container in the background (detached mode), use:
> ```bash
> docker run -d -p 8000:8000 --name nlp-app nlp-similarity-app
> ```
> Stop it later with: `docker stop nlp-app`

### What's in the Docker Setup?

| File | Purpose |
|---|---|
| `Dockerfile` | Defines the image: Python 3.12-slim base, installs deps, copies code, starts Uvicorn |
| `.dockerignore` | Excludes `.venv/`, `__pycache__/`, model binaries, and `.git` from the build context |

---

## 📡 API Reference

### `POST /similarity`

Computes the semantic similarity between two sentences.

**Request Body (JSON):**
```json
{
  "sentence1": "The cat sat on the mat.",
  "sentence2": "A cat was resting on a rug."
}
```

**Response Body (JSON):**
```json
{
  "score": 0.8731,
  "interpretation": "Highly Similar",
  "sentence1": "The cat sat on the mat.",
  "sentence2": "A cat was resting on a rug."
}
```

**Example using curl:**
```bash
curl -X POST http://localhost:8000/similarity \
     -H "Content-Type: application/json" \
     -d '{"sentence1": "I love machine learning", "sentence2": "I enjoy deep learning"}'
```

---

## 🎯 Score Interpretation

| Score Range | Label | Meaning |
|---|---|---|
| **0.8 – 1.0** | 🟢 Highly Similar | Sentences convey the same or very similar meaning |
| **0.5 – 0.8** | 🟡 Moderately Similar | Some overlap in meaning or topic |
| **0.0 – 0.5** | 🔴 Not Similar | Sentences are largely unrelated |

---

## 🛠️ Technologies Used

| Layer | Technology | Purpose |
|---|---|---|
| Backend | [FastAPI](https://fastapi.tiangolo.com/) | REST API framework |
| Server | [Uvicorn](https://www.uvicorn.org/) | ASGI server |
| NLP Model | [Sentence Transformers](https://www.sbert.net/) | Sentence embeddings |
| Pretrained Model | `all-MiniLM-L6-v2` | 384-dim dense vectors |
| Similarity Metric | [scikit-learn](https://scikit-learn.org/) | Cosine similarity |
| Templating | [Jinja2](https://jinja.palletsprojects.com/) | HTML rendering |
| Frontend | HTML5 + CSS3 + Vanilla JS | User interface |
| Validation | [Pydantic](https://docs.pydantic.dev/) | Request/response schemas |
| Containerization | [Docker](https://www.docker.com/) | Portable, dependency-free deployment |

---

## 👨‍💻 Author Notes

This project is designed to be **beginner-friendly**:
- Every file is heavily commented for clarity.
- The model is loaded **only once** at startup (not per request), making it efficient.
- The frontend communicates with the backend via the `fetch()` API, demonstrating how modern SPAs work without a page reload.
- The cosine similarity is computed using **scikit-learn**, which is a standard, well-tested library.

---

*Built with ❤️ using Python, FastAPI, and Sentence-BERT.*
