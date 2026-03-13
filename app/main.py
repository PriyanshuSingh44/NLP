# main.py
# Entry point for the FastAPI application.
# Sets up the app, configures static files, Jinja2 templates,
# registers the API routes, and serves the frontend HTML page.

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router  # Import our API route definitions

# -------------------------------------------------------------------
# Application Setup
# -------------------------------------------------------------------

# Create the FastAPI application instance with metadata for the docs
app = FastAPI(
    title="Semantic Similarity API",
    description="Computes semantic similarity between two sentences using Sentence-BERT (all-MiniLM-L6-v2).",
    version="1.0.0",
)

# Allow all origins for development. Restrict in production environments.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the /static directory so the browser can load CSS, JS, etc.
# Any file in the 'static' folder is accessible at /static/<filename>
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template rendering from the 'templates' folder
templates = Jinja2Templates(directory="templates")

# Register all API routes from routes.py (prefixed at root level)
app.include_router(router)


# -------------------------------------------------------------------
# Frontend Route
# -------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_ui(request: Request):
    """
    Serves the main HTML page (index.html) at the root URL.
    The 'request' object is passed to the template so Jinja2
    can render dynamic content if needed.
    """
    return templates.TemplateResponse("index.html", {"request": request})
