# MediTalk: AI-Powered Medical Chat Analysis System

MediTalk is an enterprise-style AI/NLP backend that analyzes conversations between patients and healthcare providers. It extracts medical entities, detects symptoms, classifies urgency, summarizes long transcripts, identifies key topics, and stores predictions for later review.

> Educational disclaimer: this project demonstrates NLP engineering concepts and is not a medical device. It should not replace licensed clinical judgment, emergency triage protocols, or regulated decision support systems.

## Why This Project Was Built

MediTalk was built to demonstrate how artificial intelligence, natural language processing, and backend engineering can be combined to solve a realistic healthcare communication problem. In many healthcare environments, important patient details are buried inside long chat transcripts, support conversations, telehealth messages, or provider notes. Reading every message manually can be slow, repetitive, and difficult to scale.

The purpose of this project is to show how an AI-powered system can help organize medical conversations into structured, searchable, and actionable insights. It is designed as a portfolio-ready learning project for developers, data scientists, and students who want hands-on experience with real-world NLP concepts such as medical entity recognition, symptom extraction, urgency classification, summarization, confidence scoring, and API-based model deployment.

This project can be used to learn how to:

- Build a production-style AI backend with FastAPI
- Process unstructured medical chat text using NLP
- Extract symptoms, diseases, medications, body parts, and medical conditions
- Classify patient conversation urgency into `Low`, `Medium`, `High`, or `Emergency`
- Generate concise summaries from long conversations
- Store AI predictions for review, auditing, and future analytics
- Design a modular inference pipeline that can use rules, scikit-learn, spaCy, and transformer models

## What This Project Does

MediTalk accepts patient-provider chat messages through a REST API and returns a structured analysis of the conversation. For each transcript, the system:

- Cleans and preprocesses the text
- Converts messages into a speaker-aware transcript
- Extracts medical entities using a medical lexicon and optional spaCy models
- Identifies symptoms and whether they are negated, such as "no chest pain"
- Detects urgency using clinical red-flag rules or optional transformer classification
- Summarizes the conversation into a short clinical-style overview
- Extracts key topics using TF-IDF
- Produces confidence scores and explanation reasons
- Stores the analysis in a database for later retrieval

## Project Overview

Healthcare chat systems generate large volumes of unstructured clinical text. MediTalk turns those transcripts into structured insights:

- Concise conversation summaries
- Symptoms and negation awareness
- Medical Named Entity Recognition for symptoms, medications, diseases, body parts, and conditions
- Urgency classification: `Low`, `Medium`, `High`, `Emergency`
- Confidence scores and explanation reasons
- REST APIs for inference and retrieval
- PostgreSQL-ready persistence with SQLite default for easy local development
- Optional Redis caching
- Optional spaCy, Hugging Face Transformers, PyTorch, and Sentence Transformers integrations

The system works offline with deterministic fallback NLP rules. When model dependencies and model flags are enabled, it upgrades to transformer summarization, zero-shot classification, spaCy NER, and sentence embeddings.

## Architecture

```text
Client / Frontend / EHR Adapter
          |
          v
FastAPI REST Layer
          |
          v
Input Validation with Pydantic
          |
          v
ConversationAnalyzer
   |-- Preprocessing and tokenization
   |-- Medical entity extraction
   |-- Symptom and negation extraction
   |-- Urgency classification
   |-- Summarization
   |-- Key topic extraction
          |
          v
SQLAlchemy Persistence + Optional Redis Cache
          |
          v
PostgreSQL / SQLite
```

## Technology Stack

| Layer | Tools |
| --- | --- |
| API | FastAPI, Uvicorn, Pydantic |
| NLP | spaCy optional, Hugging Face Transformers optional, Sentence Transformers optional |
| ML | scikit-learn TF-IDF, rule-assisted classifiers, optional PyTorch-backed transformer pipelines |
| Database | SQLAlchemy, PostgreSQL, SQLite for local development |
| Cache | Redis optional |
| Quality | pytest, Ruff-ready configuration |
| Deployment | Docker, Docker Compose |

## NLP Concepts Demonstrated

- Text preprocessing and normalization
- Tokenization and sentence splitting
- Medical lexicon matching
- Named Entity Recognition
- Negation detection for symptoms
- Confidence scoring
- Extractive summarization
- Optional transformer summarization
- Zero-shot classification
- TF-IDF topic extraction
- Optional semantic embeddings
- Model registry and lazy model loading
- Inference pipeline design

## Folder Structure

```text
.
|-- app
|   |-- api
|   |   `-- routes.py
|   |-- core
|   |   |-- cache.py
|   |   |-- config.py
|   |   `-- logging.py
|   |-- models
|   |   `-- conversation.py
|   |-- nlp
|   |   |-- analyzer.py
|   |   |-- entities.py
|   |   |-- model_registry.py
|   |   |-- preprocessing.py
|   |   |-- summarization.py
|   |   |-- topics.py
|   |   `-- urgency.py
|   |-- schemas
|   |   `-- conversation.py
|   |-- db.py
|   |-- main.py
|   `-- services.py
|-- sample_data
|-- tests
|-- Dockerfile
|-- docker-compose.yml
|-- pyproject.toml
|-- requirements.txt
|-- requirements-nlp.txt
`-- README.md
```

## Installation

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For optional transformer and spaCy support:

```bash
pip install -r requirements-nlp.txt
python -m spacy download en_core_web_sm
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Default local development uses SQLite:

```env
DATABASE_URL=sqlite:///./meditalk.db
ENABLE_TRANSFORMERS=false
ENABLE_SPACY=false
```

To use PostgreSQL and Redis:

```env
DATABASE_URL=postgresql+psycopg://meditalk:meditalk@postgres:5432/meditalk
REDIS_URL=redis://redis:6379/0
```

To enable optional NLP models:

```env
ENABLE_TRANSFORMERS=true
ENABLE_SPACY=true
SPACY_MODEL=en_core_web_sm
SUMMARY_MODEL=sshleifer/distilbart-cnn-12-6
ZERO_SHOT_MODEL=facebook/bart-large-mnli
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Run the API

```bash
uvicorn app.main:app --reload
```

Open:

- API root docs: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Health check: `http://127.0.0.1:8000/api/v1/health`

## API Documentation

### Health Check

```http
GET /api/v1/health
```

Example response:

```json
{
  "status": "ok",
  "service": "MediTalk",
  "version": "0.1.0"
}
```

### Analyze a Conversation

```http
POST /api/v1/analyze
Content-Type: application/json
```

Example request:

```json
{
  "conversation_id": "case-2026-001",
  "messages": [
    {
      "speaker": "patient",
      "text": "I have crushing chest pain and shortness of breath. I feel dizzy."
    },
    {
      "speaker": "provider",
      "text": "Please call emergency services immediately. Are you sweating or nauseated?"
    }
  ]
}
```

Example response:

```json
{
  "id": "7b766d3e-91af-4d52-9563-8a3fb62eaa07",
  "conversation_id": "case-2026-001",
  "summary": "Patient: I have crushing chest pain and shortness of breath. I feel dizzy. Provider: Please call emergency services immediately.",
  "symptoms": [
    {
      "name": "chest pain",
      "severity": "crushing",
      "negated": false,
      "confidence": 0.88
    },
    {
      "name": "shortness of breath",
      "severity": null,
      "negated": false,
      "confidence": 0.88
    }
  ],
  "entities": [
    {
      "text": "chest pain",
      "label": "SYMPTOM",
      "start": 26,
      "end": 36,
      "confidence": 0.88,
      "source": "medical_lexicon"
    }
  ],
  "urgency": {
    "level": "Emergency",
    "confidence": 0.94,
    "reasons": ["chest pain with breathing difficulty"]
  },
  "key_topics": ["chest pain", "shortness breath"],
  "recommendations": [
    "Advise immediate emergency care or local emergency services.",
    "Escalate to a licensed clinician for urgent review."
  ],
  "model_info": {
    "ner": "rule_based_medical_lexicon",
    "summary": "extractive_textrank_fallback",
    "urgency": "clinical_rules_and_keyword_scoring",
    "embeddings": "tfidf_keywords"
  },
  "created_at": "2026-05-08T10:00:00Z"
}
```

### List Analyses

```http
GET /api/v1/conversations?limit=25
```

### Get One Analysis

```http
GET /api/v1/conversations/{analysis_id}
```

## Model Workflow

1. **Preprocessing**: messages are normalized and converted into a speaker-labeled transcript.
2. **NER**: the fallback medical lexicon identifies known symptoms, diseases, medications, body parts, and conditions. Optional spaCy NER adds general entity coverage.
3. **Symptom extraction**: symptom entities are enriched with simple severity and negation detection.
4. **Urgency classification**: clinical red-flag rules classify urgency. Optional zero-shot transformers can replace this path.
5. **Summarization**: the fallback summarizer ranks important sentences. Optional Hugging Face summarization generates abstractive summaries.
6. **Topic extraction**: TF-IDF extracts key terms and phrases for context-aware review.
7. **Persistence**: structured predictions and metadata are stored in SQL.

## Sample Data

Two sample payloads are included:

- `sample_data/conversation_emergency.json`
- `sample_data/conversation_routine.json`

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d @sample_data/conversation_emergency.json
```

## Training and Inference Flow

This repository focuses on inference-ready architecture. A realistic training extension would include:

1. Collect de-identified, consented medical chat transcripts.
2. Label spans for symptoms, medications, diseases, body parts, and conditions.
3. Label transcript-level urgency classes.
4. Train or fine-tune NER models using spaCy or transformer token classification.
5. Train urgency classifiers using scikit-learn baselines and transformer encoders.
6. Evaluate with precision, recall, F1, AUROC, calibration, and confusion matrices.
7. Register model versions and deploy through `ModelRegistry`.
8. Monitor drift, latency, and reviewer overrides.

Suggested datasets for learning and prototyping:

- i2b2 clinical NLP datasets, subject to access terms
- MIMIC-III / MIMIC-IV, subject to credentialing and data use agreements
- MedDialog dataset
- Synthetic medical chat examples for non-clinical demos

## Docker Setup

Create a `.env` file:

```env
APP_NAME=MediTalk
DATABASE_URL=postgresql+psycopg://meditalk:meditalk@postgres:5432/meditalk
REDIS_URL=redis://redis:6379/0
ENABLE_TRANSFORMERS=false
ENABLE_SPACY=false
```

Start services:

```bash
docker compose up --build
```

The API will be available at:

```text
http://127.0.0.1:8000/docs
```

## Testing

```bash
pip install -r requirements.txt
pip install pytest httpx
pytest
```

## Screenshots / Placeholders

Add project screenshots here when showcasing on GitHub:

- Swagger UI with `/api/v1/analyze`
- Example JSON response
- Database table view
- Architecture diagram
- Optional frontend dashboard

## Deployment Notes

For production-like deployments:

- Use PostgreSQL instead of SQLite.
- Use Redis for repeated transcript analysis caching.
- Put the API behind a reverse proxy or managed API gateway.
- Configure TLS, request limits, authentication, and audit logs.
- Store secrets in a secret manager.
- Pin model versions and dependencies.
- Run model inference on GPU-enabled workers if using large transformer models.
- Add observability: latency, error rates, token counts, model confidence, and drift.
- Add PHI handling controls and compliance review before real healthcare use.

## Future Plans and Enhancements

MediTalk is designed as a strong foundation for a more advanced AI/NLP healthcare platform. Future improvements can expand the system from a backend prototype into a more complete, scalable, and clinically useful application.

- **Fine-tuned medical urgency classifier**: train BioClinicalBERT, PubMedBERT, ClinicalBERT, or similar transformer models on labeled medical conversations for more accurate urgency prediction.
- **Advanced medical NER**: replace or enrich the rule-based lexicon with a transformer token-classification model trained to detect symptoms, medications, diseases, procedures, allergies, and lab concepts.
- **Better negation and context detection**: integrate NegEx, medspaCy, or custom clinical context rules to distinguish confirmed symptoms from denied symptoms, family history, past history, and hypothetical concerns.
- **FHIR-compatible schemas**: map analysis results to healthcare interoperability formats such as FHIR `Condition`, `MedicationStatement`, `Observation`, and `Communication`.
- **Role-aware summarization**: generate separate summaries for patient concerns, provider recommendations, medication instructions, and follow-up actions.
- **Human review dashboard**: add a clinician or reviewer interface where predictions can be accepted, corrected, flagged, or escalated.
- **Asynchronous inference jobs**: move long-running transformer inference to background workers using Celery, Redis Queue, or FastAPI background tasks.
- **Model versioning and experiment tracking**: add model registry support, evaluation reports, A/B testing, and rollback strategies for safer ML deployment.
- **Authentication and authorization**: add JWT/OAuth2 login, role-based access control, tenant isolation, and API keys.
- **PHI de-identification pipeline**: detect and mask names, phone numbers, dates, addresses, IDs, and other sensitive health information before storage or model processing.
- **Analytics and monitoring**: track urgency trends, common symptoms, model confidence, latency, API errors, and prediction drift over time.
- **Frontend dashboard**: build a web interface for uploading transcripts, viewing extracted entities, filtering by urgency, and reviewing historical analyses.
- **Deployment hardening**: add Kubernetes manifests, CI/CD pipelines, managed PostgreSQL/Redis setup, observability, rate limiting, and secure secret management.

## Contribution Guidelines

1. Fork the repository.
2. Create a feature branch.
3. Add or update tests for behavior changes.
4. Run the test suite.
5. Open a pull request with a clear description, screenshots if relevant, and implementation notes.

## Learning Outcomes

By studying or extending MediTalk, developers can learn how to:

- Build modular FastAPI services
- Design production-style NLP inference pipelines
- Combine rules, scikit-learn, and transformers
- Validate structured API inputs with Pydantic
- Persist AI predictions with SQLAlchemy
- Add optional Redis caching
- Design model registries and fallback paths
- Document AI projects for GitHub portfolios
- Think about safety, confidence, and deployment constraints in medical NLP

## License

Add your preferred license before publishing. MIT is a common choice for portfolio projects.
