<div align="center">

# BERTokenScope

### Visualizing How Transformers Understand Language, Context, and Financial Text

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python">
  <img src="https://img.shields.io/badge/CS50AI-Attention%20Enhanced-red">
  <img src="https://img.shields.io/badge/BERT-Masked%20Language%20Modeling-yellow?logo=huggingface">
  <img src="https://img.shields.io/badge/Transformers-Explainability-purple?logo=huggingface">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit">
  <img src="https://img.shields.io/badge/FastAPI-Service-009688?logo=fastapi">
  <img src="https://img.shields.io/badge/PyTorch-Inference-EE4C2C?logo=pytorch">
  <img src="https://img.shields.io/badge/Finance%20NLP-Analytics-green">
  <img src="https://img.shields.io/badge/Model%20Comparison-BERT%20%7C%20FinBERT%20%7C%20RoBERTa-4B0082" alt="Model Comparison">
  <img src="https://img.shields.io/badge/Explainable%20AI-Token%20Insights-8A2BE2" alt="Explainable AI">
</p>

<img src="assets/bertokenscope-banner.png" alt="BERTokenScope Banner" width="100%">

<p align="center">
  <b>Transformer Attention • Masked Token Prediction • Financial NLP • Embedding Intelligence • Model Explainability</b>
</p>

</div>

---

## 🚀 Executive Summary

**BERTokenScope** is a transformer explainability and NLP intelligence platform built as an enhanced, production-style extension of the **Harvard CS50AI Attention** project.

The original CS50AI Attention assignment focuses on using BERT to predict masked words and generate static attention diagrams. BERTokenScope takes that foundation and expands it into a full portfolio-grade system for exploring how transformer models understand language, context, token relationships, and finance-specific text signals.

At its core, BERTokenScope answers one powerful question:

> **How does a transformer decide what words matter?**

Instead of treating BERT like a black box, this project opens the model up.

It helps users inspect:

- which tokens receive the most attention
- how attention changes across layers and heads
- what BERT predicts for masked words
- how financial tone, risk, uncertainty, and executive language appear in text
- how sentence/document embeddings can be compared in semantic space
- how different transformer models behave on the same input
- how removing important tokens changes model outputs

This project combines **CS50AI foundations**, **university-level machine learning and deep learning concepts**, and my broader experience building full-stack, AI, cloud, and software engineering projects.

The result is not just a course assignment.

It is an **AI interpretability platform**.

---

# 🧠 Why This Project Exists

Modern AI systems are powered by transformers.

Large language models, search systems, summarizers, chatbots, coding assistants, recommendation systems, and AI agents all rely on the same core idea:

> Tokens should not be understood alone.  
> Tokens should be understood in context.

That is where **attention** comes in.

Attention allows a model to decide which words in a sequence matter most when understanding another word. For example, in a sentence like:

```text
The company reduced guidance because demand weakened.
```

A transformer might pay strong attention between:

- `reduced` and `guidance`
- `demand` and `weakened`
- `because` and the explanation that follows

For financial text, this becomes especially useful.

Earnings calls, investor reports, filings, and analyst transcripts often contain subtle signals. A model may pick up risk language, uncertainty, confidence, caution, or forward-looking sentiment.

BERTokenScope was built to make those signals visible.

---

# 🎓 CS50AI Attention → BERTokenScope

## Original CS50AI Attention Project

The original CS50AI Attention project asks students to use BERT for masked language modeling.

The course version includes:

- loading a BERT masked language model
- identifying the `[MASK]` token
- predicting likely replacement words
- generating attention diagrams
- analyzing attention heads
- writing observations about what different heads appear to learn

The project is a strong introduction to transformer internals.

But it is intentionally limited in scope.

It focuses mainly on:

- static scripts
- local inference
- attention image generation
- manual analysis

---

## Enhanced Version: BERTokenScope

BERTokenScope expands that foundation into a more complete and realistic NLP platform.

It adds:

- an interactive Streamlit dashboard
- a FastAPI backend
- structured NLP services
- financial text intelligence
- embedding exploration
- explainability reports
- counterfactual token analysis
- model comparison tools
- runtime benchmarking
- local run tracking
- Docker Compose support
- API versioning
- safe fallback behavior
- testable components that do not require model downloads

In other words:

> CS50AI Attention taught the foundation.  
> BERTokenScope turns that foundation into an industry-style AI system.

---

# ✨ What BERTokenScope Does

## 1. Attention Explorer

The Attention Explorer helps inspect transformer attention layer by layer and head by head.

It allows users to study how tokens attend to one another across the model.

### It can help answer:

- Which tokens are receiving the strongest attention?
- Which token relationships dominate a specific layer?
- Which attention heads appear interpretable?
- Do certain heads focus on nearby tokens?
- Do certain heads focus on important domain-specific words?

### Example Use Cases

- Explore how BERT attends to verbs and objects.
- Inspect how financial risk words connect to surrounding context.
- Compare attention patterns between neutral and negative statements.
- Use attention as a teaching tool for transformer internals.

---

## 2. Masked Word Lab

The Masked Word Lab uses BERT-style masked language modeling.

Users provide a sentence containing `[MASK]`, and the model predicts the most likely replacement words.

### Example

```text
The company reported strong [MASK] growth this quarter.
```

Possible predictions might include:

```text
revenue
sales
earnings
profit
```

### Why This Matters

Masked language modeling helps show how BERT understands context.

The model is not just guessing a random word.

It is using surrounding tokens to infer what word best fits the sentence.

This is the same foundational idea behind many modern NLP systems.

---

## 3. Token Relationship Analysis

BERTokenScope identifies strong token-to-token attention links.

Instead of only showing a heatmap, it extracts meaningful relationships between tokens.

### Example Relationships

```text
risk → increased
revenue → declined
guidance → lowered
demand → weakened
margin → compressed
```

This makes attention easier to understand.

A heatmap is useful.

But a ranked list of token relationships is faster to interpret.

---

## 4. Financial NLP Intelligence

BERTokenScope includes finance-aware text analysis features.

It can inspect financial language for:

- sentiment
- risk language
- uncertainty
- executive tone
- forward-looking statements
- positive business signals
- negative business signals
- cautious or defensive wording

### Example Financial Signals

```text
Revenue increased, but management warned of margin pressure and weaker demand.
```

BERTokenScope can surface signals like:

- positive: `revenue increased`
- risk: `margin pressure`
- negative demand: `weaker demand`
- cautious tone: `warned`

This makes the project more than a general NLP demo.

It becomes useful for financial text intelligence.

---

## 5. Transcript Drift Analysis

Financial communication changes over time.

A company might sound confident in one quarter and cautious in the next.

BERTokenScope includes transcript drift analysis ideas for comparing tone across periods.

### Example Comparison

```text
Q1: "We expect strong growth across all segments."
Q2: "We remain cautious due to demand uncertainty."
```

The system can help compare:

- tone change
- risk language increase
- uncertainty increase
- sentiment drift
- executive confidence shift

This is especially useful for:

- earnings call analysis
- investor research
- financial NLP dashboards
- analyst workflow tools

---

## 6. Explainability Lab

The Explainability Lab provides scaffolding for understanding why a model output may have occurred.

It can support:

- token importance
- attention-based attribution
- strongest token links
- prediction rationale
- confidence scoring
- counterfactual analysis

The goal is not just to show the model’s answer.

The goal is to explain the model’s behavior.

---

## 7. Counterfactual Explanations

Counterfactual analysis asks:

> What changes if we remove or modify an important token?

For example:

```text
Original: The company reported weak demand.
Modified: The company reported demand.
```

If removing `weak` changes the sentiment or prediction score, then `weak` was likely important.

This helps make model behavior more understandable.

---

## 8. Embedding Explorer

BERTokenScope includes embedding exploration hooks.

Embeddings convert text into numerical vectors that represent meaning.

This allows text to be compared mathematically.

### Example Use Cases

- Compare two financial statements.
- Cluster similar transcript excerpts.
- Map companies by semantic similarity.
- Identify related risk disclosures.
- Build retrieval or search features later.

---

## 9. Company Similarity Maps

Using embeddings, BERTokenScope can support company or document similarity analysis.

For example:

- Which companies discuss similar risks?
- Which transcript excerpts sound alike?
- Which filings are semantically close?
- Which documents cluster together?

This connects transformer NLP with real-world document intelligence.

---

## 10. Model Comparison

BERTokenScope is designed to compare multiple transformer model families.

Potential compatible models include:

- BERT
- DistilBERT
- RoBERTa
- FinBERT
- sentence-transformer models

### Comparison Dimensions

- predicted tokens
- confidence scores
- latency
- output differences
- finance-specific relevance
- embedding similarity
- interpretability quality

This turns the project into a model experimentation platform.

---

## 11. Runtime Benchmarking

BERTokenScope includes benchmarking ideas for comparing model behavior and performance.

It can compare:

- inference latency
- confidence distribution
- top-k prediction differences
- fallback vs live model behavior
- model family performance

This is important because production AI systems are not only judged by accuracy.

They are also judged by speed, reliability, cost, and stability.

---

## 12. FastAPI Service

BERTokenScope includes a backend API layer for serving NLP analysis.

The FastAPI service supports a more production-ready architecture where the dashboard and API are separated.

### API Responsibilities

- masked-token prediction
- financial text analysis
- health checks
- request validation
- structured JSON responses
- API-key protected routes
- versioned endpoints
- safe error envelopes

This makes the project feel more like a real AI service rather than just a notebook or script.

---

# 🏗️ System Architecture

BERTokenScope follows a modular architecture.

```text
User
 │
 ▼
Streamlit Dashboard
 │
 │  Interactive UI for demos, analysis, charts, and explainability
 │
 ▼
FastAPI Service
 │
 │  Versioned API routes, validation, auth, health checks
 │
 ▼
NLP Service Layer
 │
 ├── Masked Token Prediction
 ├── Attention Extraction
 ├── Financial NLP Analysis
 ├── Embedding Generation
 ├── Model Comparison
 └── Explainability Reports
 │
 ▼
Model Adapter Layer
 │
 ├── BERT
 ├── DistilBERT
 ├── RoBERTa
 ├── FinBERT
 └── Fallback/Demo Mode
 │
 ▼
Local Artifacts + Run Tracking
 │
 ├── JSON outputs
 ├── SQLite metadata
 ├── Logs
 └── Analysis history
```

---

# 🔁 Core Workflow

A typical BERTokenScope workflow looks like this:

```text
1. User enters a sentence or financial text
2. Text is cleaned and tokenized
3. Model or fallback service processes the input
4. BERTokenScope extracts predictions, attention links, and language signals
5. Results are converted into structured outputs
6. Streamlit displays charts, tables, explanations, and insights
7. Optional API artifacts are saved for run history and reproducibility
```

---

# 🧬 Example Inputs

## Masked Language Example

```text
The company reported strong [MASK] growth this quarter.
```

## Financial Risk Example

```text
Management lowered guidance due to weaker demand and continued margin pressure.
```

## Executive Tone Example

```text
We remain confident in our long-term strategy, although near-term conditions remain uncertain.
```

## Transcript Drift Example

```text
Q1: We expect strong demand across all regions.
Q2: We are seeing slower demand and increased customer caution.
```

---

# 📊 Example Outputs

BERTokenScope can produce outputs such as:

```text
Top Mask Predictions:
1. revenue
2. sales
3. earnings
4. profit
5. margin
```

```text
Strongest Attention Links:
revenue → growth
guidance → lowered
demand → weaker
margin → pressure
```

```text
Financial NLP Signals:
Sentiment: Cautious
Risk Level: Elevated
Uncertainty: Medium
Executive Tone: Defensive
```

```text
Counterfactual Impact:
Removing "weaker" reduced negative tone score by 34%.
```

---

# 🧠 Key AI Concepts Demonstrated

## Transformer Attention

Attention allows a model to decide how much each token should focus on every other token.

This is the heart of transformer-based language understanding.

---

## Masked Language Modeling

Masked language modeling trains a model to predict missing words from context.

BERT was trained using this objective.

BERTokenScope uses this idea to show how context shapes prediction.

---

## Token-Level Interpretability

Instead of only seeing the final model output, BERTokenScope exposes token relationships.

This helps explain the model’s internal behavior.

---

## Financial NLP

Financial text has domain-specific language.

Words like `guidance`, `margin`, `demand`, `headwinds`, `liquidity`, and `uncertainty` carry important business meaning.

BERTokenScope adds finance-aware analysis to make transformer outputs more useful in real-world contexts.

---

## Embedding Similarity

Embeddings allow text to be represented as vectors.

This enables:

- semantic search
- clustering
- similarity scoring
- document comparison
- retrieval systems

---

## Explainable AI

Explainable AI focuses on making model behavior understandable to humans.

BERTokenScope supports this through attention analysis, token attribution, counterfactuals, and structured reports.

---

# 🧰 Tech Stack

## Core Language

- Python 3.12

## Frontend / Dashboard

- Streamlit

## Backend

- FastAPI
- Uvicorn

## NLP / ML

- Hugging Face Transformers
- BERT-compatible masked language models
- FinBERT-compatible finance models
- PyTorch
- Tokenizers
- Embedding models

## Data / Analytics

- NumPy
- pandas
- scikit-learn
- SQLite
- JSON artifacts

## Visualization

- Streamlit charts
- Attention heatmaps
- Token relationship tables
- Embedding map hooks
- Dimensionality reduction scaffolding

## DevOps / Production Readiness

- Docker
- Docker Compose
- CI checks
- API contract tests
- Structured logging
- Request IDs
- Health checks
- Rate limiting scaffolding
- Security headers
- API key authentication

---

# 📁 Project Structure

```text
BERTokenScope/
│
├── api/
│   ├── main.py
│   └── routes/
│
├── app/
│   └── streamlit_app.py
│
├── attention/
│   ├── extraction.py
│   ├── heatmaps.py
│   ├── rollout.py
│   └── token_links.py
│
├── ber_tokenscope/
│   ├── config.py
│   ├── schemas.py
│   ├── settings.py
│   └── model_adapters.py
│
├── embeddings/
│   ├── encode.py
│   ├── reduce.py
│   └── similarity.py
│
├── explainability/
│   ├── attribution.py
│   ├── counterfactuals.py
│   └── reports.py
│
├── financial_nlp/
│   ├── sentiment.py
│   ├── risk_signals.py
│   ├── uncertainty.py
│   └── transcript_drift.py
│
├── configs/
│   └── default.yaml
│
├── docs/
│   ├── architecture.md
│   ├── cs50ai-extension.md
│   ├── portfolio-deployment.md
│   └── streamlit-cloud.md
│
├── tests/
│   ├── test_api.py
│   ├── test_attention.py
│   ├── test_financial_nlp.py
│   └── test_fallbacks.py
│
├── assets/
│   ├── bertokenscope-banner.png
│   ├── dashboard-preview.gif
│   └── architecture-diagram.png
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── requirements-models.txt
├── pyproject.toml
├── README.md
└── LICENSE
```

---

# ⚡ Quick Start

## 1. Clone the Repository

```powershell
git clone https://github.com/YOUR_USERNAME/BERTokenScope.git
cd BERTokenScope
```

---

## 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

---

## 3. Install Requirements

```powershell
pip install -r requirements.txt
```

---

## 4. Run the Streamlit Dashboard

```powershell
streamlit run app/streamlit_app.py
```

Then open the local URL shown in your terminal.

Usually:

```text
http://localhost:8501
```

---

# 🧪 Running the API

Set an API key:

```powershell
$env:BERTSCOPE_API_KEY="replace-with-a-long-random-secret"
```

Run the FastAPI server:

```powershell
uvicorn api.main:app --reload
```

API will usually be available at:

```text
http://127.0.0.1:8000
```

Interactive API docs:

```text
http://127.0.0.1:8000/docs
```

---

# 🔌 Example API Endpoints

## Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "service": "BERTokenScope"
}
```

---

## Masked Token Prediction

```http
POST /api/v1/mask/predict
```

Example request:

```json
{
  "text": "The company reported strong [MASK] growth this quarter.",
  "top_k": 5
}
```

Example response:

```json
{
  "predictions": [
    {
      "token": "revenue",
      "score": 0.41
    },
    {
      "token": "sales",
      "score": 0.23
    },
    {
      "token": "earnings",
      "score": 0.14
    }
  ]
}
```

---

## Financial Text Analysis

```http
POST /api/v1/finance/analyze
```

Example request:

```json
{
  "text": "Management lowered guidance due to weaker demand and margin pressure."
}
```

Example response:

```json
{
  "sentiment": "cautious",
  "risk_level": "elevated",
  "uncertainty": "medium",
  "signals": [
    "lowered guidance",
    "weaker demand",
    "margin pressure"
  ]
}
```

---

# 🐳 Docker Usage

## Run with Docker Compose

```powershell
docker compose up --build
```

---

## Run with Optional Gateway Profile

```powershell
docker compose --profile gateway up --build
```

---

# 🧠 Model Downloads

BERTokenScope is designed to be portfolio-friendly and reliable.

By default, the public dashboard can run in an offline-safe fallback mode.

This means the demo can still work without:

- GPU access
- live Hugging Face downloads
- large model cache files
- unstable cloud inference dependencies

For live transformer inference, install optional model dependencies:

```powershell
pip install -r requirements-models.txt
```

Then enable model downloads:

```text
BERTSCOPE_ALLOW_MODEL_DOWNLOADS=true
```

---

# 🌐 Public Portfolio Deployment

BERTokenScope is prepared for a practical portfolio deployment strategy.

## Recommended Public Setup

- Deploy the Streamlit dashboard on Streamlit Community Cloud.
- Keep the dashboard in offline/fallback mode for reliability.
- Use the GitHub repo to showcase the full FastAPI backend architecture.
- Deploy the FastAPI backend separately later if live transformer serving is needed.

## Streamlit Community Cloud Settings

```text
Main file path: app/streamlit_app.py
Required secrets: none
Recommended mode: offline-safe demo
```

For more details, see:

```text
docs/streamlit-cloud.md
docs/portfolio-deployment.md
```

---

# 🧪 Testing

Run the test suite:

```powershell
pytest
```

Run tests with verbose output:

```powershell
pytest -v
```

Run a specific test file:

```powershell
pytest tests/test_attention.py
```

---

# ✅ Production-Ready Features

BERTokenScope includes several features that make it more realistic than a simple course script.

## API and Backend

- FastAPI backend
- `/api/v1` route versioning
- API-key authentication
- health checks
- structured responses
- safe error messages
- request validation
- idempotency key support scaffolding
- pagination scaffolding

## Reliability

- deterministic fallback behavior
- offline-safe demo mode
- model lazy-loading
- model warmup endpoint scaffolding
- testable components without model downloads

## Observability

- request IDs
- structured JSON logs
- Prometheus-style metrics scaffolding
- run history
- local artifacts

## Security

- API key protection
- role-aware route protection scaffolding
- CORS configuration
- security headers
- request size limits
- rate limiting scaffolding
- safe error envelopes

## Data Governance

- redaction hooks
- audit logs
- retention cleanup
- financial-use disclaimers
- local artifact control

## DevOps

- Dockerfile
- Docker Compose
- optional gateway profile
- CI checks
- linting
- formatting checks
- type-checking scaffolding
- security scan scaffolding
- release image workflow scaffolding

---

# 📈 Portfolio Impact

BERTokenScope demonstrates my ability to move from classroom AI to production-style AI engineering.

It shows:

- transformer understanding
- NLP interpretability
- financial text analytics
- dashboard development
- API-first design
- software architecture
- model-serving awareness
- testing and deployment readiness

This project sits at the intersection of:

```text
Artificial Intelligence
+
Natural Language Processing
+
Financial Analytics
+
Explainable AI
+
Full-Stack ML Engineering
```

---

# 💼 Resume Bullets

Here are strong resume-ready bullets for this project:

```text
Engineered BERTokenScope, a transformer explainability platform extending Harvard CS50AI Attention into a production-style NLP system for BERT masked-token prediction, attention visualization, financial text analytics, and model comparison.
```

```text
Built an interactive Streamlit and FastAPI-based NLP intelligence system with token-level attention exploration, finance-aware sentiment/risk analysis, embedding similarity hooks, deterministic fallback mode, Docker Compose support, and testable service components.
```

```text
Designed a portfolio-ready transformer interpretability workflow using BERT-compatible models, attention head analysis, counterfactual token explanations, structured API responses, local run tracking, and offline-safe deployment patterns.
```

---

# 🔮 Future Improvements

Potential future upgrades include:

- live hosted FastAPI backend
- full Hugging Face model serving
- FinBERT sentiment integration
- persistent PostgreSQL run history
- vector database support
- transcript upload pipeline
- PDF/filing parser
- earnings-call dashboard
- SHAP/LIME-style attribution
- WebSocket streaming for inference jobs
- Kubernetes deployment manifests
- AWS deployment using ECS or Lambda containers
- CI/CD deployment to cloud infrastructure

---

# 🧾 Disclaimer

BERTokenScope is an educational and portfolio project.

Financial NLP outputs should not be treated as investment advice. Sentiment, risk, uncertainty, and tone analysis are model-assisted signals intended for research, learning, and demonstration purposes only.

---

# 🧠 CS50AI Concepts Applied

This project directly extends the transformer-based NLP and attention concepts introduced in the **CS50AI Attention** project.

| CS50AI Attention | BERTokenScope |
|---|---|
| `[MASK]` Token Prediction | BERT Masked-Language Intelligence |
| Tokenization | Token-Level Context Exploration |
| Self-Attention Scores | Attention Maps and Token Relationship Analysis |
| 12 BERT Layers | Layer-by-Layer Transformer Inspection |
| 12 Attention Heads per Layer | Head-Level Interpretability |
| Static Attention Diagrams | Interactive Attention Visualization |
| Attention Head Analysis | Explainability Reports and Token Insights |
| Natural Language Sentences | Financial Text, Earnings Language, and Risk Signals |
| Hugging Face Transformers | Modular Model Adapter Layer |
| Single-Purpose Python Script | Streamlit Dashboard + FastAPI Backend |
| Manual Interpretation | Structured NLP Intelligence Workflow |
| Course Assignment | Production-Style Transformer Explainability Platform |

BERTokenScope demonstrates how foundational transformer and attention concepts can scale into a production-oriented NLP explainability system for analyzing language, context, financial tone, and token-level model behavior.

---

## 👤 Author

<p align="center">
  <b>Mitra Boga</b>
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/bogamitra/">
    <img src="https://img.shields.io/badge/LinkedIn-Mitra%20Boga-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn">
  </a>
  <a href="https://x.com/techtraboga">
    <img src="https://img.shields.io/badge/X-@techtraboga-black?style=for-the-badge&logo=x" alt="X">
  </a>
</p>
