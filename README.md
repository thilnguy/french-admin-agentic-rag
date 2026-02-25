# French Admin Agentic RAG

A production-ready RAG agent designed to assist with French administrative procedures. This project leverages an asynchronous architecture, structured logging, and strict type validation for reliability and scalability.

## 🚀 Features

- **Local-First "Local Brain" Architecture**: Primary agent logic runs on a fine-tuned **Qwen 2.5 7B 8-bit** model optimized for Mac M4 (MLX).
- **Asynchronous Core**: Built with `asyncio` for high-conformance concurrent request handling.
- **Hybrid Search**: Combines BM25 and Vector Search (Qdrant) with RRF Fusion for superior retrieval.
- **Agentic Workflows**: Deployment of specialized agents orchestrated via LangGraph.
- **Expert Performance**: Achieved a **9.0/10** score on strict administrative benchmarks with **88.9% clarification accuracy**.
- **Robustness**: Multi-layer guardrails (gpt-4o-mini) and 91%+ test coverage.
- **Multi-language Support**: Native support for French, English, and Vietnamese with cross-language intent classification.

## 🛠 Prerequisites

- **Python 3.13+**
- **uv** (Fast Python package installer and resolver)
- **Docker** & **Docker Compose** (for running services)
- **Redis** & **Qdrant** (if running locally without Docker)

## 📦 Installation

This project uses `uv` for dependency management.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/thilnguy/french-admin-agentic-rag.git
   cd french-admin-agentic-rag
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Install Pre-commit hooks:**
   Ensure code quality checks run before every commit.
   ```bash
   uv run pre-commit install
   ```

## ⚙️ Configuration

Copy the example environment file and configure your secrets:

```bash
cp .env.example .env
```

**Required Variables:**
- `OPENAI_API_KEY`: Your OpenAI API Key.
- `QDRANT_HOST`: Host for Qdrant (default: localhost).
- `REDIS_HOST`: Host for Redis (default: localhost).

## 🏃‍♂️ Running Locally

### Start Infrastructure (Redis & Qdrant)
You can use the provided docker-compose (if available) or run them separately.
```bash
docker run -d -p 6379:6379 redis:latest
docker run -d -p 6333:6333 qdrant/qdrant:latest
```

### Run the Application
Use the provided Makefile for convenience:
```bash
make run
# Or manually:
# uv run uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`.
Access Swagger UI at `http://localhost:8000/docs`.

## 🧪 Testing

We use `pytest` for unit and integration testing.

```bash
make test
# Or manually:
# uv run pytest tests/
```

## 🐳 Docker Deployment

Build and run the production container:

```bash
make docker-build
docker run -p 8000:8000 french-admin-agent
```

## 📂 Project Structure

```
├── .github/          # CI/CD Workflows
├── config/           # Configuration files
├── evals/            # Evaluation scripts (Ragas)
├── scripts/          # Utility scripts (test_agent.py, etc.)
├── skills/           # Agent skills (Translator, Retriever)
├── src/
│   ├── agents/       # Agent Orchestrator & Logic
│   │   ├── orchestrator.py
│   │   ├── procedure_agent.py # [NEW] Specialized Procedure Agent
│   │   ├── legal_agent.py     # [NEW] Legal Research Agent
│   │   ├── graph.py           # [NEW] LangGraph Workflow
│   │   ├── state.py           # AgentState Pydantic Model
│   │   └── intent_classifier.py
│   ├── memory/       # Redis Memory Management
│   ├── shared/       # Shared Utilities
│   │   ├── query_pipeline.py    # [NEW] Query Preprocessing
│   │   ├── language_resolver.py # [NEW] Language Handling
│   │   ├── hybrid_retriever.py  # [NEW] BM25 + Vector Search
│   │   └── guardrails.py        # Safety Checks
│   ├── utils/        # Logging & Helpers
│   ├── config.py     # Pydantic Settings
│   ├── main.py       # FastAPI Entrypoint
│   └── schemas.py    # Pydantic Models
├── tests/            # Pytest Suite
│   ├── integration/  # API Integration Tests
│   └── unit/         # Unit Tests
├── Dockerfile        # Multi-stage Docker build
├── Makefile          # Development commands
├── pyproject.toml    # Dependencies & Tool Config
└── README.md         # Documentation
```

## 📖 Documentation

- **[Fine-tuning Process](docs/finetuning_process.md)**: Detailed guide on how we fine-tuned Qwen 2.5 for French administrative tasks.
- **[Architecture Evolution](docs/architecture_evolution.md)**: How the system evolved from a monolith to a Local-First Agentic RAG.
- **[Production Roadmap](docs/production_roadmap.md)**: Current status and future plans for production readiness.
- **[Project Walkthrough](docs/project_walkthrough.md)**: A chronological log of all major improvements and security hardening.

## 🤝 Contribution

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (Pre-commit hooks will run automatically).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.
