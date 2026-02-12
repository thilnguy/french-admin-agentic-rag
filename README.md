# French Admin Agentic RAG

A production-ready RAG agent designed to assist with French administrative procedures. This project leverages an asynchronous architecture, structured logging, and strict type validation for reliability and scalability.

## 🚀 Features

- **Asynchronous Core**: Built with `asyncio` for high-conformance concurrent request handling.
- **RAG Architecture**: Uses Qdrant for vector search and Redis for conversation history/caching.
- **Robustness**: Global exception handling, Pydantic validation, and comprehensive test suite.
- **Multi-language Support**: Handles queries in French, English, and Vietnamese (with internal processing in French).
- **Production Ready**: Dockerized, CI/CD with GitHub Actions, and structured JSON logging.

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
│   │   ├── state.py      # [NEW] AgentState Pydantic Model
│   │   └── intent_classifier.py # [NEW] Intent Classification
│   ├── memory/       # Redis Memory Management
│   ├── shared/       # Shared Utilities (Guardrails)
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

## 🤝 Contribution

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (Pre-commit hooks will run automatically).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.
