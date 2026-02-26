# French Admin Agentic RAG

A production-ready, multilingual RAG agent designed to assist with French administrative procedures. Built with an asynchronous agentic architecture, a Data-Driven Topic Registry, and multi-layer guardrails.

## 🚀 Features

- **Hybrid Fast/Slow Lane Architecture**: Simple Q&A routes to a lightweight RAG pipeline; complex multi-step procedures route through a LangGraph agentic workflow.
- **Asynchronous Core**: Built with `asyncio` for high-concurrency, non-blocking request handling.
- **Hybrid Search**: Combines BM25 (sparse) and Vector Search (Qdrant) with RRF Fusion for superior retrieval.
- **Data-Driven Topic Registry**: All topic rules, mandatory variables, guardrail keywords, and few-shot exemplars are YAML-driven — no hardcoding in prompts.
- **Expert Performance**: Achieved a **9.5/10** score on a 100-case blind benchmark with **91.8% clarification accuracy**.
- **Multi-layer Guardrails**: Topic validation + hallucination detection, both grounded in retrieved legal context.
- **Multi-language Support**: Native support for French, English, and Vietnamese with cross-language intent classification and multilingual guardrail keywords.

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
- `QDRANT_HOST`: Host for Qdrant (default: `localhost`).
- `REDIS_HOST`: Host for Redis (default: `localhost`).

**Optional Model Overrides:**
- `OPENAI_MODEL`: Main generation model (default: `gpt-4o`).
- `GUARDRAIL_MODEL`: Model for topic/hallucination checks (default: `gpt-4o-mini`).
- `FAST_LLM_MODEL`: Model for lightweight tasks like query rewriting (default: `gpt-4o-mini`).

## 🏃‍♂️ Running Locally

### Start Infrastructure (Redis & Qdrant)
```bash
docker run -d -p 6379:6379 redis:latest
docker run -d -p 6333:6333 qdrant/qdrant:latest
```

### Run the Application
```bash
make run
# Or manually:
# uv run uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Access Swagger UI at `http://localhost:8000/docs`.

## 🧪 Testing

```bash
make test
# Or manually:
# uv run pytest tests/
```

## 🐳 Docker Deployment

```bash
make docker-build
docker run -p 8000:8000 french-admin-agent
```

## 📂 Project Structure

```
├── .github/          # CI/CD Workflows
├── docs/             # Architecture & release documentation
├── evals/            # LLM Judge evaluation framework
│   ├── data/         # Benchmark datasets
│   ├── results/      # Evaluation results (JSON)
│   └── runners/      # Eval scripts (llm_judge.py, etc.)
├── finetuning/       # Fine-tuning scripts & data (experimental)
├── scripts/          # Utility scripts
├── skills/           # Agent skills (Translator, Retriever)
├── src/
│   ├── agents/       # Orchestrator, ProcedureGuide, LegalAgent, Graph, State
│   ├── memory/       # Redis-backed session state management
│   ├── rules/        # Data-Driven Topic Registry (YAML + Python)
│   │   ├── topic_registry.yaml  # Topics, rules, keywords, exemplars
│   │   └── registry.py          # TopicRegistry class
│   ├── shared/       # Guardrails, QueryPipeline, LanguageResolver, HybridRetriever
│   ├── utils/        # Logging, Metrics, LLM Factory
│   ├── config.py     # Pydantic Settings (all config centralized here)
│   ├── main.py       # FastAPI Entrypoint
│   └── schemas.py    # Pydantic Models
├── tests/
│   ├── integration/  # API integration tests
│   └── unit/         # Unit tests (149+ passing)
├── Dockerfile        # Multi-stage Docker build
├── Makefile          # Development commands
├── pyproject.toml    # Dependencies & Tool Config
└── README.md
```

## 📖 Documentation

- **[Rule System Guide](docs/rule_system.md)**: How the Data-Driven Topic Registry works, YAML format, multilingual keywords.
- **[Architecture Evolution](docs/architecture_evolution.md)**: How the system evolved from monolith to multi-agent.
- **[Production Roadmap](docs/production_roadmap.md)**: Current production readiness status and future plans.
- **[Project Walkthrough](docs/project_walkthrough.md)**: Chronological log of all major improvements.
- **[Fine-tuning Process](docs/finetuning_process.md)**: Documentation of the experimental Qwen 2.5 fine-tuning.

## 🤝 Contribution

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (Pre-commit hooks will run automatically).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.
