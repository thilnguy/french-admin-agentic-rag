# 🇫🇷 Marianne AI - French Administrative Agentic RAG

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-cc1d1d.svg)]()
[![Model](https://img.shields.io/badge/Model-GPT--4o%20%7C%20Qwen%208--bit-ff69b4.svg)]()

> **Marianne AI** is a production-grade, state-of-the-art **Agentic RAG** system built to navigate the complexities of French residency laws, labor rights, and administrative procedures for foreigners.
>
> 🛑 **Zero-Hallucination Architecture**: Legal advice demands truth, not probability. Marianne uses Structural Context Fallbacks and strict pre-synthesis validation. If relevant laws aren't found, she explicitly asks for clarification rather than hallucinating answers from irrelevant legal sub-clauses.

---

## ✨ System Architecture & Key Innovations

*   **🛡️ Agentic Reasoning over Basic RAG**: Instead of blindly searching the user's noisy prompt (e.g., "I lost my wallet on the RER..."), specialized Pre-processor LLMs rewrite the query to isolate the *Core Goal* (e.g., "Procedure to replace a lost Residence Permit") while extracting vital constraints (Nationality, Language).
*   **⚖️ Systemic Fallback (Zero-Hallucination)**: An evaluator LLM grades retrieved legal texts. If Qdrant returns low-confidence matches, the Agent actively halts generation, invoking a `[DEMANDER]` decision node to ask the user for specific missing details.
*   **🔮 Dynamic Dual-Model Engine**: Hot-swap between **OpenAI API (GPT-4o)** for maximum reasoning and a **Local Fine-Tuned Model (Qwen 2.5 8-bit)** for high-privacy, zero-cost inference. Swapping is instant via the UI, instantiating LLMs dynamically *per-request* for robust thread-safety.
*   **🌍 Polyglot RAG Pipeline**: Natively fluently handles English, French, and Vietnamese. Employs translation-routing to ensure Vietnamese queries perfectly match against the French legal corpus natively stored in the DB.
*   **🎯 Two-Stage Retrieval (Cross-Encoder)**: Merges Qdrant Vector Search with the `bge-reranker-v2-m3` Cross-Encoder, compressing 20 matches down to the top 5 most semantically relevant documents.
*   **⚡ Real-Time Streaming UX (SSE)**: A sleek, ChatGPT-style interface built in Streamlit, fed by a continuous Server-Sent Events (SSE) stream from FastAPI logic blocks. 
*   **🚀 Enterprise Ready**: Fully containerized with Docker Compose. Ships with comprehensive Kubernetes (K8s) manifests (Auto-scaling HPA, Deployments) and Prometheus/Grafana/OpenTelemetry telemetry tracking.

---

## 🛠 Tech Stack

*   **Logic Routing**: Langchain, LangGraph (Asynchronous DAGs)
*   **Backend API**: FastAPI, Pydantic, Python 3.13
*   **Frontend**: Streamlit
*   **Vector Database**: Qdrant
*   **State & Caching**: Redis (Async)
*   **Observability**: Prometheus, Grafana, OpenTelemetry

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/thilnguy/french-admin-agentic-rag.git
   cd french-admin-agentic-rag
   ```

2. **Install dependencies (using `uv`):**
   ```bash
   uv sync
   ```

3. **Configure the Environment:**
   Copy the example `.env` file and insert your API keys.
   ```bash
   cp .env.example .env
   ```

4. **Run Infrastructure (Docker):**
   ```bash
   docker-compose up -d --build
   ```
   *(This boots FastAPI, Streamlit, Redis, Qdrant concurrently).*

5. **Access the Chat App:**
   Open [http://localhost:8501](http://localhost:8501) in your browser.
   *(The backend API resolves at `http://localhost:8001/docs`)*

---

## 📂 Project Structure

```
├── .github/          # CI/CD Workflows
├── docs/             # Architecture & evaluation documentation
├── evals/            # LLM Judge evaluation framework & benchmarks
├── k8s/              # Kubernetes Production Manifests (HPA, Deployments)
├── scripts/          # Legal Data processing & Vector DB upsert scripts
├── src/
│   ├── agents/       # LangGraph Orchestrator, LegalAgent, ProcedureGuide, Preprocessors
│   ├── memory/       # Redis-backed session state & TTL
│   ├── rules/        # Multi-layer Topic Registry & Rule YAMLs
│   ├── shared/       # Guardrails, Query Pipeline, Reranker
│   ├── utils/        # Telemetry, Dynamic LLM Factory, Audit logger
│   └── main.py       # FastAPI Entrypoint (SSE Streaming)
├── streamlit_app.py  # Frontend Application UI
├── docker-compose.yml 
└── README.md
```

---

## 📖 Deep Dive Documentation

*   **[Architecture Evolution](docs/architecture_evolution.md)**: Explore the timeline of how the system evolved from a simple RAG monolith to an Agentic Multi-Model ecosystem.
*   **[Rule System & Guardrails](docs/rule_system.md)**: Deep dive into the deterministic YAML-driven constraint framework mitigating model drift.
*   **[Local Fine-Tuning](docs/finetuning_process.md)**: How we quantized and customized Qwen 2.5 7B specifically for French Administrative NLP.

---

> *"Navigating foreign bureaucracy is tough. Your AI shouldn't make it harder."*
