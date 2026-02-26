# Release v1.4.0: The Production Observability Update

**Date:** 2026-02-26  
**Status:** General Availability (GA)

Version 1.4 focuses on migrating the Marianne AI system from a development sandbox to a resilient, production-ready backend. This release introduces comprehensive distributed tracing, metrics recording, continuous integration pipelines, and Kubernetes deployment manifests.

---

## 🚀 Key Features

### 1. Observability & Telemetry Pipeline
We have instituted a full "Three Pillars of Observability" stack to guarantee transparency in production:
*   **OpenTelemetry Distributed Tracing**: Auto-instrumentation has been injected into the FastAPI layer, accompanied by explicit manual span generation in critical areas such as `orchestrator_handle_query` and `guardrail_check_hallucination`. Traces are exported to an OTLP endpoint (compatible with Jaeger).
*   **Prometheus & Grafana**: A new `/metrics` endpoint exposes vital custom metrics for proactive monitoring:
    *   `LLM_REQUEST_DURATION` (Histogram)
    *   `TOKEN_USAGE` (Counter)
    *   `GUARDRAIL_REJECTIONS` (Counter by reason)
*   **Structured Audit Logging**: `AuditLogger` now writes highly structured JSON records to rotating log files, specifically capturing `user_nationality` and query intent for offline human-expert evaluation.

### 2. Kubernetes Deployment Topology
Bootstrapped the native Kubernetes configuration in `/k8s/` to facilitate High Availability (HA) deployments.
*   **Manifests included**: ConfigMaps, encrypted Secrets, persistent Qdrant PVCs, and Services.
*   **Horizontal Pod Autoscaler (HPA)**: The FastAPI deployment is configured to automatically scale longitudinally across 2 to 10 instances based on synchronous memory and CPU constraints.
*   **Ingress Configurations**: specifically engineered to handle uninterrupted Server-Sent Events (SSE) traffic over persistent HTTP/2 connections.

### 3. Asynchronous Streaming API (SSE)
*   Transitioned the backend endpoint `/chat` to `/chat/stream` utilizing FastAPI’s `StreamingResponse`. 
*   Employs LangGraph's `astream_events` to seamlessly channel multi-agent outputs, enabling immediate _Time-To-First-Token_ latency improvements.
*   Downstream Agent Nodes properly flag terminal LCEL execution chains as `final_answer` to selectively output safe tokens for the end user.

### 4. Data Refresh Automation Pipeline
*   Engineered `update_legal_data.py` as an automated Python worker script.
*   Introduced a continuous GitHub Action pipeline triggered on cron that autonomously pulls updated French government datasets from Hugging Face and upserts them directly into the Qdrant Vector database, keeping the agent’s legal knowledge fresh without engineering overhead.

---

## 🛠 Stability Improvements
*   **Graceful Degradation Layers**: Vector Database retrieval functions are newly wrapped in `try/except` failsafes. If Qdrant hangs, the LLM retrieves a graceful "missing context" warning instead of triggering a fatal HTTP 500 error cascade.
*   **Redis Failover**: If the cache backend goes temporarily offline, agent logic gracefully resets to an empty thread state rather than failing.
