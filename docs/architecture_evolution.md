# Architecture Evolution: Monolithic to Multi-Agent System

## 1. Current State: The "Router-Pipeline" Pattern
Currently, `AdminOrchestrator` acts as a monolithic controller that executes a hardcoded linear pipeline:
`Guardrail -> Translate -> Retrieve -> Generate -> Translate -> Guardrail`

| Feature | Current Implementation | Limitation |
|---------|------------------------|------------|
| **Autonomy** | None. Linear script. | Cannot handle complex multi-step reasoning (e.g., "Find form X, then tell me how to fill it"). |
| **Skills** | Python Functions (`retrieve`, `translate`) | Passive tools. Cannot self-correct (e.g., "Search failed, let me try a broader query"). |
| **State** | Redis List (Chat History) | No structured state (e.g., "clarifying missing details", "waiting for user document"). |
| **Scalability** | Rigid Class Method | Adding new capabilities (e.g., "Appointment Booking") makes `handle_query` massive and unmaintainable. |

**Verdict**: Technically **NOT** a Multi-Agent System. It is an **Agentic Pipeline**.

---

## 2. Target State: Event-Driven Multi-Agent System
To support complex administrative tasks (forms, appointments, personalized advice), we should evolve to a **Graph-Based Multi-Agent Architecture** (pattern similar to `LangGraph`).

### Core Concepts
1.  **Supervisor (Orchestrator Agent)**: The "Project Manager". Breaks down complex user requests and delegates to workers.
2.  **Worker Agents**: Specialized autonomous units.
3.  **Shared Graph State**: A structured object passed between agents containing `Messages`, `CurrentStep`, `CollectedData` (e.g., user's age, visa type).

### Proposed Agents
| Agent | Role | Tools/Autonomy |
|-------|------|----------------|
| **Supervisor** | Triage & Planning | Decides: "User needs a visa -> Delegate to Procedure Agent". "User asks simple QA -> Delegate to QA Agent". |
| **Legal Specialist** | Research & Fact-Checking | **Tools**: `Qdrant`, `Legifrance API`. <br> **Autonomy**: Can re-write search queries if results are empty. Can cross-reference multiple sources. |
| **Procedure Guide** | Interactive Step-by-Step | **Tools**: `FormFiller`, `AppointmentChecker`. <br> **Autonomy**: Maintains state of a specific process (e.g., "Step 3/5: Upload photo"). |
| **Cultural Adaptor** | UX & Translation | **Tools**: `Translation`, `ToneAdjustment`. <br> **Autonomy**: Adapts answers to cultural context (e.g., explaining implicitly understood French norms to foreigners). |

---

## 3. Migration Roadmap

### Phase 1: State Management Refactor (Weeks 1-2)
*Goal: Move from "Chat History List" to "Structured State".*
- define `AgentState` Pydantic model:
  ```python
  class AgentState(BaseModel):
      messages: List[BaseMessage]
      user_profile: UserProfile  # (visa_status, age, etc.)
      current_task: Optional[str]
      next_step: str
  ```
- This allows the agent to "remember" it is in the middle of a visa application process, even if the user asks a side question.

### Phase 2: Extract "Legal Specialist" Agent (Weeks 3-4)
*Goal: Make retrieval agentic.*
- Convert `retrieve_legal_info` from a function to an Agent loop:
  1. Generate Search Query.
  2. Execute Search.
  3. **Eval**: Are results relevant?
  4. **Loop**: If no, refine query and retry (max 3 times).
  5. Return synthesized answer.

### Phase 3: The Supervisor Router (Month 2)
*Goal: Dynamic delegation.*
- Replace hardcoded `handle_query` with a Router:
  - If intent == "QA" -> Call Legal Specialist.
  - If intent == "Procedure" -> Call Procedure Guide.
  - If intent == "Chitchat" -> Handle directly.

---

## 4. Benefit Analysis

| Metric | Current Monolith | Proposed MAS |
|--------|------------------|--------------|
| **Complexity Handling** | Low (Single turn QA) | High (Multi-turn workflows) |
| **Error Recovery** | Fail immediately | Retry/Self-correct |
| **Maintenance** | Single huge class | Small specialized modules |
| **Latency** | Low (Linear) | Higher (Multi-step reasoning) |

**Recommendation**: Start **Phase 1 (State Refactor)** immediately. The current system is brittle for anything beyond simple QA.

---

## 5. Risks & Challenges (The "No Free Lunch" Part)

While MAS offers autonomy, it introduces significant new challenges:

### 1. Latency & Cost 💸
- **Current**: 1 LLM call (Generate) + 1 Guardrail. Fast (~2-3s).
- **MAS**: Supervisor -> Worker -> Tool -> Worker -> Supervisor.
- **Risk**: Each hop adds latency. A complex query could take **10-15s** and cost **3-5x more** tokens due to internal reasoning loops.

### 2. Infinite Loops 🔄
- **Risk**: Agents getting stuck.
  - *Example*: Legal Specialist can't find info -> Refine Query -> Search -> Can't find info -> Refine Query...
- **Mitigation**: Strict `max_iterations` (e.g., 3 retries max) and "Give Up" state.

### 3. Debugging Complexity 🕸️
- **Current**: Stack trace shows exactly where it failed.
- **MAS**: Failure is emergent. "Why did the Supervisor pick the wrong worker?" or "Why did the worker decide to stop searching?" requires **distributed tracing** (LangSmith/Arize Phoenix).

### 4. State Management Consistency
- **Risk**: If the `AgentState` gets corrupted or desynchronized (e.g., Redis failure), the agent "forgets" where it is in a multi-step flow.

---

## 6. Proposed Hybrid Strategy (The "Router-First" Approach) 🛡️

To balance **Speed** (Current Monolith) with **Power** (MAS), we deploy a Hybrid system.

### Core Concept: "Fast Lane" vs "Slow Lane"

1.  **Fast Lane (Standard RAG)**:
    - **Use Case**: Simple Q&A ("How much is a passport?").
    - **Logic**: `Guardrail -> Translate -> Retrieve -> Generate`.
    - **SLA**: < 3 seconds.
    - **Cost**: Low.

2.  **Slow Lane (Agentic Graph)**:
    - **Use Case**: Complex flows ("Help me fill out Cerfa 12345", "Wait, why was my visa rejected?").
    - **Logic**: `Supervisor -> Plan -> Tool Loop -> Response`.
    - **SLA**: 10-30 seconds.
    - **Cost**: High.

### Implementation: The Intelligent Router

The `AdminOrchestrator` becomes a **Smart Router**:

```python
async def handle_query(self, query):
    # 1. Classification (Lightweight LLM or Keyword)
    intent = await classify_intent(query)

    # 2. Routing
    if intent == "SIMPLE_QA":
        return await self.run_fast_rag_pipeline(query)  # Current code
    elif intent == "COMPLEX_TASK":
        return await self.run_agentic_workflow(query)   # New LangGraph
```

### Benefits
- ✅ **90% of queries** (simple info) stay **fast and cheap**.
- ✅ **10% of queries** (hard tasks) get **full agentic power**.
- ✅ **Less risk**: If the Agent system breaks, the basic Q&A still works.
