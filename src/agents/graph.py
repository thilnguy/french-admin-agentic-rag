from typing import Literal
from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.intent_classifier import Intent
from src.agents.legal_agent import legal_agent
from src.agents.procedure_agent import procedure_agent
from langchain_core.messages import AIMessage


# Node Functions
async def legal_expert_node(state: AgentState):
    """Executes the Legal Research Agent."""
    # Prefer the goal-anchored rewritten query over raw user message
    query = state.metadata.get("current_query") or state.messages[-1].content
    response = await legal_agent.run(query, state)
    return {"messages": [AIMessage(content=response)]}


async def procedure_expert_node(state: AgentState):
    """Executes the Procedure Guide Agent."""
    # Prefer the goal-anchored rewritten query over raw user message
    query = state.metadata.get("current_query") or state.messages[-1].content
    response = await procedure_agent.run(query, state)
    return {
        "messages": [AIMessage(content=response)],
        "retrieved_docs": state.retrieved_docs,
    }


# Router Logic
def route_request(
    state: AgentState,
) -> Literal["legal_expert", "procedure_expert", "__end__"]:
    intent = state.intent
    if intent == Intent.LEGAL_INQUIRY:
        return "legal_expert"
    elif intent in [Intent.COMPLEX_PROCEDURE, Intent.FORM_FILLING]:
        return "procedure_expert"
    # Simple QA or Unknown are handled by AdminOrchestrator directly (Fast Lane)
    # But if we enter the graph, it implies we want one of these.
    # If state.intent is SIMPLE_QA but we are here, what do we do?
    # Maybe return END?
    return END


# Graph Definition
workflow = StateGraph(AgentState)

workflow.add_node("legal_expert", legal_expert_node)
workflow.add_node("procedure_expert", procedure_expert_node)

# Conditional Entry Point
# We assume the caller (Orchestrator) has already set state.intent
workflow.set_conditional_entry_point(
    route_request,
    {"legal_expert": "legal_expert", "procedure_expert": "procedure_expert", END: END},
)

workflow.add_edge("legal_expert", END)
workflow.add_edge("procedure_expert", END)

# Compile
agent_graph = workflow.compile()
