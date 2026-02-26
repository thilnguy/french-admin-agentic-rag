from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import messages_to_dict, messages_from_dict
import redis.asyncio as redis
import json
from src.config import settings
from src.agents.state import AgentState


class MemoryManager:
    def __init__(self):
        self.redis_url = settings.REDIS_URL
        # Sync client for legacy LangChain compatibility
        self.redis_url_sync = settings.REDIS_URL
        # Async client for efficient State Management
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)

    def get_session_history(self, session_id: str):
        """
        Retrieves or creates a Redis-backed chat message history for a given session.
        (Legacy method used by LangChain Runnables)
        """
        return RedisChatMessageHistory(session_id, url=self.redis_url)

    async def save_agent_state(self, session_id: str, state: AgentState):
        """
        Serializes and saves the full AgentState to Redis.
        """
        try:
            # Convert Pydantic model to dict
            state_data = state.model_dump()

            # Serialize LangChain messages to robust dict format
            state_data["messages"] = messages_to_dict(state.messages)

            await self.redis_client.set(f"agent_state:{session_id}", json.dumps(state_data))
        except Exception as e:
            from src.utils.logger import logger
            logger.error(f"Redis save failed for session {session_id}: {str(e)}")
            # Graceful degradation: fail silently so the user still gets their answer

    async def load_agent_state(self, session_id: str) -> AgentState:
        """
        Loads AgentState from Redis.
        Handles schema migrations gracefully (e.g., new fields added to AgentState).
        """
        try:
            # 1. Try to load structured state
            data = await self.redis_client.get(f"agent_state:{session_id}")
            if data:
                state_dict = json.loads(data)
                # Deserialize messages
                if "messages" in state_dict:
                    state_dict["messages"] = messages_from_dict(state_dict["messages"])
                try:
                    return AgentState(**state_dict)
                except Exception:
                    # Schema migration: state has old fields, create fresh state preserving messages
                    messages = state_dict.get("messages", [])
                    new_state = AgentState(session_id=session_id, messages=messages)
                    await self.save_agent_state(session_id, new_state)
                    return new_state

            # 2. Fallback: Check for legacy history
            # Note: accessing .messages on RedisChatMessageHistory is a sync call (blocking)
            # but acceptable for one-time migration.
            legacy_history = self.get_session_history(session_id)
            if legacy_history.messages:
                # Create new state from legacy messages
                new_state = AgentState(
                    session_id=session_id, messages=legacy_history.messages
                )
                # Persist immediately to new format
                await self.save_agent_state(session_id, new_state)
                return new_state

            # 3. Return fresh state
            return AgentState(session_id=session_id)
        except Exception as e:
            from src.utils.logger import logger
            logger.error(f"Redis load failed for session {session_id}: {str(e)}. Returning fresh state.")
            # Graceful degradation: return fresh state if Redis is unreachable
            return AgentState(session_id=session_id)

    def wrap_with_history(self, chain):
        """
        Wraps a LangChain chain with message history logic.
        """
        return RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="query",
            history_messages_key="chat_history",
        )


# Singleton instance
memory_manager = MemoryManager()
