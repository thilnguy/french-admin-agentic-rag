import os
import redis
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

class MemoryManager:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    def get_session_history(self, session_id: str):
        """
        Retrieves or creates a Redis-backed chat message history for a given session.
        """
        return RedisChatMessageHistory(session_id, url=self.redis_url)

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
