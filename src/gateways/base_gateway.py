"""
Py-Buddy: Omnichannel Base Gateway Architecture
Preserves the Py-Buddy architectural lineage of multi-platform ingress adapters.
Provides a unified abstraction for Web, Telegram, Discord, and REST consumers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import os
import sys

# Product core moved to src/core/ (docs/13 C2): react_agent resolves there,
# while this package itself resolves via src/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')))

from react_agent import ReActAgent, SessionState

class BaseIngressGateway(ABC):
    """
    Abstract Channel Gateway for omnichannel e-commerce consultations.
    All platform adapters (Web, Telegram, Discord) delegate core reasoning to this contract.
    """
    def __init__(self, agent: ReActAgent, runner: Any = None, schema_prompt: str = "",
                 channel_name: str = "generic", session_store: Any = None):
        self.agent = agent
        self.runner = runner
        self.schema_prompt = schema_prompt
        self.channel_name = channel_name
        #: F35: the store that actually persists gateway sessions. The old
        #: Telegram runner built its own store that nothing ever wrote to,
        #: so messenger conversations lost all state between turns.
        self.session_store = session_store

    @abstractmethod
    def format_response(self, result: Dict[str, Any]) -> Any:
        """Adapts the raw ReAct and Hypergraph execution result to channel-specific layout."""

    def process_incoming_query(self, user_id: str, query_text: str) -> Dict[str, Any]:
        """
        Executes unified query processing through the Neuro-Symbolic ReAct engine.
        """
        session_id = f"{self.channel_name}:{user_id}"
        if self.runner:
            # F35: restore the session's persisted belief and asked-question
            # state so a returning customer's dialogue continues rather than
            # restarting from a flat prior -- the same contract the web
            # /api/chat path honours.
            session = self.session_store.get_session(session_id) \
                if self.session_store else None
            state = self.agent.run_live_session(
                query_text,
                llm_runner=self.runner,
                schema_prompt=self.schema_prompt,
                session_id=session_id,
                belief_state=session.belief_state if session else None,
                asked_question_ids=session.asked_question_ids if session else None,
            )
            if self.session_store is not None:
                if session is None:
                    session = SessionState(session_id=session_id,
                                           user_query=query_text)
                session.user_query = query_text
                session.final_response = state.final_response
                session.execution_history = state.execution_history
                session.belief_state = state.belief_state
                session.asked_question_ids = list(state.asked_question_ids)
                self.session_store.save_session(session)
            result = {
                "final_answer": state.final_response or "Не вдалося отримати детерміновану відповідь.",
                "steps": state.execution_history,
                "termination_reason": state.termination_reason,
            }
        else:
            # Fallback when runner is not configured
            result = {
                "final_answer": f"Запит прийнято каналом {self.channel_name}: {query_text}",
                "steps": [],
                "termination_reason": None,
            }

        return {
            "channel": self.channel_name,
            "user_id": user_id,
            "query": query_text,
            "result": result,
            "formatted_payload": self.format_response(result)
        }

