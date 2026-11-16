"""
Omnichannel gateway sanitization and session continuity regressions.

Failing-before counterexamples:

- Telegram HTML parse_mode received raw answer text -- a `<`/`&` in a
  product name broke markup or injected tags;
- gateway conversations never reached the executed session store, so
  messenger sessions lost all state (including the elicited belief)
  between turns;
- the Discord runner constructed an adapter, printed "ready", and exited
  with no listener at all.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "core"))
sys.path.insert(0, os.path.join(ROOT, "src", "runners"))

from active_inference import UserBeliefState
from react_agent import SessionState
from session_store import SQLiteSessionStore

from gateways.discord_adapter import DiscordGateway
from gateways.telegram_adapter import TelegramGateway


class FakeAgent:
    """Minimal run_live_session double: records the persisted belief it
    was handed and returns a state carrying a fresh one."""

    def __init__(self):
        self.seen_beliefs = []
        self.seen_asked = []

    def run_live_session(self, user_query, llm_runner, schema_prompt,
                         session_id, belief_state=None, asked_question_ids=None):
        self.seen_beliefs.append(belief_state)
        self.seen_asked.append(asked_question_ids)
        state = SessionState(session_id=session_id, user_query=user_query,
                             belief_state=belief_state or UserBeliefState())
        # Simulate an answered question: beta mass on one dimension.
        state.belief_state.bayesian_update("price_sensitivity", 0.9)
        state.asked_question_ids = list(
            dict.fromkeys(list(asked_question_ids or []) + ["q_budget"]))
        state.final_response = "Відповідь консультанта."
        state.execution_history = []
        return state


class TestTelegramFormatting:
    def test_answer_is_escaped_inside_html_payload(self):
        gw = TelegramGateway(agent=None)
        out = gw.format_response({
            "final_answer": "Диск <script>alert(1)</script> & 2x<3 мм",
            "steps": []})
        assert "<script>" not in out["text"]
        assert "&lt;script&gt;" in out["text"]
        assert "&amp;" in out["text"]
        # Our own intentional markup stays unescaped.
        assert "<b>" in out["text"]
        assert out["parse_mode"] == "HTML"

    def test_long_answer_is_bounded_with_visible_marker(self):
        gw = TelegramGateway(agent=None)
        out = gw.format_response({"final_answer": "x" * 6000, "steps": []})
        assert len(out["text"]) <= 4096
        assert "обрізано" in out["text"]

    def test_truncation_never_splits_an_entity(self):
        gw = TelegramGateway(agent=None)
        # Put the "&" so its escaped "&amp;" spans exactly across the
        # cut boundary (the marker reserves the tail of the budget).
        marker_len = len("… [відповідь обрізано]")
        prefix_len = len(gw.format_response({"final_answer": "", "steps": []})["text"])
        answer = "y" * (4096 - marker_len - prefix_len - 1) + "&" + "z" * 200
        out = gw.format_response({"final_answer": answer, "steps": []})
        assert len(out["text"]) <= 4096
        body = out["text"].split("… [відповідь обрізано]")[0]
        assert not body.endswith("&")
        assert "&amp;" not in body[-6:]


class TestGatewaySessionPersistence:
    def test_messenger_session_reaches_the_executed_store(self, tmp_path):
        store = SQLiteSessionStore(str(tmp_path / "sess.db"))
        agent = FakeAgent()
        gw = TelegramGateway(agent=agent, runner=object(),
                             session_store=store)

        gw.process_incoming_query(user_id="42", query_text="щось купити")
        gw.process_incoming_query(user_id="42", query_text="а дешевше?")

        # The second turn received the belief persisted by the first.
        first, second = agent.seen_beliefs
        assert first is None or first.dimensions.get("price_sensitivity", 0.5) != \
            second.dimensions.get("price_sensitivity")
        assert second is not None
        assert second.dimensions["price_sensitivity"] > 0.5
        # Asked questions carry across turns too.
        assert agent.seen_asked[1] == ["q_budget"]

        saved = store.get_session("telegram:42")
        assert saved is not None
        assert saved.asked_question_ids == ["q_budget"]

    def test_discord_gateway_uses_same_store_contract(self, tmp_path):
        store = SQLiteSessionStore(str(tmp_path / "sess.db"))
        gw = DiscordGateway(agent=FakeAgent(), runner=object(),
                            session_store=store)
        gw.process_incoming_query(user_id="7", query_text="запит")
        assert store.get_session("discord:7") is not None


class TestDiscordRunnerHonesty:
    def test_runner_without_token_returns_nonzero(self, monkeypatch):
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        from run_discord import run_discord_bot
        assert run_discord_bot() != 0

    def test_runner_with_token_does_not_pretend_ready(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        from run_discord import run_discord_bot
        # No Discord listener exists in this codebase; the runner must
        # report that honestly instead of a silent pretend-ready exit(0).
        assert run_discord_bot() == 1
