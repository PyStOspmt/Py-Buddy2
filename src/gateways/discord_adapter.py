"""
Py-Buddy: Discord Gateway Adapter (Py-Buddy Lineage)
Handles Discord Bot API events, Slash commands, and Rich Embed formatting.
"""

from typing import Dict, Any, Optional
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')))
from gateways.base_gateway import BaseIngressGateway
from react_agent import ReActAgent

class DiscordGateway(BaseIngressGateway):
    """
    Adapter for Discord Bot messaging and Slash Commands.
    Translates ReAct reasoning outcomes into Discord Rich Embeds with color-coded confidence.
    """
    def __init__(self, agent: ReActAgent, runner: Any = None, schema_prompt: str = "",
                 bot_token: Optional[str] = None, session_store: Any = None):
        super().__init__(agent, runner=runner, schema_prompt=schema_prompt,
                         channel_name="discord", session_store=session_store)
        self.bot_token = bot_token or os.getenv("DISCORD_BOT_TOKEN")

    def format_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats response into a Discord Rich Embed.
        """
        final_answer = result.get("final_answer", "")
        steps = result.get("steps", [])
        
        # F55: colour the embed by the structured outcome -- the step
        # verdicts and the session's termination reason -- not by grepping
        # the answer prose for Ukrainian cue words. An incompatible
        # verdict rendered in English must still colour red, and a
        # response with no evaluated evidence must not wear a
        # "verified" badge it never earned.
        step_verdicts = [s.get("verdict") for s in steps if isinstance(s, dict)]
        termination = result.get("termination_reason") or ""
        if termination.startswith("HALLUCINATION") or termination in (
                "STEP_BUDGET_EXCEEDED", "STAGNATION_DETECTED"):
            color = 0xEF4444  # the engine itself reported a failure
            status_title = "❌ Виконання не завершено"
        elif "incompatible" in step_verdicts or "candidate_conflict" in step_verdicts:
            color = 0xEF4444  # Crimson Red for evaluated conflicts
            status_title = "❌ Виявлено несумісність (Virtual Hypergraph)"
        elif "compatible" in step_verdicts:
            color = 0x10B981  # Emerald Green for an evaluated match
            status_title = "✅ Сумісність перевірено Py-Buddy"
        elif termination.startswith("SUCCESS"):
            color = 0x6366F1  # grounded answer, no compat verdict rendered
            status_title = "💬 Відповідь Py-Buddy"
        else:
            color = 0x6B7280  # neutral -- nothing was actually evaluated
            status_title = "💬 Відповідь Py-Buddy"

        fields = []
        if steps:
            # Add ReAct reasoning chain as embed fields
            steps_preview = []
            for i, step in enumerate(steps[:3], 1):
                tool = step.get("action", {}).get("tool", "Аналіз")
                steps_preview.append(f"**Крок {i} ({tool}):** {step.get('thought', '')[:80]}...")
            fields.append({
                "name": "🔬 Траєкторія міркувань (ReAct)",
                "value": "\n".join(steps_preview) if steps_preview else "Детермінована валідація",
                "inline": False
            })

        embed = {
            "title": status_title,
            "description": final_answer,
            "color": color,
            "fields": fields,
            "footer": {
                "text": f"Py-Buddy • DeepSeek V4 Flash • Кроків: {len(steps)}"
            }
        }
        return {"embeds": [embed]}

    def handle_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handles Discord interaction payload (Slash Command)."""
        user_id = interaction_data.get("member", {}).get("user", {}).get("id", "discord_user")
        options = interaction_data.get("data", {}).get("options", [])
        query_text = options[0].get("value", "") if options else ""

        if not query_text:
            return {"type": 4, "data": {"content": "Будь ласка, введіть запит для консультації."}}

        payload = self.process_incoming_query(user_id=str(user_id), query_text=query_text)
        return {
            "type": 4,
            "data": payload["formatted_payload"]
        }
