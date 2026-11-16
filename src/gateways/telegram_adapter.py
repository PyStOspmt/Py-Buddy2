"""
Py-Buddy: Telegram Gateway Adapter (Py-Buddy Lineage)
Handles Telegram Bot API integration, message formatting, and inline trade-off keyboards.
"""

from typing import Dict, Any, Optional
import html
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')))
from gateways.base_gateway import BaseIngressGateway
from react_agent import ReActAgent

class TelegramGateway(BaseIngressGateway):
    """
    Adapter for Telegram Bot messaging.
    Translates ReAct reasoning outcomes and Pareto trade-offs into Telegram-compatible messages.
    """
    def __init__(self, agent: ReActAgent, runner: Any = None, schema_prompt: str = "",
                 bot_token: Optional[str] = None, session_store: Any = None):
        super().__init__(agent, runner=runner, schema_prompt=schema_prompt,
                         channel_name="telegram", session_store=session_store)
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")

    def format_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats response with Telegram HTML markup and inline keyboard buttons.
        """
        # F35: the answer text is untrusted content (model output citing
        # source rows). Inserted raw into HTML parse_mode, a `<` or `&` in a
        # product name broke markup or injected tags -- so it is escaped
        # before being placed next to our own intentional markup.
        final_answer = html.escape(str(result.get("final_answer", "")))
        steps = result.get("steps", [])

        # Build explanation footer if steps occurred
        reasoning_summary = ""
        if steps:
            reasoning_summary = f"\n\n<i>🔬 Перевірено через Py-Buddy Neuro-Symbolic (кроків: {len(steps)})</i>"

        text = f"🤖 <b>Консультант Py-Buddy</b>\n\n{final_answer}{reasoning_summary}"
        # Telegram rejects messages over 4096 characters; the tail is a
        # visible truncation marker, not silent data loss. The marker's own
        # length is reserved out of the budget.
        marker = "… [відповідь обрізано]"
        if len(text) > 4096 - len(marker):
            cut = text[:4096 - len(marker)]
            # Do not split an escaped entity mid-sequence ("&lt;" -> "&lt").
            if cut.rfind("&") > cut.rfind(";"):
                cut = cut[:cut.rfind("&")]
            text = cut + marker
        
        inline_keyboard = []
        # If Pareto options or suggestions are present, convert to inline buttons
        if "альтернатив" in final_answer.lower() or "парето" in final_answer.lower():
            inline_keyboard.append([
                {"text": "📊 Показати компроміси (Парето)", "callback_data": "action_show_pareto"},
                {"text": "🔄 Змінити критерії", "callback_data": "action_reset_filter"}
            ])
            
        return {
            "parse_mode": "HTML",
            "text": text,
            "reply_markup": {"inline_keyboard": inline_keyboard} if inline_keyboard else None
        }

    def handle_webhook_update(self, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Processes incoming Telegram webhook payload."""
        message = update_data.get("message", {})
        chat_id = str(message.get("chat", {}).get("id", ""))
        text = message.get("text", "").strip()

        if not chat_id or not text:
            return None

        # Route through unified gateway
        payload = self.process_incoming_query(user_id=chat_id, query_text=text)
        return {
            "method": "sendMessage",
            "chat_id": chat_id,
            **payload["formatted_payload"]
        }
