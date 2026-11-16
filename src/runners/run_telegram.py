"""
Py-Buddy: Standalone Telegram Bot Polling Runner (Py-Buddy Lineage)
Polls the Telegram Bot API for incoming updates, passes queries through the
Active-Inference Neuro-Symbolic Gateway, and sends verified responses with inline trade-off buttons.
"""

import os
import sys
import time
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')))

from server import ctx
from gateways.telegram_adapter import TelegramGateway

def run_telegram_poller():
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        print("[Telegram Bot] УВАГА: TELEGRAM_BOT_TOKEN не задано у .env файлі.")
        print("[Telegram Bot] Для запуску живого бота додайте рядок у .env:")
        print("TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz")
        return

    print("[Telegram Bot] Ініціалізація Py-Buddy Telegram Gateway...")
    gateway = TelegramGateway(
        agent=ctx.agent,
        runner=ctx.runner,
        schema_prompt=ctx.schema_prompt,
        bot_token=bot_token,
        # F35: the executed persistence path is the shared server store --
        # the runner's own store used to be created and never written to.
        session_store=ctx.session_store,
    )

    api_url = f"https://api.telegram.org/bot{bot_token}"
    offset = 0

    print("[Telegram Bot] Live Long-Polling запущено. Очікування повідомлень (Ctrl+C для виходу)...")
    while True:
        try:
            resp = requests.get(f"{api_url}/getUpdates", params={"offset": offset, "timeout": 20}, timeout=25)
            if resp.status_code != 200:
                time.sleep(3)
                continue

            data = resp.json()
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                dispatch_payload = gateway.handle_webhook_update(update)
                if dispatch_payload:
                    requests.post(f"{api_url}/sendMessage", json=dispatch_payload, timeout=10)

        except KeyboardInterrupt:
            print("\n[Telegram Bot] Зупинка бота за запитом користувача.")
            break
        except Exception as e:
            print(f"[Telegram Bot] Помилка циклу опитування: {e}")
            time.sleep(3)

if __name__ == "__main__":
    run_telegram_poller()
