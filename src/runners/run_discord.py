"""
Py-Buddy: Standalone Discord Bot Runner (Py-Buddy Lineage)
Listens for Discord message events and Slash commands, routing queries through
the Active-Inference Neuro-Symbolic Gateway and responding with Rich Embeds.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')))

from server import ctx
from gateways.discord_adapter import DiscordGateway

def run_discord_bot() -> int:
    """Wire the Discord adapter or admit there is no listener.

    F35: this runner used to construct the gateway, print "ready", and
    exit -- advertising omnichannel support with no message source at all.
    Discord delivery requires a Gateway websocket client, which is not a
    project dependency, so the honest outcome is a non-zero exit naming
    the missing component rather than a silent pretend-ready.
    """
    bot_token = os.getenv("DISCORD_BOT_TOKEN")
    if not bot_token:
        print("[Discord Bot] УВАГА: DISCORD_BOT_TOKEN не задано у .env файлі.")
        print("[Discord Bot] Для запуску живого бота додайте рядок у .env:")
        print("DISCORD_BOT_TOKEN=MTA...your_discord_token_here")
        return 2

    DiscordGateway(
        agent=ctx.agent,
        runner=ctx.runner,
        schema_prompt=ctx.schema_prompt,
        bot_token=bot_token,
        session_store=ctx.session_store,
    )
    print("[Discord Bot] ПОМИЛКА: адаптер сформовано, але реал-time listener "
          "(Discord Gateway websocket) у проєкті відсутній.")
    print("[Discord Bot] Канал не запущено -- чесний вихід замість "
          "імітації роботи.")
    return 1

if __name__ == "__main__":
    sys.exit(run_discord_bot())
