"""
Py-Buddy: OpenRouter API Client (DeepSeek / Llama 3.3 / Qwen)
Provides ultra-low-cost, high-throughput LLM reasoning via OpenRouter.
"""

import os
import json
import re
import time
import random
import hashlib
import threading
import requests
from typing import Dict, List, Any, Optional, Tuple


_MODELS_CACHE_LOCK = threading.Lock()
_MODELS_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_MODELS_CACHE_TTL_S = 120.0


def _clear_models_cache() -> None:
    """Clear the in-memory live OpenRouter model catalogue cache (tests/reset)."""
    with _MODELS_CACHE_LOCK:
        _MODELS_CACHE.clear()


def _parse_price_per_mtok(raw_val: Any) -> Optional[float]:
    try:
        per_tok = float(raw_val)
    except (TypeError, ValueError):
        return None
    if per_tok < 0:
        return None
    return round(per_tok * 1_000_000.0, 4)


def fetch_openrouter_models(
    api_key: str,
    *,
    validate_key: bool = True,
    timeout: float = 8.0,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Validate an OpenRouter key and fetch the live available model catalogue.

    Caches successful listings in memory keyed strictly by SHA-256 of the key
    (never storing plaintext keys on disk). Returns `(models, error_message)`.
    """
    key = (api_key or "").strip()
    if not key:
        return [], "OpenRouter API-ключ не задано."

    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    now = time.time()
    with _MODELS_CACHE_LOCK:
        cached = _MODELS_CACHE.get(key_hash)
        if cached and (now - cached[0]) < _MODELS_CACHE_TTL_S:
            return list(cached[1]), None

    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/PyStOspmt/Py-Buddy2",
        "X-Title": "Py-Buddy",
    }

    if validate_key:
        try:
            key_resp = requests.get(
                "https://openrouter.ai/api/v1/key",
                headers=headers,
                timeout=min(timeout, 5.0),
            )
            if key_resp.status_code in (401, 403):
                return [], f"Недійсний або відхилений OpenRouter API-ключ (HTTP {key_resp.status_code})."
            if key_resp.status_code >= 500:
                return [], f"Сервер авторизації OpenRouter недоступний (HTTP {key_resp.status_code})."
        except Exception as exc:
            return [], f"Не вдалося перевірити ключ OpenRouter: {exc}"

    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code in (401, 403):
            return [], f"OpenRouter відхилив запит каталогу моделей (HTTP {resp.status_code})."
        if resp.status_code != 200:
            return [], f"Помилка OpenRouter /models (HTTP {resp.status_code})."
        payload = resp.json()
        raw_models = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(raw_models, list):
            return [], "Некоректний формат каталогу моделей від OpenRouter."
    except Exception as exc:
        return [], f"Не вдалося завантажити моделі з OpenRouter: {exc}"

    parsed: List[Dict[str, Any]] = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
        p_in = _parse_price_per_mtok(pricing.get("prompt"))
        p_out = _parse_price_per_mtok(pricing.get("completion"))
        is_free = (
            mid.endswith(":free")
            or (p_in == 0.0 and p_out == 0.0)
        )
        parsed.append({
            "id": mid,
            "name": str(item.get("name") or mid),
            "context_length": int(item.get("context_length") or 0),
            "is_free": bool(is_free),
            "prompt_price": 0.0 if is_free else p_in,
            "completion_price": 0.0 if is_free else p_out,
            "description": str(item.get("description") or "")[:240],
        })

    parsed.sort(key=lambda m: (0 if m["is_free"] else 1, m["id"].lower()))
    with _MODELS_CACHE_LOCK:
        _MODELS_CACHE[key_hash] = (now, parsed)
    return parsed, None


def _env_fallbacks(env: Optional[Dict[str, str]] = None,
                    exclude: Optional[str] = None) -> List[str]:
    """Read OPENROUTER_FALLBACKS without importing dotenv at module scope."""
    raw = (env or os.environ).get("OPENROUTER_FALLBACKS", "")
    out: List[str] = []
    for part in raw.split(","):
        name = part.strip()
        if name and name != exclude and name not in out:
            out.append(name)
    return out


class OpenRouterAgentRunner:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        app_title: str = "Py-Buddy",
        require_key: bool = True,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        env_model = os.environ.get("OPENROUTER_MODEL")

        # Search CWD and parent directories up to repo root (src/core -> src -> root) for .env.
        _module_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths = [
            ".env",
            os.path.join(_module_dir, "..", "..", ".env"),
            os.path.join(_module_dir, "..", ".env"),
            os.path.join(_module_dir, ".env"),
        ]
        for c_path in candidate_paths:
            if os.path.exists(c_path):
                with open(c_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not self.api_key and line.startswith("OPENROUTER_API_KEY="):
                            self.api_key = line.strip().split("=", 1)[1].strip("\"'")
                        if not env_model and line.startswith("OPENROUTER_MODEL="):
                            env_model = line.strip().split("=", 1)[1].strip("\"'")
                if self.api_key:
                    break

        if require_key and not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set. Provide it in .env or environment.")

        self.model_name = model_name or env_model or "deepseek/deepseek-v4-flash-0731"
        if model_name and env_model and model_name != env_model:
            self.fallback_models = [m for m in [env_model] if m != self.model_name]
        else:
            self.fallback_models = _env_fallbacks(exclude=self.model_name)
        self.app_title = app_title
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # Model that produced the last successful step (pool may rotate).
        self.last_model: Optional[str] = None

    def _ensure_api_key(self) -> str:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set. Provide it in .env or environment.")
        return self.api_key

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Direct text-to-text generation via OpenRouter API."""
        self._ensure_api_key()
        models_to_try = [self.model_name] + [m for m in self.fallback_models if m != self.model_name]
        for model in models_to_try:
            for attempt in range(3):
                try:
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": 4000
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/PyStOspmt/Py-Buddy2",
                        "X-Title": "Py-Buddy"
                    }
                    resp = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        data=json.dumps(payload),
                        timeout=35
                    )
                    if resp.status_code in (401, 403):
                        return ""
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices", [])
                        if choices:
                            msg = choices[0].get("message", {})
                            content = msg.get("content") or msg.get("reasoning_content") or ""
                            if content.strip():
                                self.last_model = model
                                return content.strip()
                    elif resp.status_code != 429 and resp.status_code < 500:
                        return ""
                except Exception:
                    time.sleep(1.0)
                    continue
        return ""

    def answer_once(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        price_in_per_mtok: float = 0.0,
        price_out_per_mtok: float = 0.0,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> Tuple[str, float]:
        """Single-turn plain-text completion returning (text, cost_usd)."""
        self._ensure_api_key()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/PyStOspmt/Py-Buddy2",
            "X-Title": self.app_title,
            "Content-Type": "application/json",
        }
        models = [self.model_name]
        for extra in (self.fallback_models or []):
            if extra and extra not in models:
                models.append(extra)
        last_err: Optional[str] = None
        for attempt in range(4):
            model = models[attempt % len(models)] if models else self.model_name
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            try:
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=45)
                if resp.status_code in (401, 403):
                    last_err = f"OpenRouter API error {resp.status_code}: {resp.text[:200]}"
                    break
                if resp.status_code == 429 or resp.status_code >= 500 or resp.status_code == 400:
                    last_err = f"OpenRouter API error {resp.status_code}: {resp.text[:200]}"
                    time.sleep(min(8.0, 2.0 ** attempt) + random.uniform(0, 0.5))
                    continue
                if resp.status_code != 200:
                    last_err = f"OpenRouter API error {resp.status_code}: {resp.text[:200]}"
                    continue
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    last_err = f"OpenRouter response without choices: {resp.text[:200]}"
                    time.sleep(min(8.0, 2.0 ** attempt) + random.uniform(0, 0.5))
                    continue
                usage = data.get("usage") or {}
                in_tok = int(usage.get("prompt_tokens") or 0)
                out_tok = int(usage.get("completion_tokens") or 0)
                self.total_prompt_tokens += in_tok
                self.total_completion_tokens += out_tok
                self.last_model = model
                cost = (in_tok * float(price_in_per_mtok) + out_tok * float(price_out_per_mtok)) / 1_000_000.0
                msg = choices[0].get("message") or {}
                text = str(msg.get("content") or "").strip()
                if text:
                    return text, cost
                last_err = "Empty content from model"
            except Exception as exc:
                last_err = str(exc)
                time.sleep(min(8.0, 2.0 ** attempt) + random.uniform(0, 0.5))
        raise RuntimeError(f"OpenRouter answer_once unavailable: {last_err}")

    @staticmethod
    def _parse_step(content: str) -> Optional[Dict[str, Any]]:
        """
        Parse one model step out of its text content.

        F02: an explicit `final_answer:` marker is a limited format-recovery
        attempt; anything else unparseable is a typed protocol error, never
        a wrapped final answer.
        """
        parsed: Optional[Dict[str, Any]] = None
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
        if fence:
            try:
                parsed = json.loads(fence.group(1))
            except Exception:
                parsed = None
        if parsed is None:
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = None
        if parsed is None:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    parsed = None
        if parsed is not None:
            return parsed

        bare = re.search(
            r"final_answer\s*[:=]\s*(.+)", content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if bare:
            return {
                "thought": "Bare-text fallback",
                "final_answer": bare.group(1).strip().strip("\"'"),
                "recommended_items": [],
            }
        # Unparseable output is returned as a typed protocol error rather than a fake answer.
        return {
            "protocol_error": "unparseable_model_output",
            "detail": content[:200],
        }

    @staticmethod
    def _build_react_messages(
        system_instruction: str,
        user_query: str,
        dialog_history: List[Dict[str, str]],
        current_step: int,
    ) -> List[Dict[str, str]]:
        """Assemble chat messages in strict chronological order (prior turns -> current query -> step trace)."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_instruction},
        ]
        prior_turns: List[Dict[str, str]] = []
        current_trace: List[Dict[str, str]] = []
        for h in dialog_history:
            content_str = str(h.get("content") or "")
            if not current_trace and content_str.startswith("[Попередній хід діалогу"):
                prior_turns.append({"role": h["role"], "content": content_str})
            else:
                current_trace.append({"role": h["role"], "content": content_str})

        messages.extend(prior_turns)
        messages.append({"role": "user", "content": f"Поточний запит користувача: {user_query}"})
        messages.extend(current_trace)

        if current_step >= 5:
            step_prompt = (
                f"Поточний крок ReAct: {current_step}. УВАГА: Це завершальний крок! "
                "Якщо в історії вже є необхідні дані або спостереження (Observations), ОБОВ'ЯЗКОВО сформуй 'final_answer' замість виклику нових інструментів!"
            )
        else:
            step_prompt = f"Поточний крок ReAct: {current_step}. Надай наступну дію у СУВОРОМУ форматі JSON (thought, action/final_answer)."

        messages.append({"role": "user", "content": step_prompt})
        return messages

    def generate_react_step(
        self,
        schema_prompt: str,
        user_query: str,
        dialog_history: List[Dict[str, str]],
        current_step: int,
        tools_description: str
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "provider_error": "OPENROUTER_API_KEY is not set. Provide it in .env, environment, or browser header.",
                "error_code": "API_KEY_MISSING",
            }
        self._ensure_api_key()
        system_instruction = f"""Ти — ядро універсального нейро-символьного консультанта платформи Py-Buddy з контролем фактів Zero-Hallucination Gate.
Твоє завдання — допомагати користувачеві виключно в межах підключеного активного каталогу товарів та бази даних.
ВАЖЛИВО: Твоя поточна спеціалізація, асортимент та всі доступні товари визначаються ВИКЛЮЧНО схемою та даними поточної бази даних нижче! Ніколи не стверджуй, що ти спеціалізуєшся на комп'ютерній техніці, якщо активний каталог присвячений іншій сфері (наприклад, харчові продукти, автозапчастини, загальний маркетплейс тощо).

{schema_prompt}

ДОСТУПНІ ІНСТРУМЕНТИ:
{tools_description}

ПРАВИЛА РОБОТИ (ReAct):
0. РОЗУМНИЙ ВИКЛИК АГЕНТА ТА ДІАЛОГ:
   - Якщо запит користувача є розмовою, привітанням, питанням про твоє ім'я/роль/можливості ("хто ти", "як тебе звуть", "що робиш"), або загальним запитанням про асортимент ("що ви продаєте", "який асортимент", "які товари є") БЕЗ потреби шукати конкретні екстремуми чи окремі артикули — ВІДРАЗУ повертай 'final_answer' на кроці 1 без виклику інструментів. При питанні про асортимент стисло перелічи категорії товарів відповідно до наведеної вище схеми активного каталогу. Спілкуйся природно, привітно та виключно українською мовою.
   - Якщо користувач шукає конкретні товари, запитує про наявність, точні ціни, характеристики або сумісність — спочатку виклич потрібний інструмент ('catalog_search', 'sql_query', 'hypergraph_check', 'pareto_relax', 'active_inference_step').
   - Якщо шуканого товару немає в базі даних або категорія не підтримується — ввічливо поясни це на основі каталогу і запропонуй корисні альтернативи серед наявних категорій.
1. Не вигадуй характеристики, наявність чи ціни. Всі дані бери ВИКЛЮЧНО через виклики інструментів. Якщо валюта джерела визначена у схемі або рядках (наприклад, грн / USD / EUR) — вказуй її біля кожної ціни; якщо валюта в джерелі не задана — не вигадуй її.
2. Якщо потрібні товари — використовуй 'catalog_search' (для швидкого пошуку за назвою/моделлю з точним підрахунком загальної кількості `total`, розбивкою за категоріями і сортуванням `price_asc`/`price_desc`) або 'sql_query' (для складних JOIN/EAV-умов).
   - Формуй запити з урахуванням наміру користувача та схеми підключеної бази даних.
   - ЗАВЖДИ перевіряй поля спостереження `truncated`, `ordered`, `total`, `showing` і `notice`: якщо `truncated=true` і `ordered=false`, вибірка є неповною у довільному порядку — НІКОЛИ не стверджуй за нею про «найдешевший», «найдорожчий» чи відсутність товару, а уточни запит з `ORDER BY ... ASC/DESC` або через `catalog_search`.
   - При пошуку екстремумів разом із кількістю в `sql_query` поєднуй реальні рядки (`id, name, price`) з `(SELECT COUNT(*) ...) AS total_matching` та `ORDER BY price ASC/DESC`, або використовуй `catalog_search`. Не обмежуйся голим `SELECT COUNT(*), MIN(price)` без `id, name, price`, якщо називаєш конкретний товар.
   - Якщо один і той самий товар або його модифікації пропонуються різними продавцями чи за різними цінами, аналізуй пропозиції та вказуй клієнту найвигідніший варіант.
   - Якщо інструмент 'sql_query' повертає 'fuzzy_suggestions', обов'язково проаналізуй ці схожі товари з бази даних та запропонуй їх клієнту.
   - Для значень з апострофом у SQL використовуй стандартний подвійний апостроф '' (наприклад, 'Пам''ять' або 'м''ясо').
3. Якщо користувач збирає або перевіряє сумісність деталей/компонентів — ОБОВ'ЯЗКОВО виклич 'hypergraph_check' зі списком їхніх ID.
4. Якщо бюджет замалий і товари не знайдено — виклич 'pareto_relax'.
5. Якщо вподобання незрозумілі — виклич 'active_inference_step'.
6. Формат твоєї відповіді повинен бути СУВОРИМ JSON такого виду:
Якщо потрібен інструмент:
{{
  "thought": "Коротке міркування, що треба перевірити",
  "action": "НАЗВА_ІНСТРУМЕНТУ",
  "action_input": {{ "параметр": "значення" }}
}}
Якщо ти маєш усю інформацію і готовий дати фінальну відповідь клієнту:
{{
  "thought": "Усі перевірки виконано, формую відповідь",
  "final_answer": "Текст зрозумілої відповіді покупцеві",
  "recommended_items": [
    {{ "id": "id_товару", "name": "назва", "price": 12345.0 }}
  ]
}}
7. СЕМАНТИКА ТА СИНОНІМИ (DOMAIN-AGNOSTIC):
   - Користувач може вживати розмовні форми, скорочення, синоніми або сленг. Завжди зіставляй намір користувача з реальними категоріями, брендами та полями активного каталогу, наведеними у схемі вище.
   - Якщо точна категорія чи сутність невідома — скористайся інструментом 'ontograph_navigate' або перевір перелік категорій через 'sql_query'.
   - Поняття 'топовий', 'найкращий', 'потужний', 'преміальний' означають оптимізацію за якісними характеристиками, рейтингом чи продуктивністю (ORDER BY ... DESC).
   - Поняття 'дешевий', 'бюджетний', 'економний' означають сортування за ціною (ORDER BY price ASC).
"""

        messages = self._build_react_messages(
            system_instruction, user_query, dialog_history, current_step
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/PyStOspmt/Py-Buddy2",
            "X-Title": self.app_title,
            "Content-Type": "application/json"
        }

        # The free tier is rate-limited: OpenRouter occasionally returns 429/5xx,
        # or a 200 body with no `choices` (upstream failure). Retry transient
        # failures with exponential backoff + jitter; count tokens only on a
        # successful parse.
        #
        # Model fallback pool: free models get withdrawn without notice, and a
        # single pinned model turns the whole live arm into a coin flip. The
        # runner therefore walks the pool (primary + OPENROUTER_FALLBACKS) per
        # attempt, recording which model actually answered so the transcript
        # stays attributable.
        models = [self.model_name]
        for extra in (self.fallback_models or []):
            if extra and extra not in models:
                models.append(extra)
        last_err: Optional[str] = None
        for attempt in range(4):  # initial attempt + 3 retries
            model = models[attempt % len(models)] if models else self.model_name
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=45)
                if response.status_code == 400 and ("structured-outputs" in response.text or "response_format" in response.text):
                    # Provider does not support structured-outputs: retry immediately without response_format
                    payload_no_rf = dict(payload)
                    payload_no_rf.pop("response_format", None)
                    response = requests.post(self.api_url, headers=headers, json=payload_no_rf, timeout=45)

                if response.status_code in (401, 403):
                    last_err = f"OpenRouter API error {response.status_code}: {response.text[:200]}"
                    break
                if response.status_code == 429 or response.status_code >= 500 or response.status_code == 400:
                    last_err = f"OpenRouter API error {response.status_code}: {response.text[:200]}"
                    time.sleep(min(8.0, 2.0 ** attempt) + random.uniform(0, 0.5))
                    continue
                if response.status_code != 200:
                    last_err = f"OpenRouter API error {response.status_code}: {response.text[:200]}"
                    continue

                res_json = response.json()
                choices = res_json.get("choices") or []
                if not choices:
                    last_err = f"OpenRouter response without choices: {response.text[:200]}"
                    time.sleep(min(8.0, 2.0 ** attempt) + random.uniform(0, 0.5))
                    continue

                usage = res_json.get("usage", {})
                self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                self.total_completion_tokens += usage.get("completion_tokens", 0)
                self.last_model = model

                msg_obj = choices[0].get("message") or {}
                # F32: provider reasoning is internal chain-of-thought, never
                # client-visible content. Falling back to it used to leak
                # hidden reasoning into the answer path.
                raw_content = msg_obj.get("content") or ""

                # Check for direct tool_calls structure if content is missing
                if not raw_content and msg_obj.get("tool_calls"):
                    tc = msg_obj["tool_calls"][0]
                    fn = tc.get("function", {})
                    try:
                        fn_args = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        fn_args = {}
                    return {
                        "thought": "Виклик інструменту від моделі через tool_calls",
                        "action": fn.get("name"),
                        "action_input": fn_args
                    }

                content = str(raw_content).strip() if raw_content is not None else ""
                if not content:
                    last_err = "Empty content from model"
                    continue

                parsed = self._parse_step(content)
                if parsed is not None:
                    return parsed
            except Exception as e:
                last_err = str(e)
                time.sleep(min(8.0, 2.0 ** attempt) + random.uniform(0, 0.5))
                continue

        # F02/F08: a provider outage is a provider_error outcome, not a fake
        # apology answer. The agent turns this into an honest client message
        # and the trace records PROVIDER_ERROR.
        return {
            "provider_error": f"OpenRouter unavailable after retries: {last_err}",
        }
