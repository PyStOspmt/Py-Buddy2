"""
Py-Buddy: Zero-Hallucination Grounding Gate & Post-Generation Verifier.
Verifies product mentions, positional entity-price bindings, arithmetic sums,
sentence-level attribute claims, and tool verdict preservation against per-run evidence.
"""

import os
import re
import sqlite3
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from catalog_query import GroundingSet
from claim_verifier import split_sentence_spans, verify_answer_against_rows
from react_tools import agent_grounding
from source_currency import format_price_with_currency
from source_projection import build_source_projection, quote_ident

#: Currency-tagged price token pattern shared with answer verification.
PRICE_TOKEN_RE = re.compile(
    r"(?:[$€][ \t\u00a0]*(\d[\d \t\u00a0'.,]*\d|\d)"
    r"|(\d[\d \t\u00a0'.,]*\d|\d)[ \t\u00a0]*(?:грн\b|грн\.?|UAH\b|₴|USD\b|EUR\b|[$€]))",
    re.IGNORECASE,
)


def _price_match_raw(match: re.Match) -> str:
    return match.group(1) or match.group(2) or ""

_CONFLICT_NOTICE_RE = re.compile(
    r"\b(?:несуміс\w*|не\s*сумісн\w*|конфлікт\w*|не\s+можна|неможлив\w*)\b",
    re.IGNORECASE | re.UNICODE,
)
_NEGATIVE_STATUS_RE = re.compile(
    r"\b(?:на\s+жаль|немає|недоступн\w*|не\s+знайдено|жодн\w*|"
    r"несуміс\w*|не\s*сумісн\w*|послаблен\w*|не\s+виконано|не\s+задовольня\w*)\b",
    re.IGNORECASE | re.UNICODE,
)
_PRE_NEG_RE = re.compile(
    r"\b(?:на\s+жаль|немає|недоступн\w*|не\s+знайдено|жодн\w*)\b",
    re.IGNORECASE | re.UNICODE,
)


def norm_name(value: object) -> str:
    """Normalize text for case- and diacritic-insensitive entity comparisons."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_word(needle_norm: str, haystack_norm: str) -> bool:
    """Whole-word containment for normalized product names (minimum length 4)."""
    if len(needle_norm) < 4:
        return False
    return re.search(r"(?<!\w)" + re.escape(needle_norm) + r"(?!\w)", haystack_norm) is not None


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Return character spans for sentences without splitting decimals, abbreviations, or ordinals."""
    return split_sentence_spans(text or "")


def catalog_names(agent: Any) -> List[Tuple[str, str]]:
    """Return cached (name, normalized_name) pairs read lazily from agent.db_path."""
    cache = getattr(agent, "_catalog_names_cache", None)
    if cache is None:
        pairs: List[Tuple[str, str]] = []
        db = getattr(agent, "db_path", None)
        if db and os.path.exists(db):
            try:
                proj = build_source_projection(db)
                table = proj.entity_table
                name_res = proj.resolve("name")
                if table and name_res.resolved:
                    n_col = quote_ident(name_res.column)
                    t_id = quote_ident(table)
                    ro_uri = "file:%s?mode=ro" % os.path.abspath(db).replace("\\", "/")
                    conn = sqlite3.connect(ro_uri, uri=True)
                    try:
                        pairs = [
                            (str(r[0]), norm_name(str(r[0])))
                            for r in conn.execute(
                                f"SELECT {n_col} FROM {t_id} WHERE {n_col} IS NOT NULL"
                            )
                        ]
                    finally:
                        conn.close()
            except Exception:
                pairs = []
        agent._catalog_names_cache = pairs
    return agent._catalog_names_cache


def ungrounded_catalog_mentions(agent: Any, text: str, gs: GroundingSet) -> List[str]:
    """Identify catalogue entity names appearing in text outside this run's evidence."""
    needle = norm_name(text)
    if not needle:
        return []
    grounded_keys = {norm_name(n) for n in gs.names}
    hits = {
        name
        for name, norm in catalog_names(agent)
        if norm not in grounded_keys and contains_word(norm, needle)
    }
    return sorted(hits)


def parse_price_token(raw: str) -> Optional[float]:
    """Parse Ukrainian/formatted numeric strings into float."""
    txt = re.sub(r"[\s']", "", str(raw))
    if "," in txt and "." in txt:
        txt = txt.replace(".", "").replace(",", ".")
    elif "," in txt:
        txt = txt.replace(",", ".")
    try:
        return float(txt)
    except ValueError:
        return None


def rows_ground_price(price: float, rows: List[Dict[str, Any]]) -> bool:
    """Return True if any grounded row matches the target price within rounding tolerance."""
    for row in rows:
        for key in ("price", "cost", "amount"):
            try:
                if row.get(key) is None:
                    continue
                if abs(float(row[key]) - float(price)) <= 0.51:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _format_price_suffix(row: Dict[str, Any], currency: Optional[str] = None) -> str:
    """Format a grounded price (or seller range on conflict) from row evidence."""
    try:
        price = float(row["price"]) if row.get("price") is not None else None
    except (TypeError, ValueError):
        price = None
    if price is not None:
        return f" — {format_price_with_currency(price, currency)}"
    conflicts: List[float] = []
    for raw in row.get("price_conflicts") or ():
        try:
            if raw is not None:
                conflicts.append(float(raw))
        except (TypeError, ValueError):
            continue
    if conflicts:
        lo, hi = min(conflicts), max(conflicts)
        if lo == hi:
            return f" — {format_price_with_currency(lo, currency)}"
        lo_str = format_price_with_currency(lo, currency)
        hi_str = format_price_with_currency(hi, currency)
        return f" — від {lo_str} до {hi_str}"
    return ""


def render_from_rows(
    rows: List[Dict[str, Any]],
    currency: Optional[str] = None,
) -> str:
    """Render a strictly grounded list of up to 5 items directly from verified rows."""
    lines = []
    for row in rows[:5]:
        name = row.get("name") or row.get("title") or row.get("id", "?")
        lines.append(f"- {name}{_format_price_suffix(row, currency=currency)}")
    return (
        "Знайшов у каталозі за вашим запитом:\n" + "\n".join(lines)
        if lines
        else "За вашим запитом у каталозі нічого не знайдено — ціну назвати неможливо."
    )


def grounded_synthesis(
    items: List[Dict[str, Any]],
    note: str = "",
    currency: Optional[str] = None,
) -> str:
    """Render verified recommendations strictly from grounded catalog rows."""
    lines = ["Ось що реально є в каталозі за вашими критеріями:"]
    for r in items:
        lines.append(f"- {r.get('name')}{_format_price_suffix(r, currency=currency)}")
    if note:
        lines.append(note)
    return "\n".join(lines)


def compute_arithmetic_closures(rows: List[Dict[str, Any]]) -> Set[float]:
    """Calculate valid subset sums and differences across bounded grounded prices."""
    prices: List[float] = []
    for r in rows:
        for key in ("price", "cost", "amount"):
            if r.get(key) is not None:
                try:
                    p = float(r[key])
                    if p > 0:
                        prices.append(p)
                    break
                except (TypeError, ValueError):
                    pass

    closures: Set[float] = set(prices)
    n = min(len(prices), 6)
    for i in range(n):
        for j in range(i + 1, n):
            closures.add(round(prices[i] + prices[j], 2))
            closures.add(round(abs(prices[i] - prices[j]), 2))
            for k in range(j + 1, n):
                closures.add(round(prices[i] + prices[j] + prices[k], 2))
                for m in range(k + 1, n):
                    closures.add(round(prices[i] + prices[j] + prices[k] + prices[m], 2))
    return closures


def extract_query_numbers(query_text: str) -> Set[float]:
    """Extract numeric budgets or thresholds explicitly supplied by the user."""
    res: Set[float] = set()
    if not query_text:
        return res
    for m in PRICE_TOKEN_RE.finditer(query_text):
        p = parse_price_token(_price_match_raw(m))
        if p is not None:
            res.add(p)
    for raw_num in re.findall(r"\b(\d{3,7}(?:[.,]\d+)?)\b", query_text):
        p = parse_price_token(raw_num)
        if p is not None:
            res.add(p)
    return res


def extract_tool_statuses(
    execution_history: List[Dict[str, Any]],
) -> Tuple[List[str], Optional[str], Optional[str]]:
    """Extract hypergraph conflict notes, verdict label, and Pareto relaxation notice."""
    hypergraph_conflicts: List[str] = []
    compat_verdict: Optional[str] = None
    relax_notice: Optional[str] = None
    for rec in execution_history:
        tname = rec.get("tool_name")
        obs = rec.get("observation") if isinstance(rec.get("observation"), dict) else {}
        if tname == "hypergraph_check":
            v = obs.get("verdict")
            if v in ("incompatible", "candidate_conflict") or obs.get("is_compatible") is False:
                compat_verdict = str(v or "incompatible")
                for c in obs.get("candidate_conflicts") or obs.get("conflicts") or []:
                    if c and str(c) not in hypergraph_conflicts:
                        hypergraph_conflicts.append(str(c))
        elif tname == "pareto_relax":
            if (
                obs.get("relaxed") is True
                or obs.get("nvfg_status") == "RELAXED"
                or obs.get("success") is False
            ):
                rt = str(obs.get("relaxation_text") or "").strip()
                if rt:
                    relax_notice = rt
                elif relax_notice is None:
                    relax_notice = (
                        "На жаль, точних варіантів за всіма заданими обмеженнями в каталозі немає."
                    )
    return hypergraph_conflicts, compat_verdict, relax_notice


def preserve_tool_status(
    text: str,
    hypergraph_conflicts: List[str],
    compat_verdict: Optional[str],
    relax_notice: Optional[str],
    pre_text: str = "",
) -> str:
    """Ensure conflict or relaxation verdicts survive response correction and pruning."""
    tail = [s for s in re.split(r"[.!?\n]+", text or "") if s.strip()]
    tail_sent = tail[-1] if tail else ""

    if hypergraph_conflicts or compat_verdict in ("incompatible", "candidate_conflict"):
        if not _CONFLICT_NOTICE_RE.search(tail_sent):
            conflict_msg = (
                "Висновок: компоненти несумісні за перевіркою каталогу ("
                + "; ".join(hypergraph_conflicts[:2])
                + ")."
                if hypergraph_conflicts
                else "Висновок: компоненти несумісні за даними каталогу."
            )
            return f"{text}\n{conflict_msg}" if text else conflict_msg
        return text

    had_neg_in_pre = bool(_PRE_NEG_RE.search(pre_text or ""))
    if (relax_notice is not None and not _NEGATIVE_STATUS_RE.search(tail_sent)) or (
        had_neg_in_pre and not _NEGATIVE_STATUS_RE.search(text or "")
    ):
        notice = (
            relax_notice
            if relax_notice
            else "На жаль, точних варіантів за всіма заданими обмеженнями в каталозі немає."
        )
        return f"{text}\n{notice}" if text else notice
    return text


def _item_prices(entry: Dict[str, Any]) -> List[float]:
    vals: List[float] = []
    if entry.get("price") is not None:
        vals.append(float(entry["price"]))
    for cp in entry.get("price_conflicts") or ():
        if cp is not None:
            vals.append(float(cp))
    return vals


def _validate_positional_prices(
    text: str,
    allowed_by_name: Dict[str, Dict[str, Any]],
    user_numbers: List[float],
    *,
    repair: bool,
    currency: Optional[str] = None,
) -> Tuple[str, bool, bool]:
    """Validate (and optionally repair) currency tokens positionally against preceding items."""
    mention_positions: List[Tuple[int, str]] = []
    for name in allowed_by_name:
        for m in re.finditer(re.escape(name), text or "", re.IGNORECASE):
            mention_positions.append((m.start(), name))
    mention_positions.sort()
    spans = sentence_spans(text)

    def _sentence_products(pos: int) -> List[str]:
        for start, end in spans:
            if start <= pos < end:
                seg = norm_name(text[start:end])
                return [n for n in allowed_by_name if norm_name(n) in seg]
        return []

    corrected_text = text
    price_corrected = False
    uncorrectable_stray = False

    for _m in list(PRICE_TOKEN_RE.finditer(text or "")):
        _value = parse_price_token(_price_match_raw(_m))
        if _value is None:
            continue

        preceding = [n for p, n in mention_positions if p < _m.start()]
        if preceding and any(
            abs(_value - vp) <= 0.51 for vp in _item_prices(allowed_by_name[preceding[-1]])
        ):
            continue

        if any(abs(_value - u) <= 1.0 for u in user_numbers):
            continue

        sent_products = _sentence_products(_m.start())
        if len(sent_products) >= 2:
            per_item = [_item_prices(allowed_by_name[n]) for n in sent_products]
            if all(per_item):
                sums = {0.0}
                for p_list in per_item:
                    sums = {s + p for s in sums for p in p_list}
                if any(abs(_value - tot) <= 2.0 for tot in sums):
                    continue

        if repair:
            bound = next(
                (allowed_by_name[n] for n in sent_products if _item_prices(allowed_by_name[n])),
                None,
            )
            if bound is not None:
                fix_p = _item_prices(bound)[0]
                corrected_text = corrected_text.replace(
                    _m.group(0),
                    format_price_with_currency(fix_p, currency),
                    1,
                )
                price_corrected = True
                continue

        uncorrectable_stray = True
        break

    return corrected_text, price_corrected, uncorrectable_stray


def _prune_orphaned_price_continuations(
    sentences: List[Any],
    allowed_by_name: Dict[str, Dict[str, Any]],
    user_numbers: List[float],
) -> str:
    """Drop productless price lines whose owning item sentence was pruned or re-attributed."""
    kept: List[str] = []
    dropped_item_context = False

    for sv in sentences:
        if sv.verdict == "unsupported":
            if sv.products:
                dropped_item_context = True
            continue
        if sv.products:
            dropped_item_context = False
            kept.append(sv.text)
            continue
        if dropped_item_context and PRICE_TOKEN_RE.search(sv.text or ""):
            continue
        # Ensure a bare price sentence stays anchored to its surviving preceding item.
        if PRICE_TOKEN_RE.search(sv.text or ""):
            candidate = "\n".join(kept + [sv.text])
            _, _, stray = _validate_positional_prices(
                candidate, allowed_by_name, user_numbers, repair=False
            )
            if stray:
                continue
        kept.append(sv.text)

    return "\n".join(kept).strip()


def _extract_tool_counts(execution_history: List[Dict[str, Any]]) -> List[int]:
    """Collect exact integer cardinalities returned by executed tools."""
    counts: Set[int] = set()
    for rec in execution_history or ():
        obs = rec.get("observation")
        if not isinstance(obs, dict):
            obs = rec.get("result")
        if not isinstance(obs, dict):
            continue
        for key in ("row_count", "pareto_front_count", "evaluated_rules", "total"):
            val = obs.get(key)
            if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
                counts.add(val)
        for key in ("rows", "hits", "products", "pareto_front"):
            seq = obs.get(key)
            if isinstance(seq, list):
                counts.add(len(seq))
    return sorted(counts)


def _mentioned_rows_in_prose_order(
    synthesis_text: str,
    allowed: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract grounded rows explicitly named in prose, sorted by first appearance."""
    needle = norm_name(synthesis_text)
    if not needle:
        return []
    candidates: List[Tuple[int, int, int, Dict[str, Any]]] = []
    seen_names: Set[str] = set()
    for idx, row in enumerate(allowed):
        raw_name = str(row.get("name") or "")
        norm_n = norm_name(raw_name)
        if not norm_n or norm_n in seen_names:
            continue
        matches = list(re.finditer(r"(?<!\w)" + re.escape(norm_n) + r"(?!\w)", needle))
        if not matches and len(norm_n) >= 4:
            pos = needle.find(norm_n)
            if pos >= 0:
                matches = []
                seen_names.add(norm_n)
                candidates.append((pos, pos + len(norm_n), idx, row))
                continue
        if matches:
            seen_names.add(norm_n)
            m0 = matches[0]
            candidates.append((m0.start(), m0.end(), idx, row))
    # Longer spans win ties at the same start offset to avoid substring shadowing.
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    occupied: List[Tuple[int, int]] = []
    ordered: List[Dict[str, Any]] = []
    for start, end, _idx, row in candidates:
        if any(os_s <= start and end <= os_e for os_s, os_e in occupied):
            continue
        occupied.append((start, end))
        ordered.append(dict(row))
    return ordered


def _collect_grounding_context(
    agent: Any,
    state: Any,
    synthesis_text: str,
    recommended_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble per-session grounding evidence and positional validation results."""
    currency = getattr(agent, "currency", None)
    hypergraph_conflicts, compat_verdict, relax_notice = extract_tool_statuses(
        state.execution_history
    )
    tool_counts = _extract_tool_counts(state.execution_history)
    gs = agent_grounding(agent, state)
    allowed = gs.projected_recommendations()
    allowed_by_name = {r["name"]: r for r in allowed}

    ungrounded_mentions = ungrounded_catalog_mentions(agent, synthesis_text, gs)
    user_numbers = extract_query_numbers(state.user_query)
    corrected_text, price_corrected, uncorrectable_stray = _validate_positional_prices(
        synthesis_text,
        allowed_by_name,
        user_numbers,
        repair=True,
        currency=currency,
    )
    effective_text = corrected_text if price_corrected else synthesis_text

    mentioned_verified = _mentioned_rows_in_prose_order(effective_text, allowed)
    rec_names = [
        str(r.get("name", "")) if isinstance(r, dict) else str(r or "")
        for r in (recommended_items or [])
    ]
    if mentioned_verified:
        verified = mentioned_verified
    else:
        verified = [dict(allowed_by_name[n]) for n in rec_names if n in allowed_by_name]
    fabricated = sorted({n for n in rec_names if n} - set(allowed_by_name))
    return {
        "currency": currency,
        "hypergraph_conflicts": hypergraph_conflicts,
        "compat_verdict": compat_verdict,
        "relax_notice": relax_notice,
        "tool_counts": tool_counts,
        "gs": gs,
        "allowed": allowed,
        "allowed_by_name": allowed_by_name,
        "ungrounded_mentions": ungrounded_mentions,
        "user_numbers": user_numbers,
        "effective_text": effective_text,
        "price_corrected": price_corrected,
        "uncorrectable_stray": uncorrectable_stray,
        "verified": verified,
        "fabricated": fabricated,
    }


def _format_evidence_summary(
    evidence_rows: List[Dict[str, Any]],
    limit: int = 8,
    currency: Optional[str] = None,
) -> str:
    """Format grounded rows compactly for single-retry ClaimVerifier feedback."""
    if not evidence_rows:
        return (
            "Підтверджені дані інструментів у цій сесії відсутні — "
            "якщо точних значень немає, чесно повідом про відсутність даних."
        )
    lines: List[str] = []
    for row in evidence_rows[:limit]:
        name = row.get("name") or row.get("id") or "позиція"
        price = row.get("price")
        parts: List[str] = [str(name)]
        if price is not None:
            try:
                parts.append(f"ціна={format_price_with_currency(float(price), currency)}")
            except (TypeError, ValueError):
                parts.append(f"ціна={price}")
        specs = row.get("specs") if isinstance(row.get("specs"), dict) else {}
        extra_kv = []
        for k, v in list(specs.items())[:5]:
            if v is not None and str(v).strip():
                extra_kv.append(f"{k}={v}")
        for k in ("category", "brand", "stock_quantity"):
            if row.get(k) is not None and str(row.get(k)).strip() and len(extra_kv) < 6:
                extra_kv.append(f"{k}={row[k]}")
        if extra_kv:
            parts.append(", ".join(extra_kv))
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)


def diagnose_grounding_feedback(
    agent: Any,
    state: Any,
    synthesis_text: str,
    recommended_items: List[Dict[str, Any]],
) -> Optional[Tuple[str, int]]:
    """Inspect a candidate LLM answer without mutating ``state``.

    Returns ``(repair_feedback_prompt, issue_count)`` when ClaimVerifier or the
    grounding checks detect unsupported assertions, or ``None`` when clean.
    """
    ctx = _collect_grounding_context(agent, state, synthesis_text, recommended_items)
    evidence_rows = ctx["gs"].verification_rows()
    report = verify_answer_against_rows(
        ctx["effective_text"],
        evidence_rows,
        state.user_query,
        tool_conflicts=ctx["hypergraph_conflicts"],
        compatibility_verdict=ctx["compat_verdict"],
        tool_counts=ctx["tool_counts"],
    )

    issues: List[str] = []
    if ctx["uncorrectable_stray"]:
        issues.append(
            "У тексті вказана ціна, якої немає у вибраних рядках каталогу."
        )
    if ctx["fabricated"]:
        issues.append(
            "У recommended_items передано непідтверджені назви: "
            + ", ".join(ctx["fabricated"][:5])
        )
    if ctx["ungrounded_mentions"]:
        issues.append(
            "У тексті згадано товари каталогу поза результатами інструментів: "
            + ", ".join(ctx["ungrounded_mentions"][:5])
        )
    for sv in report.sentences:
        if sv.verdict == "unsupported":
            why = ", ".join(sv.unsupported) or "непідтверджене твердження"
            issues.append(f"Речення «{sv.text}»: {why}")

    if not issues:
        return None

    feedback = (
        "Твоя чернетка відповіді не пройшла перевірку фактів (ClaimVerifier). "
        "Виявлені розбіжності:\n"
        + "\n".join(f"- {item}" for item in issues[:6])
        + "\n\nПідтверджені факти з уже виконаних інструментів:\n"
        + _format_evidence_summary(evidence_rows, currency=ctx["currency"])
        + "\n\nПереформулюй фінальну відповідь ОДНИМ чистим JSON з ключем `final_answer`, "
        "використовуючи ТІЛЬКИ наведені вище факти без вигаданих чисел, атрибутів чи цін; "
        "якщо потрібних даних немає у каталозі — прямо скажи, що інформації бракує."
    )
    return feedback, len(issues)


def finalize_grounded_response(
    agent: Any,
    state: Any,
    synthesis_text: str,
    recommended_items: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    """Run positional entity-price validation and ClaimVerifier over candidate output."""
    ctx = _collect_grounding_context(agent, state, synthesis_text, recommended_items)
    currency = ctx["currency"]
    hypergraph_conflicts = ctx["hypergraph_conflicts"]
    compat_verdict = ctx["compat_verdict"]
    relax_notice = ctx["relax_notice"]
    tool_counts = ctx["tool_counts"]
    gs = ctx["gs"]
    allowed = ctx["allowed"]
    allowed_by_name = ctx["allowed_by_name"]
    ungrounded_mentions = ctx["ungrounded_mentions"]
    user_numbers = ctx["user_numbers"]
    synthesis_text = ctx["effective_text"]
    uncorrectable_stray = ctx["uncorrectable_stray"]
    verified = ctx["verified"]
    fabricated = ctx["fabricated"]

    if uncorrectable_stray:
        if verified:
            rows = verified[:3] or allowed[:3]
            state.is_terminated = True
            state.termination_reason = "SUCCESS_GROUNDED_PRICE_CORRECTED"
            state.recommended_items = rows
            state.final_response = preserve_tool_status(
                render_from_rows(rows, currency=currency),
                hypergraph_conflicts,
                compat_verdict,
                relax_notice,
                synthesis_text,
            )
            return True, state.final_response
        state.is_terminated = True
        state.termination_reason = "HALLUCINATION_GATE_VIOLATION"
        safe_fallback = (
            "У відповіді була ціна, якої немає в каталозі, тому її не показую. "
            "Уточніть, будь ласка, запит — я не називаю ціни, яких немає в базі."
        )
        state.final_response = safe_fallback
        return False, safe_fallback

    if (fabricated or ungrounded_mentions) and not verified:
        if allowed:
            verified = allowed[:3]
            state.termination_reason = "SUCCESS_GROUNDED_AFTER_CORRECTION"
            note = (
                " Частину згаданого не вдалося підтвердити в каталозі, "
                "тому показую лише те, що реально є в наявності."
            )
            state.is_terminated = True
            state.recommended_items = verified
            state.final_response = preserve_tool_status(
                grounded_synthesis(verified, note, currency=currency),
                hypergraph_conflicts,
                compat_verdict,
                relax_notice,
                synthesis_text,
            )
            return True, state.final_response
        state.is_terminated = True
        state.termination_reason = "HALLUCINATION_GATE_VIOLATION"
        safe_fallback = (
            "Не зміг знайти жодного товару, який підтверджується даними каталогу. "
            "Уточніть, будь ласка, запит — я не стану називати ціни або "
            "характеристики, яких немає в базі."
        )
        state.final_response = safe_fallback
        return False, safe_fallback

    state.is_terminated = True
    if fabricated or ungrounded_mentions:
        state.termination_reason = "SUCCESS_GROUNDED_WITH_CORRECTION"
        removed = len(fabricated) + len(ungrounded_mentions)
        note = (
            f" Частину згаданого ({removed} поз.) не вдалося підтвердити "
            "в каталозі, тому відповідь показує лише перевірені товари."
        )
        state.recommended_items = verified
        state.final_response = preserve_tool_status(
            grounded_synthesis(verified, note, currency=currency),
            hypergraph_conflicts,
            compat_verdict,
            relax_notice,
            synthesis_text,
        )
        return True, state.final_response

    evidence_rows = gs.verification_rows()
    state.pre_verifier_response = synthesis_text
    report = verify_answer_against_rows(
        synthesis_text,
        evidence_rows,
        state.user_query,
        tool_conflicts=hypergraph_conflicts,
        compatibility_verdict=compat_verdict,
        tool_counts=tool_counts,
    )
    state.verification_report = report.to_dict()
    if report.n_unsupported:
        corrected = _prune_orphaned_price_continuations(
            report.sentences, allowed_by_name, user_numbers
        )
        if corrected and verified:
            corrected = preserve_tool_status(
                corrected,
                hypergraph_conflicts,
                compat_verdict,
                relax_notice,
                synthesis_text,
            )
            state.termination_reason = "SUCCESS_GROUNDED_WITH_CORRECTION"
            state.recommended_items = verified
            state.final_response = corrected
            return True, corrected
        rows = verified or allowed[:3]
        if not rows:
            state.termination_reason = "HALLUCINATION_GATE_VIOLATION"
            state.is_terminated = True
            safe_fallback = (
                "Частину тверджень не вдалося підтвердити даними каталогу, "
                "а перевірених позицій для відповіді немає."
            )
            state.final_response = safe_fallback
            return False, safe_fallback
        state.termination_reason = "SUCCESS_GROUNDED_AFTER_CORRECTION"
        state.is_terminated = True
        state.recommended_items = rows
        note = (
            " Частину тверджень не вдалося підтвердити отриманими "
            "даними, тому відповідь показує лише перевірене."
        )
        state.final_response = preserve_tool_status(
            grounded_synthesis(rows, note, currency=currency),
            hypergraph_conflicts,
            compat_verdict,
            relax_notice,
            synthesis_text,
        )
        return True, state.final_response

    state.termination_reason = "SUCCESS_GROUNDED"
    state.recommended_items = verified
    state.final_response = preserve_tool_status(
        synthesis_text,
        hypergraph_conflicts,
        compat_verdict,
        relax_notice,
        synthesis_text,
    )
    return True, state.final_response
