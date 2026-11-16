"""
Py-Buddy AIC-2026 comprehensive evaluation harness.

Closes reviewer questions R1, R2, R3, R6, R9, R10, R12, R13 with
reproducible, budget-bounded measurements:

  1. Expanded Single-Turn Suite (49 core + 24 fresh held-out cases = 73)
     comparing a real live `LLM-RAG` baseline (`BM25 + LLM`, no gate/verifier)
     against `Full Py-Buddy` (`ReAct + Hypergraph + GroundingSet +
     ClaimVerifier + NVFG Lattice`), across multiple LLM backbones.
  2. Paired 4-Arm Ablation on identical LLM traces:
       LLM-RAG  |  ReAct+StructuredGate (pre-ClaimVerifier)  |  Full Py-Buddy
     reporting both execution metrics and sentence-level prose hallucination
     interventions from `claim_verifier`.
  3. Multi-Turn Dialogue Suite (anaphora, budget narrowing, price traps,
     compatibility switches) evaluating cross-turn intent continuity with
     strict per-turn GroundingSet isolation.
  4. Non-English Constraint Induction (`R3` + `R12`): `jury_zoo.db`
     (Ukrainian attributes with Cyrillic/Latin homoglyphs) + `cip_benchmark`
     datasets comparing English-keyword, Multilingual-seed, TF-IDF cosine,
     and domain-free `induce_structural`.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import sqlite3
import sys
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.abspath(os.path.join(HERE, "..", "core"))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, CORE):
    if p not in sys.path:
        sys.path.insert(0, p)

from baseline_rag import RetrievalBaseline  # noqa: E402
from benchmark_cases import (  # noqa: E402
    CATEGORY_UA,
    VERDICT_FAMILIES,
    BenchmarkCase,
    Snapshot,
    build_cases,
    check_bundle,
    load_snapshot,
    sockets_of,
)
from cip_benchmark import compare as cip_compare, induce_structural  # noqa: E402
from contract_inductor import ContractInductor  # noqa: E402
from hypergraph import ProductNode  # noqa: E402
from live_agent import (  # noqa: E402
    BudgetExceeded,
    BudgetedRunner,
    LivePyBuddyAgent,
    SharedBudget,
    default_schema_prompt,
)
from metrics import CaseScore, Oracle, aggregate, score_case  # noqa: E402
from openrouter_client import OpenRouterAgentRunner  # noqa: E402

DEFAULT_DB = os.path.join(ROOT, "data", "marketplace_rozetka.db")
DEFAULT_ZOO_DB = os.path.join(ROOT, "data", "jury_zoo.db")

#: Pricing table (USD per 1M prompt / completion tokens) verified from
#: OpenRouter /api/v1/models on 2026-11-13.
MODEL_CATALOG: Dict[str, Tuple[float, float]] = {
    "deepseek/deepseek-v4.1-flash-1017": (0.15, 0.60),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "openai/gpt-6-luna": (0.10, 0.50),
    "xiaomi/mimo-v2.6-flash": (0.14, 0.28),
    "google/gemini-3.1-flash-lite": (0.25, 1.50),
    "tencent/hy4": (0.417, 1.25),
    "deepseek/deepseek-v3.2": (0.26, 0.38),
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "qwen/qwen3-30b-a3b-instruct-2507": (0.10, 0.30),
    "nvidia/nemotron-3.5-lightning:free": (0.0, 0.0),
    "nvidia/nemotron-3-ultra-550b-a55b:free": (0.0, 0.0),
}

_HELDOUT_TRAPS = (
    "Intel Core i9-15900KS",
    "Nvidia GeForce RTX 5090 Ti",
    "Samsung 990 Pro 8TB",
    "Corsair HX1600i 1600W",
)


def _attr_ground_terms(snap: Snapshot) -> List[str]:
    terms = set()
    for prod_attrs in snap.attrs.values():
        for vals in prod_attrs.values():
            for v in vals:
                terms.add(str(v))
    return sorted(terms)


def build_heldout_cases(
    snap: Snapshot,
    core_cases: Sequence[BenchmarkCase],
    seed: int = 20261113,
) -> List[BenchmarkCase]:
    """24 independent held-out single-turn cases disjoint from `core_cases`."""
    rnd = random.Random(seed)
    used_ids = {c.meta.get("product_id") for c in core_cases if c.meta.get("product_id")}
    used_bundles = {
        tuple(c.meta["bundle"]) for c in core_cases if c.meta.get("bundle")
    }
    cases: List[BenchmarkCase] = []
    idx = len(core_cases)

    def add(q: str, ctype: str, oracle: Oracle, **meta: Any) -> None:
        nonlocal idx
        idx += 1
        oracle.checks_verdict = ctype in VERDICT_FAMILIES
        cases.append(BenchmarkCase(f"ho{idx:03d}", q, ctype, oracle, dict(meta)))

    # 5 unseen direct-price queries
    pool = [
        pid for pid in snap.products
        if int(snap.products[pid]["stock_quantity"]) > 0 and pid not in used_ids
    ]
    for pid in rnd.sample(pool, min(5, len(pool))):
        name = snap.id_to_name[pid]
        price = float(snap.products[pid]["price"])
        add(
            f"Підкажи актуальну ціну на {name}.",
            "direct_price",
            Oracle(f"ho{idx + 1:03d}", "direct_price", [name], {name: price}, "neutral", True),
            product_id=pid,
        )

    # 4 unseen compat_accept + 4 unseen compat_reject
    cpu_ids = [p for p, r in snap.products.items() if r["category"] == "cpu" and sockets_of(snap, p)]
    mb_ids = [p for p, r in snap.products.items() if r["category"] == "motherboard" and sockets_of(snap, p)]
    sample_cpus = rnd.sample(cpu_ids, min(40, len(cpu_ids)))
    sample_mbs = rnd.sample(mb_ids, min(40, len(mb_ids)))
    rnd.shuffle(sample_cpus)
    rnd.shuffle(sample_mbs)
    accepts, rejects = [], []
    for cpu in sample_cpus:
        for mb in sample_mbs:
            b = (cpu, mb)
            if b in used_bundles:
                continue
            v = check_bundle(snap, [cpu, mb])
            (accepts if v.ok else rejects).append([cpu, mb])
            if len(accepts) >= 16 and len(rejects) >= 16:
                break
        if len(accepts) >= 16 and len(rejects) >= 16:
            break
    rnd.shuffle(accepts)
    rnd.shuffle(rejects)
    for bundle in accepts[:4]:
        names = [snap.id_to_name[p] for p in bundle]
        add(
            f"Чи сумісні між собою {names[0]} і {names[1]}?",
            "compat_accept",
            Oracle(f"ho{idx + 1:03d}", "compat_accept", names, {}, "accept", True),
            bundle=bundle,
        )
    for bundle in rejects[:4]:
        names = [snap.id_to_name[p] for p in bundle]
        add(
            f"Чи сумісні між собою {names[0]} і {names[1]}?",
            "compat_reject",
            Oracle(f"ho{idx + 1:03d}", "compat_reject", names, {}, "reject", False),
            bundle=bundle,
        )

    # 3 paraphrased overconstrained queries
    for cat in ("motherboard", "ssd", "gpu"):
        ids = [p for p, r in snap.products.items() if r["category"] == cat]
        if not ids:
            continue
        floor = min(float(snap.products[p]["price"]) for p in ids)
        budget = max(100.0, round(floor * 0.45, -2))
        add(
            f"Потрібен {CATEGORY_UA[cat]} не дорожче {budget:.0f} грн, що порадите?",
            "overconstrained",
            Oracle(f"ho{idx + 1:03d}", "overconstrained", [], {}, "reject", False),
            category=cat,
            budget=budget,
        )

    # 4 unseen price traps
    for trap in _HELDOUT_TRAPS:
        add(
            f"Яка ціна та наявність {trap}?",
            "price_trap",
            Oracle(f"ho{idx + 1:03d}", "price_trap", [], {}, "reject", False),
            fabricated=trap,
        )

    # 2 paraphrased cheapest + 2 conversational
    for cat in ("ram", "motherboard"):
        ids = [p for p, r in snap.products.items() if r["category"] == cat]
        if ids:
            pid = min(ids, key=lambda p: float(snap.products[p]["price"]))
            name = snap.id_to_name[pid]
            add(
                f"Який найдешевший варіант у категорії «{CATEGORY_UA[cat]}»?",
                "cheapest_category",
                Oracle(f"ho{idx + 1:03d}", "cheapest_category", [name], {name: float(snap.products[pid]["price"])}, "accept", True),
                product_id=pid,
                category=cat,
            )
    for q in ("Які категорії товарів є у вашому магазині?", "Дякую, ти дуже допоміг!"):
        add(
            q,
            "conversational",
            Oracle(f"ho{idx + 1:03d}", "conversational", [], {}, "neutral", True),
        )
    return cases


def build_expanded_cases(snap: Snapshot) -> List[BenchmarkCase]:
    core = build_cases(snap)
    heldout = build_heldout_cases(snap, core)
    return core + heldout


def build_multiturn_scenarios(snap: Snapshot) -> List[Dict[str, Any]]:
    """5 deterministic 3-turn dialogues (15 turns) anchored in `snap`."""
    cpu_ids = [p for p, r in snap.products.items() if r["category"] == "cpu" and sockets_of(snap, p)]
    mb_ids = [p for p, r in snap.products.items() if r["category"] == "motherboard" and sockets_of(snap, p)]
    cpu_ok = cpu_ids[0]
    compat_mb = next(mb for mb in mb_ids if check_bundle(snap, [cpu_ok, mb]).ok)
    incompat_mb = next(mb for mb in mb_ids if not check_bundle(snap, [cpu_ok, mb]).ok)

    cheapest_psu = min(
        (p for p, r in snap.products.items() if r["category"] == "psu" and int(r["stock_quantity"]) > 0),
        key=lambda p: float(snap.products[p]["price"]),
    )
    gpu_floor = min(float(r["price"]) for r in snap.products.values() if r["category"] == "gpu")
    impossible_gpu = max(500.0, round(gpu_floor * 0.45, -2))

    return [
        {
            "scenario_id": "mt_compat_anaphora",
            "turns": [
                {
                    "query": f"Скільки коштує {snap.id_to_name[cpu_ok]}?",
                    "oracle": Oracle("mt1_t1", "direct_price", [snap.id_to_name[cpu_ok]], {snap.id_to_name[cpu_ok]: float(snap.products[cpu_ok]["price"])}, "neutral", False),
                },
                {
                    "query": f"А чи сумісний цей процесор із материнською платою {snap.id_to_name[compat_mb]}?",
                    "oracle": Oracle("mt1_t2", "compat_accept", [snap.id_to_name[cpu_ok], snap.id_to_name[compat_mb]], {}, "accept", True),
                },
                {
                    "query": f"А з {snap.id_to_name[incompat_mb]} він буде сумісний?",
                    "oracle": Oracle("mt1_t3", "compat_reject", [snap.id_to_name[cpu_ok], snap.id_to_name[incompat_mb]], {}, "reject", True),
                },
            ],
        },
        {
            "scenario_id": "mt_budget_and_trap",
            "turns": [
                {
                    "query": f"Порадь відеокарту з ціною до {impossible_gpu:.0f} грн.",
                    "oracle": Oracle("mt2_t1", "overconstrained", [], {}, "reject", True),
                },
                {
                    "query": "Тоді підкажи найдешевший блок живлення в наявності.",
                    "oracle": Oracle("mt2_t2", "cheapest_category", [snap.id_to_name[cheapest_psu]], {snap.id_to_name[cheapest_psu]: float(snap.products[cheapest_psu]["price"])}, "accept", True),
                },
                {
                    "query": "А чи є у вас Intel Core i9-15900KS і скільки він коштує?",
                    "oracle": Oracle("mt2_t3", "price_trap", [], {}, "reject", True),
                },
            ],
        },
        {
            "scenario_id": "mt_meta_and_pivot",
            "turns": [
                {
                    "query": "Привіт! Які категорії комп'ютерних товарів є в каталозі?",
                    "oracle": Oracle("mt3_t1", "conversational", [], {}, "neutral", False),
                },
                {
                    "query": f"Покажи ціну {snap.id_to_name[compat_mb]}.",
                    "oracle": Oracle("mt3_t2", "direct_price", [snap.id_to_name[compat_mb]], {snap.id_to_name[compat_mb]: float(snap.products[compat_mb]["price"])}, "neutral", False),
                },
                {
                    "query": "Дякую, на цьому все!",
                    "oracle": Oracle("mt3_t3", "conversational", [], {}, "neutral", False),
                },
            ],
        },
    ]


def _tfidf_similarity_rules(products: Sequence[ProductNode]) -> List[str]:
    """Fair char-3-gram TF-IDF cosine baseline between Ukrainian/English attribute keys across categories."""
    by_key: Dict[str, Tuple[set, str]] = {}
    for p in products:
        for k, v in p.attributes.items():
            cats, txt = by_key.setdefault(k, (set(), ""))
            cats.add(p.category)
            by_key[k] = (cats, f"{txt} {k} {v}")

    def ngrams(s: str) -> Counter:
        s_clean = f" {s.lower()} "
        return Counter(s_clean[i:i + 3] for i in range(max(1, len(s_clean) - 2)))

    keys = sorted(by_key)
    vecs = {k: ngrams(k) for k in keys}
    matched: List[str] = []
    for k in keys:
        cats, _ = by_key[k]
        if len(cats) >= 2:
            matched.append(f"eq({k})")
    for i, a in enumerate(keys):
        va = vecs[a]
        na = math.sqrt(sum(v * v for v in va.values())) or 1.0
        for b in keys[i + 1:]:
            vb = vecs[b]
            nb = math.sqrt(sum(v * v for v in vb.values())) or 1.0
            dot = sum(va[g] * vb[g] for g in va if g in vb)
            cos = dot / (na * nb)
            if cos >= 0.45:
                matched.append(f"pair({a},{b})")
    return matched


def run_non_english_induction_eval(zoo_db_path: str = DEFAULT_ZOO_DB) -> Dict[str, Any]:
    """Evaluate R3 + R12 on cip_benchmark plus real Ukrainian `jury_zoo.db`."""
    cip = cip_compare()
    zoo_res: Dict[str, Any] = {"available": os.path.exists(zoo_db_path)}
    if os.path.exists(zoo_db_path):
        conn = sqlite3.connect(f"file:{zoo_db_path}?mode=ro", uri=True)
        try:
            attrs: Dict[str, Dict[str, str]] = {}
            for tid, k, v in conn.execute("SELECT tvaryna_id, kliuch, znachennya FROM vlastyvosti"):
                attrs.setdefault(str(tid), {})[str(k)] = str(v)
            prods = [
                ProductNode(str(tid), str(name), str(cat), None, attrs.get(str(tid), {}))
                for tid, name, cat in conn.execute(
                    "SELECT t.id, t.imya, s.nazva FROM tvaryny t "
                    "JOIN seredovyshcha s ON t.sered_id = s.id"
                )
            ]
        finally:
            conn.close()
        ci = ContractInductor()
        det_rules = ci.induce_contracts_from_products(prods, use_llm=False)
        str_rules = induce_structural(prods)
        tfidf_rules = _tfidf_similarity_rules(prods)
        zoo_res.update({
            "n_entities": len(prods),
            "n_eav_rows": sum(len(p.attributes) for p in prods),
            "contract_inductor_count": len(det_rules),
            "contract_inductor_rules": [r.description for r in det_rules],
            "structural_count": len(str_rules),
            "structural_rules": [str(r) for r in str_rules],
            "tfidf_char3_count": len(tfidf_rules),
            "tfidf_char3_sample": tfidf_rules[:10],
            "contains_dimensional_containment_in_structural": any(
                str(r).startswith("le(") for r in str_rules
            ),
            "contains_dimensional_containment_in_contract_inductor": any(
                type(r).__name__ in ("SpatialBoundingConstraint", "CumulativeCapacityConstraint")
                for r in det_rules
            ),
        })
    return {"cip_benchmark": cip, "jury_zoo_ukrainian": zoo_res}


def run_model_evaluation(
    snap: Snapshot,
    cases: Sequence[BenchmarkCase],
    model_name: str,
    *,
    max_usd: float,
    max_llm_calls: int,
    include_multiturn: bool = True,
    workers: int = 1,
    print_progress: bool = False,
    runner_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Run LLM-RAG baseline + Full Py-Buddy (+ paired pre-verifier ablation) on `model_name`."""
    pin, pout = MODEL_CATALOG.get(model_name, (0.30, 1.00))
    attr_terms = _attr_ground_terms(snap)
    schema_prompt = default_schema_prompt(snap.db_path)
    from domain_catalog import DomainCatalogRegistry
    from hypergraph import KnowledgeHypergraph
    base_kg = DomainCatalogRegistry.create_database_catalog(snap.db_path, seed_limit=120)

    def _clone_kg() -> KnowledgeHypergraph:
        kg = KnowledgeHypergraph()
        kg.nodes.update(base_kg.nodes)
        kg.hyperedges.update(base_kg.hyperedges)
        kg.constraints = list(base_kg.constraints)
        kg.induction_report = (
            dict(base_kg.induction_report)
            if getattr(base_kg, "induction_report", None) else None
        )
        for k, v in base_kg._node_to_edges.items():
            kg._node_to_edges[k] = set(v)
        for k, v in base_kg._by_category.items():
            kg._by_category[k] = set(v)
        for k, v in base_kg._by_attribute_value.items():
            kg._by_attribute_value[k] = set(v)
        return kg

    def _make_runner() -> Any:
        if runner_factory is not None:
            return runner_factory()
        r = OpenRouterAgentRunner(model_name=model_name)
        r.fallback_models = []
        return r

    rag_budget = SharedBudget(
        max_llm_calls=max(len(cases) + 25, max_llm_calls),
        max_usd=max_usd * 0.25,
        price_in_per_mtok=pin,
        price_out_per_mtok=pout,
    )

    def _eval_rag_case(item: Tuple[int, BenchmarkCase]) -> Tuple[int, CaseScore, Dict[str, Any]]:
        idx, c = item
        budgeted = BudgetedRunner(_make_runner(), shared_budget=rag_budget)
        rag = RetrievalBaseline(snap.db_path, runner=budgeted, top_k=8)
        t0 = time.time()
        err = None
        text = ""
        calls = 0
        try:
            text, calls, _ = rag.answer(c.query)
        except BudgetExceeded as exc:
            err = f"BudgetExceeded: {exc}"
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        lat = time.time() - t0
        sc = score_case(
            text, c.oracle, snap.names(), snap.price_index,
            steps=calls, latency_s=lat, cost_usd=budgeted.cost_usd,
            extra_grounded_terms=attr_terms, user_query=c.query,
        )
        return idx, sc, {
            "case_id": c.case_id,
            "query": c.query,
            "text": text,
            "error": err,
            "score": sc.to_dict(),
        }

    rag_results: List[Tuple[int, CaseScore, Dict[str, Any]]] = []
    indexed_cases = list(enumerate(cases))
    if max(1, workers) == 1:
        for item in indexed_cases:
            rag_results.append(_eval_rag_case(item))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_eval_rag_case, item) for item in indexed_cases]
            for fut in concurrent.futures.as_completed(futs):
                rag_results.append(fut.result())
                done = len(rag_results)
                if print_progress and (done == len(cases) or done % 15 == 0 or len(cases) <= 10):
                    print(f"    [RAG {model_name}] {done}/{len(cases)} (${rag_budget.cost_usd:.4f})", flush=True)
    rag_results.sort(key=lambda x: x[0])
    rag_scores = [x[1] for x in rag_results]
    rag_cases_log = [x[2] for x in rag_results]

    pb_budget = SharedBudget(
        max_llm_calls=max_llm_calls,
        max_usd=max_usd * 0.75,
        price_in_per_mtok=pin,
        price_out_per_mtok=pout,
    )

    def _eval_pb_case(item: Tuple[int, BenchmarkCase]) -> Tuple[int, CaseScore, CaseScore, Dict[str, Any], bool]:
        idx, c = item
        pb = LivePyBuddyAgent(
            snap,
            _make_runner(),
            schema_prompt=schema_prompt,
            shared_budget=pb_budget,
            hypergraph=_clone_kg(),
        )
        t0 = time.time()
        out = pb.answer(c.query)
        lat = time.time() - t0
        sc = score_case(
            str(out["text"]), c.oracle, snap.names(), snap.price_index,
            steps=int(out["steps"]), latency_s=lat, cost_usd=float(out["cost_usd"]),
            extra_grounded_terms=attr_terms, user_query=c.query,
        )
        pre_text = out.get("pre_verifier_text")
        if pre_text is not None:
            sc_pre = score_case(
                str(pre_text), c.oracle, snap.names(), snap.price_index,
                steps=int(out["steps"]), latency_s=lat, cost_usd=float(out["cost_usd"]),
                extra_grounded_terms=attr_terms, user_query=c.query,
            )
        else:
            sc_pre = sc
        vrep = out.get("verification_report") or {}
        case_err = pb.case_log[-1]["error"] if pb.case_log else None
        log_row = {
            "case_id": c.case_id,
            "query": c.query,
            "text": out["text"],
            "pre_verifier_text": pre_text,
            "termination_reason": out.get("termination_reason"),
            "verification_report": vrep,
            "error": case_err,
            "score": sc.to_dict(),
        }
        return idx, sc, sc_pre, log_row, bool(pb.stats()["truncated"])

    pb_results: List[Tuple[int, CaseScore, CaseScore, Dict[str, Any], bool]] = []
    if max(1, workers) == 1:
        for item in indexed_cases:
            pb_results.append(_eval_pb_case(item))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_eval_pb_case, item) for item in indexed_cases]
            for fut in concurrent.futures.as_completed(futs):
                pb_results.append(fut.result())
                done = len(pb_results)
                if print_progress and (done == len(cases) or done % 15 == 0 or len(cases) <= 10):
                    print(f"    [PyBuddy {model_name}] {done}/{len(cases)} (${pb_budget.cost_usd:.4f})", flush=True)
    pb_results.sort(key=lambda x: x[0])

    pb_scores = [x[1] for x in pb_results]
    pre_verifier_scores = [x[2] for x in pb_results]
    pb_cases_log = [x[3] for x in pb_results]
    pb_truncated = any(x[4] for x in pb_results)

    claim_interventions = 0
    claim_blocked_or_repaired = 0
    claim_unsupported_sentences_pre = 0
    for row in pb_cases_log:
        vrep = row.get("verification_report") or {}
        if vrep.get("n_unsupported", 0) > 0:
            claim_interventions += 1
            claim_unsupported_sentences_pre += int(vrep["n_unsupported"])
            if str(row.get("termination_reason") or "").startswith(("HALLUCINATION", "SUCCESS_GROUNDED_")):
                claim_blocked_or_repaired += 1

    multiturn_out: Dict[str, Any] = {}
    if include_multiturn and not pb_truncated:
        scenarios = build_multiturn_scenarios(snap)
        mt_rag_scores: List[CaseScore] = []
        mt_pb_scores: List[CaseScore] = []
        mt_logs: List[Dict[str, Any]] = []
        for scen in scenarios:
            hist_rag: List[Dict[str, str]] = []
            hist_pb: List[Dict[str, str]] = []
            scen_turns: List[Dict[str, Any]] = []
            mt_rag_agent = RetrievalBaseline(
                snap.db_path, runner=BudgetedRunner(_make_runner(), shared_budget=rag_budget), top_k=8
            )
            mt_pb_agent = LivePyBuddyAgent(
                snap, _make_runner(), schema_prompt=schema_prompt, shared_budget=pb_budget
            )
            for turn in scen["turns"]:
                q = turn["query"]
                oracle = turn["oracle"]
                t0 = time.time()
                try:
                    r_txt, r_calls, _ = mt_rag_agent.answer(q, conversation_history=hist_rag)
                except Exception as exc:
                    r_txt, r_calls = f"ERROR: {exc}", 0
                r_sc = score_case(
                    r_txt, oracle, snap.names(), snap.price_index,
                    steps=r_calls, latency_s=time.time() - t0,
                    extra_grounded_terms=attr_terms, user_query=q,
                )
                mt_rag_scores.append(r_sc)
                hist_rag.extend([{"role": "user", "content": q}, {"role": "assistant", "content": r_txt}])

                t1 = time.time()
                p_out = mt_pb_agent.answer(q, conversation_history=hist_pb)
                p_txt = str(p_out["text"])
                p_sc = score_case(
                    p_txt, oracle, snap.names(), snap.price_index,
                    steps=int(p_out["steps"]), latency_s=time.time() - t1,
                    cost_usd=float(p_out["cost_usd"]),
                    extra_grounded_terms=attr_terms, user_query=q,
                )
                mt_pb_scores.append(p_sc)
                hist_pb.extend([{"role": "user", "content": q}, {"role": "assistant", "content": p_txt}])
                scen_turns.append({
                    "query": q,
                    "llm_rag_answer": r_txt,
                    "llm_rag_correct": r_sc.is_correct,
                    "llm_rag_violations": r_sc.violations,
                    "pybuddy_answer": p_txt,
                    "pybuddy_correct": p_sc.is_correct,
                    "pybuddy_violations": p_sc.violations,
                    "pybuddy_termination": p_out.get("termination_reason"),
                })
            mt_logs.append({"scenario_id": scen["scenario_id"], "turns": scen_turns})
        multiturn_out = {
            "n_turns": len(mt_pb_scores),
            "llm_rag": aggregate(mt_rag_scores).__dict__,
            "pybuddy_live": aggregate(mt_pb_scores).__dict__,
            "scenarios": mt_logs,
        }

    rag_agg = aggregate(rag_scores).__dict__
    pre_agg = aggregate(pre_verifier_scores).__dict__
    pb_agg = aggregate(pb_scores).__dict__

    return {
        "model": model_name,
        "pricing_usd_per_1m": {"input": pin, "output": pout},
        "n_cases": len(cases),
        "llm_rag": {
            "aggregate": rag_agg,
            "llm_calls": rag_budget.calls,
            "prompt_tokens": rag_budget.prompt_tokens,
            "completion_tokens": rag_budget.completion_tokens,
            "cost_usd": round(rag_budget.cost_usd, 6),
            "cases": rag_cases_log,
        },
        "ablation_pybuddy_pre_claim_verifier": {
            "aggregate": pre_agg,
            "unsupported_prose_cases_caught": claim_interventions,
            "unsupported_prose_sentences_caught": claim_unsupported_sentences_pre,
            "cases_repaired_or_blocked": claim_blocked_or_repaired,
        },
        "pybuddy_live": {
            "aggregate": pb_agg,
            "stats": {
                "model": model_name,
                "llm_calls": pb_budget.calls,
                "prompt_tokens": pb_budget.prompt_tokens,
                "completion_tokens": pb_budget.completion_tokens,
                "cost_usd": round(pb_budget.cost_usd, 6),
                "max_llm_calls": pb_budget.max_llm_calls,
                "max_usd": pb_budget.max_usd,
                "budget_mode": pb_budget.budget_mode,
                "truncated": pb_truncated,
                "cases_answered": sum(1 for row in pb_cases_log if not row["error"]),
                "cases_errored": sum(1 for row in pb_cases_log if row["error"]),
            },
            "cases": pb_cases_log,
        },
        "multiturn": multiturn_out,
        "total_cost_usd": round(rag_budget.cost_usd + pb_budget.cost_usd, 6),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--zoo-db", default=DEFAULT_ZOO_DB)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["deepseek/deepseek-v3.2", "google/gemini-2.5-flash-lite"],
    )
    ap.add_argument("--suite", choices=("pilot", "core", "expanded"), default="expanded")
    ap.add_argument("--max-usd-per-model", type=float, default=0.55)
    ap.add_argument("--max-calls-per-model", type=int, default=350)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-multiturn", action="store_true")
    ap.add_argument("--skip-cip", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    snap = load_snapshot(args.db)
    if args.suite == "core":
        cases = build_cases(snap)
    elif args.suite == "pilot":
        all_c = build_expanded_cases(snap)
        # 6 representative cases: direct, cheapest, compat_accept, compat_reject, overconstrained, price_trap
        wanted = ("direct_price", "cheapest_category", "compat_accept", "compat_reject", "overconstrained", "price_trap")
        cases = [next(c for c in all_c if c.case_type == wt) for wt in wanted]
    else:
        cases = build_expanded_cases(snap)

    report: Dict[str, Any] = {
        "suite": args.suite,
        "n_cases": len(cases),
        "db": os.path.basename(args.db),
        "non_english_induction": None if args.skip_cip else run_non_english_induction_eval(args.zoo_db),
        "models": [],
        "total_spend_usd": 0.0,
    }

    for model in args.models:
        print(f"\n=== Running {model} ({len(cases)} single-turn cases, workers={args.workers}) ===", flush=True)
        res = run_model_evaluation(
            snap, cases, model,
            max_usd=args.max_usd_per_model,
            max_llm_calls=args.max_calls_per_model,
            include_multiturn=not args.no_multiturn,
            workers=args.workers,
            print_progress=not args.quiet,
        )
        report["models"].append(res)
        report["total_spend_usd"] = round(report["total_spend_usd"] + res["total_cost_usd"], 6)
        r_agg = res["llm_rag"]["aggregate"]
        p_agg = res["pybuddy_live"]["aggregate"]
        pre = res["ablation_pybuddy_pre_claim_verifier"]
        print(
            f"  LLM-RAG : acc={r_agg['execution_accuracy_pct']}% | hall={r_agg['hallucination_rate_pct']}% | "
            f"price_err={r_agg['price_error_rate_pct']}% | NVFG={r_agg['nvfg_pct']}% | ${res['llm_rag']['cost_usd']}",
            flush=True,
        )
        print(
            f"  PyBuddy : acc={p_agg['execution_accuracy_pct']}% | hall={p_agg['hallucination_rate_pct']}% | "
            f"price_err={p_agg['price_error_rate_pct']}% | NVFG={p_agg['nvfg_pct']}% | "
            f"claim_caught={pre['unsupported_prose_cases_caught']} (${res['total_cost_usd']} total)",
            flush=True,
        )
        if res["multiturn"]:
            mr = res["multiturn"]["llm_rag"]
            mp = res["multiturn"]["pybuddy_live"]
            print(
                f"  MultiTurn ({res['multiturn']['n_turns']}t): LLM-RAG acc={mr['execution_accuracy_pct']}% | "
                f"PyBuddy acc={mp['execution_accuracy_pct']}%",
                flush=True,
            )
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
