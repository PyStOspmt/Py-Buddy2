# Архітектурна карта платформи Py-Buddy
## Навігатор по модулях та технічній документації

Цей індекс містить повну карту всіх компонентів, вихідних файлів системи та детальної документації кожного модуля.

---

### Карта документів та зв'язок із файлами коду

| № | Документ | Основні теми | Відповідні файли вихідного коду |
| :--- | :--- | :--- | :--- |
| **01** | [**01_GLOBAL_ARCHITECTURE.md**](01_GLOBAL_ARCHITECTURE.md) | Загальна архітектура, філософія, поділ на Нейронний та Символічний контури, генеральна Mermaid-діаграма даних | Весь проєкт, [`src/core/server.py`](../../src/core/server.py) |
| **02** | [**02_SYSTEM_KERNEL.md**](02_SYSTEM_KERNEL.md) | Ядро системи: Fast Cascaded Router (Tier 0/1/2), ReAct агент, телеметрія по кроках, SQL AST Sanitizer, Grounding Gate & ClaimVerifier | [`src/core/fast_router.py`](../../src/core/fast_router.py)<br>[`src/core/react_agent.py`](../../src/core/react_agent.py)<br>[`src/core/sanitizer.py`](../../src/core/sanitizer.py)<br>[`src/core/grounding_gate.py`](../../src/core/grounding_gate.py)<br>[`src/core/claim_verifier.py`](../../src/core/claim_verifier.py)<br>[`src/core/openrouter_client.py`](../../src/core/openrouter_client.py)<br>[`src/core/router.py`](../../src/core/router.py) |
| **03** | [**03_SYMBOLIC_CORE_HYPERGRAPH_LATTICE.md**](03_SYMBOLIC_CORE_HYPERGRAPH_LATTICE.md) | Символічне ядро: n-арний віртуальний гіперграф, Equivalence/Sum/Fit контракти, Open Basket принцип, ґратка релаксації, фронт Парето | [`src/core/hypergraph.py`](../../src/core/hypergraph.py)<br>[`src/core/relaxation_lattice.py`](../../src/core/relaxation_lattice.py)<br>[`src/core/pareto_relaxer.py`](../../src/core/pareto_relaxer.py)<br>[`src/core/attribute_algebra.py`](../../src/core/attribute_algebra.py) |
| **04** | [**04_FCA_AND_DYNAMIC_TAXONOMY.md**](04_FCA_AND_DYNAMIC_TAXONOMY.md) | Пам'ять та ієрархія каталогу: Formal Concept Analysis (FCA), ґратка понять $\underline{\mathfrak{B}}(G, M, I)$, подолання Horizon Truncation | [`src/core/taxonomy_fca.py`](../../src/core/taxonomy_fca.py)<br>[`tests/test_taxonomy_fca.py`](../../tests/test_taxonomy_fca.py) |
| **05** | [**05_DATA_FABRIC_AND_PROFILER.md**](05_DATA_FABRIC_AND_PROFILER.md) | Фабрика даних: автопрофайлер схем, детекція EAV, інваріант $ZCCS = 0$, структурна індукція та Entity Resolution | [`src/core/profiler.py`](../../src/core/profiler.py)<br>[`src/core/domain_catalog.py`](../../src/core/domain_catalog.py)<br>[`src/core/source_eav_discovery.py`](../../src/core/source_eav_discovery.py)<br>[`src/core/contract_inductor.py`](../../src/core/contract_inductor.py)<br>[`src/core/entity_resolution.py`](../../src/core/entity_resolution.py) |
| **06** | [**06_API_OMNICHANNEL_AND_UI.md**](06_API_OMNICHANNEL_AND_UI.md) | Інтерфейси: FastAPI REST сервіс, сесії SQLite WAL, шлюзи Telegram/Discord, веб-консоль з таймінгами | [`src/core/server.py`](../../src/core/server.py)<br>[`src/core/session_store.py`](../../src/core/session_store.py)<br>[`src/gateways/`](../../src/gateways/)<br>[`web/`](../../web/) |

---

### Швидкий старт для розробників
- **Бенчмарки та відтворювані звіти:** [`docs/BENCHMARKS.md`](../BENCHMARKS.md)
- **Запуск бекенд-сервера:** `python src/core/server.py` (доступно на `http://127.0.0.1:8000`)
- **Запуск повного набору тестів:** `pytest tests/ -v`
- **Перевірка інваріантів нульової галюцинації:** `pytest tests/test_fast_router.py tests/test_adaptive_query_recovery.py tests/test_taxonomy_fca.py -v`
