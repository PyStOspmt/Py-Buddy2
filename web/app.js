/**
 * Py-Buddy · Neuro-Symbolic Console
 *
 * Vanilla JS client for the FastAPI backend. Design notes:
 *  - The model is never trusted: every rendered price/name comes from the
 *    backend's verified payload (`recommended_items`), gate verdicts are
 *    surfaced verbatim from `termination_reason`.
 *  - All dynamic strings go through escapeHtml() before touching innerHTML.
 *  - Every network call has a timeout and an explicit error surface; a dead
 *    backend shows in the connection pill instead of a silently frozen UI.
 */
"use strict";

const TIER_LABELS = {
  tier0_conversational: "Tier-0 · Direct LLM",
  tier1_fast_path: "Tier-1 · Fast SQL",
  tier2_react_reasoning: "Tier-2 · Deep ReAct",
};

// F53: a fixed palette, not a fixed vocabulary -- the category string is
// hashed into it, so no category name needs to be known in advance and an
// unknown one gets a stable colour of its own rather than a wrong match.
const CATEGORY_PALETTE = [
  "#6366f1", "#3b82f6", "#ec4899", "#10b981", "#f59e0b",
  "#14b8a6", "#8b5cf6", "#ef4444", "#84cc16", "#06b6d4",
];

class PyBuddyConsole {
  constructor() {
    this.state = {
      domain: "pc_hardware",
      catalog: [],
      domainNames: Object.create(null),
      domainCurrencies: Object.create(null),
      paretoSliderDomain: null,
      paretoCategory: "",
      paretoPlotPoints: [],
      paretoSelectedId: null,
      paretoHoverId: null,
      // F35: no fixture product ids are preloaded -- the bundle starts
      // empty and is built from whatever the actual catalogue contains.
      bundle: [],
      catalogTotal: null,
      nameById: Object.create(null),
      lastChat: null,
      lastPareto: null,
      sessionCost: 0,
      pending: false,
      sessionId: PyBuddyConsole.loadSessionId(),
      model: null,
      chatHistory: [],
      availableModels: [],
      activeModelFilter: "all",
      modelSearchQuery: "",
    };

    const $ = (id) => document.getElementById(id);
    this.dom = {
      chatForm: $("chat-form"),
      chatInput: $("chat-input"),
      chatHistory: $("chat-history"),
      sendBtn: $("send-btn"),
      clearChatBtn: $("clear-chat-btn"),
      btnCopyChatMd: $("btn-copy-chat-md"),
      btnCopyChatJson: $("btn-copy-chat-json"),
      sessionInfo: $("session-info"),
      scenarioChips: $("scenario-chips"),
      domainSelect: $("domain-select"),
      chipGate: $("chip-gate"),
      chipGateText: $("chip-gate-text"),
      chipModelText: $("chip-model-text"),
      btnModelSelector: $("btn-model-selector"),
      chipNodesText: $("chip-nodes-text"),
      connDot: $("conn-dot"),
      connText: $("conn-text"),
      tabs: document.querySelectorAll(".tab"),
      catalogSearch: $("catalog-search"),
      catalogCount: $("catalog-count"),
      catalogList: $("catalog-list"),
      bundleChips: $("bundle-chips"),
      bundleCount: $("bundle-count"),
      bundleStatus: $("bundle-status"),
      clearBundleBtn: $("clear-bundle-btn"),
      conflictBox: $("conflict-box"),
      conflictList: $("conflict-list"),
      hypergraphSvg: $("hypergraph-svg"),
      budgetSlider: $("budget-slider"),
      perfSlider: $("perf-slider"),
      budgetValue: $("budget-value"),
      perfValue: $("perf-value"),
      paretoCategory: $("pareto-category"),
      paretoCanvas: $("pareto-canvas"),
      paretoTooltip: $("pareto-tooltip"),
      paretoPointCard: $("pareto-point-card"),
      paretoExplanation: $("pareto-explanation"),
      paretoFront: $("pareto-front"),
      refreshTelemetryBtn: $("refresh-telemetry-btn"),
      invariantsList: $("invariants-list"),
      rawJson: $("raw-json"),
      kpiLatency: $("kpi-latency"),
      kpiTier: $("kpi-tier"),
      kpiSteps: $("kpi-steps"),
      kpiCost: $("kpi-cost"),
      kpiModel: $("kpi-model"),
      kpiGate: $("kpi-gate"),
      kpiDomain: $("kpi-domain"),
      kpiScale: $("kpi-scale"),
      sbTier: $("sb-tier"),
      sbLatency: $("sb-latency"),
      sbModel: $("sb-model"),
      sbGate: $("sb-gate"),
      sbCost: $("sb-cost"),
      sbDomain: $("sb-domain"),
      sbEngine: $("sb-answer-engine"),
      modelModal: $("model-modal"),
      modalCloseBtn: $("modal-close-btn"),
      modelSearchInput: $("model-search-input"),
      modelSearchClear: $("model-search-clear"),
      modalModelsCount: $("modal-models-count"),
      modalModelsList: $("modal-models-list"),
      modelFilterTags: $("model-filter-tags"),
      toastContainer: $("toast-container"),
      btnRefreshTaxonomy: $("btn-refresh-taxonomy"),
      btnExpandTaxonomy: $("btn-expand-taxonomy"),
      btnCollapseTaxonomy: $("btn-collapse-taxonomy"),
      taxonomySummary: $("taxonomy-summary"),
      ontographTree: $("ontograph-tree"),
      btnRefreshInspection: $("btn-refresh-inspection"),
      inspectionSummary: $("inspection-summary"),
      inspectionBody: $("inspection-body"),
      regDomainId: $("reg-domain-id"),
      regDbPath: $("reg-db-path"),
      regFile: $("reg-file"),
      regLlmAssist: $("reg-llm-assist"),
      regName: $("reg-name"),
      regAdminToken: $("reg-admin-token"),
      btnRegisterDomain: $("btn-register-domain"),
      regStatus: $("reg-status"),
      semState: $("sem-state"),
      semFresh: $("sem-fresh"),
      semStatusBody: $("sem-status-body"),
      semQ: $("sem-q"),
      semMode: $("sem-mode"),
      semK: $("sem-k"),
      semHyde: $("sem-hyde"),
      semMeta: $("sem-meta"),
      semHits: $("sem-hits"),
      btnRefreshSemantic: $("btn-refresh-semantic"),
      btnSemSync: $("btn-sem-sync"),
      btnSemSearch: $("btn-sem-search"),
    };

    this.welcomeHtml = this.dom.chatHistory.innerHTML;
    // One operator token for every admin-gated action: remembered in
    // localStorage after first use and reused by the conductor button,
    // so nobody hunts for "the field below" again.
    if (this.dom.regAdminToken) {
      const saved = localStorage.getItem("pb_admin_token") || "";
      if (saved && !this.dom.regAdminToken.value)
        this.dom.regAdminToken.value = saved;
      this.dom.regAdminToken.addEventListener("change", () => {
        if (this.dom.regAdminToken.value)
          localStorage.setItem("pb_admin_token",
            this.dom.regAdminToken.value);
      });
    }
    this.dom.sessionInfo.textContent = "session: " + this.state.sessionId.slice(0, 12);
    this.bindEvents();
    this.init();
  }

  /* F30: the server owns session identity. A new session is minted on the
     server and returned; adopt it so the conversation keeps its memory. */
  adoptSessionId(id) {
    if (!id || id === this.state.sessionId) return;
    this.state.sessionId = id;
    sessionStorage.setItem("pybuddy_session_id", id);
    this.dom.sessionInfo.textContent = "session: " + id.slice(0, 12);
  }

  static loadSessionId() {
    try {
      let id = sessionStorage.getItem("pybuddy_session_id");
      if (!id) {
        id = "web_" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
        sessionStorage.setItem("pybuddy_session_id", id);
      }
      return id;
    } catch (_) {
      return "web_" + Math.random().toString(36).slice(2, 10);
    }
  }

  debounce(fn, wait) {
    let t;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn.apply(this, args), wait);
    };
  }

  /* ------------------------------------------------------------------ */
  /* Wiring                                                              */
  /* ------------------------------------------------------------------ */
  bindEvents() {
    this.dom.chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const text = this.dom.chatInput.value.trim();
      if (!text || this.state.pending) return;
      this.dom.chatInput.value = "";
      this.dom.chatInput.style.height = "auto";
      this.sendChat(text);
    });

    this.dom.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.dom.chatForm.requestSubmit();
      }
    });
    this.dom.chatInput.addEventListener("input", () => {
      const el = this.dom.chatInput;
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 132) + "px";
    });

    this.dom.scenarioChips.addEventListener("click", (e) => {
      const actionBtn = e.target.closest("[data-demo-action]");
      if (actionBtn) {
        this.runDemoAction(actionBtn.getAttribute("data-demo-action"));
        return;
      }
      const btn = e.target.closest("[data-query]");
      if (btn && !this.state.pending) this.sendChat(btn.getAttribute("data-query"));
    });

    this.dom.clearChatBtn.addEventListener("click", () => {
      this.dom.chatHistory.innerHTML = this.welcomeHtml;
      this.state.sessionCost = 0;
      this.state.chatHistory = [];
      this.updateCostOnly();
    });

    if (this.dom.btnCopyChatMd) {
      this.dom.btnCopyChatMd.addEventListener("click", () => this.exportChatMarkdown());
    }
    if (this.dom.btnCopyChatJson) {
      this.dom.btnCopyChatJson.addEventListener("click", () => this.exportChatJson());
    }
    if (this.dom.btnModelSelector) {
      this.dom.btnModelSelector.addEventListener("click", () => this.openModelSelector());
    }
    if (this.dom.modalCloseBtn) {
      this.dom.modalCloseBtn.addEventListener("click", () => this.closeModelSelector());
    }
    if (this.dom.modelModal) {
      this.dom.modelModal.addEventListener("click", (e) => {
        if (e.target === this.dom.modelModal) this.closeModelSelector();
      });
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") this.closeModelSelector();
      });
    }
    if (this.dom.modelModal) {
      this.dom.modelModal.addEventListener("click", (e) => {
        if (e.target === this.dom.modelModal) this.closeModelSelector();
      });
    }
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.dom.modelModal && !this.dom.modelModal.hidden) {
        this.closeModelSelector();
      }
    });
    if (this.dom.modelSearchInput) {
      this.dom.modelSearchInput.addEventListener("input", (e) => {
        this.state.modelSearchQuery = e.target.value.trim().toLowerCase();
        if (this.dom.modelSearchClear) {
          this.dom.modelSearchClear.hidden = !this.state.modelSearchQuery;
        }
        this.renderModelsList();
      });
    }
    if (this.dom.modelSearchClear) {
      this.dom.modelSearchClear.addEventListener("click", () => {
        this.dom.modelSearchInput.value = "";
        this.state.modelSearchQuery = "";
        this.dom.modelSearchClear.hidden = true;
        this.renderModelsList();
        this.dom.modelSearchInput.focus();
      });
    }
    if (this.dom.modelFilterTags) {
      this.dom.modelFilterTags.addEventListener("click", (e) => {
        const btn = e.target.closest(".filter-pill");
        if (!btn) return;
        this.dom.modelFilterTags.querySelectorAll(".filter-pill").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        this.state.activeModelFilter = btn.getAttribute("data-filter") || "all";
        this.renderModelsList();
      });
    }
    if (this.dom.modalModelsList) {
      this.dom.modalModelsList.addEventListener("click", (e) => {
        const card = e.target.closest("[data-model-id]");
        if (!card) return;
        const modelId = card.getAttribute("data-model-id");
        this.selectModel(modelId);
      });
    }

    this.dom.domainSelect.addEventListener("change", (e) => this.switchDomain(e.target.value));

    this.dom.tabs.forEach((tab) => {
      tab.addEventListener("click", () => this.activateTab(tab.getAttribute("data-tab")));
    });

    this.dom.catalogSearch.addEventListener("input", this.debounce((e) => {
      this.fetchCatalog(e.target.value.trim());
    }, 250));

    this.dom.catalogList.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-add-id]");
      if (btn) {
        this.addToBundle(btn.getAttribute("data-add-id"));
        return;
      }
      const moreBtn = e.target.closest("[data-load-more-catalog]");
      if (moreBtn) {
        this.fetchCatalog((this.dom.catalogSearch.value || "").trim(), true);
      }
    });

    this.dom.catalogList.addEventListener("scroll", () => {
      const el = this.dom.catalogList;
      if (
        !this.state.catalogLoadingMore &&
        this.state.catalogTotal != null &&
        this.state.catalog.length > 0 &&
        this.state.catalog.length < this.state.catalogTotal &&
        el.scrollTop + el.clientHeight >= el.scrollHeight - 140
      ) {
        this.fetchCatalog((this.dom.catalogSearch.value || "").trim(), true);
      }
    });

    this.dom.chatHistory.addEventListener("click", (e) => {
      const addBtn = e.target.closest("[data-add-id]");
      if (addBtn) this.addToBundle(addBtn.getAttribute("data-add-id"));

      const copyBtn = e.target.closest("[data-copy-turn]");
      if (copyBtn) {
        const idx = Number(copyBtn.getAttribute("data-copy-turn"));
        this.copyTurnMarkdown(idx);
      }

      const rawBtn = e.target.closest("[data-toggle-raw]");
      if (rawBtn) {
        const wrap = rawBtn.closest(".cv-diff-wrap");
        const grid = wrap && wrap.querySelector(".cv-diff-grid");
        if (grid) {
          grid.hidden = !grid.hidden;
          rawBtn.textContent = grid.hidden
            ? "Показати необрізану відповідь (ДО vs ПІСЛЯ)"
            : "Сховати необрізану відповідь (ДО vs ПІСЛЯ)";
        }
      }
    });

    this.dom.bundleChips.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-remove-id]");
      if (btn) this.removeFromBundle(btn.getAttribute("data-remove-id"));
    });
    this.dom.clearBundleBtn.addEventListener("click", () => {
      this.state.bundle = [];
      this.renderBundleChips();
      this.validateBundle();
    });

    const onSlider = this.debounce(() => this.fetchPareto(), 300);
    this.dom.budgetSlider.addEventListener("input", () => {
      this.dom.budgetValue.textContent = this.formatPrice(Number(this.dom.budgetSlider.value));
      onSlider();
    });
    this.dom.perfSlider.addEventListener("input", () => {
      this.dom.perfValue.textContent = this.dom.perfSlider.value;
      onSlider();
    });
    if (this.dom.paretoCategory) {
      this.dom.paretoCategory.addEventListener("change", () => {
        this.state.paretoCategory = this.dom.paretoCategory.value || "";
        this.state.paretoSelectedId = null;
        this.fetchPareto();
      });
    }
    if (this.dom.paretoCanvas) {
      this.dom.paretoCanvas.addEventListener("mousemove", (e) => this._handleParetoHover(e));
      this.dom.paretoCanvas.addEventListener("mouseleave", () => {
        if (this.dom.paretoTooltip) this.dom.paretoTooltip.hidden = true;
        if (this.state.paretoHoverId !== null) {
          this.state.paretoHoverId = null;
          if (this.state.lastPareto) this.drawPareto(this.state.lastPareto);
        }
      });
      this.dom.paretoCanvas.addEventListener("click", (e) => {
        const hit = this._paretoPointAt(e);
        if (hit) this.selectParetoPoint(hit.c, true);
      });
    }
    if (this.dom.paretoFront) {
      this.dom.paretoFront.addEventListener("click", (e) => {
        const row = e.target.closest("tr[data-item-id]");
        if (!row || !this.state.lastPareto) return;
        const pid = row.getAttribute("data-item-id");
        const cand = (this.state.lastPareto.all_candidates || []).find((x) => String(x.item_id) === String(pid));
        if (cand) this.selectParetoPoint(cand, true);
      });
    }
    if (this.dom.paretoPointCard) {
      this.dom.paretoPointCard.addEventListener("click", (e) => {
        const btn = e.target.closest("[data-add-bundle]");
        if (!btn) return;
        this.addToBundle(btn.getAttribute("data-add-bundle"));
      });
    }

    this.dom.refreshTelemetryBtn.addEventListener("click", () => {
      this.fetchTelemetry();
      this.refreshStatus();
    });

    if (this.dom.btnRefreshTaxonomy) {
      this.dom.btnRefreshTaxonomy.addEventListener("click", () => this.fetchTaxonomy());
      if (this.dom.btnExpandTaxonomy) {
        this.dom.btnExpandTaxonomy.addEventListener("click", () => this.setOntoExpanded(true));
      }
      if (this.dom.btnCollapseTaxonomy) {
        this.dom.btnCollapseTaxonomy.addEventListener("click", () => this.setOntoExpanded(false));
      }
      this.dom.btnRefreshInspection.addEventListener("click", () => this.fetchInspection());
      this.dom.btnRegisterDomain.addEventListener("click", () => this.registerDomain());
    }
    if (this.dom.inspectionBody) {
      this.dom.inspectionBody.addEventListener("click", (e) => {
        const btn = e.target.closest(".btn-rule-status");
        if (btn) this.toggleConstraintStatus(btn);
      });
    }

    if (this.dom.btnRefreshSemantic) {
      this.dom.btnRefreshSemantic.addEventListener("click", () => this.fetchSemanticStatus());
      this.dom.btnSemSync.addEventListener("click", () => this.refreshSemanticSync());
      this.dom.btnSemSearch.addEventListener("click", () => this.runSemanticSearch());
      this.dom.semQ.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); this.runSemanticSearch(); }
      });
    }

    if (this.dom.ontographTree) {
      this.dom.ontographTree.addEventListener("click", (e) => {
        const filterBtn = e.target.closest(".btn-node-filter");
        if (filterBtn) {
          e.stopPropagation();
          const filterName = filterBtn.getAttribute("data-filter-name");
          if (filterName) {
            this.dom.catalogSearch.value = filterName;
            this.fetchCatalog(filterName);
            this.activateTab("catalog");
            this.dom.chatInput.value = `Розкажи про наявність та особливості товарів у категорії "${filterName}"`;
            this.showToast(`<svg class=\"ic\" width=\"12\" height=\"12\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\" style=\"vertical-align:-1px\"><circle cx=\"10.5\" cy=\"10.5\" r=\"6.5\"/><path d=\"M15.5 15.5 21 21\"/></svg> Каталог відфільтровано за розділом: ${filterName}`);
          }
          return;
        }

        const header = e.target.closest(".onto-header");
        if (!header) return;
        const nodeEl = header.closest(".onto-node");
        if (!nodeEl) return;
        const childrenEl = nodeEl.querySelector(":scope > .onto-children");
        const toggleEl = header.querySelector(".onto-toggle");
        if (childrenEl) {
          childrenEl.classList.toggle("collapsed");
          if (toggleEl) {
            toggleEl.textContent = childrenEl.classList.contains("collapsed") ? "▶" : "▼";
          }
        }
      });
    }
  }

  /* ------------------------------------------------------------------ */
  /* Boot / transport / status                                           */
  /* ------------------------------------------------------------------ */
  async init() {
    // F53: the domain list and display names are fetched from the
    // backend's own registry before anything renders -- the page ships no
    // domain vocabulary of its own.
    await this.fetchDomains();
    this.renderScenarioChips();
    this.renderBundleChips();
    this.dom.kpiDomain.textContent = this.domainName(this.state.domain);
    this.dom.kpiScale.textContent = this.scaleLabel();
    this.dom.sbDomain.textContent = "⧉ " + this.state.domain;

    await this.refreshStatus();
    await this.fetchCatalog();
    await this.validateBundle();
    await this.fetchPareto();
    await this.fetchTelemetry();

    setInterval(() => this.refreshStatus(), 30000);
  }

  /**
   * Populate the domain selector and display-name map from the backend
   * registry. On failure the selector keeps a single option -- the
   * domain id itself -- so the UI degrades visibly instead of showing a
   * list of catalogues that do not exist.
   */
  async fetchDomains() {
    try {
      const data = await this.api("/api/domains");
      const domains = (data && data.domains) || [];
      const names = Object.create(null);
      const currencies = Object.create(null);
      const options = [];
      domains.forEach((d) => {
        if (!d || !d.id) return;
        names[d.id] = d.name || d.id;
        currencies[d.id] = d.currency || null;
        options.push(
          '<option value="' + this.escapeHtml(d.id) + '">' +
            this.escapeHtml(d.name || d.id) + "</option>"
        );
      });
      if (options.length) {
        this.dom.domainSelect.innerHTML = options.join("");
      }
      this.state.domainNames = names;
      this.state.domainCurrencies = currencies;
      // The server's active_domain is a global default -- it must not
      // override a domain the user already picked (domainLocked), or the
      // UI would "spontaneously" jump catalogues on every refresh.
      if (data && data.active_domain && !this.state.domainLocked) {
        this.state.domain = data.active_domain;
      }
      this.dom.domainSelect.value = this.state.domain;
      this.dom.budgetValue.textContent = this.formatPrice(Number(this.dom.budgetSlider.value));
    } catch (_) {
      // leave the selector as-is; state.domain stays whatever it was
    }
  }

  /**
   * Fetch wrapper: JSON in/out, timeout via AbortController, FastAPI
   * `detail` errors surfaced as readable messages.
   */
  async api(path, opts = {}) {
    const timeout = opts.timeout || 30000;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeout);
    try {
      const headers = Object.assign(
        {},
        opts.body ? { "Content-Type": "application/json" } : {},
        opts.headers || {}
      );
      const res = await fetch(path, {
        method: opts.method || "GET",
        headers: Object.keys(headers).length ? headers : undefined,
        body: opts.body ? JSON.stringify(opts.body) : undefined,
        signal: ctrl.signal,
      });
      let data = null;
      try { data = await res.json(); } catch (_) { /* non-JSON body */ }
      if (!res.ok) {
        let detail = res.statusText;
        if (data && data.detail !== undefined) {
          detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
        }
        throw new Error("HTTP " + res.status + " — " + detail);
      }
      return data;
    } catch (err) {
      if (err && err.name === "AbortError") {
        throw new Error("Перевищено час очікування (" + Math.round(timeout / 1000) + " с).");
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  setConnection(online, text) {
    this.dom.connDot.className = "dot " + (online ? "" : "dot-err");
    this.dom.connText.textContent = text || (online ? "online" : "offline");
  }

  async refreshStatus() {
    try {
      const q = this.state.domain ? "?domain=" + encodeURIComponent(this.state.domain) : "";
      const s = await this.api("/api/status" + q);
      this.setConnection(true);
      const gate = s.invariants && s.invariants.zero_hallucination_gate;
      const gateActive = gate === "ACTIVE";
      this.dom.chipGateText.textContent = "Gate: " + (gate || "?");
      this.dom.chipGate.className = "chip " + (gateActive ? "chip-ok" : "chip-err");
      if (s.model) {
        this.dom.chipModelText.textContent = "model: " + s.model;
        this.state.model = s.model;
      } else {
        this.dom.chipModelText.textContent = "model: —";
      }
      this.dom.chipNodesText.textContent = "nodes: " + (s.nodes_count != null ? s.nodes_count : "—");
      if (s.active_domain && s.active_domain !== this.state.domain
          && !this.state.domainLocked) {
        this.state.domain = s.active_domain;
        this.dom.domainSelect.value = s.active_domain;
      }
    } catch (_) {
      this.setConnection(false);
    }
  }

  /* ------------------------------------------------------------------ */
  /* Chat                                                                */
  /* ------------------------------------------------------------------ */
  async sendChat(text) {
    this.state.chatHistory.push({
      role: "user",
      content: text,
      timestamp: new Date().toISOString()
    });
    this.appendUserMessage(text);
    const loadingId = this.appendLoading();
    this.state.pending = true;
    this.dom.sendBtn.disabled = true;
    try {
      const data = await this.api("/api/chat", {
        method: "POST",
        timeout: 180000,
        body: {
          query: text,
          session_id: this.state.sessionId,
          domain: this.state.domain,
          model: this.state.model || undefined,
        },
      });
      this.removeNode(loadingId);
      this.adoptSessionId(data.session_id);
      this.state.chatHistory.push({
        role: "assistant",
        data: data,
        timestamp: new Date().toISOString()
      });
      this.appendBotMessage(data);
      this.state.lastChat = data;
      this.state.sessionCost += Number(data.token_cost || 0);
      this.setConnection(true);
      this.updateAfterAnswer(data);
    } catch (err) {
      this.removeNode(loadingId);
      this.appendErrorMessage(err && err.message ? err.message : String(err));
      this.setConnection(false);
    } finally {
      this.state.pending = false;
      this.dom.sendBtn.disabled = false;
      this.dom.chatInput.focus();
    }
  }

  nearBottom() {
    const el = this.dom.chatHistory;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 90;
  }

  scrollToBottom(force) {
    const el = this.dom.chatHistory;
    if (force || this.nearBottom()) el.scrollTop = el.scrollHeight;
  }

  appendNode(className, innerHtml) {
    const el = document.createElement("article");
    el.className = className;
    el.innerHTML = innerHtml;
    this.dom.chatHistory.appendChild(el);
    this.scrollToBottom();
    return el;
  }

  removeNode(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  appendUserMessage(text) {
    this.appendNode(
      "msg msg-user",
      '<div class="msg-role">Ви</div><div class="msg-body">' +
        this.escapeHtml(text) +
        "</div>"
    );
  }

  appendSystemMessage(text) {
    this.appendNode(
      "msg msg-assistant",
      '<div class="msg-role">Система</div><div class="msg-body"><p>' +
        text +
        "</p></div>"
    );
  }

  appendErrorMessage(message) {
    this.appendNode(
      "msg msg-error",
      '<div class="msg-role">Помилка</div><div class="msg-body">' +
        this.escapeHtml(message) +
        "</div>"
    );
  }

  appendLoading() {
    const id = "loading-" + Date.now() + "-" + Math.floor(Math.random() * 1e4);
    const el = document.createElement("article");
    el.className = "msg msg-assistant";
    el.id = id;
    el.innerHTML =
      '<div class="msg-role">Консультант</div><div class="msg-body">' +
      '<span class="typing"><span class="typing-dots"><i></i><i></i><i></i></span>' +
      "Аналіз каталогу та перевірка гейтом…</span></div>";
    this.dom.chatHistory.appendChild(el);
    this.scrollToBottom(true);
    return id;
  }

  /** The gate verdict, rendered as a badge. Never re-derived, never hidden. */
  gateBadge(data) {
    const tr = data.termination_reason;
    if (tr === "DETERMINISTIC_HYPERGRAPH") {
      return { badge: '<span class="badge badge-info">◆ Гіперграф · детерміновано</span>', short: "— (гіперграф)", engine: "hypergraph" };
    }
    if (data.tier === "tier1_fast_path") {
      return { badge: '<span class="badge badge-info">◆ Tier-1 SQL · детерміновано</span>', short: "— (SQL)", engine: "tier1_sql" };
    }
    const map = {
      SUCCESS_GROUNDED: ['<span class="badge badge-ok">✓ Гейт: заземлено</span>', "заземлено", "react+gate"],
      SUCCESS_GROUNDED_WITH_CORRECTION: ['<span class="badge badge-warn">✓ Гейт: виправлено (частково)</span>', "виправлено", "react+gate"],
      SUCCESS_GROUNDED_AFTER_CORRECTION: ['<span class="badge badge-warn">✓ Гейт: перебудовано після помилки</span>', "перебудовано", "react+gate"],
      HALLUCINATION_GATE_VIOLATION: ['<span class="badge badge-err"><svg class=\"ic\" width=\"12\" height=\"12\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\" style=\"vertical-align:-1px\"><circle cx=\"12\" cy=\"12\" r=\"9\"/><path d=\"M9 9l6 6M15 9l-6 6\"/></svg> Гейт: відмова · непідтверджені твердження</span>', "ВІДМОВА", "react+gate"],
      STEP_BUDGET_EXCEEDED: ['<span class="badge badge-warn"><svg class=\"ic\" width=\"12\" height=\"12\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\" style=\"vertical-align:-1px\"><path d=\"M12 4 2.5 20h19z\"/><path d=\"M12 10v4M12 17.5v.5\"/></svg> Ліміт кроків ReAct</span>', "ліміт кроків", "react"],
      STAGNATION_DETECTED: ['<span class="badge badge-warn">⚠ Зупинено: стагнація</span>', "стагнація", "react"],
      UNPARSEABLE_OUTPUT: ['<span class="badge badge-err">⛔ Не розпізнано вихід моделі</span>', "нерозпізнано", "react"],
    };
    if (tr && map[tr]) {
      return { badge: map[tr][0], short: map[tr][1], engine: map[tr][2] };
    }
    if (tr) {
      return {
        badge: '<span class="badge badge-warn">' + this.escapeHtml(tr) + "</span>",
        short: this.escapeHtml(tr),
        engine: "react",
      };
    }
    // No tool calls -> the model answered directly (Tier-0).
    return { badge: '<span class="badge">○ Tier-0 · без інструментів</span>', short: "не застосовувався", engine: "tier0_llm" };
  }

  tierBadge(data) {
    if (!data.tier) return "";
    const label = TIER_LABELS[data.tier] || data.tier;
    return '<span class="badge badge-info"><svg class=\"ic\" width=\"12\" height=\"12\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\" style=\"vertical-align:-1px\"><path d=\"M13 2 4.5 13.5H11L10 22l8.5-11.5H12z\"/></svg> ' + this.escapeHtml(label) + "</span>";
  }


  appendBotMessage(data) {
    const gate = this.gateBadge(data);
    const stepsCount = data.steps ? data.steps.length : 0;
    const latencyMs = Math.round(Number(data.latency_seconds || 0) * 1000);
    const cost = Number(data.token_cost || 0);

    const vui = window.PyBuddyVerifierUI || null;
    const esc = (s) => this.escapeHtml(s);
    const md = (s) => this.renderMarkdown(s);

    const metaBits = [];
    metaBits.push(gate.badge);
    metaBits.push(this.renderNvfgPill(data.nvfg));
    if (vui) {
      const vBadge = vui.renderVerifierBadge(data);
      if (vBadge) metaBits.push(vBadge);
    }
    const tb = this.tierBadge(data);
    if (tb) metaBits.push(tb);
    if (data.model) metaBits.push('<span class="badge"><rect x="2.5" y="4" width="19" height="13" rx="1.5"/><path d="M9 21h6M12 17v4"/> ' + this.escapeHtml(data.model) + "</span>");
    metaBits.push('<span class="badge">' + latencyMs + " мс</span>");
    metaBits.push('<span class="badge">$' + cost.toFixed(4) + "</span>");
    if (stepsCount) metaBits.push('<span class="badge">' + stepsCount + " крок(и)</span>");
    const turnIdx = this.state.chatHistory.length - 1;
    metaBits.push(
      '<button class="msg-toolbar-btn" type="button" data-copy-turn="' +
        turnIdx +
        '" title="Копіювати крок ReAct з траєкторією"><svg class=\"ic\" width=\"12\" height=\"12\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\" style=\"vertical-align:-1px\"><path d=\"M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z\"/><path d=\"M14 3v5h5\"/><path d=\"M9 13h6M9 17h4\"/></svg> Траєкторія</button>'
    );

    const answerHtml = vui
      ? vui.annotateProvenance(md(data.final_answer || ""), data.recommended_items, this.state.domain, esc)
      : md(data.final_answer || "");
    const relaxHtml = vui ? vui.renderRelaxationCard(data, esc) : "";
    const verifierHtml = vui ? vui.renderVerificationPanel(data, esc, md) : "";

    const body =
      answerHtml +
      relaxHtml +
      verifierHtml +
      this.renderRecommendations(data.recommended_items, data.currency !== undefined ? data.currency : this.activeCurrency(data.domain)) +
      this.renderStageTimings(data.stage_timings) +
      this.renderSteps(data.steps);

    this.appendNode(
      "msg msg-assistant",
      '<div class="msg-role">Консультант</div>' +
        '<div class="msg-body">' + body + "</div>" +
        '<div class="msg-meta">' + metaBits.join("") + "</div>"
    );
  }

  renderNvfgPill(nv) {
    if (!nv) {
      return '<span class="badge nvfg-pill is-na" title="Солвер обмежень для цієї відповіді не запускався">NVFG · не застосовано</span>';
    }
    const esc = (s) => this.escapeHtml(String(s));
    if (nv.indicator === null || nv.indicator === undefined) {
      return '<span class="badge nvfg-pill is-unk" title="Заземлення відповіді не встановлено">NVFG · не визначено</span>';
    }
    const cls = nv.indicator ? "is-ok" : "is-err";
    const basis = nv.indicator ? "заземлено, інформативно"
      : nv.grounded === false ? "гейт відхилив відповідь" : "інформативного вмісту немає";
    const relax = nv.relaxed && nv.relaxation_text ? " · " + esc(nv.relaxation_text) : "";
    const regret = typeof nv.relative_regret === "number"
      ? " · регрес " + (nv.relative_regret * 100).toFixed(2) + " %" : "";
    return `<span class="badge nvfg-pill ${cls}" title="статус солвера: ${esc(nv.status)}">` +
      `NVFG = ${nv.indicator ? 1 : 0} · ${basis}${relax}${regret}</span>`;
  }

  renderStageTimings(st) {
    if (!st || typeof st !== "object") return "";
    const total = Number(st.total_time_ms || 0);
    const llm = Number(st.llm_time_ms || 0);
    const tool = Number(st.tool_time_ms || 0);
    const gate = Number(st.gate_time_ms || 0);
    const router = Number(st.router_time_ms || 0);

    if (total <= 0) return "";
    const llmPct = total > 0 ? ((llm / total) * 100).toFixed(1) : "0.0";

    const parts = [];
    if (llm > 0) {
      parts.push('<div class="timing-pill pill-llm"><svg class=\"ic\" width=\"12\" height=\"12\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\" style=\"vertical-align:-1px\"><rect x=\"5\" y=\"5\" width=\"14\" height=\"14\" rx=\"1.5\"/><rect x=\"10\" y=\"10\" width=\"4\" height=\"4\"/><path d=\"M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3\"/></svg> LLM міркування: ' + (llm / 1000).toFixed(2) + ' с (' + llmPct + '%)</div>');
    }
    if (tool > 0) {
      parts.push('<div class="timing-pill pill-tool"><svg class=\"ic\" width=\"12\" height=\"12\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\" style=\"vertical-align:-1px\"><path d=\"M13 2 4.5 13.5H11L10 22l8.5-11.5H12z\"/></svg> БД &amp; Інструменти: ' + Math.round(tool) + ' мс</div>');
    }
    if (gate > 0) {
      parts.push('<div class="timing-pill pill-gate"><svg class=\"ic\" width=\"12\" height=\"12\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\" style=\"vertical-align:-1px\"><path d=\"M12 2.5 4.5 5.5v6c0 5 3.2 8.4 7.5 10 4.3-1.6 7.5-5 7.5-10v-6z\"/><path d=\"m8.8 12 2.2 2.2 4.2-4.5\"/></svg> Gate: ' + Math.round(gate) + ' мс</div>');
    }
    if (router > 0) {
      parts.push('<div class="timing-pill pill-router">⏱️ Роутер: ' + Math.round(router) + ' мс</div>');
    }

    if (!parts.length) return "";
    return '<div class="stage-timings-bar" title="Прозорий розподіл часу виконання за стадіями">' + parts.join("") + '</div>';
  }

  renderRecommendations(items, currency) {
    if (!Array.isArray(items) || !items.length) return "";
    const cur = currency !== undefined ? currency : this.activeCurrency();
    const cards = items.slice(0, 8).map((it) => {
      const name = it.name || it.id || "";
      const price = it.price != null ? this.formatPrice(it.price, cur) : "";
      const sub = [it.category, it.id].filter(Boolean).map((s) => String(s)).join(" · ");
      const addBtn = it.id
        ? '<button class="btn btn-ghost product-add" type="button" title="Додати до комплекту" data-add-id="' +
          this.escapeHtml(String(it.id)) + '">+</button>'
        : "";
      return (
        '<div class="rec-card"><div class="rec-info">' +
        '<div class="rec-name">' + this.escapeHtml(name) + "</div>" +
        (sub ? '<div class="rec-sub">' + this.escapeHtml(sub) + "</div>" : "") +
        "</div>" +
        (price ? '<div class="rec-price">' + price + "</div>" : "") +
        addBtn +
        "</div>"
      );
    });
    return '<div class="rec-cards">' + cards.join("") + "</div>";
  }

  renderSteps(steps) {
    if (!Array.isArray(steps) || !steps.length) return "";
    const items = steps.map((s, i) => {
      const tool = s.tool_name || (s.action && s.action.tool) || "міркування";
      let time = "";
      if (s.llm_latency_s != null && s.tool_latency_s != null) {
        time = '<span class="step-time">LLM: ' + Number(s.llm_latency_s).toFixed(1) + 'с · Tool: ' + Math.round(Number(s.tool_latency_s) * 1000) + 'мс</span>';
      } else if (s.latency_s != null) {
        time = '<span class="step-time">' + Math.round(Number(s.latency_s) * 1000) + ' мс</span>';
      }
      const fields = [];
      const push = (label, value, isError) => {
        if (value === undefined || value === null || value === "") return;
        const text = typeof value === "object" ? JSON.stringify(value, null, 1) : String(value);
        fields.push(
          '<div class="step-field"><span class="step-field-label">' + label + '</span>' +
          '<span class="step-field-val' + (isError ? " is-error" : "") + '">' +
          this.escapeHtml(this.truncate(text, 700)) +
          "</span></div>"
        );
      };
      push("thought:", s.thought);
      push("action:", s.action || s.args);
      push("observation:", s.observation);
      push("model_step:", s.model_step);
      push("error:", s.error, true);
      return (
        '<div class="react-step"><div class="react-step-head">' +
        "<span>#" + (s.step != null ? s.step : i + 1) + " · " + this.escapeHtml(String(tool)) + "</span>" +
        time +
        "</div>" + fields.join("") +
        "</div>"
      );
    });
    return (
      '<details class="react-steps"><summary>Траєкторія ReAct (' +
      steps.length +
      " кроків)</summary>" +
      items.join("") +
      "</details>"
    );
  }

  /* ------------------------------------------------------------------ */
  /* Rendering helpers                                                   */
  /* ------------------------------------------------------------------ */
  fmt(v) {
    if (v == null) return "";
    if (typeof v === "object") {
      const vals = Object.values(v).filter((x) => x != null && x !== "");
      return vals.length ? vals.join(" · ") : "";
    }
    return String(v);
  }

  escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  truncate(text, n) {
    const s = String(text);
    return s.length > n ? s.slice(0, n) + " …" : s;
  }

  activeCurrency(dom) {
    const key = dom || this.state.domain;
    const map = this.state.domainCurrencies;
    return map && map[key] ? map[key] : null;
  }

  formatCurrencyUnit(currency, style = "symbol") {
    if (!currency) return "";
    const code = String(currency).trim().toUpperCase();
    if (!code || code === "NONE" || code === "NULL") return "";
    if (code === "UAH") return style === "word" ? "грн" : "₴";
    return code;
  }

  formatPrice(value, currencyOverride) {
    if (value === null || value === undefined || value === "") return "—";
    const n = Number(value);
    if (!isFinite(n)) return "—";
    const cur = currencyOverride !== undefined ? currencyOverride : this.activeCurrency();
    const unit = this.formatCurrencyUnit(cur, "symbol");
    const numStr = n.toLocaleString("uk-UA", { maximumFractionDigits: 2 });
    return unit ? numStr + " " + unit : numStr;
  }

  inlineMarkdown(escaped) {
    return escaped
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  /** Escape first, format second: user/model text can never inject markup. */
  renderMarkdown(raw) {
    const pre = String(raw || "")
      .replace(/(^|\n)([ \t]*(?:\*\*|__)?\d{1,3}\.(?:\*\*|__)?)[ \t]*\r?\n(?=[ \t]*\S)/g, "$1$2 ")
      .replace(/([:;,.!?)\S])[ \t]+(?=\d{1,2}\.\s+(?:\*\*|__))/g, "$1\n");
    const lines = this.escapeHtml(pre).split(/\r?\n/);
    let html = "";
    let listTag = null;
    const closeList = () => {
      if (listTag) {
        html += `</${listTag}>`;
        listTag = null;
      }
    };
    for (const line of lines) {
      const ul = line.match(/^\s*[-*•]\s+(.*)$/);
      if (ul) {
        if (listTag && listTag !== "ul") closeList();
        if (!listTag) {
          html += "<ul>";
          listTag = "ul";
        }
        html += "<li>" + this.inlineMarkdown(ul[1]) + "</li>";
        continue;
      }
      const ol = line.match(/^\s*(?:\*\*|__)?(\d{1,3})\.(?:\*\*|__)?\s+(.*)$/);
      if (ol) {
        const ordNum = Number(ol[1]);
        if (listTag && listTag !== "ol") closeList();
        if (!listTag) {
          html += ordNum > 1 ? `<ol start="${ordNum}">` : "<ol>";
          listTag = "ol";
        }
        html += "<li>" + this.inlineMarkdown(ol[2]) + "</li>";
        continue;
      }
      closeList();
      if (line.trim() === "") continue;
      html += "<p>" + this.inlineMarkdown(line) + "</p>";
    }
    closeList();
    return html || "<p>—</p>";
  }


  /* ------------------------------------------------------------------ */
  /* Catalog                                                             */
  /* ------------------------------------------------------------------ */
  async fetchCatalog(search = "", append = false) {
    const dom = this.state.domain;
    const pageSize = 100;
    const offset = append ? (this.state.catalog || []).length : 0;
    if (append) {
      if (this.state.catalogLoadingMore) return;
      if (this.state.catalogTotal != null && offset >= this.state.catalogTotal) return;
      this.state.catalogLoadingMore = true;
      const moreBtn = this.dom.catalogList.querySelector("[data-load-more-catalog]");
      if (moreBtn) {
        moreBtn.disabled = true;
        moreBtn.textContent = "Завантаження…";
      }
    } else if ((search || "") === (this.dom.catalogSearch.value || "").trim()) {
      this.state.catalogLoadingMore = false;
      this.dom.catalogList.innerHTML = '<div class="list-note">Завантаження каталогу…</div>';
      this.dom.catalogCount.textContent = "…";
    }
    try {
      const data = await this.api(
        "/api/catalog?domain=" + encodeURIComponent(dom) +
        "&search=" + encodeURIComponent(search) +
        "&limit=" + pageSize +
        "&offset=" + offset
      );
      if (dom !== this.state.domain) return;
      const cur = (this.dom.catalogSearch.value || "").trim();
      if ((search || "") !== cur) return;
      const incoming = data.products || [];
      if (append && offset > 0) {
        const seen = new Set((this.state.catalog || []).map((p) => String(p.id)));
        for (const p of incoming) {
          if (p && !seen.has(String(p.id))) {
            this.state.catalog.push(p);
            seen.add(String(p.id));
          }
        }
      } else {
        this.state.catalog = incoming;
        this.dom.catalogList.scrollTop = 0;
      }
      this.state.catalogRelaxation = data.relaxation || null;
      this.state.catalogTotal = Number.isFinite(data.total) ? data.total : null;
      if (!search && this.dom.kpiScale) {
        this.dom.kpiScale.textContent = this.scaleLabel();
      }
      if (Object.prototype.hasOwnProperty.call(data, "currency")) {
        this.state.domainCurrencies[dom] = data.currency || null;
      }
      this.state.catalog.forEach((p) => {
        if (p && p.id) this.state.nameById[p.id] = p.name;
      });
      const total = data.total || 0;
      const shown = this.state.catalog.length;
      this.dom.catalogCount.textContent =
        shown < total
          ? `${shown.toLocaleString("uk-UA")} з ${total.toLocaleString("uk-UA")}`
          : `${total.toLocaleString("uk-UA")} знайдено`;
      this.renderCatalog();
      if (!append) this.renderScenarioChips();
      this.setConnection(true);
    } catch (err) {
      if (!append) {
        this.dom.catalogCount.textContent = "—";
        this.dom.catalogList.innerHTML =
          '<div class="list-note is-error">Не вдалося завантажити каталог: ' +
          this.escapeHtml(err.message || String(err)) +
          "</div>";
      }
      this.setConnection(false);
    } finally {
      this.state.catalogLoadingMore = false;
    }
  }

  renderCatalog() {
    const items = this.state.catalog;
    if (!items.length) {
      const rel = this.state.catalogRelaxation;
      if (rel && rel.items && rel.items.length) {
        // Evidence-bound relaxation: zero lexical hits stays zero, and
        // each associative candidate shows the EAV axis that carried it.
        const rows = rel.items.map((i) => {
          const via = i.axis
            ? "через " + i.axis + " = " + (i.matched_value || "?")
            : i.via;
          return '<div class="product-item"><div class="product-info">' +
            '<div class="product-name">' + this.escapeHtml(i.name || i.entity_id) + "</div>" +
            '<div class="product-sub">≈ ' + this.escapeHtml(via) + "</div>" +
            "</div></div>";
        }).join("");
        this.dom.catalogList.innerHTML =
          '<div class="list-note">Прямих збігів немає — асоціативні ' +
          'кандидати (EBR):</div>' + rows;
      } else {
        this.dom.catalogList.innerHTML =
          '<div class="list-note">Нічого не знайдено. Спробуйте інший запит.</div>';
      }
      return;
    }
    const total = this.state.catalogTotal || items.length;
    const moreFooter = items.length < total
      ? '<div class="catalog-more-wrap"><button class="btn btn-ghost catalog-more-btn" type="button" data-load-more-catalog>' +
        `Показати ще (+${Math.min(100, total - items.length)} · завантажено ${items.length.toLocaleString("uk-UA")} з ${total.toLocaleString("uk-UA")})` +
        "</button></div>"
      : "";
    this.dom.catalogList.innerHTML = items
      .map((p) => {
        const vendor = p.attributes && p.attributes.vendor ? " · " + p.attributes.vendor : "";
        const inBundle = this.state.bundle.indexOf(p.id) !== -1;
        return (
          '<div class="product-item">' +
          '<div class="product-info">' +
          '<div class="product-name">' + this.escapeHtml(p.name) + "</div>" +
          '<div class="product-sub">' + this.escapeHtml(String(p.category || "") + vendor) + "</div>" +
          "</div>" +
          '<div class="product-price">' + this.formatPrice(p.price) + "</div>" +
          '<button class="btn product-add" type="button" data-add-id="' +
          this.escapeHtml(String(p.id)) + '" title="Додати до комплекту"' +
          (inBundle ? " disabled" : "") + ">" + (inBundle ? "✓" : "+") + "</button>" +
          "</div>"
        );
      })
      .join("") + moreFooter;
  }

  /* ------------------------------------------------------------------ */
  /* Bundle compatibility                                                */
  /* ------------------------------------------------------------------ */
  scaleLabel() {
    // F35: the badge reports the measured catalogue size, not a hardcoded
    // marketing figure. Before the first catalog fetch, unknown stays unknown.
    if (this.state.catalogTotal == null) {
      return "База: обсяг не завантажено";
    }
    return "База: " + this.state.catalogTotal.toLocaleString("uk-UA") + " товарів";
  }

  addToBundle(id) {
    if (!id || this.state.bundle.indexOf(id) !== -1) return;
    this.state.bundle.push(id);
    this.renderBundleChips();
    this.validateBundle();
    this.activateTab("bundle");
    this.renderCatalog();
  }

  removeFromBundle(id) {
    this.state.bundle = this.state.bundle.filter((x) => x !== id);
    this.renderBundleChips();
    this.validateBundle();
    this.renderCatalog();
  }

  renderBundleChips() {
    const ids = this.state.bundle;
    this.dom.bundleCount.hidden = ids.length === 0;
    this.dom.bundleCount.textContent = String(ids.length);
    if (!ids.length) {
      this.dom.bundleChips.innerHTML = '<span class="muted">Кошик порожній — додайте товари з каталогу</span>';
      return;
    }
    this.dom.bundleChips.innerHTML = ids
      .map((id) => {
        const name = this.state.nameById[id] || id;
        return (
          '<span class="bundle-chip"><span title="' + this.escapeHtml(name) + '">' +
          this.escapeHtml(name) +
          '</span><button type="button" data-remove-id="' +
          this.escapeHtml(String(id)) +
          '" title="Прибрати">×</button></span>'
        );
      })
      .join("");
  }


  async validateBundle() {
    const ids = this.state.bundle;
    const dom = this.state.domain;
    if (!ids.length) {
      this.dom.bundleStatus.textContent = "—";
      this.dom.bundleStatus.className = "verdict";
      this.dom.conflictBox.hidden = true;
      this.renderSvgGraph({ nodes: [], links: [] });
      return;
    }
    try {
      const data = await this.api("/api/validate_bundle", {
        method: "POST",
        body: { product_ids: ids, domain: dom },
      });
      if (dom !== this.state.domain) return;
      (data.products || []).forEach((p) => {
        if (p && p.id) this.state.nameById[p.id] = p.name;
      });
      // F15: verdict is three-state — "unknown" (is_compatible=null, no
      // rules evaluated) must not be painted as incompatibility.
      const verdict = data.verdict ||
        (data.is_compatible === true ? "compatible"
         : data.is_compatible === false ? "incompatible" : "unknown");
      if (verdict === "compatible") {
        this.dom.bundleStatus.textContent = "✓ Комплект сумісний";
        this.dom.bundleStatus.className = "verdict ok";
        this.dom.conflictBox.hidden = true;
      } else if (verdict === "incompatible") {
        this.dom.bundleStatus.textContent = "✗ Виявлено несумісність";
        this.dom.bundleStatus.className = "verdict err";
        this.dom.conflictBox.hidden = false;
        this.dom.conflictList.innerHTML = (data.conflicts || [])
          .map((c) => "<li>" + this.escapeHtml(String(c)) + "</li>")
          .join("");
      } else if (verdict === "candidate_conflict") {
        this.dom.bundleStatus.textContent = "⚠ Ймовірний конфлікт";
        this.dom.bundleStatus.className = "verdict warn";
        this.dom.conflictBox.hidden = false;
        this.dom.conflictList.innerHTML =
          (data.candidate_conflicts || data.conflicts || [])
            .map((c) => "<li>" + this.escapeHtml(String(c)) + "</li>")
            .join("");
      } else {
        this.dom.bundleStatus.textContent =
          "— Сумісність не підтверджена";
        this.dom.bundleStatus.className = "verdict warn";
        this.dom.conflictBox.hidden = false;
        const gaps = (data.coverage_gaps || [])
          .map((c) => "<li>" + this.escapeHtml(String(c)) + "</li>")
          .join("");
        this.dom.conflictList.innerHTML = gaps ||
          '<li class="muted">Немає правил сумісності для цих товарів — ' +
          "вердикт невідомий, це не доведення сумісності.</li>";
      }
      this.renderSvgGraph(data.graph_data || { nodes: [], links: [] });
      this.renderBundleChips();
      this.setConnection(true);
    } catch (err) {
      this.dom.bundleStatus.textContent = "помилка перевірки";
      this.dom.bundleStatus.className = "verdict err";
      this.dom.conflictBox.hidden = false;
      this.dom.conflictList.innerHTML =
        '<li class="muted">' + this.escapeHtml(err.message || String(err)) + "</li>";
      this.setConnection(false);
    }
  }

  categoryColor(category) {
    const key = String(category || "");
    let h = 0;
    for (let i = 0; i < key.length; i++) {
      h = (h * 31 + key.charCodeAt(i)) >>> 0;
    }
    return CATEGORY_PALETTE[h % CATEGORY_PALETTE.length];
  }

  domainName(id) {
    // F53: names come from /api/domains (profile-derived on the backend),
    // never from a fixture-bound lookup table baked into the page.
    return this.state.domainNames[id] || id;
  }

  renderSvgGraph(graphData) {
    const nodes = graphData.nodes || [];
    const links = graphData.links || [];
    const svg = this.dom.hypergraphSvg;
    if (!nodes.length) {
      svg.innerHTML =
        '<g class="hg-empty"><circle cx="300" cy="165" r="24" class="hg-empty-ring"/>' +
        '<path d="M290 165h20M300 155v20" class="hg-empty-plus"/>' +
        '<text x="300" y="215" text-anchor="middle" class="hg-empty-text">' +
        "Кошик порожній — додайте товари для перевірки сумісності</text></g>";
      return;
    }
    const W = 600, H = 380;
    const cx = W / 2, cy = H / 2;
    const R = Math.min(W, H) * 0.30;
    const coords = {};
    nodes.forEach((n, i) => {
      const a = (i / nodes.length) * 2 * Math.PI - Math.PI / 2;
      coords[n.id] = { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a), a, n };
    });

    let out = "";
    links.forEach((l, idx) => {
      const s = coords[l.source], t = coords[l.target];
      if (!s || !t) return;
      const mx = (s.x + t.x) / 2 + (cx - (s.x + t.x) / 2) * 0.35;
      const my = (s.y + t.y) / 2 + (cy - (s.y + t.y) / 2) * 0.35;
      const cls = { conflict: "hg-edge-conflict", warn: "hg-edge-warn",
                    unknown: "hg-edge-unknown" }[l.status] || "hg-edge-ok";
      out += '<path class="hg-edge ' + cls + '" d="M' + s.x.toFixed(1) + ',' +
        s.y.toFixed(1) + ' Q' + mx.toFixed(1) + ',' + my.toFixed(1) + ' ' +
        t.x.toFixed(1) + ',' + t.y.toFixed(1) + '"><title>' +
        this.escapeHtml(l.label || "") + "</title></path>";
      if (l.label) {
        const bx = 0.25 * s.x + 0.5 * mx + 0.25 * t.x;
        const by = 0.25 * s.y + 0.5 * my + 0.25 * t.y;
        const dx = t.x - s.x, dy = t.y - s.y;
        const len = Math.hypot(dx, dy) || 1;
        const nx = -dy / len, ny = dx / len;
        const sign = idx % 2 === 0 ? 1 : -1;
        const lx = Math.max(82, Math.min(518, bx + nx * sign * 14));
        const ly = Math.max(16, Math.min(366, by + ny * sign * 11 - 3));
        let shortLab = String(l.label).trim();
        const bm = shortLab.match(/\[([^\]]+)\]/);
        if (bm && shortLab.length > 26) {
          shortLab = "[" + bm[1] + "] " + (l.status === "conflict" ? "несумісні" : l.status === "warn" ? "ризик" : "правило");
        } else if (shortLab.length > 28) {
          shortLab = shortLab.slice(0, 27) + "…";
        }
        out += '<text class="hg-edge-lab" x="' + lx.toFixed(1) + '" y="' +
          ly.toFixed(1) + '" text-anchor="middle"><title>' +
          this.escapeHtml(l.label) + "</title>" +
          this.escapeHtml(shortLab) + "</text>";
      }
    });

    nodes.forEach((n) => {
      const c = coords[n.id];
      const color = this.categoryColor(n.category);
      const abbr = String(n.category || "·").slice(0, 3).toUpperCase();
      const cos = Math.cos(c.a), sin = Math.sin(c.a);
      const anchor = Math.abs(cos) < 0.35 ? "middle" : (cos > 0 ? "start" : "end");
      const lx = cos * 48, ly = sin * 48;
      const name = n.name && n.name.length > 26 ? n.name.slice(0, 24) + "…" : (n.name || "");
      out += '<g class="hg-node" transform="translate(' + c.x.toFixed(1) + "," +
        c.y.toFixed(1) + ')">' +
        '<line class="hg-lead" x1="' + (cos * 21).toFixed(1) + '" y1="' +
          (sin * 21).toFixed(1) + '" x2="' + (cos * 42).toFixed(1) +
          '" y2="' + (sin * 42).toFixed(1) + '"/>' +
        '<circle r="19" class="hg-node-fill"/>' +
        '<circle r="19" fill="none" stroke="' + color + '" stroke-width="1.6"/>' +
        '<text class="hg-abbr" y="3.5" text-anchor="middle">' + this.escapeHtml(abbr) + "</text>" +
        '<text class="hg-name" x="' + lx.toFixed(1) + '" y="' + (ly - 2).toFixed(1) +
          '" text-anchor="' + anchor + '">' + this.escapeHtml(name) + "</text>" +
        '<text class="hg-price" x="' + lx.toFixed(1) + '" y="' + (ly + 11).toFixed(1) +
          '" text-anchor="' + anchor + '">' + this.escapeHtml(this.formatPrice(n.price)) + "</text>" +
        "<title>" + this.escapeHtml(n.name || "") + "</title></g>";
    });
    svg.innerHTML = out;
  }


  /* ------------------------------------------------------------------ */
  /* Pareto frontier                                                     */
  /* ------------------------------------------------------------------ */
  syncParetoCategories(data) {
    const sel = this.dom.paretoCategory;
    if (!sel || !data) return;
    const cats = Array.isArray(data.categories) ? data.categories : [];
    const current = data.selected_category || this.state.paretoCategory || "";
    const esc = (s) => this.escapeHtml(String(s));
    const opts = ['<option value="">Усі категорії</option>']
      .concat(cats.map((c) => `<option value="${esc(c)}"${c === current ? " selected" : ""}>${esc(c)}</option>`));
    sel.innerHTML = opts.join("");
    sel.value = cats.includes(current) ? current : "";
    if (sel.selectedIndex < 0 && sel.options.length > 0) sel.selectedIndex = 0;
    this.state.paretoCategory = sel.value;
    sel.disabled = cats.length <= 1;
    if (typeof sel._refreshSkin === "function") sel._refreshSkin();
  }

  syncParetoSliders(dom, data) {
    const scopeKey = `${dom}::${(data && data.selected_category) || "*"}`;
    if (!data || this.state.paretoSliderDomain === scopeKey) return false;
    this.state.paretoSliderDomain = scopeKey;
    const cur = data.currency !== undefined ? data.currency : this.activeCurrency(dom);
    if (Object.prototype.hasOwnProperty.call(data, "currency")) {
      this.state.domainCurrencies[dom] = data.currency || null;
    }
    let changed = false;
    const pr = data.price_range;
    if (pr && isFinite(pr.min) && isFinite(pr.max)) {
      const minPrice = Math.max(0, Math.floor(pr.min));
      let hi = Math.ceil(pr.max);
      if (hi <= minPrice) hi = minPrice + 10;
      const rawRange = Math.max(10, hi - Math.floor(minPrice * 0.25));
      const step = rawRange <= 50 ? 1 : rawRange <= 500 ? 10 : rawRange <= 5000 ? 50 : rawRange <= 30000 ? 100 : 500;
      const lo = minPrice <= 100 ? 0 : Math.max(0, Math.floor((minPrice * 0.25) / step) * step);
      const span = hi - minPrice;
      const mid = Math.min(hi, Math.max(minPrice, Math.round((minPrice + span * 0.6) / step) * step));
      const curVal = Number(this.dom.budgetSlider.value);
      this.dom.budgetSlider.min = String(lo);
      this.dom.budgetSlider.max = String(hi);
      this.dom.budgetSlider.step = String(step);
      if (!isFinite(curVal) || curVal < lo || curVal > hi || this.state.domain !== dom || Math.abs(curVal - mid) > Math.max(span, 1) * 0.2) {
        this.dom.budgetSlider.value = String(mid);
        changed = Math.abs(curVal - mid) > Math.max(1, step * 0.5);
      }
      this.dom.budgetSlider.disabled = false;
      this.dom.budgetValue.textContent = this.formatPrice(Number(this.dom.budgetSlider.value), cur);
    } else {
      this.dom.budgetSlider.disabled = true;
      this.dom.budgetValue.textContent = "—";
    }
    const sr = data.performance_range;
    if (sr && isFinite(sr.min) && isFinite(sr.max)) {
      const sLo = Math.floor(sr.min);
      const sHi = Math.max(sLo, Math.ceil(sr.max));
      const sStep = sHi - sLo <= 2 && sHi > sLo ? 0.5 : 1;
      const sVal = Math.max(sLo, Math.min(sHi, sHi > sLo ? Math.round(sLo + (sHi - sLo) * 0.5) : sLo));
      const oldP = Number(this.dom.perfSlider.value);
      this.dom.perfSlider.min = String(sLo);
      this.dom.perfSlider.max = String(Math.max(sLo + 1, sHi));
      this.dom.perfSlider.step = String(sStep);
      this.dom.perfSlider.value = String(sVal);
      this.dom.perfSlider.disabled = sHi === sLo;
      this.dom.perfValue.textContent = String(sVal);
      if (Math.abs(oldP - sVal) > 0) changed = true;
    } else {
      this.dom.perfSlider.disabled = true;
      this.dom.perfValue.textContent = "—";
    }
    return changed;
  }

  async fetchPareto() {
    const dom = this.state.domain;
    try {
      const data = await this.api("/api/pareto", {
        method: "POST",
        body: {
          budget_max: Number(this.dom.budgetSlider.value),
          min_performance: Number(this.dom.perfSlider.value),
          domain: dom,
          category: this.state.paretoCategory || null,
        },
      });
      if (dom !== this.state.domain) return;
      this.syncParetoCategories(data);
      if (this.syncParetoSliders(dom, data)) {
        return await this.fetchPareto();
      }
      this.state.lastPareto = data;
      let extraNote = "";
      if (data.total_source_items && (data.all_candidates || []).length > 0 && !data.selected_category && data.total_source_items > (data.all_candidates.length + (data.excluded_without_price || 0) + (data.excluded_without_performance || 0))) {
        extraNote = `\nПримітка: оцінено початковий зріз (${data.all_candidates.length} з ${data.total_source_items} позицій); для фокусного аналізу оберіть категорію вище.`;
      }
      this.dom.paretoExplanation.textContent = (data.explanation || "—") + extraNote;
      this.drawPareto(data);
      this.renderParetoFront(data);
      const selected = this.state.paretoSelectedId
        ? (data.all_candidates || []).find((x) => String(x.item_id) === String(this.state.paretoSelectedId))
        : null;
      this.selectParetoPoint(selected || null, false);
      this.setConnection(true);
    } catch (err) {
      this.dom.paretoExplanation.textContent =
        "Не вдалося обчислити фронт: " + (err.message || String(err));
      this.setConnection(false);
    }
  }

  paretoScopeFront(data) {
    return data.pareto_front_feasible || [];
  }

  renderParetoFront(data) {
    const el = this.dom.paretoFront;
    const front = [...this.paretoScopeFront(data)].sort((a, b) => (Number(a.price) || 0) - (Number(b.price) || 0));
    if (!front.length) {
      el.innerHTML = data.front_scope === "no_feasible_item"
        ? '<div class="pf-empty">Жоден товар каталогу не задовольняє обидві вимоги одночасно. Пунктирна лінія на графіку — довідковий фронт усього каталогу.</div>'
        : "";
      return;
    }
    const esc = (s) => this.escapeHtml(String(s));
    const budget = Number(data.budget_max);
    const minPerf = Number(data.min_performance);
    const variantOf = (c) => (data.best_a && data.best_a.item_id === c.item_id ? "A"
      : data.best_b && data.best_b.item_id === c.item_id ? "B" : "");
    const cur = data.currency !== undefined ? data.currency : this.activeCurrency();
    const rows = front.map((c, i) => {
      const fits = Number(c.price) <= budget && Number(c.performance_score) >= minPerf;
      const selCls = String(this.state.paretoSelectedId) === String(c.item_id) ? " is-selected" : "";
      return `<tr class="${fits ? "pf-fit" : "pf-miss"}${selCls}" data-item-id="${esc(c.item_id)}"><td class="mono">${i + 1}</td><td>${esc(c.name)}</td>` +
        `<td class="mono">${esc(this.formatPrice(c.price, cur))}</td><td class="mono">${esc(c.performance_score)}</td>` +
        `<td class="mono">${fits ? "у межах" : "поза межами"}</td><td class="pf-tag">${variantOf(c)}</td></tr>`;
    }).join("");
    el.innerHTML = '<table class="pf-table"><thead><tr><th>№</th><th>Товар</th><th>Ціна</th>' +
      '<th>Бал</th><th>Статус</th><th>Вар.</th></tr></thead><tbody>' + rows + "</tbody></table>";
  }

  drawPareto(data) {
    const canvas = this.dom.paretoCanvas;
    const rect = canvas.parentElement.getBoundingClientRect();
    if (rect.width < 20 || rect.height < 20) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = rect.width;
    const H = rect.height;

    ctx.clearRect(0, 0, W, H);
    if (!data) return;
    const all = data.all_candidates || [];
    const front = this.paretoScopeFront(data);
    this.state.paretoPlotPoints = [];
    if (!all.length) {
      ctx.fillStyle = "#7d8a95";
      ctx.font = "12px 'Segoe UI', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Немає кандидатів для цих обмежень", W / 2, H / 2);
      return;
    }

    const padL = 56, padR = 16, padT = 16, padB = 38;
    const prices = all.map((c) => Number(c.price) || 0);
    const perfs = all.map((c) => Number(c.performance_score) || 0);
    const pMin = Math.min(...prices);
    const pMax = Math.max(...prices);
    const sMin = Math.min(...perfs);
    const sMax = Math.max(...perfs);
    const flatPerf = Math.abs(sMax - sMin) < 1e-6;
    const pSpan = Math.max(pMax - pMin, 1e-6);
    const sSpan = flatPerf ? 1 : (sMax - sMin);
    const midY = (padT + H - padB) / 2;
    const toX = (p) => padL + ((p - pMin) / pSpan) * (W - padL - padR);
    const toY = (s) => flatPerf ? midY : H - padB - ((s - sMin) / sSpan) * (H - padT - padB);
    const fmtP = (v) => v >= 1000 ? (v / 1000).toFixed(v % 1000 ? 1 : 0) + "k" : Number.isInteger(v) ? String(v) : v.toFixed(1);

    ctx.font = "9px Consolas, monospace";
    ctx.strokeStyle = "rgba(125,138,149,0.18)";
    ctx.fillStyle = "#56636d";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const y = padT + ((H - padT - padB) * g) / 4;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
      ctx.textAlign = "right";
      if (!flatPerf) {
        ctx.fillText((sMax - (sSpan * g) / 4).toFixed(0), padL - 7, y + 3);
      } else if (g === 2) {
        ctx.fillText(sMin.toFixed(0), padL - 7, y + 3);
      }
      const x = padL + ((W - padL - padR) * g) / 4;
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, H - padB); ctx.stroke();
      ctx.textAlign = "center";
      ctx.fillText(fmtP(pMin + (pSpan * g) / 4), x, H - padB + 12);
    }
    ctx.strokeStyle = "#2b3a46";
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
    ctx.stroke();
    ctx.fillStyle = "#56636d"; ctx.textAlign = "center";
    const axisCur = data.currency !== undefined ? data.currency : this.activeCurrency();
    const axisUnit = this.formatCurrencyUnit(axisCur, "symbol");
    ctx.fillText(axisUnit ? `ціна, ${axisUnit} →` : "ціна →", (padL + W - padR) / 2, H - 8);
    ctx.save();
    ctx.translate(11, (padT + H - padB) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("продуктивність →", 0, 0); ctx.restore();

    const budget = Number(this.dom.budgetSlider.value) || 0;
    const minPerf = Number(this.dom.perfSlider.value) || 0;
    ctx.setLineDash([4, 4]);
    if (budget >= pMin && budget <= pMax) {
      const bx = toX(budget);
      ctx.strokeStyle = "rgba(63,193,212,.55)";
      ctx.beginPath(); ctx.moveTo(bx, padT); ctx.lineTo(bx, H - padB); ctx.stroke();
      ctx.fillStyle = "#3fc1d4"; ctx.textAlign = "left";
      ctx.fillText("бюджет", Math.min(bx + 4, W - padR - 42), padT + 9);
    }
    if (!flatPerf && minPerf >= sMin && minPerf <= sMax) {
      const py = toY(minPerf);
      ctx.strokeStyle = "rgba(232,161,60,.5)";
      ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(W - padR, py); ctx.stroke();
      ctx.fillStyle = "#e8a13c"; ctx.textAlign = "left";
      ctx.fillText("мін. продуктивність", padL + 4, py - 4);
    } else if (flatPerf) {
      ctx.fillStyle = "#7d8a95"; ctx.textAlign = "left";
      ctx.fillText(`шкала продуктивності в джерелі не диференційована (${sMin.toFixed(0)})`, padL + 6, midY - 12);
    }
    ctx.setLineDash([]);

    const frontIds = new Set(front.map((pt) => String(pt.item_id)));
    const refIds = new Set((data.pareto_front || []).map((pt) => String(pt.item_id)));
    all.forEach((c) => {
      const px = toX(Number(c.price) || 0);
      const py = toY(Number(c.performance_score) || 0);
      const pid = String(c.item_id);
      this.state.paretoPlotPoints.push({ x: px, y: py, c });
      if (!frontIds.has(pid) && !refIds.has(pid)) {
        ctx.fillStyle = "rgba(125,138,149,.52)";
        ctx.beginPath();
        ctx.arc(px, py, 3.0, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    const reference = [...(data.pareto_front || [])].sort((a, b) => (Number(a.price) || 0) - (Number(b.price) || 0));
    if (reference.length > 1) {
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = "rgba(125,138,149,0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      reference.forEach((pt, i) => {
        const x = toX(Number(pt.price) || 0), y = toY(Number(pt.performance_score) || 0);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }
    const sorted = [...front].sort((a, b) => (Number(a.price) || 0) - (Number(b.price) || 0));
    if (sorted.length > 1) {
      ctx.beginPath();
      sorted.forEach((pt, i) => {
        const x = toX(Number(pt.price) || 0), y = toY(Number(pt.performance_score) || 0);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.save();
      ctx.shadowColor = "rgba(61,220,132,.5)"; ctx.shadowBlur = 8;
      ctx.strokeStyle = "#3ddc84"; ctx.lineWidth = 1.8; ctx.stroke();
      ctx.restore();
      ctx.lineTo(toX(sorted[sorted.length - 1].price), H - padB);
      ctx.lineTo(toX(sorted[0].price), H - padB);
      ctx.closePath();
      ctx.fillStyle = "rgba(61,220,132,.07)"; ctx.fill();
    }

    sorted.forEach((pt) => {
      const x = toX(Number(pt.price) || 0), y = toY(Number(pt.performance_score) || 0);
      ctx.fillStyle = "#0b0e12";
      ctx.beginPath(); ctx.arc(x, y, 4.6, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#3ddc84"; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.arc(x, y, 4.6, 0, Math.PI * 2); ctx.stroke();
    });

    const marker = (c, color, tag) => {
      if (!c) return;
      const x = toX(Number(c.price) || 0), y = toY(Number(c.performance_score) || 0);
      ctx.save();
      ctx.shadowColor = color; ctx.shadowBlur = 10;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(x, y, 5.2, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
      ctx.font = "bold 9px Consolas, monospace"; ctx.textAlign = "left";
      ctx.fillStyle = color;
      ctx.fillText(tag, x + 8, y - 6);
    };
    marker(data.best_a, "#3fc1d4", "A");
    marker(data.best_b, "#e8a13c", "B");

    const drawRing = (targetId, color, radius) => {
      if (!targetId) return;
      const pt = this.state.paretoPlotPoints.find((p) => String(p.c.item_id) === String(targetId));
      if (!pt) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    };
    drawRing(this.state.paretoHoverId, "rgba(255,255,255,0.75)", 7.5);
    drawRing(this.state.paretoSelectedId, "#ffffff", 9.0);
  }

  _paretoPointAt(e) {
    const canvas = this.dom.paretoCanvas;
    const pts = this.state.paretoPlotPoints || [];
    if (!canvas || !pts.length) return null;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best = null;
    let bestD2 = 14 * 14;
    for (const pt of pts) {
      const dx = pt.x - mx;
      const dy = pt.y - my;
      const d2 = dx * dx + dy * dy;
      if (d2 <= bestD2) {
        bestD2 = d2;
        best = pt;
      }
    }
    return best;
  }

  _handleParetoHover(e) {
    const hit = this._paretoPointAt(e);
    const tip = this.dom.paretoTooltip;
    const data = this.state.lastPareto;
    if (!hit || !tip || !data) {
      if (tip) tip.hidden = true;
      if (this.state.paretoHoverId !== null) {
        this.state.paretoHoverId = null;
        if (data) this.drawPareto(data);
      }
      return;
    }
    const c = hit.c;
    const pid = String(c.item_id);
    if (this.state.paretoHoverId !== pid) {
      this.state.paretoHoverId = pid;
      this.drawPareto(data);
    }
    const esc = (s) => this.escapeHtml(String(s));
    const cur = data.currency !== undefined ? data.currency : this.activeCurrency();
    tip.innerHTML =
      `<strong>${esc(c.name)}</strong>` +
      (c.category ? `<div class="muted">${esc(c.category)}</div>` : "") +
      `<div class="mono">${esc(this.formatPrice(c.price, cur))} · продукт. ${esc(c.performance_score)}</div>` +
      `<div class="muted">Клікніть, щоб подивитися статус</div>`;
    const wrap = this.dom.paretoCanvas.parentElement.getBoundingClientRect();
    const left = Math.min(Math.max(8, hit.x + 12), Math.max(8, wrap.width - 268));
    const top = Math.max(8, hit.y - 48);
    tip.style.left = `${Math.round(left)}px`;
    tip.style.top = `${Math.round(top)}px`;
    tip.hidden = false;
  }

  paretoDominator(cand, all) {
    const p = Number(cand.price);
    const s = Number(cand.performance_score);
    let dom = null;
    for (const o of all || []) {
      if (String(o.item_id) === String(cand.item_id)) continue;
      const op = Number(o.price);
      const os = Number(o.performance_score);
      if (op <= p && os >= s && (op < p || os > s)) {
        if (!dom || os > Number(dom.performance_score) || (os === Number(dom.performance_score) && op < Number(dom.price))) {
          dom = o;
        }
      }
    }
    return dom;
  }

  selectParetoPoint(cand, redraw = true) {
    const card = this.dom.paretoPointCard;
    if (!card) return;
    if (!cand) {
      this.state.paretoSelectedId = null;
      card.innerHTML = '<span class="muted">Наведіть або натисніть на будь-яку точку графіка (зелену чи сіру), щоб побачити товар і чому він у такому статусі.</span>';
      return;
    }
    const data = this.state.lastPareto;
    if (!data) return;
    this.state.paretoSelectedId = String(cand.item_id);
    if (redraw) {
      this.drawPareto(data);
      this.renderParetoFront(data);
    }
    const esc = (s) => this.escapeHtml(String(s));
    const cur = data.currency !== undefined ? data.currency : this.activeCurrency();
    const budget = Number(data.budget_max);
    const minPerf = Number(data.min_performance);
    const p = Number(cand.price);
    const s = Number(cand.performance_score);
    const onRefFront = (data.pareto_front || []).some((x) => String(x.item_id) === String(cand.item_id));
    const dom = onRefFront ? null : this.paretoDominator(cand, data.all_candidates);

    let reason = "На глобальному Парето-фронті каталогу.";
    if (data.best_a && String(data.best_a.item_id) === String(cand.item_id)) {
      reason = "Варіант A: найкраща продуктивність серед допустимих точок фронту.";
    } else if (data.best_b && String(data.best_b.item_id) === String(cand.item_id)) {
      reason = "Варіант B: найдешевший баланс на допустимому Парето-фронті.";
    } else if (dom) {
      reason = `Домінований: «${dom.name}» дешевший/не дорожчий (${this.formatPrice(dom.price, cur)}) і не гірший за продуктивністю (${dom.performance_score}).`;
    } else if (p > budget || s < minPerf) {
      const why = [];
      if (p > budget) why.push(`дорожчий за бюджет (${this.formatPrice(budget, cur)})`);
      if (s < minPerf) why.push(`нижче мін. продуктивності (${minPerf})`);
      reason = `На глобальному фронті, але поза поточними повзунками: ${why.join(" і ")}.`;
    }

    card.innerHTML =
      `<div>` +
      `<strong>${esc(cand.name)}</strong>` +
      (cand.category ? ` <span class="muted">· ${esc(cand.category)}</span>` : "") +
      `<div class="ppc-meta">${esc(this.formatPrice(cand.price, cur))} · продукт. ${esc(cand.performance_score)}</div>` +
      `<div class="ppc-reason">${esc(reason)}</div>` +
      `</div>` +
      `<button type="button" class="btn btn-sm" data-add-bundle="${esc(cand.item_id)}">+ У сумісність</button>`;
  }


  /* ------------------------------------------------------------------ */
  /* Telemetry                                                           */
  /* ------------------------------------------------------------------ */
  async fetchTelemetry() {
    try {
      const data = await this.api("/api/telemetry");
      const inv = data.active_invariants || {};
      this.dom.invariantsList.innerHTML = Object.entries(inv)
        .map(([key, v]) =>
          '<div class="invariant-item">' +
          '<span class="inv-name">' + this.escapeHtml(key) + "</span>" +
          '<span class="inv-desc">' + this.escapeHtml(v.desc || "") + "</span>" +
          '<span class="inv-badge">' + this.escapeHtml(v.status || "?") + "</span>" +
          "</div>"
        )
        .join("");
      if (!Object.keys(inv).length) {
        this.dom.invariantsList.innerHTML = '<div class="list-note">Немає даних телеметрії.</div>';
      }
      this.setConnection(true);
    } catch (err) {
      this.dom.invariantsList.innerHTML =
        '<div class="list-note is-error">' +
        this.escapeHtml(err.message || String(err)) +
        "</div>";
      this.setConnection(false);
    }
  }

  updateAfterAnswer(data) {
    this.updateStatusBar(data);
    this.updateKpis(data);
    try {
      this.dom.rawJson.textContent = JSON.stringify(data, null, 2);
    } catch (_) {
      this.dom.rawJson.textContent = String(data);
    }
  }

  updateCostOnly() {
    this.dom.sbCost.textContent = "$" + this.state.sessionCost.toFixed(4);
    this.dom.kpiCost.textContent = "$" + this.state.sessionCost.toFixed(4);
  }

  updateStatusBar(data) {
    const gate = this.gateBadge(data);
    const tier = data.tier ? (TIER_LABELS[data.tier] || data.tier) : "—";
    const latencyMs = Math.round(Number(data.latency_seconds || 0) * 1000);
    const latencyText = latencyMs < 1000 ? latencyMs + " мс" : (latencyMs / 1000).toFixed(2) + " с";

    this.dom.sbTier.innerHTML = "<svg class='ic' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round' aria-hidden='true' style='vertical-align:-1px'><path d='M13 2 4.5 13.5H11L10 22l8.5-11.5H12z'/></svg> " + tier;
    this.dom.sbLatency.innerHTML = "<svg class='ic' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round' aria-hidden='true' style='vertical-align:-1px'><circle cx='12' cy='12' r='9'/><path d='M12 7v5l3.5 2'/></svg> " + latencyText;
    this.dom.sbModel.innerHTML = "<svg class='ic' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round' aria-hidden='true' style='vertical-align:-1px'><rect x='2.5' y='4' width='19' height='13' rx='1.5'/><path d='M9 21h6M12 17v4'/></svg> " + (data.model || "—");
    this.dom.sbGate.innerHTML = "<svg class='ic' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round' aria-hidden='true' style='vertical-align:-1px'><path d='M12 2.5 4.5 5.5v6c0 5 3.2 8.4 7.5 10 4.3-1.6 7.5-5 7.5-10v-6z'/><path d='m8.8 12 2.2 2.2 4.2-4.5'/></svg> gate: " + gate.short;
    this.dom.sbGate.className = "sb-item" +
      (gate.short === "ВІДМОВА" ? " sb-gate-err" :
        (gate.short === "заземлено" || gate.short.indexOf("виправлено") === 0 ||
         gate.short === "перебудовано" ? " sb-gate-ok" : ""));
    this.dom.sbEngine.textContent = "engine: " + gate.engine;
    this.updateCostOnly();
  }

  updateKpis(data) {
    const latencyMs = Math.round(Number(data.latency_seconds || 0) * 1000);
    this.dom.kpiLatency.textContent =
      latencyMs < 1000 ? latencyMs + " мс" : (latencyMs / 1000).toFixed(3) + " с";
    this.dom.kpiTier.textContent = data.tier
      ? (TIER_LABELS[data.tier] || data.tier)
      : "детермінований граф";
    this.dom.kpiSteps.textContent = String(data.steps ? data.steps.length : 0);
    this.dom.kpiCost.textContent = "$" + this.state.sessionCost.toFixed(4);
    this.dom.kpiModel.textContent = data.model || "— (без виклику моделі)";
    const gate = this.gateBadge(data);
    this.dom.kpiGate.textContent = "gate: " + gate.short;
    this.dom.kpiDomain.textContent = this.domainName(this.state.domain);
    this.dom.kpiScale.textContent = this.scaleLabel();
  }


  /* ------------------------------------------------------------------ */
  /* Domain switch / tabs                                                */
  /* ------------------------------------------------------------------ */
  async switchDomain(domain) {
    const seq = (this._switchSeq = (this._switchSeq || 0) + 1);
    try {
      const switched = await this.api("/api/switch_domain", {
        method: "POST",
        body: { domain, session_id: this.state.sessionId },
      });
      if (seq !== this._switchSeq) return;
      this.adoptSessionId(switched.session_id);
      this.state.domain = domain;
      this.state.domainLocked = true;
      if (this.dom.domainSelect.value !== domain) {
        this.dom.domainSelect.value = domain;
      }
      if (typeof this.dom.domainSelect._refreshSkin === "function") {
        this.dom.domainSelect._refreshSkin();
      }
      this.state.bundle = [];
      this.state.catalog = [];
      this.state.catalogTotal = null;
      this.state.lastPareto = null;
      this.state.paretoSliderDomain = null;
      this.state.paretoCategory = "";
      this.state.paretoSelectedId = null;
      this.state.paretoHoverId = null;
      this.state.paretoPlotPoints = [];
      if (this.dom.paretoCategory) {
        this.dom.paretoCategory.innerHTML = '<option value="" selected>Усі категорії</option>';
        this.dom.paretoCategory.selectedIndex = 0;
        this.dom.paretoCategory.disabled = true;
        if (typeof this.dom.paretoCategory._refreshSkin === "function") {
          this.dom.paretoCategory._refreshSkin();
        }
      }
      if (this.dom.paretoFront) this.dom.paretoFront.innerHTML = "";
      if (this.dom.paretoExplanation) this.dom.paretoExplanation.textContent = "Обчислення Парето-фронту…";
      this.selectParetoPoint(null, false);
      this.drawPareto(null);
      if (this.dom.paretoTooltip) this.dom.paretoTooltip.hidden = true;
      if (this.dom.taxonomySummary) {
        this.dom.taxonomySummary.textContent = "Таксономія: " + this.domainName(domain) + "…";
      }
      this.renderScenarioChips();
      this.renderBundleChips();
      this.dom.kpiDomain.textContent = this.domainName(domain);
      this.dom.kpiScale.textContent = this.scaleLabel();
      this.dom.sbDomain.textContent = "⧉ " + domain;
      this.appendSystemMessage(
        "Активний каталог перемкнуто: <strong>" +
          this.escapeHtml(this.domainName(domain)) +
          "</strong>."
      );
      this.setConnection(true);
      const jobs = [
        this.refreshStatus(),
        this.fetchCatalog(this.dom.catalogSearch.value.trim()),
        this.validateBundle(),
        this.fetchPareto(),
        this.fetchTelemetry(),
        this.fetchTaxonomy(),
      ];
      // Inspection is expensive (rebuilds the source projection); refresh it
      // only while its tab is actually on screen.
      if (document.getElementById("panel-inspection").classList.contains("active")) {
        jobs.push(this.fetchInspection());
      }
      if (document.getElementById("panel-semantic").classList.contains("active")) {
        jobs.push(this.fetchSemanticStatus());
      }
      await Promise.all(jobs);
    } catch (err) {
      this.appendErrorMessage("Не вдалося перемкнути каталог: " + (err.message || String(err)));
      this.setConnection(false);
      this.dom.domainSelect.value = this.state.domain;
    }
  }

  activateTab(name) {
    this.dom.tabs.forEach((t) => {
      t.classList.toggle("active", t.getAttribute("data-tab") === name);
    });
    document.querySelectorAll(".tab-panel").forEach((p) => {
      p.classList.toggle("active", p.id === "panel-" + name);
    });
    document.body.classList.toggle("insp-active", name === "inspection");
    if (name === "pareto") {
      if (this.state.lastPareto) this.drawPareto(this.state.lastPareto);
      else this.fetchPareto();
    } else if (name === "telemetry") {
      this.fetchTelemetry();
    } else if (name === "taxonomy") {
      this.fetchTaxonomy();
    } else if (name === "inspection") {
      this.fetchInspection();
    } else if (name === "semantic") {
      this.fetchSemanticStatus();
    }
  }

  /* ------------------------------------------------------------------ */
  /* OntoGraph Taxonomy Tree                                             */
  /* ------------------------------------------------------------------ */
  async fetchTaxonomy() {
    if (!this.dom.ontographTree) return;
    const dom = this.state.domain;
    this.dom.ontographTree.innerHTML = '<div class="list-note">Побудова динамічного графу таксономії…</div>';
    try {
      const data = await this.api("/api/taxonomy?domain=" + encodeURIComponent(dom));
      if (dom !== this.state.domain) return;
      if (!data || !data.tree) {
        this.dom.ontographTree.innerHTML = '<div class="list-note">Таксономія недоступна для цього домену.</div>';
        return;
      }
      if (this.dom.taxonomySummary) {
        this.dom.taxonomySummary.textContent = `Таксономія: ${this.domainName(data.domain)} (${data.tree.item_count} товарів)`;
      }
      this.dom.ontographTree.innerHTML = this.renderOntoNode(data.tree, true);
    } catch (err) {
      this.dom.ontographTree.innerHTML =
        '<div class="list-note is-error">Помилка побудови OntoGraph: ' +
        this.escapeHtml(err.message || String(err)) +
        '</div>';
    }
  }

  renderScenarioChips() {
    if (!this.dom.scenarioChips) return;
    // F53: chips are generated from the categories the active catalogue
    // actually carries -- the page asserts no product names, brands or
    // fixture vocabulary of its own. Before the first catalog load only
    // catalogue-generic prompts exist.
    const catList = this.state.catalog || [];
    const posPrices = catList
      .filter((p) => p && p.price !== null && p.price !== undefined && p.price !== "")
      .map((p) => Number(p.price))
      .filter((n) => Number.isFinite(n) && n > 0)
      .sort((a, b) => a - b);
    const hasPrices = posPrices.length > 0;
    const unitWord = this.formatCurrencyUnit(this.activeCurrency(), "word");
    const unitTail = unitWord ? " " + unitWord : "";
    const budgetHint = hasPrices
      ? Math.max(1, Math.round(posPrices[Math.floor(posPrices.length * 0.4)]))
      : null;

    const list = [
      { label: "Огляд каталогу", query: "Які категорії товарів є в цьому каталозі?" },
    ];
    if (hasPrices) {
      list.push(
        { label: "Найдешевший товар", query: "Який найдешевший товар є в наявності?" },
        { label: "Підбір за бюджетом", query: `Порадь товар у межах бюджету ${budgetHint}${unitTail}` },
      );
    } else {
      list.push(
        { label: "Ключові атрибути", query: "Які характеристики та атрибути є у позицій цього каталогу?" },
      );
    }
    const seen = Object.create(null);
    for (const p of catList) {
      const cat = p && p.category ? String(p.category) : "";
      if (cat && !seen[cat]) {
        seen[cat] = true;
        list.push({
          label: "" + cat,
          query: "Що є в наявності у категорії «" + cat + "»?",
        });
      }
      if (Object.keys(seen).length >= 3) break;
    }

    // Demo-ready verification & relaxation triggers derived strictly from loaded catalog data.
    const firstItem = catList.find((p) => p && p.name);
    const verifyQuery = firstItem
      ? "Назви " + (hasPrices ? "точну ціну та всі " : "всі ") + "числові характеристики для «" + firstItem.name + "» без округлень."
      : "Порівняй доступні позиції каталогу з точними числовими характеристиками.";
    const tightBudget = hasPrices ? Math.max(1, Math.floor(posPrices[0] * 0.4)) : null;
    const relaxQuery = hasPrices
      ? `Підбери найкращий варіант суворо до ${tightBudget}${unitTail} — бюджет не порушувати, інші параметри можна послабити.`
      : "Знайди найбільш релевантні позиції за сукупністю доступних характеристик без вигаданих чисел.";

    let html = '<span class="scenario-label">Сценарії:</span>';
    for (const sc of list) {
      html += `<button class="chip-btn" type="button" data-query="${this.escapeHtml(sc.query)}">${this.escapeHtml(sc.label)}</button>`;
    }
    html += `<button class="chip-btn chip-demo" type="button" data-query="${this.escapeHtml(verifyQuery)}" title="${hasPrices ? "Перевірка spec+ціни по рядку джерела" : "Перевірка атрибутів по рядку джерела"}">✂ ${hasPrices ? "Перевірка spec+ціни" : "Перевірка фактів"}</button>`;
    html += `<button class="chip-btn chip-demo" type="button" data-query="${this.escapeHtml(relaxQuery)}" title="${hasPrices ? "Демонстрація жорсткого бюджету, релаксації та regret" : "Підбір без вигаданих числових порогів"}">⚖ ${hasPrices ? "Релаксація + regret" : "Підбір за атрибутами"}</button>`;
    html += `<button class="chip-btn chip-demo" type="button" data-demo-action="onboard" title="Підключити нове зовнішнє джерело даних та запустити live-індукцію схеми">⧉ Нове джерело</button>`;
    this.dom.scenarioChips.innerHTML = html;
  }

  /**
   * Demo helper: open Schema Inspection & onboarding panel without hardcoding
   * any fixture domain identifier or bypassing admin authentication (F29/F53).
   */
  runDemoAction(action) {
    if (action === "onboard") {
      this.activateTab("inspection");
      if (this.dom.regStatus && !this.dom.regStatus.textContent.trim()) {
        this.dom.regStatus.textContent =
          "Оберіть SQLite-файл (напр. data/*.db) або завантажте CSV/JSONL для live-індукції схеми.";
      }
      if (this.dom.regDbPath) this.dom.regDbPath.focus();
    }
  }

  setOntoExpanded(expand) {
    if (!this.dom.ontographTree) return;
    const nodes = this.dom.ontographTree.querySelectorAll(".onto-node");
    nodes.forEach((nodeEl) => {
      const isRoot = nodeEl.classList.contains("onto-level-0");
      const childrenEl = nodeEl.querySelector(":scope > .onto-children");
      const toggleEl = nodeEl.querySelector(":scope > .onto-header .onto-toggle");
      if (!childrenEl) return;
      const shouldCollapse = !expand && !isRoot;
      childrenEl.classList.toggle("collapsed", shouldCollapse);
      if (toggleEl) toggleEl.textContent = shouldCollapse ? "▶" : "▼";
    });
  }

  renderOntoNode(node, isRoot = false) {
    if (!node) return "";
    const hasChildren = Array.isArray(node.children) && node.children.length > 0;
    const startCollapsed = !isRoot && hasChildren;
    const levelClass = "onto-level-" + (node.level != null ? node.level : 0);
    const countLabel = (node.item_count || 0) + " шт";

    let priceLabel = "";
    if (node.price_min != null && node.price_max != null && (node.price_min > 0 || node.price_max > 0)) {
      priceLabel = `<span class="onto-price">${this.formatPrice(node.price_min)} – ${this.formatPrice(node.price_max)}</span>`;
    }

    let paramsLabel = "";
    if (Array.isArray(node.parameters) && node.parameters.length > 0) {
      const tags = node.parameters
        .slice(0, 4)
        .map((p) => `<span class="onto-param-tag">${this.escapeHtml(p)}</span>`)
        .join("");
      if (tags) {
        paramsLabel = `<div class="onto-params">${tags}</div>`;
      }
    }

    const title = isRoot
      ? `<svg class=\"ic\" width=\"12\" height=\"12\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\" style=\"vertical-align:-1px\"><path d=\"M12 2 3 7v10l9 5 9-5V7z\"/><path d=\"M3 7l9 5 9-5M12 12v10\"/></svg> Увесь каталог (${this.domainName(this.state.domain)})`
      : this.escapeHtml(node.name || node.id || "Розділ");
    const toggleIcon = hasChildren ? (startCollapsed ? "▶" : "▼") : "•";

    let childrenHtml = "";
    if (hasChildren) {
      childrenHtml = `<div class="onto-children${startCollapsed ? " collapsed" : ""}">${node.children.map((c) => this.renderOntoNode(c, false)).join("")}</div>`;
    }

    const actionBtn = !isRoot
      ? `<button type="button" class="btn-node-filter" data-filter-name="${this.escapeHtml(node.name)}" title="Фільтрувати каталог">Показати</button>`
      : "";

    return `
      <div class="onto-node ${levelClass}">
        <div class="onto-header" data-node-id="${this.escapeHtml(node.id || "")}">
          <span class="onto-toggle">${toggleIcon}</span>
          <span class="onto-title">${title}</span>
          <span class="onto-count">${countLabel}</span>
          ${priceLabel}
          ${actionBtn}
        </div>
        ${paramsLabel}
        ${childrenHtml}
      </div>
    `.trim();
  }

  /* ------------------------------------------------------------------ */
  /* Source Introspection: як система прочитала БД                        */
  /* ------------------------------------------------------------------ */
  async fetchInspection() {
    if (!this.dom.inspectionBody) return;
    const dom = this.state.domain;
    this.dom.inspectionBody.innerHTML =
      '<div class="list-note">Профілювання джерела та збір інтроспекції…</div>';
    try {
      const d = await this.api(
        "/api/domains/" + encodeURIComponent(dom) + "/inspection"
      );
      if (dom !== this.state.domain) return;
      if (this.dom.inspectionSummary) {
        this.dom.inspectionSummary.textContent =
          `${d.name} (${d.state}) · entity: ${d.projection && d.projection.entity_table ? d.projection.entity_table : "?"} · ` +
          `вузлів графа: ${d.hypergraph ? d.hypergraph.seeded_nodes : 0}`;
      }
      this.dom.inspectionBody.innerHTML = this.renderInspection(d);
      this._initBuildMap();
    } catch (err) {
      this.dom.inspectionBody.innerHTML =
        '<div class="list-note is-error">Помилка інтроспекції: ' +
        this.escapeHtml(err.message || String(err)) + "</div>";
    }
  }

  askRuleToken(retryBtn) {
    const msgEl = document.getElementById("insp-rule-msg");
    if (!msgEl) return;
    msgEl.innerHTML =
      "Для зміни статусу правила потрібен admin-токен: " +
      '<input id="insp-rule-tok" type="password" class="input sem-tok" placeholder="X-Admin-Token"> ' +
      '<button id="insp-rule-tok-go" class="btn-mini">OK</button>';
    const go = () => {
      const el = document.getElementById("insp-rule-tok");
      const v = el && el.value.trim();
      if (!v) return;
      localStorage.setItem("pb_admin_token", v);
      if (this.dom.regAdminToken) this.dom.regAdminToken.value = v;
      if (retryBtn) this.toggleConstraintStatus(retryBtn);
    };
    const btnGo = document.getElementById("insp-rule-tok-go");
    const inp = document.getElementById("insp-rule-tok");
    if (btnGo) btnGo.addEventListener("click", go);
    if (inp) inp.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  }

  async toggleConstraintStatus(btn) {
    const ruleId = btn.getAttribute("data-rule-id");
    const nextStatus = btn.getAttribute("data-next-status");
    if (!ruleId || !nextStatus) return;
    const msgEl = document.getElementById("insp-rule-msg");
    btn.disabled = true;
    try {
      await this.api(
        "/api/domains/" + encodeURIComponent(this.state.domain) + "/constraints/status",
        {
          method: "POST",
          headers: { "X-Admin-Token": this.adminToken() },
          body: { rule_id: ruleId, status: nextStatus },
        },
      );
      await this.fetchInspection();
    } catch (err) {
      btn.disabled = false;
      const msg = String((err && err.message) || err);
      if (msg.startsWith("HTTP 403") || msg.startsWith("HTTP 401")) {
        this.askRuleToken(btn);
      } else if (msgEl) {
        msgEl.textContent = "Помилка оновлення правила: " + msg;
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Semantic index panel: lifecycle status, sync trigger, semantic     */
  /* search. Every field comes from /api/domains/{id}/semantic verbatim  */
  /* -- the panel adds no interpretation of its own.                    */
  /* ------------------------------------------------------------------ */
  async fetchSemanticStatus() {
    if (!this.dom.semStatusBody) return;
    const dom = this.state.domain;
    try {
      const d = await this.api(
        "/api/domains/" + encodeURIComponent(dom) + "/semantic");
      if (dom !== this.state.domain) return;
      this.renderSemanticStatus(d);
      // While a build is in flight keep polling so progress stays live.
      clearTimeout(this._semPoll);
      if (d.state === "building") {
        this._semPoll = setTimeout(() => {
          if (document.getElementById("panel-semantic")
              .classList.contains("active")) {
            this.fetchSemanticStatus();
          }
        }, 2500);
      }
    } catch (err) {
      this.dom.semStatusBody.innerHTML =
        '<div class="list-note is-error">Статус семантики: ' +
        this.escapeHtml(err.message || String(err)) + "</div>";
    }
  }

  renderSemanticStatus(d) {
    const esc = (s) => this.escapeHtml(String(s));
    const state = String(d.state || "—");
    const stCls = state === "ready" ? "ok"
      : (state === "failed" || state === "unavailable") ? "err" : "warn";
    this.dom.semState.textContent = "idx: " + state;
    this.dom.semState.className = "verdict " + stCls;
    this.dom.semState.title = d.reason || "";
    const fresh = d.fresh;
    this.dom.semFresh.textContent = fresh === true ? "fresh"
      : fresh === false ? "stale" : "freshness: —";
    this.dom.semFresh.className = "verdict " +
      (fresh === true ? "ok" : fresh === false ? "warn" : "");

    const p = d.progress || {};
    const idx = d.index || {};
    const cov = d.coverage || {};
    const ent = d.entity || {};
    const rep = d.last_sync || {};
    const mb = (b) => b == null ? "—" : (b / 1048576).toFixed(1) + " MB";
    const pct = (p.total > 0)
      ? Math.round(100 * (p.processed || 0) / p.total) : 0;

    let html = '<div class="insp-grid">';
    html += `<div class="insp-row"><span>Стан</span><span class="mono ${stCls === "ok" ? "insp-ok" : stCls === "err" ? "insp-err" : "insp-warn"}">${esc(state)}${d.reason ? " — " + esc(d.reason) : ""}</span></div>`;
    if (p.total) {
      html += `<div class="insp-row"><span>Прогрес</span><span class="mono">` +
        `${esc(p.processed || 0)}/${esc(p.total)} документів · embedded ${esc(p.embedded || 0)}` +
        `</span></div>`;
      html += `<div class="sem-bar"><div class="sem-bar-fill" style="width:${pct}%"></div></div>`;
    }
    const entTxt = ent.table
      ? esc(ent.table) + (ent.ambiguous
          ? ' <span class="insp-warn">НЕОДНОЗНАЧНО</span>' : "")
      : "—";
    html += `<div class="insp-row"><span>Сутність (обрано структурно)</span><span class="mono">${entTxt}</span></div>`;
    if (ent.alternatives && ent.alternatives.length) {
      html += `<div class="insp-row"><span>Альтернативи</span><span class="mono insp-dim">${esc(ent.alternatives.join(", "))}</span></div>`;
    }
    if (idx.live != null) {
      html += `<div class="insp-row"><span>Індекс</span><span class="mono">` +
        `${esc(idx.live)} live · ${esc(idx.dim)}d · gen ${esc(idx.generation)}` +
        ` · ${mb(idx.disk_bytes)} на диску</span></div>`;
      html += `<div class="insp-row"><span>Модель</span><span class="mono insp-dim" title="${esc(idx.model_id || "")}">${esc((idx.model_id || "—").slice(0, 60))}</span></div>`;
    }
    if (cov.source_rows != null) {
      html += `<div class="insp-row"><span>Покриття</span><span class="mono">` +
        `джерело ${esc(cov.source_rows)} · проіндексовано ${esc(cov.indexed ?? "?")}` +
        ` · обрізано ${esc(cov.truncated_docs ?? 0)}/${esc(cov.model_truncated_docs ?? "?")}` +
        ` · null-ref ${esc(cov.null_ref_rows ?? 0)} · orphan ${esc(cov.orphan_rows ?? 0)}</span></div>`;
    }
    if (rep.source_rows != null) {
      html += `<div class="insp-row"><span>Остання синхронізація</span><span class="mono">` +
        `+${esc(rep.inserted ?? 0)} ~${esc(rep.updated ?? 0)} -${esc(rep.deleted ?? 0)}` +
        ` · без змін ${esc(rep.unchanged ?? 0)} · нових векторів ${esc(rep.embedded_texts ?? "?")}` +
        ` · ${esc(rep.elapsed_s != null ? rep.elapsed_s.toFixed(1) + " с" : "?")}` +
        `${rep.complete === false ? " · НЕПОВНА" : ""}` +
        `${rep.error ? " · " + esc(rep.error) : ""}</span></div>`;
    }
    html += "</div>";
    this.dom.semStatusBody.innerHTML = html;
  }

  async refreshSemanticSync() {
    const dom = this.state.domain;
    this.dom.btnSemSync.disabled = true;
    try {
      await this.api(
        "/api/domains/" + encodeURIComponent(dom) + "/refresh", {
          method: "POST",
          headers: { "X-Admin-Token": this.adminToken() },
          body: { rebuild: false },
        });
      this.dom.semMeta.textContent =
        "синхронізацію прийнято (202) — статус оновлюється…";
      this.fetchSemanticStatus();
      setTimeout(() => this.fetchSemanticStatus(), 2500);
    } catch (err) {
      const msg = String(err && err.message || err);
      if (msg.startsWith("HTTP 403") || msg.startsWith("HTTP 401")) {
        this.askSemToken();
      } else {
        this.dom.semMeta.textContent = "sync: " + msg;
      }
    } finally {
      this.dom.btnSemSync.disabled = false;
    }
  }

  askSemToken() {
    // Same inline-token pattern as the build map: enter once, every
    // admin action reuses it.
    this.dom.semMeta.innerHTML =
      "потрібен admin-токен: " +
      '<input id="sem-tok" type="password" class="input sem-tok" ' +
      'placeholder="X-Admin-Token"> ' +
      '<button id="sem-tok-go" class="btn-mini">OK</button>';
    const go = () => {
      const el = document.getElementById("sem-tok");
      const v = el && el.value.trim();
      if (!v) return;
      localStorage.setItem("pb_admin_token", v);
      if (this.dom.regAdminToken) this.dom.regAdminToken.value = v;
      this.refreshSemanticSync();
    };
    document.getElementById("sem-tok-go").addEventListener("click", go);
    document.getElementById("sem-tok").addEventListener("keydown",
      (e) => { if (e.key === "Enter") go(); });
  }

  async runSemanticSearch() {
    const q = (this.dom.semQ.value || "").trim();
    if (!q || this.dom.btnSemSearch.disabled) return;
    const dom = this.state.domain;
    this.dom.semMeta.textContent = "пошук…";
    this.dom.semHits.innerHTML = "";
    this.dom.btnSemSearch.disabled = true;
    try {
      const d = await this.api(
        "/api/domains/" + encodeURIComponent(dom) + "/search?q=" +
        encodeURIComponent(q) + "&k=" + this.dom.semK.value +
        "&mode=" + this.dom.semMode.value +
        (this.dom.semHyde.checked ? "&hyde=1" : ""),
        { timeout: 120000 });
      if (dom !== this.state.domain) return;
      this.renderSemSearch(d);
      // A search can flip a stale index into refresh -- reflect it.
      this.renderSemanticStatus(d.index);
    } catch (err) {
      this.dom.semMeta.textContent =
        "помилка: " + (err.message || String(err));
    } finally {
      this.dom.btnSemSearch.disabled = false;
    }
  }

  renderSemSearch(d) {
    const esc = (s) => this.escapeHtml(String(s));
    const conf = d.confidence || {};
    const confTxt = ["top1_sim", "top1_z", "margin"]
      .filter((k) => conf[k] != null)
      .map((k) => `${k}=${Number(conf[k]).toFixed(3)}`).join(" · ");
    let meta = `status: ${esc(d.status)} · k=${esc(d.k)} · mode ${esc(d.mode)}`;
    if (confTxt) meta += " · " + confTxt;
    if (d.notes && d.notes.length) {
      meta += " · " + d.notes.map(esc).join(" · ");
    }
    this.dom.semMeta.textContent = meta;
    const hits = d.hits || [];
    if (!hits.length) {
      this.dom.semHits.innerHTML =
        '<div class="list-note">Збігів немає (чесна відсутність, не вигадка).</div>';
      return;
    }
    this.dom.semHits.innerHTML = hits.map((h, i) => {
      const ch = [
        h.dense_rank != null ? `dense#${h.dense_rank}` : null,
        h.dense_sim != null ? `sim ${Number(h.dense_sim).toFixed(3)}` : null,
        h.lexical_rank != null ? `lex#${h.lexical_rank}` : null,
      ].filter(Boolean).join(" · ");
      return '<div class="sem-hit">' +
        `<div class="sem-hit-head"><span class="sem-hit-n">${i + 1}</span>` +
        `<span class="sem-hit-name">${esc(h.name || h.entity_id)}</span>` +
        `<span class="sem-hit-score mono">${h.score != null ? Number(h.score).toFixed(4) : ""}</span></div>` +
        `<div class="sem-hit-meta mono insp-dim">id ${esc(h.entity_id)}` +
        (ch ? " · " + esc(ch) : "") + "</div>" +
        `<div class="sem-hit-snip">${esc(h.snippet || "")}</div>` +
        "</div>";
    }).join("");
  }

  renderFiduciary(f, att) {
    const esc = (s) => this.escapeHtml(String(s));
    const status = String(f.status || "—");
    const cls = status === "CERTIFIED" ? "insp-ok" : status === "VIOLATION" ? "insp-err" : "insp-warn";
    const fid = typeof f.fid === "number" ? f.fid.toFixed(6) : "—";
    const channels = (f.channels || [])
      .map((c) => `${esc(c.table)}.${esc(c.column)} → ${esc(c.concept)}`).join("<br>");
    let html = '<div class="section-label">Сертифікат інваріантності довіри (FIC)</div><div class="insp-grid">';
    html += `<div class="insp-row"><span>Статус</span><span class="mono ${cls}">${esc(status)}` +
      `${f.reason ? " — " + esc(f.reason) : ""}</span></div>`;
    html += `<div class="insp-row"><span>FID (зсув рангу)</span><span class="mono">${esc(fid)}</span></div>`;
    html += `<div class="insp-row"><span>Оцінено позицій</span><span class="mono">${esc(f.evaluated_items || 0)}` +
      ` · переміщено ${(f.moved_items || []).length} · порушень ${(f.violations || []).length}</span></div>`;
    html += `<div class="insp-row"><span>Комерційні канали</span><span class="mono insp-dim">${channels || "не виявлено"}</span></div>`;
    html += `<div class="insp-row"><span>Межі перевірки</span><span class="insp-dim">${esc(f.scope || "—")}</span></div>`;
    if (att && att.source) {
      const sha = String(att.source.sha256 || "");
      const digest = String(att.digest || "");
      html += `<div class="insp-row"><span>Атестація</span><span class="mono insp-ok">${esc((att.result || {}).status || "—")}` +
        ` · ${esc(att.issued_at || "—")}</span></div>`;
      html += `<div class="insp-row"><span>sha256 джерела</span><span class="mono insp-dim" title="${esc(sha)}">` +
        `${esc(sha.slice(0, 16) || "—")}…</span></div>`;
      html += `<div class="insp-row"><span>digest атестації</span><span class="mono insp-dim" title="${esc(digest)}">` +
        `${esc(digest.slice(0, 16) || "—")}…</span></div>`;
    }
    return html + "</div>";
  }

  displaySourcePath(rawPath) {
    if (!rawPath) return "—";
    const norm = String(rawPath).replace(/\\/g, "/");
    for (const marker of ["/data/", "/tests/fixtures/", "/tests/"]) {
      const idx = norm.lastIndexOf(marker);
      if (idx !== -1) return norm.slice(idx + 1);
    }
    const parts = norm.split("/").filter(Boolean);
    return parts.length > 2 ? parts.slice(-2).join("/") : norm;
  }

  renderInspection(d) {
    const esc = (s) => this.escapeHtml(String(s));
    let html = "";

    // --- Джерело / стан -------------------------------------------------
    html += '<div class="section-label">Джерело та стан</div>';
    html += '<div class="insp-grid">';
    html += `<div class="insp-row"><span>Стан</span><span class="mono ${d.state === "ready" ? "insp-ok" : "insp-warn"}">${esc(d.state)}${d.state_reason ? " — " + esc(d.state_reason) : ""}</span></div>`;
    html += `<div class="insp-row"><span>Реєстрація</span><span class="mono">${esc(d.registered_via)}</span></div>`;
    html += `<div class="insp-row"><span>Файл БД</span><span class="mono insp-dim">${esc(this.displaySourcePath(d.db_path))}</span></div>`;
    if (d.source_version) {
      html += `<div class="insp-row"><span>Версія джерела</span><span class="mono">mtime ${esc(d.source_version[0])} · ${esc(d.source_version[1])} B</span></div>`;
    }
    html += "</div>";
    if (d.fiduciary) {
      html += this.renderFiduciary(d.fiduciary, d.attestation);
    }

    // --- Таблиці джерела ------------------------------------------------
    if (d.tables && Object.keys(d.tables).length) {
      html += '<div class="section-label">Виявлені таблиці джерела</div>';
      html += '<div class="insp-tables">';
      for (const [tname, t] of Object.entries(d.tables)) {
        const cols = Object.entries(t.columns || {})
          .map(([c, meta]) => `${esc(c)}<i class="insp-dim">:${esc(meta.semantic_type || meta.type || "")}</i>`)
          .join(" · ");
        html += `<div class="insp-table-card"><div class="insp-table-head"><code>${esc(tname)}</code> <span class="mono insp-dim">${esc(t.row_count)} рядків</span></div><div class="insp-table-cols">${cols}</div></div>`;
      }
      html += "</div>";
    }

    // --- Карта побудови (анімація кроків) --------------------------------
    if (d.tables && Object.keys(d.tables).length && d.projection && !d.projection.error) {
      html += '<div class="section-label">Карта побудови — як система читала схему</div>';
      html += this.renderBuildMap(d);
    }

    // --- Блок-схема проєкції (SVG) --------------------------------------
    if (d.tables && Object.keys(d.tables).length && d.projection && !d.projection.error) {
      html += '<div class="section-label">Схема проєкції (блок-схема)</div>';
      html += this.renderProjectionDiagram(d);
    }

    // --- Проєкція ролей -------------------------------------------------
    const proj = d.projection;
    if (proj && !proj.error) {
      html += '<div class="section-label">Семантична проєкція (ролі → колонки)</div>';
      html += `<div class="insp-row"><span>Entity-таблиця</span><span class="mono insp-ok">${esc(proj.entity_table || "не визначено")}</span></div>`;
      for (const [role, r] of Object.entries(proj.roles || {})) {
        const cls = r.resolved ? "insp-ok" : "insp-unknown";
        const tgt = r.resolved ? `${esc(r.table)}.${esc(r.column)}` : "∅ нерезольвлено";
        html += `<div class="insp-role"><div class="insp-role-head"><span class="mono">${esc(role)}</span><span class="${cls}">${tgt}</span></div>`;
        const cands = r.candidates || [];
        if (cands.length) {
          html += '<div class="insp-cands">';
          for (const c of cands) {
            const veto = c.vetoed ? "insp-veto" : "";
            const ev = Array.isArray(c.evidence) ? c.evidence.map((e) => esc(this.fmt(e))).join(", ") : "";
            html += `<div class="insp-cand ${veto}"><code>${esc(c.table)}.${esc(c.column)}</code>${c.vetoed ? " ⛔" : ""}<span class="insp-dim">${ev}</span></div>`;
          }
          html += "</div>";
        }
        html += "</div>";
      }
      if (proj.eav_tables && Object.keys(proj.eav_tables).length) {
        html += '<div class="section-label">EAV-таблиці (віртуальні атрибути)</div>';
        for (const [t, m] of Object.entries(proj.eav_tables)) {
          const keys = m.discovered_attributes || [];
          const efk = m.entity_fk || m.entity_column;
          const total = m.keys_total != null ? m.keys_total : keys.length;
          const cover = keys.length < total ? ` (сховище: ${keys.length})` : " — повний словник";
          const det = m.detection === "structural" ? "структурно" : "name-hint";
          html += `<div class="insp-table-card"><div class="insp-table-head"><code>${esc(t)}</code> <span class="mono insp-dim">${esc(efk)} → ${esc(m.key_column)} → ${esc(m.value_column)} · ${esc(det)} · ключів: ${esc(total)}${esc(cover)}</span></div>`;
          if (keys.length) {
            html += `<div class="insp-keys insp-keys-scroll">${keys.map((k) => `<span class="onto-param-tag">${esc(k)}</span>`).join("")}</div>`;
          }
          html += "</div>";
        }
      }
      if (d.llm_mapping) {
        html += '<div class="section-label">LLM-асист (пропозиції → верифікація даними)</div>';
        const lm = d.llm_mapping;
        if (!lm.ran) {
          html += `<div class="insp-row"><span>Статус</span><span class="mono insp-dim">${esc(lm.reason || "не запускався")}</span></div>`;
        } else {
          for (const [role, p] of Object.entries(lm.proposals || {})) {
            const ok = String(p.verdict || "").startsWith("resolved");
            html += `<div class="insp-row"><span class="mono">${esc(role)} → ${esc(p.column)}</span><span class="${ok ? "insp-ok" : "insp-unknown"}">${esc(p.verdict)}</span></div>`;
          }
        }
      } else if (d.llm_assist) {
        html += '<div class="section-label">LLM-асист</div><div class="insp-row"><span>Статус</span><span class="mono insp-dim">увімкнено, але модель недоступна / ще не запускалась</span></div>';
      }
      if (Array.isArray(proj.unassigned_numeric) && proj.unassigned_numeric.length) {
        html += `<div class="insp-row"><span>Числові колонки без ролі</span><span class="mono insp-warn">${proj.unassigned_numeric.map(esc).join(", ")}</span></div>`;
      }
    } else if (proj && proj.error) {
      html += `<div class="list-note is-error">Проєкція не вдалася: ${esc(proj.error)}</div>`;
    }

    // --- Гіперграф / контракти ------------------------------------------
    if (d.hypergraph) {
      html += '<div class="section-label">Гіперграф (робочий простір правил)</div>';
      const rep = d.hypergraph.induction_report;
      html += `<div class="insp-row"><span>Вузлів у графі</span><span class="mono">${esc(d.hypergraph.seeded_nodes)} (seed+dynamic)</span></div>`;
      html += `<div class="insp-row"><span>Контрактів</span><span class="mono">${esc(d.hypergraph.constraint_count)}${rep ? ` — індуковано ${esc(rep.induced)}, відкинуто ${esc(rep.dropped)}` : ""}</span></div>`;
      const cons = d.hypergraph.constraints || [];
      if (cons.length) {
        html += '<div id="insp-rule-msg" class="insp-dim" aria-live="polite"></div>';
        html += '<div class="insp-cons">';
        for (const c of cons.slice(0, 30)) {
          const cls = c.status === "candidate" ? "insp-cand-status" : "insp-ok";
          const nextStatus = c.status === "verified" ? "candidate" : "verified";
          const btnLabel = c.status === "verified" ? "В candidate" : "Підтвердити (verified)";
          const btnHtml = c.rule_id
            ? `<button type="button" class="btn-mini btn-rule-status" data-rule-id="${esc(c.rule_id)}" data-next-status="${nextStatus}">${btnLabel}</button>`
            : "";
          const keys = Object.entries(c)
            .filter(([k]) => !["rule_id", "type", "status", "description"].includes(k))
            .map(([k, v]) => `${esc(k)}=${esc(Array.isArray(v) ? v.join("/") : this.fmt(v))}`)
            .join(" · ");
          html += `<div class="insp-con"><div class="insp-con-head"><span class="mono">${esc(c.type)}</span> <span class="${cls}">[${esc(c.status || "?")}]</span>${btnHtml}</div><div class="insp-dim">${keys}${c.description ? " — " + esc(c.description) : ""}</div></div>`;
        }
        if (cons.length > 30) html += `<div class="insp-dim">… ще ${cons.length - 30}</div>`;
        html += "</div>";
      }
    }

    // --- Таксономія ------------------------------------------------------
    if (d.taxonomy) {
      html += '<div class="section-label">OntoGraph: агрегована таксономія</div>';
      html += `<div class="insp-row"><span>Усього товарів</span><span class="mono insp-ok">${esc(d.taxonomy.item_count)}</span></div>`;
      const cats = (d.taxonomy.top_categories || [])
        .map((c) => `${esc(c.name)} <span class="insp-dim">(${esc(c.item_count)}${c.children ? ", +" + esc(c.children) + " під" : ""})</span>`)
        .join(" · ");
      html += `<div class="insp-row"><span>Категорії</span><span>${cats || "—"}</span></div>`;
    }

    // --- Банк питань -----------------------------------------------------
    if (d.question_bank && d.question_bank.length) {
      html += '<div class="section-label">Банк діагностичних питань</div>';
      html += '<div class="insp-cons">';
      for (const q of d.question_bank) {
        const badge = q.adapted ? '<span class="insp-ok">адаптовано</span>' : '<span class="insp-dim">базове</span>';
        html += `<div class="insp-con"><span class="mono">${esc(q.id)}</span> ${badge}<div class="insp-dim">${esc(q.text)}${q.target_dimension ? " → " + esc(q.target_dimension) : ""}</div></div>`;
      }
      html += "</div>";
    }

    // --- Пошуковий бекенд -------------------------------------------------
    if (d.search_index) {
      html += '<div class="section-label">Пошуковий індекс</div>';
      const si = d.search_index;
      html += `<div class="insp-row"><span>FTS5</span><span class="mono ${si.status === "ready" ? "insp-ok" : "insp-warn"}">${esc(si.status)}${si.indexed_rows ? " · " + esc(si.indexed_rows) + " рядків за " + esc(si.built_seconds) + " с" : ""}${si.fail_reason ? " — " + esc(si.fail_reason) : ""}</span></div>`;
    }

    // --- Промпт моделі (буквально) ----------------------------------------
    html += '<div class="section-label">Схема, яку бачить модель (literal schema_prompt)</div>';
    html += `<details class="insp-details"><summary>Розгорнути промпт (${(d.schema_prompt || "").length} символів)</summary><pre class="raw-json">${esc(d.schema_prompt || "—")}</pre></details>`;

    return html;
  }

  /**
   * Accurate block-diagram of the projection, rendered as inline SVG.
   * Draws exactly what the projection asserts: the entity table box with
   * role->column bindings, every other source table as a box, EAV tables
   * with a labelled edge onto the entity, and unresolved roles as a
   * dashed red row -- nothing is invented for the picture.
   */
  renderProjectionDiagram(d) {
    const esc = (s) => this.escapeHtml(String(s));
    const proj = d.projection;
    const entity = proj.entity_table;
    const roles = proj.roles || {};
    const eav = proj.eav_tables || {};
    const W = 740, pad = 12;

    const roleRows = Object.entries(roles).map(([role, r]) => ({
      text: r.resolved ? `${role} → ${r.column}` : `${role} → не розв'язано`,
      ok: !!r.resolved,
    }));
    const rowH = 15;
    const entH = 34 + Math.max(roleRows.length, 1) * rowH + 8;
    const entW = 250, entX = 24, entY = 24;

    const others = Object.entries(d.tables).filter(([t]) => t !== entity);
    const boxW = 216, boxH = 92, gapX = 18, gapY = 14;
    const cols = Math.max(1, Math.floor((W - entX - entW - 56) / (boxW + gapX)));
    const oRows = Math.ceil(others.length / cols);
    const H = Math.max(entY + entH + 28, entY + oRows * (boxH + gapY) + 16, 220);

    let svg = `<svg viewBox="0 0 ${W} ${H}" class="insp-svg pd-svg" xmlns="http://www.w3.org/2000/svg">`;
    svg += `<defs><marker id="pd-arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0.5 L7.5 4 L0 7.5 Z" class="pd-arr-fill"/></marker></defs>`;

    // entity card
    svg += `<g class="pd-card"><rect x="${entX}" y="${entY}" width="${entW}" height="${entH}" rx="7" class="pd-ent"/>`;
    svg += `<text x="${entX + pad}" y="${entY + 18}" class="pd-head">${esc(entity)}</text>`;
    svg += `<text x="${entX + pad}" y="${entY + 31}" class="pd-dim">${esc(d.tables[entity].row_count)} рядків · entity</text>`;
    svg += `<line x1="${entX}" y1="${entY + 38}" x2="${entX + entW}" y2="${entY + 38}" class="pd-sep"/>`;
    roleRows.forEach((r, i) => {
      const y = entY + 52 + i * rowH;
      svg += `<circle cx="${entX + pad}" cy="${y - 3.5}" r="2.6" class="${r.ok ? "pd-dot-ok" : "pd-dot-miss"}"/>`;
      svg += `<text x="${entX + pad + 9}" y="${y}" class="${r.ok ? "pd-role" : "pd-unres"}">${esc(r.text)}</text>`;
    });
    svg += "</g>";

    // table cards + curved edges
    others.forEach(([tname, t], i) => {
      const cx = entX + entW + 56 + (i % cols) * (boxW + gapX);
      const cy = entY + Math.floor(i / cols) * (boxH + gapY);
      const isEav = !!eav[tname];
      svg += `<g class="pd-card"><rect x="${cx}" y="${cy}" width="${boxW}" height="${boxH}" rx="7" class="${isEav ? "pd-eav" : "pd-tbl"}"/>`;
      svg += `<text x="${cx + pad}" y="${cy + 18}" class="pd-head pd-head-sm">${esc(tname)}</text>`;
      svg += `<text x="${cx + pad}" y="${cy + 31}" class="pd-dim">${esc(t.row_count)} рядків${isEav ? " · EAV" : ""}</text>`;
      const colNames = Object.keys(t.columns || {}).slice(0, 3);
      colNames.forEach((c, j) => {
        const meta = (t.columns || {})[c] || {};
        svg += `<text x="${cx + pad}" y="${cy + 46 + j * 13}" class="pd-col">${esc(c)}<tspan class="pd-dim"> :${esc(meta.semantic_type || meta.type || "")}</tspan></text>`;
      });
      if (Object.keys(t.columns || {}).length > 3) {
        svg += `<text x="${cx + pad}" y="${cy + 46 + 3 * 13}" class="pd-dim">…${Object.keys(t.columns).length - 3}</text>`;
      }
      svg += "</g>";

      // bezier edge: eav/other table -> entity
      const x1 = cx, y1 = cy + boxH / 2;
      const x2 = entX + entW, y2 = entY + entH / 2;
      const c1x = x1 - Math.abs(x1 - x2) * 0.45, c2x = x2 + Math.abs(x1 - x2) * 0.45;
      const m = eav[tname];
      const lab = m ? `${esc(m.entity_fk || m.entity_column || "?")} · ${esc(m.key_column || "?")}→${esc(m.value_column || "?")}` : "";
      svg += `<path class="pd-edge" d="M${x1},${y1} C${c1x},${y1} ${c2x},${y2} ${x2},${y2}" marker-end="url(#pd-arr)"/>`;
      if (lab) {
        const lx = (x1 + x2) / 2, ly = (y1 + y2) / 2;
        const wlab = Math.min(lab.length * 5.2 + 14, 190);
        svg += `<rect x="${lx - wlab / 2}" y="${ly - 9}" width="${wlab}" height="15" rx="7" class="pd-lab-bg"/>`;
        svg += `<text x="${lx}" y="${ly + 2}" class="pd-edge-lab" text-anchor="middle">${lab}</text>`;
      }
    });
    svg += "</svg>";
    return `<div class="insp-diagram">${svg}</div>`;
  }

  /* ------------------------------------------------------------------ */
  adminToken() {
    return (this.dom.regAdminToken && this.dom.regAdminToken.value)
      || localStorage.getItem("pb_admin_token") || "";
  }

  /* Build map: replayable "neural canvas" of the projection pipeline     */
  /* ------------------------------------------------------------------ */

  renderBuildMap(d) {
    const esc = (s) => this.escapeHtml(String(s));
    const proj = d.projection || {};
    const tables = Object.entries(d.tables || {});
    const entity = proj.entity_table;
    const roles = proj.roles || {};
    const eav = proj.eav_tables || {};
    const inferred = proj.inferred_links || [];
    const trace = proj.build_trace || [];
    this._bmTrace = trace;
    this._bmData = d; // click-detail cards read the full payload

    // step index: which trace event reveals which element id
    const stepOf = {};
    trace.forEach((ev, i) => {
      (ev.targets || []).forEach((t) => {
        if (stepOf[t] === undefined) stepOf[t] = i;
      });
    });
    const step = (id) =>
      stepOf[id] === undefined ? "end" : String(stepOf[id]);

    const W = 880, H = 560, cx = 330, cy = 290, R = 195;
    const pos = {};
    const others = tables.filter(([t]) => t !== entity);
    others.forEach(([t], i) => {
      const a = (i / Math.max(others.length, 1)) * Math.PI * 2 - Math.PI / 2;
      pos[t] = { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a) };
    });
    if (entity) pos[entity] = { x: cx, y: cy };
    const rolePos = {};
    Object.keys(roles).forEach((r, i) => {
      rolePos[r] = { x: 730, y: 60 + i * 56 };
    });

    let svg = `<svg id="buildmap-svg" viewBox="0 0 ${W} ${H}" class="bm-svg" xmlns="http://www.w3.org/2000/svg">`;
    const desc = {}; // element id -> hover description
    const edge = (id, x1, y1, x2, y2, cls, label) => {
      svg += `<g class="bm-edge ${cls}" data-bm="${step(id)}" data-tid="${id}" data-desc="${esc(desc[id] || label || "")}">` +
        `<path d="M${x1},${y1} C${x1 + (x2 - x1) * 0.4},${y1} ${x2 - (x2 - x1) * 0.4},${y2} ${x2},${y2}" fill="none"/>` +
        (label ? `<text x="${(x1 + x2) / 2}" y="${(y1 + y2) / 2 - 4}">${esc(label)}</text>` : "") +
        "</g>";
    };

    // edges first so nodes draw over them
    for (const [role, res] of Object.entries(roles)) {
      if (!res.resolved || !entity || !rolePos[role]) continue;
      const rp = rolePos[role];
      const id = "edge:role:" + role;
      desc[id] = `${role} → ${res.table}.${res.column}`;
      edge(id, pos[entity].x, pos[entity].y, rp.x - 110, rp.y,
        "bm-edge-role", role);
    }
    for (const [tname, tm] of tables) {
      for (const [c, cm] of Object.entries(tm.columns || {})) {
        if (!cm.fk_target || !pos[tname]) continue;
        const tgt = String(cm.fk_target).split(".")[0];
        if (!pos[tgt]) continue;
        const id = `edge:fk:${tname}:${tgt}`;
        desc[id] = `FK ${tname}.${c} → ${cm.fk_target}`;
        edge(id, pos[tname].x, pos[tname].y, pos[tgt].x, pos[tgt].y,
          "bm-edge-fk", "fk");
      }
    }
    for (const t of Object.keys(eav)) {
      if (!pos[t] || !entity) continue;
      const id = "edge:eav:" + t;
      const m = eav[t];
      desc[id] = `EAV: ${m.entity_fk || m.entity_column || "?"} → ` +
        `${m.key_column || "?"} → ${m.value_column || "?"} ` +
        `(${m.detection || "?"})`;
      edge(id, pos[entity].x, pos[entity].y, pos[t].x, pos[t].y,
        "bm-edge-eav", "eav");
    }
    for (const l of inferred) {
      if (!pos[l.child_table] || !pos[l.parent_table]) continue;
      const id = `edge:inf:${l.child_table}:${l.parent_table}`;
      desc[id] = `inferred link ${l.child_table}.${l.child_column} → ` +
        `${l.parent_table}.${l.parent_column} · overlap ${l.overlap}`;
      edge(id, pos[l.child_table].x, pos[l.child_table].y,
        pos[l.parent_table].x, pos[l.parent_table].y,
        "bm-edge-inf", `∞ ${l.overlap}`);
    }
    // axis-bound roles: a role that lives on an EAV key axis
    const axisRoles = proj.axis_roles || {};
    for (const [role, ar] of Object.entries(axisRoles)) {
      if (!pos[ar.table] || !rolePos[role]) continue;
      const id = `edge:axis:${ar.table}:${role}`;
      const rp = rolePos[role];
      desc[id] = `${role} ← EAV-вісь ${ar.table}.${ar.key_column}=` +
        `"${ar.axis}" · словник ${ar.vocab_size} · ` +
        `покриття ${Math.round((ar.coverage || 0) * 100)}%`;
      edge(id, pos[ar.table].x, pos[ar.table].y, rp.x - 110, rp.y,
        "bm-edge-axis", `axis:${ar.axis}`);
    }

    // table nodes (neurons)
    for (const [tname, tm] of tables) {
      const p = pos[tname];
      const rows = tm.row_count || 0;
      const r = Math.max(15, Math.min(30, 14 + Math.log10(rows + 1) * 5));
      const isEnt = tname === entity;
      const isEav = !!eav[tname];
      const id = "table:" + tname;
      const cls = "bm-tbl" + (isEnt ? " bm-entity" : "") +
        (isEav ? " bm-eav" : "");
      svg += `<g class="bm-node ${cls}" data-bm="${step(id)}" data-tid="${id}" ` +
        `data-desc="${esc(tname)} · ${rows} рядків` +
        `${isEnt ? " · entity" : ""}${isEav ? " · EAV" : ""}">` +
        `<circle cx="${p.x}" cy="${p.y}" r="${r}"/>` +
        (isEnt ? `<circle cx="${p.x}" cy="${p.y}" r="${r + 4}" class="bm-halo"/>` : "") +
        `<text x="${p.x}" y="${p.y + r + 13}" class="bm-txt">${esc(tname)}</text>` +
        `<text x="${p.x}" y="${p.y + 3}" class="bm-txt bm-rows">${esc(rows)}</text>` +
        "</g>";
    }

    // role chips (what the projection resolved)
    for (const [role, res] of Object.entries(roles)) {
      const rp = rolePos[role];
      const id = "role:" + role;
      const axis = (proj.axis_roles || {})[role];
      const ok = !!res.resolved || !!axis;
      svg += `<g class="bm-node ${ok ? "bm-role" : "bm-missing"}" ` +
        `data-bm="${step(id)}" data-tid="${id}" data-desc="${esc(
          res.resolved ? `${role} → ${res.table}.${res.column}`
          : axis ? `${role} ← вісь ${axis.table}."${axis.axis}"`
          : `${role}: не розв'язано`)}">` +
        `<rect x="${rp.x - 108}" y="${rp.y - 13}" width="216" height="26" rx="13"/>` +
        `<text x="${rp.x}" y="${rp.y + 4}" class="bm-txt">${esc(role)}${ok ? "" : " ∅"}</text>` +
        "</g>";
    }
    svg += "</svg>";

    const log = trace.map((ev) =>
      `<div class="bm-log-row" data-seq="${ev.seq}">` +
      `<span class="bm-log-seq">${ev.seq}</span>` +
      `<span class="bm-log-lab">${esc(ev.label)}</span>` +
      (ev.detail ? `<div class="bm-log-det">${esc(ev.detail)}</div>` : "") +
      "</div>").join("");

    return `<div class="buildmap">
      <div class="bm-controls">
        <button id="bm-play" class="btn-mini">▶ програти</button>
        <button id="bm-restart" class="btn-mini">↻ з початку</button>
        <input id="bm-slider" type="range" min="0" max="${Math.max(trace.length, 1)}" value="0"/>
        <span id="bm-step" class="mono insp-dim">0/${trace.length}</span>
        <button id="bm-conductor" class="btn-mini" title="Запустити LLM-диригента: гіпотези → верифікація даними"><svg class=\"ic\" width=\"12\" height=\"12\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\" style=\"vertical-align:-1px\"><path d=\"M13 2 4.5 13.5H11L10 22l8.5-11.5H12z\"/></svg> диригент</button>
        <select id="bm-embed-model" class="bm-select" title="Модель ембедінгу для векторного каналу"></select>
        <button id="bm-vector" class="btn-mini" title="Побудувати векторний індекс (opt-in канал парафрази)">⊕ вектор</button>
        <span class="bm-reanalyze" title="Перебудувати проєкцію заново з обраними параметрами — автоаналіз при реєстрації не торкається">
          <svg class=\"ic\" width=\"12\" height=\"12\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.8\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\" style=\"vertical-align:-1px\"><path d=\"M20 11A8 8 0 1 0 18.6 16\"/><path d=\"M20 5v6h-6\"/></svg> <label><input id="bm-re-llm" type="checkbox"/>диригент</label>
          <label><input id="bm-re-vec" type="checkbox"/>вектор</label>
          <button id="bm-reanalyze" class="btn-mini">переаналізувати</button>
        </span>
      </div>
      <div class="bm-wrap">
        ${svg}
        <div class="bm-side">
          <div id="bm-detail" class="bm-detail">Наведи на вузол чи синапс…</div>
          <div id="bm-log" class="bm-log">${log || '<div class="bm-log-det">трейс порожній</div>'}</div>
        </div>
      </div>
    </div>`;
  }

  _initBuildMap() {
    const svg = document.getElementById("buildmap-svg");
    if (!svg) return;
    const slider = document.getElementById("bm-slider");
    const stepLabel = document.getElementById("bm-step");
    const detail = document.getElementById("bm-detail");
    const log = document.getElementById("bm-log");
    const els = [...svg.querySelectorAll("[data-bm]")];
    const total = (this._bmTrace || []).length;
    let timer = null;

    // Click a neuron to expand it: the detail panel becomes a card with
    // the evidence behind that node -- columns+samples for tables,
    // candidates+vetoes for roles, vocab for EAV axes.
    const esc = (s) => this.escapeHtml(String(s));
    const detailCard = (tid) => {
      const d = this._bmData || {};
      const proj = d.projection || {};
      const eavM = proj.eav_tables || {};
      if (tid.startsWith("table:")) {
        const t = tid.slice(6);
        const tm = (d.tables || {})[t] || {};
        const em = eavM[t];
        let h = `<div class="bm-det-title">${esc(t)}</div>` +
          `<div class="bm-det-sub">${tm.row_count || 0} рядків` +
          (t === proj.entity_table ? " · entity" : "") +
          (em ? " · EAV" : "") + `</div>`;
        if (em) {
          h += `<div class="bm-det-sec">EAV: ` +
            `${esc(em.entity_fk || em.entity_column || "?")} → ` +
            `${esc(em.key_column || "?")} → ${esc(em.value_column || "?")}` +
            ` · ${esc(em.detection || "?")} · ` +
            `${em.keys_total ?? "?"} осей</div>` +
            `<div class="bm-det-samp">` +
            esc((em.discovered_attributes || []).slice(0, 30).join(", ")) +
            `</div>`;
        }
        h += Object.entries(tm.columns || {}).map(([c, cm]) =>
          `<div class="bm-det-col"><span class="mono">${esc(c)}</span> ` +
          `<span class="insp-dim">${esc(cm.semantic_type || cm.type || "")}` +
          (cm.fk_target ? ` → ${esc(cm.fk_target)}` : "") +
          (cm.distinct_count != null ? ` · ~${cm.distinct_count}` : "") +
          `</span>` +
          ((cm.sample_values || []).length
            ? `<div class="bm-det-samp">` +
              esc(cm.sample_values.slice(0, 4).join(" · ")) + `</div>`
            : "") +
          `</div>`).join("");
        return { html: h, entityTable: t === proj.entity_table };
      }
      if (tid.startsWith("role:")) {
        const r = tid.slice(5);
        const res = (proj.roles || {})[r] || {};
        const axis = (proj.axis_roles || {})[r];
        let h = `<div class="bm-det-title">${esc(r)}</div>`;
        if (res.resolved)
          h += `<div class="bm-det-sub">→ ${esc(res.table)}.` +
            `${esc(res.column)}</div>`;
        else if (axis)
          h += `<div class="bm-det-sub">← вісь ${esc(axis.table)}.` +
            `"${esc(axis.axis)}" · словник ${axis.vocab_size} · ` +
            `покриття ${Math.round((axis.coverage || 0) * 100)}%</div>`;
        else
          h += `<div class="bm-det-sub">не розв'язано — колонки немає ` +
            `або докази неоднозначні</div>`;
        (res.candidates || []).forEach((cd) => {
          h += `<div class="bm-det-col${cd.vetoed ? " bm-veto" : ""}">` +
            `<span class="mono">${esc(cd.table)}.${esc(cd.column)}</span>` +
            (cd.vetoed ? " ⛔ veto" : "") +
            (cd.evidence || []).map((e) =>
              `<div class="bm-det-samp">${esc(e.kind)}: ` +
              `${esc(e.detail)}</div>`).join("") +
            `</div>`;
        });
        return { html: h };
      }
      return null; // edges keep the one-line hover description
    };

    els.forEach((el) => {
      el.classList.add("bm-ghost");
      el.addEventListener("mouseenter", () => {
        if (detail) detail.textContent =
          el.dataset.desc || el.querySelector("title")?.textContent || "—";
      });
      el.addEventListener("click", () => {
        if (!detail) return;
        const card = detailCard(el.dataset.tid || "");
        if (!card) return;
        els.forEach((e2) => e2.classList.remove("bm-sel"));
        el.classList.add("bm-sel");
        detail.innerHTML = card.html +
          (card.entityTable
            ? `<div id="bm-ent-samp" class="bm-det-sec">…</div>` : "");
        detail.classList.add("bm-det-card");
        if (card.entityTable) {
          const slot = detail.querySelector("#bm-ent-samp");
          this.api("/api/catalog?domain=" +
            encodeURIComponent(this.state.domain) + "&limit=8")
            .then((c) => {
              if (!slot) return;
              slot.innerHTML = `<div class="bm-det-sec">Сутності:</div>` +
                (c.products || []).map((p) =>
                  `<div class="bm-det-col mono">${esc(p.name || p.id)}</div>`)
                .join("");
            }).catch(() => { if (slot) slot.remove(); });
        }
      });
    });

    const apply = (cur) => {
      els.forEach((el) => {
        const s = el.dataset.bm;
        const on = s === "end" ? cur >= total : Number(s) < cur;
        el.classList.toggle("bm-on", on);
      });
      if (cur > 0 && cur <= total) {
        const ev = this._bmTrace[cur - 1];
        const row = log && log.querySelector(`[data-seq="${ev.seq}"]`);
        if (log) log.querySelectorAll(".bm-cur").forEach((r) =>
          r.classList.remove("bm-cur"));
        if (row) { row.classList.add("bm-cur");
          const sc = row.closest(".scroll-area") || row.parentElement;
          if (sc) sc.scrollTop = row.offsetTop - sc.clientHeight / 2; }
      }
      if (slider) slider.value = cur;
      if (stepLabel) stepLabel.textContent = `${cur}/${total}`;
    };

    // current-event pulse: flash the elements this event touched
    const flash = (cur) => {
      if (cur <= 0 || cur > total) return;
      const ev = this._bmTrace[cur - 1];
      const kind = ev.kind || "";
      const cls = (kind === "veto" || kind === "llm_veto" ||
        kind === "link_rejected" || kind === "role_missing")
        ? "bm-flash-veto" : "bm-flash";
      (ev.targets || []).forEach((t) => {
        const el = svg.querySelector(`[data-tid="${t}"]`);
        if (el) { el.classList.add(cls); setTimeout(() => el.classList.remove(cls), 700); }
      });
    };

    let cur = 0;
    const advance = () => {
      if (cur >= total) { clearInterval(timer); timer = null;
        const b = document.getElementById("bm-play");
        if (b) b.textContent = "▶ програти";
        return; }
      cur += 1;
      apply(cur);
      flash(cur);
    };
    const play = document.getElementById("bm-play");
    if (play) play.addEventListener("click", () => {
      if (timer) { clearInterval(timer); timer = null;
        play.textContent = "▶ програти"; return; }
      if (cur >= total) cur = 0;
      play.textContent = "⏸ пауза";
      timer = setInterval(advance, 550);
    });
    const restart = document.getElementById("bm-restart");
    if (restart) restart.addEventListener("click", () => {
      cur = 0; apply(0);
    });
    if (slider) slider.addEventListener("input", () => {
      cur = Number(slider.value); apply(cur);
    });

    // Inline admin-token prompt: no hunting for a field at the bottom --
    // enter it right here once and every admin action reuses it.
    const askToken = (action) => {
      if (!detail) return;
      detail.innerHTML =
        `<div class="bm-det-title">Admin-токен</div>` +
        `<div class="bm-det-sub">Введи один раз — запам'ятається ` +
        `для всіх адмін-дій.</div>` +
        `<input id="bm-tok" type="password" placeholder="admin token" ` +
        `class="bm-tok"/>` +
        `<button id="bm-tok-go" class="btn-mini">OK</button>`;
      const go = () => {
        const v = detail.querySelector("#bm-tok").value.trim();
        if (!v) return;
        localStorage.setItem("pb_admin_token", v);
        if (this.dom.regAdminToken) this.dom.regAdminToken.value = v;
        action(v);
      };
      detail.querySelector("#bm-tok-go").addEventListener("click", go);
      detail.querySelector("#bm-tok").addEventListener("keydown",
        (e) => { if (e.key === "Enter") go(); });
      detail.querySelector("#bm-tok").focus();
    };

    // Admin actions try WITHOUT a token first -- loopback deployments
    // (PYBUDDY_LOCAL_ADMIN) accept them as-is; only a real 403 asks for
    // the token, then retries the same action once it is saved.
    const is403 = (err) => String(err && err.message || "").startsWith(
      "HTTP 403");

    const cond = document.getElementById("bm-conductor");
    if (cond) cond.addEventListener("click", async () => {
      cond.disabled = true;
      cond.innerHTML = "<svg class='ic' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M13 2 4.5 13.5H11L10 22l8.5-11.5H12z'/></svg> диригент…";
      try {
        const r = await this.api(
          "/api/domains/" + encodeURIComponent(this.state.domain) +
          "/conductor?max_rounds=8", {
            method: "POST",
            headers: { "X-Admin-Token": this.adminToken() },
          });
        const acc = r.links_accepted || 0;
        if (detail) detail.textContent =
          `Диригент: ${r.rounds || 0} раундів, ` +
          `прийнято зв'язків: ${acc}, ролей: ${r.roles_resolved || 0}` +
          `, осей: ${r.axis_roles_resolved || 0}` +
          (r.reason ? " · " + r.reason : "");
        await this.fetchInspection(); // canvas picks up new trace/links
      } catch (err) {
        if (is403(err)) { askToken(() => cond.click()); }
        else if (detail) detail.textContent =
          "Диригент: " + (err.message || String(err));
      } finally {
        cond.disabled = false;
        cond.innerHTML = "<svg class='ic' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M13 2 4.5 13.5H11L10 22l8.5-11.5H12z'/></svg> диригент";
      }
    });

    // Embedding model selector + vector-index trigger. The channel is
    // opt-in per source: explicit build, honest stats in the panel.
    const sel = document.getElementById("bm-embed-model");
    const vecBtn = document.getElementById("bm-vector");
    if (sel) {
      this.api("/api/settings/embedding").then((cfg) => {
        sel.innerHTML = (cfg.choices || []).map((m) =>
          `<option value="${esc(m)}"${m === cfg.model
            ? " selected" : ""}>${esc(m)}</option>`).join("");
      }).catch(() => {});
      sel.addEventListener("change", () => {
        this.api("/api/settings/embedding?model=" +
          encodeURIComponent(sel.value), {
            method: "POST",
            headers: { "X-Admin-Token": this.adminToken() },
          }).catch((err) => {
            if (is403(err)) askToken(() =>
              sel.dispatchEvent(new Event("change")));
          });
      });
    }
    if (vecBtn) vecBtn.addEventListener("click", async () => {
      vecBtn.disabled = true;
      vecBtn.textContent = "⊕ індекс…";
      try {
        const s = await this.api(
          "/api/domains/" + encodeURIComponent(this.state.domain) +
          "/vector_index", {
            method: "POST",
            headers: { "X-Admin-Token": this.adminToken() },
            timeout: 120000,
          });
        if (detail) detail.textContent =
          s.status === "ready"
            ? `Векторний індекс: ${s.cells} клітин ` +
              `(${s.model}${s.sampled ? ", вибірка" : ""}) ` +
              `за ${s.elapsed_s}с — канал активний`
            : `Вектор недоступний: ${s.reason || s.status}`;
      } catch (err) {
        if (is403(err)) { askToken(() => vecBtn.click()); }
        else if (detail) detail.textContent =
          "Вектор: " + (err.message || String(err));
      } finally {
        vecBtn.disabled = false;
        vecBtn.textContent = "⊕ вектор";
      }
    });

    // Re-analyze: explicit "run it again with MY parameters". The
    // auto-projection from registration is bypassed server-side
    // (fresh=True); the checkboxes choose the optional stages and the
    // result panel says exactly what each stage produced.
    const reBtn = document.getElementById("bm-reanalyze");
    if (reBtn) reBtn.addEventListener("click", async () => {
      const llm = document.getElementById("bm-re-llm").checked;
      const vec = document.getElementById("bm-re-vec").checked;
      reBtn.disabled = true;
      reBtn.textContent = "♻ аналіз…";
      try {
        const r = await this.api(
          "/api/domains/" + encodeURIComponent(this.state.domain) +
          "/reanalyze?llm_assist=" + (llm ? 1 : 0) +
          "&vector=" + (vec ? 1 : 0), {
            method: "POST",
            headers: { "X-Admin-Token": this.adminToken() },
          });
        const jobId = r.job && r.job.job_id;
        // Poll the init-job stream like registration does; the trace
        // shows the projection being rebuilt live.
        const deadline = Date.now() + 10 * 60 * 1000;
        while (Date.now() < deadline) {
          await new Promise((res) => setTimeout(res, 2500));
          const jobs = await this.api("/api/init_jobs");
          const job = (jobs.jobs || []).find(
            (j) => j.job_id === jobId);
          if (!job) break;
          if (detail) {
            const tr = job.trace || [];
            const last = tr.length ? tr[tr.length - 1] : null;
            detail.textContent = "Переаналіз… " +
              (job.stage || job.status) +
              (last ? " · " + last.label : "");
          }
          if (job.status === "done") {
            if (detail) detail.textContent =
              "Переаналізовано — канва оновлюється";
            this.fetchInspection();
            break;
          }
          if (job.status === "failed" || job.status === "cancelled") {
            if (detail) detail.textContent =
              "Переаналіз: " + job.status +
              (job.error ? " — " + job.error : "");
            break;
          }
        }
      } catch (err) {
        if (is403(err)) { askToken(() => reBtn.click()); }
        else if (detail) detail.textContent =
          "Переаналіз: " + (err.message || String(err));
      } finally {
        reBtn.disabled = false;
        reBtn.textContent = "переаналізувати";
      }
    });

    apply(total ? 0 : total); // no trace -> show everything
  }

  /**
   * Demo/Research path: register an arbitrary SQLite source through the
   * admin-gated endpoint, then poll the async init job until the engine
   * build finishes and switch the UI to the new domain.
   */
  async registerDomain() {
    const id = this.dom.regDomainId.value.trim();
    const dbPath = this.dom.regDbPath.value.trim();
    const name = this.dom.regName.value.trim();
    const token = this.dom.regAdminToken.value;
    const file = this.dom.regFile.files && this.dom.regFile.files[0];
    // Token stays optional client-side: loopback local-admin deployments
    // (PYBUDDY_LOCAL_ADMIN=1) accept empty; a hardened server answers 403
    // and the catch below points the user at the token field.
    if (!id || (!dbPath && !file)) {
      this.dom.regStatus.textContent = "Потрібні domain_id і шлях АБО файл джерела.";
      return;
    }
    this.dom.btnRegisterDomain.disabled = true;
    try {
      this.dom.regStatus.textContent = "Реєстрація та запуск ініціалізації…";
      let r;
      if (file) {
        // Multipart upload -- the API helper is JSON-only, so fetch directly.
        const fd = new FormData();
        fd.append("file", file);
        fd.append("domain_id", id);
        if (name) fd.append("name", name);
        fd.append("llm_assist", this.dom.regLlmAssist.checked ? "true" : "false");
        const res = await fetch("/api/domains/upload", {
          method: "POST",
          headers: { "X-Admin-Token": token },
          body: fd,
        });
        r = await res.json();
        if (!res.ok) throw new Error("HTTP " + res.status + " — " + (r.detail || res.statusText));
      } else {
        r = await this.api("/api/register_domain", {
          method: "POST",
          headers: { "X-Admin-Token": token },
          body: {
            domain_id: id,
            db_path: dbPath,
            name: name || null,
            async_init: true,
            llm_assist: !!this.dom.regLlmAssist.checked,
          },
          timeout: 30000,
        });
      }
      localStorage.setItem("pb_admin_token", token); // reuse elsewhere
      this.dom.regStatus.textContent = r.message || "Прийнято.";
      // Poll the async init job until the engine build resolves.
      const deadline = Date.now() + 10 * 60 * 1000;
      while (Date.now() < deadline) {
        await new Promise((res) => setTimeout(res, 3000));
        const jobs = await this.api("/api/init_jobs");
        const job = (jobs.jobs || []).find((j) => j.domain_id === id);
        if (!job || job.status === "done") {
          await this.fetchDomains();
          await this.switchDomain(id);
          this.dom.regStatus.textContent = "Ініціалізовано. Домен активний — інтроспекція оновлена.";
          this.fetchInspection();
          return;
        }
        if (job.status === "failed" || job.status === "cancelled") {
          this.dom.regStatus.textContent =
            "Ініціалізація " + job.status + ": " + (job.error || job.status);
          return;
        }
        const tr = job.trace || [];
        const last = tr.length ? tr[tr.length - 1] : null;
        this.dom.regStatus.textContent =
          "Ініціалізація триває… (" + (job.status || "queued") +
          (job.stage ? " · " + job.stage : "") + ")" +
          (last ? "\n🔨 крок " + tr.length + ": " + last.label : "");
      }
      this.dom.regStatus.textContent = "Таймаут очікування ініціалізації (10 хв).";
    } catch (err) {
      const msg = err && err.message || String(err);
      this.dom.regStatus.textContent = msg.startsWith("HTTP 403")
        ? "Помилка: 403 — сервер вимагає admin-токен (поле нижче)."
        : "Помилка: " + msg;
    } finally {
      this.dom.btnRegisterDomain.disabled = false;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Chat Export & Telemetry Logging                                    */
  /* ------------------------------------------------------------------ */
  exportChatMarkdown() {
    if (!this.state.chatHistory || this.state.chatHistory.length === 0) {
      this.showToast("⚠️ Діалог порожній для експорту");
      return;
    }
    const lines = [];
    lines.push("# Py-Buddy: Експериментальний лог сесії");
    lines.push(`- **Дата / Час:** ${new Date().toLocaleString("uk-UA")}`);
    lines.push(`- **Активний каталог:** ${this.domainName(this.state.domain)} (\`${this.state.domain}\`)`);
    lines.push(`- **Активна модель:** \`${this.state.model || "—"}\``);
    lines.push(`- **ID Сесії:** \`${this.state.sessionId}\``);
    lines.push(`- **Загальна вартість токенів:** $${Number(this.state.sessionCost || 0).toFixed(4)}`);
    lines.push(`- **Кількість повідомлень:** ${this.state.chatHistory.length}`);
    lines.push("\n---\n");

    let turnIndex = 1;
    for (const msg of this.state.chatHistory) {
      if (msg.role === "user") {
        lines.push(`### Повідомлення #${turnIndex} (Користувач)`);
        lines.push(`> ${msg.content}\n`);
      } else if (msg.role === "assistant") {
        const d = msg.data || {};
        lines.push(`### Відповідь #${turnIndex} (Консультант)`);
        lines.push(`**Фінальна відповідь:**`);
        lines.push(`> ${(d.final_answer || "").replace(/\n/g, "\n> ")}\n`);
        
        lines.push("**Телеметрія:**");
        const latencyMs = Math.round(Number(d.latency_seconds || 0) * 1000);
        lines.push(`- **Рівень маршрутизації:** \`${d.tier || "—"}\``);
        lines.push(`- **Модель:** \`${d.model || "—"}\``);
        lines.push(`- **Час виконання:** ${latencyMs} мс`);
        lines.push(`- **Вартість токенів:** $${Number(d.token_cost || 0).toFixed(4)}`);
        lines.push(`- **Zero-Hallucination Gate:** \`${d.termination_reason || "DETERMINISTIC"}\``);

        if (Array.isArray(d.recommended_items) && d.recommended_items.length) {
          lines.push("\n**Рекомендовані товари:**");
          const expCur = d.currency !== undefined ? d.currency : this.activeCurrency(d.domain);
          for (const it of d.recommended_items) {
            const pTxt = it.price != null ? " — " + this.formatPrice(it.price, expCur) : "";
            lines.push(`- ${it.name || it.id}${pTxt}`);
          }
        }

        if (Array.isArray(d.steps) && d.steps.length) {
          lines.push(`\n**Траєкторія ReAct (${d.steps.length} крок(и)):**`);
          for (const s of d.steps) {
            lines.push(`- **Крок ${s.step || 1}** · \`${s.tool_name || "Tool"}\``);
            if (s.thought) lines.push(`  - *Thought:* ${s.thought}`);
            if (s.action) {
              const actStr = typeof s.action === "string" ? s.action : JSON.stringify(s.action);
              lines.push(`  - *Action:* \`${actStr}\``);
            }
            if (s.observation) {
              const obsStr = typeof s.observation === "string" ? s.observation : JSON.stringify(s.observation);
              lines.push(`  - *Observation:* ${obsStr}`);
            }
          }
        }
        lines.push("\n---\n");
        turnIndex++;
      }
    }

    const text = lines.join("\n");
    this.copyToClipboard(text, "✓ Повний звіт діалогу (Markdown) скопійовано в буфер обміну!");
  }

  exportChatJson() {
    if (!this.state.chatHistory || this.state.chatHistory.length === 0) {
      this.showToast("⚠️ Діалог порожній для експорту");
      return;
    }
    const payload = {
      export_timestamp: new Date().toISOString(),
      session_id: this.state.sessionId,
      domain: this.state.domain,
      model: this.state.model,
      total_cost: this.state.sessionCost,
      history: this.state.chatHistory
    };
    const jsonStr = JSON.stringify(payload, null, 2);
    this.copyToClipboard(jsonStr, "✓ Сирі дані діалогу (JSON) скопійовано в буфер обміну!");
  }

  copyTurnMarkdown(index) {
    const item = this.state.chatHistory[index];
    if (!item || item.role !== "assistant") return;
    const d = item.data || {};
    const lines = [];
    lines.push(`### Крок ReAct консультанта Py-Buddy`);
    lines.push(`- **Модель:** \`${d.model || "—"}\``);
    lines.push(`- **Час виконання:** ${Math.round(Number(d.latency_seconds || 0) * 1000)} мс`);
    lines.push(`- **Рівень:** \`${d.tier || "—"}\``);
    lines.push(`- **Gate:** \`${d.termination_reason || "DETERMINISTIC"}\``);
    lines.push(`\n**Відповідь:**\n> ${(d.final_answer || "").replace(/\n/g, "\n> ")}`);

    if (Array.isArray(d.steps) && d.steps.length) {
      lines.push(`\n**Траєкторія інструментів:**`);
      for (const s of d.steps) {
        lines.push(`- **${s.tool_name || "Tool"}:**`);
        if (s.thought) lines.push(`  - Thought: ${s.thought}`);
        if (s.action) lines.push(`  - Action: ${JSON.stringify(s.action)}`);
        if (s.observation) lines.push(`  - Observation: ${JSON.stringify(s.observation)}`);
      }
    }
    this.copyToClipboard(lines.join("\n"), "✓ Траєкторію кроку скопійовано!");
  }

  async copyToClipboard(text, successMsg) {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
      }
      this.showToast(successMsg);
    } catch (err) {
      this.showToast("Помилка копіювання: " + String(err));
    }
  }

  showToast(message) {
    const container = document.getElementById("toast-container");
    if (!container) return;
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transition = "opacity 0.25s ease";
      setTimeout(() => toast.remove(), 250);
    }, 3200);
  }

  /* ------------------------------------------------------------------ */
  /* Model Selector Modal (Cline / Roo Code Style)                       */
  /* ------------------------------------------------------------------ */
  async openModelSelector() {
    if (!this.dom.modelModal) return;
    this.dom.modelModal.hidden = false;
    if (this.dom.modelSearchInput) {
      this.dom.modelSearchInput.focus();
    }
    if (!this.state.availableModels || this.state.availableModels.length === 0) {
      await this.fetchAvailableModels();
    } else {
      this.renderModelsList();
    }
  }

  closeModelSelector() {
    if (!this.dom.modelModal) return;
    this.dom.modelModal.hidden = true;
  }

  async fetchAvailableModels(retries = 1) {
    try {
      this.dom.modalModelsCount.textContent = "Завантаження каталогу моделей з OpenRouter...";
      const res = await this.api("/api/models");
      this.state.availableModels = res.models || [];
      // backend may answer with an empty catalogue while its model cache
      // is still warming up -- retry once instead of showing "0 з 0"
      if (this.state.availableModels.length === 0 && retries > 0) {
        await new Promise((r) => setTimeout(r, 2000));
        return this.fetchAvailableModels(retries - 1);
      }
      if (res.active_model) {
        this.state.model = res.active_model;
        this.dom.chipModelText.textContent = "model: " + res.active_model;
      }
      this.renderModelsList();
    } catch (err) {
      this.dom.modalModelsCount.textContent = "Помилка завантаження: " + (err.message || String(err));
    }
  }

  renderModelsList() {
    if (!this.dom.modalModelsList) return;
    const all = this.state.availableModels || [];
    const query = (this.state.modelSearchQuery || "").toLowerCase();
    const filter = this.state.activeModelFilter || "all";

    const filtered = all.filter((m) => {
      const id = (m.id || "").toLowerCase();
      const name = (m.name || "").toLowerCase();
      const desc = (m.description || "").toLowerCase();

      if (query && !id.includes(query) && !name.includes(query) && !desc.includes(query)) {
        return false;
      }

      if (filter === "free") {
        return Boolean(m.is_free);
      }
      if (filter !== "all") {
        return id.startsWith(filter + "/") || id.includes(filter);
      }
      return true;
    });

    this.dom.modalModelsCount.textContent = `Доступно: ${filtered.length} з ${all.length} моделей`;

    if (filtered.length === 0) {
      this.dom.modalModelsList.innerHTML = `
        <div style="padding: 28px; text-align: center; color: var(--muted); font-size: 13px;">
          Моделей за фільтром не знайдено
        </div>`;
      return;
    }

    const currentModel = this.state.model || "";

    const html = filtered.map((m) => {
      const isActive = m.id === currentModel;
      const ctxK = m.context_length ? `${Math.round(m.context_length / 1024)}k ctx` : "";
      const freeBadge = m.is_free ? '<span class="model-badge-free">FREE</span>' : "";
      const activeBadge = isActive ? '<span class="model-badge-active">✓ Активна</span>' : "";
      const priceStr = (!m.is_free && m.prompt_price != null) ? `$${m.prompt_price.toFixed(2)}/1M` : "";

      return `
        <div class="model-item ${isActive ? "active" : ""}" data-model-id="${this.escapeHtml(m.id)}">
          <div class="model-info-left">
            <div class="model-name-row">
              <span class="model-title-text">${this.escapeHtml(m.name || m.id)}</span>
              ${freeBadge}
              ${activeBadge}
            </div>
            <span class="model-id-text">${this.escapeHtml(m.id)}</span>
          </div>
          <div class="model-meta-right">
            ${priceStr ? `<span class="model-ctx-badge">${priceStr}</span>` : ""}
            ${ctxK ? `<span class="model-ctx-badge">${ctxK}</span>` : ""}
          </div>
        </div>
      `;
    }).join("");

    this.dom.modalModelsList.innerHTML = html;
  }

  async selectModel(modelId) {
    try {
      this.state.model = modelId;
      this.dom.chipModelText.textContent = "model: " + modelId;
      this.dom.sbModel.innerHTML = "<svg class='ic' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><rect x='2.5' y='4' width='19' height='13' rx='1.5'/><path d='M9 21h6M12 17v4'/></svg> " + modelId;
      this.dom.kpiModel.textContent = modelId;
      this.closeModelSelector();

      await this.api("/api/set_model", {
        method: "POST",
        body: { model: modelId }
      });

      this.showToast(`✓ Модель перемкнуто: ${modelId}`);
    } catch (err) {
      this.showToast(`Помилка зміни моделі: ${err.message || String(err)}`);
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.pyBuddyConsole = new PyBuddyConsole();
});


/* Custom select: рідний <select> лишається джерелом правди й отримує
   той самий 'change', а UI малюється своїм dropdown-ом. */
function skinSelect(sel) {
  if (sel._skinned) return;
  sel._skinned = true;
  sel.classList.add("sel-hidden");

  const wrap = document.createElement("div");
  wrap.className = "csel";
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "csel-btn";
  const list = document.createElement("div");
  list.className = "csel-list";
  list.hidden = true;
  wrap.appendChild(btn);
  wrap.appendChild(list);
  sel.parentNode.insertBefore(wrap, sel.nextSibling);

  const label = () => {
    const idx = sel.selectedIndex >= 0 ? sel.selectedIndex : 0;
    const o = sel.options[idx];
    btn.textContent = o ? o.textContent : "—";
    btn.disabled = Boolean(sel.disabled);
  };
  sel._refreshSkin = label;

  const close = () => { list.hidden = true; };
  const open = () => {
    label();
    list.innerHTML = "";
    Array.from(sel.options).forEach((o) => {
      const it = document.createElement("button");
      it.type = "button";
      it.className = "csel-opt" + (o.selected ? " active" : "");
      it.textContent = o.textContent;
      it.addEventListener("click", () => {
        sel.value = o.value;
        sel.dispatchEvent(new Event("change", { bubbles: true }));
        label(); close();
      });
      list.appendChild(it);
    });
    list.hidden = false;
  };

  btn.addEventListener("click", () => (list.hidden ? open() : close()));
  sel.addEventListener("change", label);
  new MutationObserver(label).observe(sel, { childList: true });
  document.addEventListener("click", (e) => { if (!wrap.contains(e.target)) close(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
  label();
}

function skinAllSelects(root) {
  (root || document).querySelectorAll("select").forEach(skinSelect);
}

document.addEventListener("DOMContentLoaded", () => {
  // селекти, що з'являються пізніше, обробляються спостерігачем
  skinAllSelects(document);
  new MutationObserver((muts) => {
    for (const m of muts) {
      for (const n of m.addedNodes) {
        if (n.nodeType !== 1) continue;
        if (n.tagName === "SELECT") skinSelect(n);
        else skinAllSelects(n);
      }
    }
  }).observe(document.body, { childList: true, subtree: true });
});

// Колесо миші над range-слайдером керує його значенням,
// а не скролить батьківську панель.
document.addEventListener("wheel", (e) => {
  const s = e.target.closest("input[type=range]");
  if (!s) return;
  e.preventDefault();
  const step = Number(s.step) || 1;
  const dir = e.deltaY < 0 ? 1 : -1;
  const min = Number(s.min) || 0, max = Number(s.max) || 100;
  s.value = Math.min(max, Math.max(min, (Number(s.value) || 0) + dir * step));
  s.dispatchEvent(new Event("input", { bubbles: true }));
  s.dispatchEvent(new Event("change", { bubbles: true }));
}, { passive: false });

/* Ресайз сайдбара перетягуванням роздільника; ширина зберігається. */
document.addEventListener("DOMContentLoaded", () => {
  const ws = document.querySelector(".workspace");
  const sb = document.querySelector(".sidebar");
  if (!ws || !sb) return;
  const rez = document.createElement("div");
  rez.className = "sb-resizer";
  sb.before(rez);
  const maxW = () => Math.min(900, Math.max(260, window.innerWidth - 320));
  const saved = parseInt(localStorage.getItem("pybuddy-sb-w") || "0", 10);
  if (saved) sb.style.width = Math.min(saved, maxW()) + "px";
  window.addEventListener("resize", () => {
    if (parseInt(sb.style.width, 10) > maxW()) sb.style.width = maxW() + "px";
  });
  let drag = false;
  rez.addEventListener("mousedown", (e) => {
    drag = true; e.preventDefault();
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
  });
  document.addEventListener("mousemove", (e) => {
    if (!drag) return;
    const w = Math.min(maxW(), Math.max(260,
      ws.getBoundingClientRect().right - e.clientX));
    sb.style.width = w + "px";
  });
  document.addEventListener("mouseup", () => {
    if (!drag) return;
    drag = false;
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
    const w = parseInt(sb.style.width, 10);
    if (w) localStorage.setItem("pybuddy-sb-w", String(w));
  });
});
