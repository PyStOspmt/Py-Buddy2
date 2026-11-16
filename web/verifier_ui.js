/**
 * Sentence-level ClaimVerifier, numeric provenance, and NVFG relaxation UI.
 * Pure presentation helpers over authoritative backend payloads
 * (`verification_report`, `pre_verifier_response`, `recommended_items`, `nvfg`).
 */
(function (global) {
  "use strict";

  function normNumStr(raw) {
    return String(raw || "")
      .replace(/[\s\u00a0\u202f]+/g, "")
      .replace(",", ".");
  }

  function renderVerifierBadge(data) {
    if (!data || typeof data !== "object") return "";
    const vr = data.verification_report;
    if (!vr) {
      if (data.tier === "tier1_fast_path" || data.termination_reason === "DETERMINISTIC_HYPERGRAPH") {
        return '<span class="badge cv-pill is-det" title="Детермінований шлях без вільної прози LLM">◆ Верифікатор: без прози</span>';
      }
      return "";
    }
    const sup = Number(vr.n_supported || 0);
    const unsup = Number(vr.n_unsupported || 0);
    const noClaims = Number(vr.n_no_claims || 0);
    if (unsup > 0) {
      return (
        `<span class="badge cv-pill is-stripped" title="Пореченнєвий верифікатор заблокував/зрізав непідтверджені речення">` +
        `✂ ВЕРИФІКАТОР: ${sup} перевірено · ${unsup} зрізано</span>`
      );
    }
    if (sup > 0) {
      return (
        `<span class="badge cv-pill is-ok" title="Кожне фактичне речення підтверджено рядками цього рану">` +
        `✓ ВЕРИФІКАТОР: ${sup} claims перевірено · 0 зрізано</span>`
      );
    }
    return (
      `<span class="badge cv-pill is-neutral" title="У прозі немає числових або spec-токенів (${noClaims} реч.)">` +
      `○ ВЕРИФІКАТОР: ${noClaims} реч. без claims · 0 зрізано</span>`
    );
  }

  function buildProvenanceIndex(items, domain) {
    const map = new Map();
    if (!Array.isArray(items)) return map;
    const tbl = domain ? `${domain} · row` : "row";
    for (const item of items) {
      if (!item || typeof item !== "object") continue;
      const name = String(item.name || item.id || "Позиція");
      const idPart = item.id != null ? ` · #${item.id}` : "";
      const addKey = (val, label) => {
        if (val === null || val === undefined || val === "") return;
        const n = Number(val);
        if (!Number.isFinite(n)) return;
        const tip = `${name}${idPart} (${tbl}) ← ${label}: ${n}`;
        for (const k of [String(n), n.toFixed(2), String(Math.round(n))]) {
          if (!map.has(k)) map.set(k, tip);
        }
      };
      if (item.price != null) addKey(item.price, "price");
      for (const [k, v] of Object.entries(item)) {
        if (k === "id" || k === "price" || k === "name" || k === "category") continue;
        if (typeof v === "number" && Number.isFinite(v)) addKey(v, k);
      }
      if (item.attributes && typeof item.attributes === "object") {
        for (const [ak, av] of Object.entries(item.attributes)) {
          if (typeof av === "number" && Number.isFinite(av)) addKey(av, ak);
        }
      }
    }
    return map;
  }

  function annotateProvenance(renderedHtml, items, domain, esc) {
    if (!renderedHtml || !Array.isArray(items) || !items.length) return renderedHtml;
    const prov = buildProvenanceIndex(items, domain);
    if (!prov.size) return renderedHtml;

    // Split by HTML tags so attributes/tags themselves are never mutated.
    const numRe = /\b\d{1,3}(?:[\s\u00a0\u202f]\d{3})*(?:[.,]\d+)?\b|\b\d+(?:[.,]\d+)?\b/g;
    return renderedHtml
      .split(/(<[^>]+>)/g)
      .map((chunk) => {
        if (!chunk || chunk[0] === "<") return chunk;
        return chunk.replace(numRe, (match) => {
          const norm = normNumStr(match);
          const n = Number(norm);
          const tip = prov.get(norm) || (Number.isFinite(n) ? prov.get(String(n)) : null);
          if (!tip) return match;
          return `<span class="prov-num" title="Прованієнс: ${esc(tip)}">${match}</span>`;
        });
      })
      .join("");
  }

  function renderRelaxationCard(data, esc) {
    const nv = data && data.nvfg;
    let stepRes = null;
    if (Array.isArray(data && data.steps)) {
      for (let i = data.steps.length - 1; i >= 0; i--) {
        const s = data.steps[i];
        if (s && s.tool_name === "pareto_relax" && s.result && typeof s.result === "object") {
          stepRes = s.result;
          break;
        }
      }
    }
    const status = (nv && nv.status) || (stepRes && stepRes.nvfg_status) || null;
    const relaxed = Boolean((nv && nv.relaxed) || (stepRes && stepRes.relaxed));
    const prot = (nv && nv.protected && nv.protected.length ? nv.protected : null) ||
      (stepRes && Array.isArray(stepRes.protected) ? stepRes.protected : []);
    if (!relaxed && status !== "UNSATISFIABLE" && !(prot && prot.length)) {
      return "";
    }

    const relText = (nv && nv.relaxation_text) || (stepRes && stepRes.relaxation_text) || "";
    const relObj = (nv && nv.relaxation) || (stepRes && stepRes.relaxation) || null;
    const relReg = (nv && typeof nv.relative_regret === "number")
      ? nv.relative_regret
      : (stepRes && typeof stepRes.relative_regret === "number" ? stepRes.relative_regret : null);

    let relaxedSummary = relText;
    if (!relaxedSummary && relObj && Array.isArray(relObj.steps) && relObj.steps.length) {
      relaxedSummary = relObj.steps
        .map((st) => `${st.name || st.attr}: ${st.old_bound} → ${st.new_bound}${st.unit ? " " + st.unit : ""}`)
        .join("; ");
    }
    if (!relaxedSummary && status === "UNSATISFIABLE") {
      relaxedSummary = "Жодна допустима релаксація не задовольняє захищені вимоги (чесна відмова).";
    }

    const cls = status === "UNSATISFIABLE" ? "is-unsat" : "is-relaxed";
    const title = status === "UNSATISFIABLE"
      ? "⛔ Нерозв’язний запит (Hard Constraints)"
      : "⚖ Спільна релаксація обмежень (NVFG)";

    return (
      `<div class="relax-card ${cls}">` +
      `<div class="relax-head"><span>${title}</span>` +
      `<span class="badge">${esc(status || "RELAXED")}</span></div>` +
      `<div class="relax-grid">` +
      (relaxedSummary
        ? `<div class="relax-row"><span class="relax-k">Послаблено:</span><span class="relax-v">${esc(relaxedSummary)}</span></div>`
        : "") +
      (relReg !== null
        ? `<div class="relax-row"><span class="relax-k">Regret (втрата):</span><span class="relax-v">${(relReg * 100).toFixed(2)} %</span></div>`
        : "") +
      `<div class="relax-row"><span class="relax-k">Захищені (hard):</span><span class="relax-v">${
        prot && prot.length ? esc(prot.join(", ")) : "не зафіксовано"
      }</span></div>` +
      `</div></div>`
    );
  }

  function renderVerificationPanel(data, esc, renderMd) {
    const vr = data && data.verification_report;
    const rawPre = data && data.pre_verifier_response;
    if (!vr && !rawPre) return "";

    const sup = vr ? Number(vr.n_supported || 0) : 0;
    const unsup = vr ? Number(vr.n_unsupported || 0) : 0;
    const noClaims = vr ? Number(vr.n_no_claims || 0) : 0;
    const frac = vr && typeof vr.supported_fraction === "number"
      ? `${Math.round(vr.supported_fraction * 100)}%`
      : "—";

    const sents = (vr && Array.isArray(vr.sentences) ? vr.sentences : []).map((sv) => {
      const verdict = sv.verdict || "no_claims";
      const txt = esc(sv.text || "");
      if (verdict === "unsupported") {
        const tokens = Array.isArray(sv.unsupported) && sv.unsupported.length
          ? `<div class="cv-tokens">непідтверджені claims: ${sv.unsupported.map((u) => `<code>${esc(u)}</code>`).join(" ")}</div>`
          : "";
        return (
          `<div class="cv-sent is-unsupported">` +
          `<span class="cv-tag">✂ Зрізано</span>` +
          `<div class="cv-sent-body"><del class="cv-stripped">${txt}</del>${tokens}</div>` +
          `</div>`
        );
      }
      if (verdict === "verified") {
        const uniqProds = Array.isArray(sv.products) ? [...new Set(sv.products)] : [];
        const prods = uniqProds.length
          ? `<div class="cv-prods">рядок джерела: ${uniqProds.map((p) => `<span>${esc(p)}</span>`).join(", ")}</div>`
          : "";
        return (
          `<div class="cv-sent is-verified">` +
          `<span class="cv-tag">✓ Перевірено</span>` +
          `<div class="cv-sent-body"><span>${txt}</span>${prods}</div>` +
          `</div>`
        );
      }
      return (
        `<div class="cv-sent is-neutral">` +
        `<span class="cv-tag">○ Без claims</span>` +
        `<div class="cv-sent-body"><span>${txt}</span></div>` +
        `</div>`
      );
    }).join("");

    let diffBlock = "";
    if (rawPre && String(rawPre).trim()) {
      const rawHtml = highlightPreResponse(rawPre, vr && vr.sentences, esc);
      const finalHtml = renderMd ? renderMd(data.final_answer || "") : `<p>${esc(data.final_answer || "")}</p>`;
      diffBlock =
        `<div class="cv-diff-wrap">` +
        `<button type="button" class="btn btn-ghost btn-xs cv-diff-toggle" data-toggle-raw>` +
        `Показати необрізану відповідь (ДО vs ПІСЛЯ)</button>` +
        `<div class="cv-diff-grid" hidden>` +
        `<div class="cv-diff-col is-before"><div class="cv-diff-title">Сира відповідь моделі (ДО верифікатора)</div><div class="cv-diff-body">${rawHtml}</div></div>` +
        `<div class="cv-diff-col is-after"><div class="cv-diff-title">Фінальна відповідь клієнту (ПІСЛЯ)</div><div class="cv-diff-body">${finalHtml}</div></div>` +
        `</div></div>`;
    }

    const openAttr = unsup > 0 ? " open" : "";
    const sumCls = unsup > 0 ? "is-warn" : "is-ok";
    const summaryLabel = unsup > 0
      ? `✂ Пореченнєвий верифікатор: ${sup} підтверджено · ${unsup} зрізано (точність claims ${frac})`
      : `✓ Пореченнєвий верифікатор: ${sup} підтверджено · ${noClaims} без claims · 0 зрізано`;

    return (
      `<details class="cv-panel ${sumCls}"${openAttr}>` +
      `<summary class="cv-summary">${summaryLabel}</summary>` +
      `<div class="cv-body">${sents || '<div class="list-note">Звіт без розбивки на речення.</div>'}${diffBlock}</div>` +
      `</details>`
    );
  }

  function formatInlinePlain(text, esc) {
    return esc(text || "")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  function highlightPreResponse(rawText, sentences, esc) {
    if (!Array.isArray(sentences) || !sentences.length) {
      return `<div class="cv-diff-line">${formatInlinePlain(rawText, esc)}</div>`;
    }
    return sentences.map((sv) => {
      const t = formatInlinePlain((sv.text || "").replace(/\s*\r?\n\s*/g, " "), esc);
      if (sv.verdict === "unsupported") {
        return `<div class="cv-diff-line"><del class="cv-stripped">${t}</del></div>`;
      }
      if (sv.verdict === "verified") {
        return `<div class="cv-diff-line"><span class="cv-raw-ok">${t}</span></div>`;
      }
      return `<div class="cv-diff-line"><span>${t}</span></div>`;
    }).join("");
  }

  global.PyBuddyVerifierUI = {
    renderVerifierBadge,
    annotateProvenance,
    renderRelaxationCard,
    renderVerificationPanel,
  };
})(window);
