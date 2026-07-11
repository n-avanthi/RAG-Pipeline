import { useState, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:8000";

const TOPIC_COLORS = [
  "#60a5fa","#34d399","#f59e0b","#f87171",
  "#a78bfa","#22d3ee","#fb923c","#e879f9",
  "#4ade80","#38bdf8","#fbbf24","#f472b6",
];
const getColor = (id) => TOPIC_COLORS[id % TOPIC_COLORS.length];

function formatBytes(b) {
  if (b < 1024) return `${b} B`;
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1048576).toFixed(1)} MB`;
}

// ── Tooltip — smart positioning (vertical + horizontal) ───────────────────────
// On mouseEnter: measures available space above/below the anchor, opens toward
// whichever side has more room, and clamps horizontally within the viewport.
// HEADER_H must match the app-header height (50px).
const HEADER_H   = 50;
const TOOLTIP_W  = 248;
const TOOLTIP_H  = 160; // generous estimate — real height varies
const V_MARGIN   = 10;  // gap between anchor and tooltip box
const H_MARGIN   = 12;  // min distance from viewport edges

function Tooltip({ text, children }) {
  const [show,    setShow]    = useState(false);
  const [pos,     setPos]     = useState({ above: true, offsetX: 0 });
  const anchorRef = useRef(null);

  const handleEnter = () => {
    if (anchorRef.current) {
      const rect      = anchorRef.current.getBoundingClientRect();
      const spaceAbove = rect.top - HEADER_H;
      const spaceBelow = window.innerHeight - rect.bottom;
      // const above      = spaceAbove >= TOOLTIP_H + V_MARGIN || spaceAbove >= spaceBelow;
      const above = spaceAbove >= TOOLTIP_H + V_MARGIN && spaceAbove >= spaceBelow;

      // Horizontal clamp: keep box inside viewport
      const centerX   = rect.left + rect.width / 2;
      const leftEdge  = centerX - TOOLTIP_W / 2;
      const rightEdge = centerX + TOOLTIP_W / 2;
      let offsetX = 0;
      if (leftEdge  < H_MARGIN)                    offsetX =  H_MARGIN - leftEdge;
      else if (rightEdge > window.innerWidth - H_MARGIN) offsetX = (window.innerWidth - H_MARGIN) - rightEdge;

      setPos({ above, offsetX });
    }
    setShow(true);
  };

  // Arrow points toward the anchor — flips with the box
  const boxStyle = pos.above
    ? { bottom: `calc(100% + ${V_MARGIN}px)`, top: "auto",
        transform: `translateX(calc(-50% + ${pos.offsetX}px))` }
    : { top:    `calc(100% + ${V_MARGIN}px)`, bottom: "auto",
        transform: `translateX(calc(-50% + ${pos.offsetX}px))` };

  return (
    <span
      className="tooltip-wrap"
      ref={anchorRef}
      onMouseEnter={handleEnter}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <span
          className={`tooltip-box${pos.above ? "" : " tooltip-below"}`}
          style={boxStyle}
        >
          {text}
        </span>
      )}
    </span>
  );
}

// ── Confidence bar ────────────────────────────────────────────────────────────
function ConfidenceBar({ score, max = 1 }) {
  const pct   = Math.min(100, Math.round((score / (max || 1)) * 100));
  const color = pct > 66 ? "#34d399" : pct > 40 ? "#60a5fa" : "#f59e0b";
  const tier  = score >= 0.60 ? "strong" : score >= 0.45 ? "moderate" : "weak";
  return (
    <div className="conf-wrap">
      <div className="conf-track">
        <div className="conf-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="conf-label" title={`cosine similarity: ${score.toFixed(3)}`}>
        {score.toFixed(3)} <span style={{ opacity: 0.5, fontSize: "0.68em" }}>{tier}</span>
      </span>
    </div>
  );
}

// ── Source card ───────────────────────────────────────────────────────────────
function SourceCard({ source, index, maxScore }) {
  const [open, setOpen] = useState(false);
  const color   = getColor(source.cluster_id);
  const preview = source.text.length > 220 ? source.text.slice(0, 220) + "…" : source.text;

  return (
    <div className="source-card" style={{ "--src-color": color }}>
      <div className="source-header" onClick={() => setOpen(v => !v)}>
        <div className="source-meta-row">
          <span className="source-idx">{index + 1}</span>
          <span className="source-file">{source.source_doc}</span>
          {source.is_bridge_chunk && (
            <Tooltip text="This passage is relevant to multiple topic areas in your knowledge base.">
              <span className="bridge-tag">multi-topic</span>
            </Tooltip>
          )}
          <span className="source-topic" style={{ color }}>{source.cluster_theme}</span>
          <span className="source-chevron">{open ? "▾" : "▸"}</span>
        </div>
        <ConfidenceBar score={source.similarity_score ?? 0} max={maxScore} />
      </div>
      <div className={`source-body${open ? " source-body-open" : ""}`}>
        <p className="source-text">{open ? source.text : preview}</p>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// UPLOAD TAB
// ─────────────────────────────────────────────────────────────────────────────
function UploadTab({ onPipelineReady, pipelineRunning }) {
  const [documents,    setDocuments]    = useState([]);
  const [uploading,    setUploading]    = useState(false);
  const [uploadError,  setUploadError]  = useState("");
  const [dragging,     setDragging]     = useState(false);
  const [processing,   setProcessing]   = useState(pipelineRunning);
  const [done,         setDone]         = useState(false);
  const [processError, setProcessError] = useState("");
  const [logs,         setLogs]         = useState([]);
  const [currentStep,  setCurrentStep]  = useState("");
  const fileInputRef = useRef(null);
  const logsEndRef   = useRef(null);
  const sseRef       = useRef(null);

  useEffect(() => { logsEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [logs]);

  const fetchDocs = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/documents`);
      if (!r.ok) return;
      const d = await r.json();
      setDocuments(d.documents || []);
    } catch { }
  }, []);

  useEffect(() => { fetchDocs(); }, [fetchDocs]);

  useEffect(() => {
    if (!processing) return;
    const id = setInterval(async () => {
      try {
        const r = await fetch(`${API}/api/process/status`);
        const d = await r.json();
        if (d.step) setCurrentStep(d.step);
        if (d.done && !d.running) {
          setProcessing(false); setDone(true);
          clearInterval(id); sseRef.current?.close();
          onPipelineReady();
        }
        if (d.error) {
          setProcessError(d.error); setProcessing(false);
          clearInterval(id); sseRef.current?.close();
        }
      } catch { }
    }, 2000);
    return () => clearInterval(id);
  }, [processing, onPipelineReady]);

  const startSSE = () => {
    if (sseRef.current) sseRef.current.close();
    const es = new EventSource(`${API}/api/process/stream`);
    es.onmessage = (e) => {
      try {
        const entry = JSON.parse(e.data);
        setLogs(prev => [...prev.slice(-200), entry]);
        if (entry.msg?.startsWith("Step:")) setCurrentStep(entry.msg.replace("Step: ", ""));
      } catch { }
    };
    es.onerror = () => es.close();
    sseRef.current = es;
  };

  const uploadFiles = async (files) => {
    if (!files || files.length === 0) return;
    setUploadError(""); setUploading(true);
    const form = new FormData();
    Array.from(files).forEach(f => form.append("files", f));
    try {
      const r    = await fetch(`${API}/api/upload`, { method: "POST", body: form });
      const text = await r.text();
      let d;
      try { d = JSON.parse(text); } catch { throw new Error(`Server error: ${text.slice(0, 200)}`); }
      if (!r.ok) throw new Error(d.error || "Upload failed");
      setDocuments(d.documents || []);
      if (d.rejected?.length)
        setUploadError(`Skipped: ${d.rejected.map(x => `${x.name} (${x.reason})`).join(", ")}`);
    } catch (e) { setUploadError(e.message); }
    finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const deleteDoc = async (name) => {
    try {
      const r = await fetch(`${API}/api/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error);
      setDocuments(d.documents || []);
    } catch (e) { setUploadError(e.message); }
  };

  const processDocuments = async () => {
    setProcessError(""); setLogs([]); setCurrentStep(""); setDone(false); setProcessing(true);
    startSSE();
    try {
      const r    = await fetch(`${API}/api/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force_rerun: true }),
      });
      const text = await r.text();
      let d;
      try { d = JSON.parse(text); } catch { throw new Error(`Server error: ${text.slice(0, 200)}`); }
      if (!r.ok) throw new Error(d.error || "Failed to start pipeline");
    } catch (e) {
      setProcessError(e.message); setProcessing(false); sseRef.current?.close();
    }
  };

  const onDrop = (e) => { e.preventDefault(); setDragging(false); uploadFiles(e.dataTransfer.files); };

  const STEPS = [
    "Chunking documents",
    "Reducing dimensions (UMAP)",
    "Fitting GMM",
    "Generating cluster profiles (LLM)",
    "Loading pipeline into memory",
  ];
  const stepIdx = STEPS.findIndex(s => s === currentStep);

  return (
    <div className="page-body">
      <div className="page-header">
        <h2 className="page-title">Knowledge Base</h2>
        <p className="page-sub">Upload PDF or TXT documents. QuickRead indexes and organises them automatically.</p>
      </div>

      <div className={`dropzone${dragging ? " dz-over" : ""}`}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}>
        <div className="dz-inner">
          <div className="dz-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
          </div>
          <p className="dz-main">Drop files here</p>
          <p className="dz-hint">PDF · TXT · up to 200 MB</p>
          <button className="btn-ghost" onClick={() => fileInputRef.current?.click()} disabled={uploading || processing}>
            {uploading ? "Uploading…" : "Browse files"}
          </button>
        </div>
        <input ref={fileInputRef} type="file" accept=".pdf,.txt" multiple style={{ display: "none" }}
          onChange={e => uploadFiles(e.target.files)} />
      </div>

      {uploadError && <div className="alert-error">{uploadError}</div>}

      {documents.length > 0 && (
        <div className="file-list">
          <div className="file-list-header">{documents.length} document{documents.length !== 1 ? "s" : ""}</div>
          {documents.map(f => (
            <div key={f.name} className="file-row">
              <span className={`ext-pill ext-${f.extension}`}>{f.extension.toUpperCase()}</span>
              <span className="file-name">{f.name}</span>
              <span className="file-size">{formatBytes(f.size_bytes)}</span>
              <button className="file-del" onClick={() => deleteDoc(f.name)} disabled={processing}>✕</button>
            </div>
          ))}
        </div>
      )}

      <button className="btn-primary" onClick={processDocuments} disabled={documents.length === 0 || processing}>
        {processing ? <><span className="spin" /> Indexing…</> : done ? "Re-index documents" : "Build Knowledge Base"}
      </button>

      {documents.length === 0 && !processing && (
        <p className="muted-text">Upload at least one document to get started.</p>
      )}
      {processError && <div className="alert-error"><b>Error:</b> {processError}</div>}

      {(processing || done || logs.length > 0) && (
        <div className="log-panel">
          <div className="log-panel-top">
            <span>{done ? "✓ Indexing complete" : processing ? (currentStep || "Starting…") : "Log"}</span>
            {processing && <span className="spin-sm" />}
          </div>
          {processing && (
            <div className="step-list">
              {STEPS.map((s, i) => (
                <div key={s} className={`step-item ${i < stepIdx ? "s-done" : i === stepIdx ? "s-active" : "s-pending"}`}>
                  <span className="step-icon">{i < stepIdx ? "✓" : i === stepIdx ? "›" : "·"}</span>
                  <span>{s}</span>
                </div>
              ))}
            </div>
          )}
          <div className="log-scroll">
            {logs.length === 0 && <span className="log-dim">Waiting…</span>}
            {logs.map((e, i) => (
              <div key={i} className={`log-row log-${e.level || "info"}`}>
                <span className="log-ts">{e.ts}</span>
                <span>{e.msg}</span>
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SEARCH TAB
// ─────────────────────────────────────────────────────────────────────────────
function SearchTab({ pipelineReady, onGoToUpload, corpusStats }) {
  const [query,           setQuery]           = useState("");
  const [status,          setStatus]          = useState("idle");
  const [answer,          setAnswer]          = useState(null);
  const [displayedAnswer, setDisplayedAnswer] = useState("");
  const [sources,         setSources]         = useState([]);
  const [routing,         setRouting]         = useState(null);
  const [clusterProfiles, setClusterProfiles] = useState({});
  const topicSummaryRef = useRef("");
  const [errorMsg,        setErrorMsg]        = useState("");
  const [sourcesOpen,     setSourcesOpen]     = useState(false);
  const [relatedQs,       setRelatedQs]       = useState([]);
  const [relatedLoading,  setRelatedLoading]  = useState(false);
  const [history,         setHistory]         = useState([]);
  const [totalChunks,     setTotalChunks]     = useState(0);
  const answerRef = useRef(null);
  const rafRef    = useRef(null);

  useEffect(() => {
    if (!answer) return;
    setDisplayedAnswer("");
    let i = 0;
    const tick = () => {
      i = Math.min(i + 4, answer.length);
      setDisplayedAnswer(answer.slice(0, i));
      if (i < answer.length) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [answer]);

  useEffect(() => {
    if (status === "done") answerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [status]);

  // Fetch cluster profiles once — powers chip tooltips + topic summary
  useEffect(() => {
    if (!pipelineReady) return;
    (async () => {
      try {
        const r = await fetch(`${API}/api/clusters`);
        const d = await r.json();
        const lookup = {};
        (d.profiles || []).forEach(p => { lookup[p.cluster_id] = p; });
        setClusterProfiles(lookup);
        topicSummaryRef.current = (d.profiles || []).slice(0, 8)
          .map(p => {
            const ents = p.key_entities.filter(e => e !== "—").slice(0, 4).join(", ");
            return `- Theme: "${p.theme}" | Key concepts: ${ents} | Distinct edge: ${p.contrastive_edge?.slice(0, 120) ?? ""}`;
          }).join("\n");
      } catch { }
    })();
  }, [pipelineReady]);

  // Related questions routed through suggested_questions with cross-cluster constraint
  const fetchRelated = async (q) => {
    setRelatedLoading(true);
    try {
      const topicContext = topicSummaryRef.current
        ? `Original question: "${q}"\n\nKnowledge base topics:\n${topicSummaryRef.current}`
        : `Original question: "${q}"`;
      const r = await fetch(`${API}/api/suggested_questions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic_summary: topicContext }),
      });
      const d = await r.json();
      setRelatedQs((d.questions || []).slice(0, 3));
    } catch { setRelatedQs([]); }
    finally { setRelatedLoading(false); }
  };

  // ── RELEVANCE FILTER ───────────────────────────────────────────────────────
  // Drop sources whose similarity score is "weak" (< 0.45) AND whose cluster
  // doesn't match the primary retrieved content. Prevents the toxicity/LoRA
  // chunks appearing for a RAG-specific question. We keep at least 3 sources.
  const filterSources = (rawSources) => {
    if (!rawSources || rawSources.length <= 3) return rawSources;
    const WEAK_THRESHOLD = 0.45;
    // Find the max score to determine the dominant cluster
    const maxScore     = Math.max(...rawSources.map(s => s.similarity_score || 0));
    const dominantCluster = rawSources.find(s => s.similarity_score === maxScore)?.cluster_id;

    // Keep: score >= threshold OR same cluster as top result
    const filtered = rawSources.filter(s =>
      (s.similarity_score ?? 0) >= WEAK_THRESHOLD ||
      s.cluster_id === dominantCluster
    );
    // Always show at least 3
    return filtered.length >= 3 ? filtered : rawSources.slice(0, 3);
  };

  const submit = async (q) => {
    const trimmed = (typeof q === "string" ? q : query).trim();
    if (!trimmed || status === "loading") return;
    setQuery(trimmed);
    setStatus("loading");
    setAnswer(null); setDisplayedAnswer(""); setSources([]);
    setRouting(null); setErrorMsg(""); setSourcesOpen(false); setRelatedQs([]);
    try {
      const r = await fetch(`${API}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: trimmed, top_k: 3, diversity_lambda: 0.5, generate_answer: true }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || "Query failed");
      setAnswer(data.answer || "(No answer generated)");
      setRouting(data.routing);
      setTotalChunks(data.total_chunks || 0);
      const rawSources = (data.results || []).flatMap(c =>
        c.chunks.map(ch => ({ ...ch, cluster_id: c.cluster_id, cluster_theme: c.theme }))
      );
      setSources(filterSources(rawSources));
      setHistory(prev => [
        { query: trimmed, ts: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) },
        ...prev.filter(h => h.query !== trimmed),
      ].slice(0, 8));
      setStatus("done");
      fetchRelated(trimmed);
    } catch (e) { setErrorMsg(e.message); setStatus("error"); }
  };

  const maxScore = sources.length ? Math.max(...sources.map(s => s.similarity_score || 0)) : 1;

  const [suggested,        setSuggested]        = useState([]);
  const [suggestedLoading, setSuggestedLoading] = useState(true);

  useEffect(() => {
    if (!pipelineReady) return;
    (async () => {
      setSuggestedLoading(true);
      try {
        await new Promise(r => setTimeout(r, 400));
        const summary = topicSummaryRef.current;
        if (!summary) { setSuggested([]); return; }
        const r2 = await fetch(`${API}/api/suggested_questions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ topic_summary: summary }),
        });
        const d2 = await r2.json();
        setSuggested(d2.questions || []);
      } catch { setSuggested([]); }
      finally { setSuggestedLoading(false); }
    })();
  }, [pipelineReady]);

  if (!pipelineReady) {
    return (
      <div className="page-body">
        <div className="empty-state">
          <div className="empty-icon">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2">
              <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
          </div>
          <p className="empty-title">Knowledge base not ready</p>
          <p className="empty-sub">Upload and index your documents before searching.</p>
          <button className="btn-primary" style={{ width: "auto", padding: "10px 28px" }} onClick={onGoToUpload}>
            Set up Knowledge Base
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="search-layout">
      {/* Sidebar */}
      <aside className="search-sidebar">
        {corpusStats && (
          <div className="sidebar-stats">
            <div className="stat-row">
              <span className="stat-val">{corpusStats.chunks?.toLocaleString() ?? "—"}</span>
              <span className="stat-lbl">passages indexed</span>
            </div>
            <div className="stat-row">
              <span className="stat-val">{corpusStats.clusters ?? "—"}</span>
              <span className="stat-lbl">topic areas</span>
            </div>
          </div>
        )}
        <div className="sidebar-section">
          <div className="sidebar-label">{history.length > 0 ? "Recent searches" : "Try asking"}</div>
          {history.length > 0
            ? history.map((h, i) => (
                <button key={i} className="hist-btn" onClick={() => submit(h.query)}>
                  <span className="hist-q">{h.query.length > 52 ? h.query.slice(0, 52) + "\u2026" : h.query}</span>
                  <span className="hist-ts">{h.ts}</span>
                </button>
              ))
            : suggestedLoading
              ? [1,2,3].map(i => (
                  <div key={i} className="hist-btn hist-skeleton">
                    <span className="skel-line" style={{ width: `${60 + i * 10}%` }} />
                  </div>
                ))
              : suggested.map((q, i) => (
                  <button key={i} className="hist-btn" onClick={() => submit(q)}>
                    <span className="hist-q">{q.length > 52 ? q.slice(0, 52) + "\u2026" : q}</span>
                  </button>
                ))
          }
        </div>
      </aside>

      {/* Main */}
      <main className="search-main">
        <div className="page-body" style={{ maxWidth: "none" }}>
          <div className={`qbox${status === "loading" ? " qbox-busy" : ""}`}>
            <textarea
              className="qbox-ta"
              placeholder="Ask anything about your knowledge base…"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(); }}
              rows={3}
              disabled={status === "loading"}
            />
            <div className="qbox-footer">
              <span className="muted-text">⌘ Return to search</span>
              <button className="btn-primary btn-sm" onClick={() => submit()} disabled={!query.trim() || status === "loading"}>
                {status === "loading" ? <><span className="spin" /> Searching…</> : "Search"}
              </button>
            </div>
          </div>

          {status === "error" && <div className="alert-error">{errorMsg}</div>}

          {status === "done" && (
            <div className="results" ref={answerRef}>
              {routing && (
                <div className="routing-strip">
                  <span className="routing-lbl">Searched across</span>
                  {routing.selected_clusters.map(cid => {
                    const profile     = clusterProfiles[cid];
                    const tooltipText = profile?.contrastive_edge || routing.reasoning;
                    return (
                      <Tooltip key={cid} text={tooltipText}>
                        <span className="topic-chip" style={{ "--chip-color": getColor(cid) }}>
                          <span className="chip-dot" style={{ background: getColor(cid) }} />
                          {routing.cluster_themes[String(cid)] || `Topic ${cid}`}
                        </span>
                      </Tooltip>
                    );
                  })}
                  <span className="routing-meta">{totalChunks} passages · {routing.latency_s}s</span>
                </div>
              )}

              <div className="answer-card">
                <div className="answer-top">
                  <span className="answer-eyebrow">Answer</span>
                  <span className="answer-trust">
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                    {sources.length} source{sources.length !== 1 ? "s" : ""} · {routing?.selected_clusters?.length ?? 0} topic{routing?.selected_clusters?.length !== 1 ? "s" : ""}
                  </span>
                </div>
                <p className="answer-body">
                  {displayedAnswer}
                  {displayedAnswer.length < (answer?.length ?? 0) && <span className="cursor" />}
                </p>
              </div>

              {sources.length > 0 && (
                <div className="sources-section">
                  <button className="sources-toggle" onClick={() => setSourcesOpen(v => !v)}>
                    <span className="toggle-caret">{sourcesOpen ? "▾" : "▸"}</span>
                    {sourcesOpen ? "Hide" : "View"} {sources.length} source passage{sources.length !== 1 ? "s" : ""}
                  </button>
                  {sourcesOpen && (
                    <div className="sources-list">
                      {sources.map((s, i) => (
                        <SourceCard key={i} source={s} index={i} maxScore={maxScore} />
                      ))}
                    </div>
                  )}
                </div>
              )}

              {(relatedLoading || relatedQs.length > 0) && (
                <div className="related-block">
                  <div className="related-header">Explore further</div>
                  {relatedLoading && (
                    <div className="related-loading">
                      <span className="spin-sm" />
                      <span className="muted-text">Generating suggestions…</span>
                    </div>
                  )}
                  {relatedQs.map((q, i) => (
                    <button key={i} className="related-row" onClick={() => submit(q)}>
                      <span className="related-arrow">→</span>
                      <span>{q}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ROOT
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [tab,            setTab]            = useState("upload");
  const [pipelineReady,  setPipelineReady]  = useState(false);
  const [pipelineRunning,setPipelineRunning]= useState(false);
  const [docCount,       setDocCount]       = useState(0);
  const [corpusStats,    setCorpusStats]    = useState(null);

  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${API}/api/health`);
        const d = await r.json();
        setPipelineReady(d.pipeline_ready);
        setDocCount(d.documents_count ?? 0);
        setPipelineRunning(d.pipeline_running ?? false);
      } catch { }
    };
    check();
    const id = setInterval(check, 10_000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!pipelineReady) return;
    (async () => {
      try {
        const r2 = await fetch(`${API}/api/clusters`);
        const d2 = await r2.json();
        const total = (d2.profiles || []).reduce((s, p) => s + (p.n_members || 0), 0);
        setCorpusStats({ clusters: d2.n_clusters ?? "—", chunks: total || null });
      } catch { }
    })();
  }, [pipelineReady]);

  const handlePipelineReady = useCallback(() => {
    setPipelineReady(true);
    setTimeout(() => setTab("search"), 1200);
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
          </div>
          <span className="brand-name">QuickRead</span>
          <span className="brand-sep" />
          <span className="brand-tagline">Knowledge Search</span>
        </div>

        <nav className="nav">
          <button className={`nav-tab${tab === "upload" ? " nav-tab-active" : ""}`} onClick={() => setTab("upload")}>
            Knowledge Base
            {docCount > 0 && <span className="nav-badge">{docCount}</span>}
          </button>
          <button className={`nav-tab${tab === "search" ? " nav-tab-active" : ""}`} onClick={() => setTab("search")}>
            Search
            {pipelineReady && <span className="nav-ready" />}
          </button>
        </nav>

        <div className="header-right">
          {pipelineReady
            ? <div className="status-pill status-on"><span className="status-dot" />Ready</div>
            : <div className="status-pill status-off">Not indexed</div>
          }
        </div>
      </header>

      <div className="app-body">
        {tab === "upload" && <UploadTab onPipelineReady={handlePipelineReady} pipelineRunning={pipelineRunning} />}
        {tab === "search" && <SearchTab pipelineReady={pipelineReady} onGoToUpload={() => setTab("upload")} corpusStats={corpusStats} />}
      </div>
    </div>
  );
}