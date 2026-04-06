import { useState, useEffect, useRef, useCallback } from "react";
import "./index.css";

const API = "http://localhost:8000";

const CLUSTER_COLORS = [
  "#2563eb","#16a34a","#d97706","#dc2626",
  "#7c3aed","#0891b2","#ea580c","#db2777",
];
const getColor = (id) => CLUSTER_COLORS[id % CLUSTER_COLORS.length];

function formatBytes(b) {
  if (b < 1024) return `${b} B`;
  if (b < 1048576) return `${(b/1024).toFixed(1)} KB`;
  return `${(b/1048576).toFixed(1)} MB`;
}

// ── Upload Tab ────────────────────────────────────────────────────────────────
function UploadTab({ onPipelineReady, pipelineRunning }) {
  const [documents, setDocuments]       = useState([]);
  const [uploading, setUploading]       = useState(false);
  const [uploadError, setUploadError]   = useState("");
  const [dragging, setDragging]         = useState(false);
  const [processing, setProcessing]     = useState(pipelineRunning);
  const [done, setDone]                 = useState(false);
  const [processError, setProcessError] = useState("");
  const [logs, setLogs]                 = useState([]);
  const [currentStep, setCurrentStep]   = useState("");
  const fileInputRef = useRef(null);
  const logsEndRef   = useRef(null);
  const sseRef       = useRef(null);

  // Scroll logs to bottom on new entry
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const fetchDocs = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/documents`);
      if (!r.ok) return;
      const d = await r.json();
      setDocuments(d.documents || []);
    } catch { /* backend not up yet */ }
  }, []);

  useEffect(() => { fetchDocs(); }, [fetchDocs]);

  // Poll /api/process/status while processing
  useEffect(() => {
    if (!processing) return;
    const id = setInterval(async () => {
      try {
        const r = await fetch(`${API}/api/process/status`);
        const d = await r.json();
        if (d.step) setCurrentStep(d.step);
        if (d.done && !d.running) {
          setProcessing(false);
          setDone(true);
          clearInterval(id);
          sseRef.current?.close();
          onPipelineReady();
        }
        if (d.error) {
          setProcessError(d.error);
          setProcessing(false);
          clearInterval(id);
          sseRef.current?.close();
        }
      } catch { /* ignore */ }
    }, 2000);
    return () => clearInterval(id);
  }, [processing, onPipelineReady]);

  const startSSE = () => {
    if (sseRef.current) sseRef.current.close();
    const es = new EventSource(`${API}/api/process/stream`);
    es.onmessage = (e) => {
      try {
        const entry = JSON.parse(e.data);
        setLogs((prev) => [...prev.slice(-200), entry]);
        if (entry.msg?.startsWith("Step:")) {
          setCurrentStep(entry.msg.replace("Step: ", ""));
        }
      } catch { /* ignore malformed */ }
    };
    es.onerror = () => es.close();
    sseRef.current = es;
  };

  const uploadFiles = async (files) => {
    if (!files || files.length === 0) return;
    setUploadError("");
    setUploading(true);
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    try {
      const r    = await fetch(`${API}/api/upload`, { method: "POST", body: form });
      const text = await r.text();
      let d;
      try { d = JSON.parse(text); }
      catch { throw new Error(`Server error: ${text.slice(0, 200)}`); }
      if (!r.ok) throw new Error(d.error || "Upload failed");
      setDocuments(d.documents || []);
      if (d.rejected?.length)
        setUploadError(`Skipped: ${d.rejected.map((x) => `${x.name} (${x.reason})`).join(", ")}`);
    } catch (e) {
      setUploadError(e.message);
    } finally {
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
    setProcessError("");
    setLogs([]);
    setCurrentStep("");
    setDone(false);
    setProcessing(true);
    startSSE();
    try {
      const r    = await fetch(`${API}/api/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force_rerun: true }),
      });
      const text = await r.text();
      let d;
      try { d = JSON.parse(text); }
      catch { throw new Error(`Server error: ${text.slice(0, 200)}`); }
      if (!r.ok) throw new Error(d.error || "Failed to start pipeline");
      // Pipeline is now running in background — status polling handles the rest
    } catch (e) {
      setProcessError(e.message);
      setProcessing(false);
      sseRef.current?.close();
    }
  };

  const onDrop = (e) => {
    e.preventDefault(); setDragging(false);
    uploadFiles(e.dataTransfer.files);
  };

  return (
    <div className="tab-body">
      <h2 className="tab-title">Upload documents</h2>
      <p className="tab-sub">Add PDF or TXT files, then click <b>Process</b> to embed and cluster them.</p>

      {/* Drop zone */}
      <div
        className={`dropzone${dragging ? " dropzone-over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <div className="dz-content">
          <div className="dz-icon-wrap">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
          </div>
          <p className="dz-main">Drag files here</p>
          <p className="dz-sub">PDF and TXT · up to 200 MB per batch</p>
          <button
            className="btn-outline"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading || processing}
          >
            {uploading ? "Uploading…" : "Browse files"}
          </button>
        </div>
        <input ref={fileInputRef} type="file" accept=".pdf,.txt" multiple
          style={{ display: "none" }} onChange={(e) => uploadFiles(e.target.files)} />
      </div>

      {uploadError && <div className="alert alert-error">{uploadError}</div>}

      {/* File list */}
      {documents.length > 0 && (
        <div className="file-list">
          <div className="file-list-head">
            {documents.length} file{documents.length !== 1 ? "s" : ""} ready to process
          </div>
          {documents.map((f) => (
            <div key={f.name} className="file-row">
              <span className={`ext-badge ext-${f.extension}`}>{f.extension.toUpperCase()}</span>
              <span className="file-name">{f.name}</span>
              <span className="file-size">{formatBytes(f.size_bytes)}</span>
              <button className="file-del" onClick={() => deleteDoc(f.name)}
                disabled={processing} title="Remove">✕</button>
            </div>
          ))}
        </div>
      )}

      {/* Process button */}
      <button className="btn-primary" onClick={processDocuments}
        disabled={documents.length === 0 || processing}>
        {processing
          ? <><span className="spin" /> Processing…</>
          : done ? "Re-process documents" : "Process documents"}
      </button>

      {documents.length === 0 && !processing && (
        <p className="hint-text">Upload at least one document to continue.</p>
      )}

      {processError && <div className="alert alert-error"><b>Error:</b> {processError}</div>}

      {/* Live log panel */}
      {(processing || done || logs.length > 0) && (
        <div className="log-panel">
          <div className="log-panel-head">
            <span>{done ? "✓ Complete" : processing ? `Running — ${currentStep || "starting…"}` : "Log"}</span>
            {processing && <span className="spin spin-dark" />}
          </div>
          <div className="log-body">
            {logs.length === 0 && <span className="log-line log-dim">Waiting for output…</span>}
            {logs.map((entry, i) => (
              <div key={i} className={`log-line log-${entry.level || "info"}`}>
                <span className="log-ts">{entry.ts}</span>
                <span>{entry.msg}</span>
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}
    </div>
  );
}

// ── Chat Tab ──────────────────────────────────────────────────────────────────
function ChatTab({ pipelineReady, onGoToUpload }) {
  const [query, setQuery]             = useState("");
  const [status, setStatus]           = useState("idle");
  const [answer, setAnswer]           = useState(null);
  const [sources, setSources]         = useState([]);
  const [routing, setRouting]         = useState(null);
  const [errorMsg, setErrorMsg]       = useState("");
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const answerRef = useRef(null);

  useEffect(() => {
    if (status === "done") answerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [status]);

  const submit = async () => {
    const q = query.trim();
    if (!q || status === "loading") return;
    setStatus("loading");
    setAnswer(null); setSources([]); setRouting(null); setErrorMsg(""); setSourcesOpen(false);
    try {
      const r = await fetch(`${API}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: 3, diversity_lambda: 0.5, generate_answer: true }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || "Query failed");
      setAnswer(data.answer || "(No answer generated)");
      setRouting(data.routing);
      setSources((data.results || []).flatMap((c) =>
        c.chunks.map((ch) => ({ ...ch, cluster_id: c.cluster_id, cluster_theme: c.theme }))
      ));
      setStatus("done");
    } catch (e) {
      setErrorMsg(e.message);
      setStatus("error");
    }
  };

  if (!pipelineReady) {
    return (
      <div className="tab-body">
        <div className="empty-state">
          <p className="empty-title">No documents processed yet</p>
          <p className="empty-sub">Upload and process your documents before asking questions.</p>
          <button className="btn-primary" style={{ width: "auto", padding: "10px 24px" }} onClick={onGoToUpload}>
            Go to Upload
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="tab-body">
      <h2 className="tab-title">Chat with your documents</h2>
      <p className="tab-sub">Ask a question — the system retrieves relevant passages across your document clusters.</p>

      <div className={`query-box${status === "loading" ? " query-box-busy" : ""}`}>
        <textarea className="query-ta"
          placeholder="e.g. What are the economic and environmental impacts of EV batteries?"
          value={query} onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(); }}
          rows={3} disabled={status === "loading"} />
        <div className="query-footer">
          <span className="hint-text">⌘ Return to send</span>
          <button className="btn-primary btn-sm" onClick={submit}
            disabled={!query.trim() || status === "loading"}>
            {status === "loading" ? <><span className="spin" /> Thinking…</> : "Send"}
          </button>
        </div>
      </div>

      {status === "error" && <div className="alert alert-error">{errorMsg}</div>}

      {status === "done" && (
        <div className="result-area" ref={answerRef}>
          {routing && (
            <div className="routing-row">
              <span className="routing-label">Sources searched:</span>
              {routing.selected_clusters.map((cid) => (
                <span key={cid} className="cluster-pill"
                  style={{ borderColor: getColor(cid), color: getColor(cid) }}>
                  {routing.cluster_themes[String(cid)] || `Cluster ${cid}`}
                </span>
              ))}
              <span className="latency">{routing.latency_s}s</span>
            </div>
          )}

          <div className="answer-card">
            <div className="answer-label">Answer</div>
            <p className="answer-text">{answer}</p>
          </div>

          {sources.length > 0 && (
            <div className="sources-wrap">
              <button className="sources-toggle" onClick={() => setSourcesOpen((v) => !v)}>
                {sourcesOpen ? "▾" : "▸"} {sources.length} source{sources.length !== 1 ? "s" : ""} from {routing?.selected_clusters.length ?? 0} cluster{routing?.selected_clusters.length !== 1 ? "s" : ""}
              </button>
              {sourcesOpen && (
                <div className="sources-list">
                  {sources.map((s, i) => (
                    <div key={i} className="source-card" style={{ borderLeftColor: getColor(s.cluster_id) }}>
                      <div className="source-meta">
                        <span className="source-file">{s.source_doc}</span>
                        <span className="source-cluster" style={{ color: getColor(s.cluster_id) }}>{s.cluster_theme}</span>
                        <span className="source-sim">sim {s.similarity_score?.toFixed(3) ?? "—"}</span>
                        {s.is_bridge_chunk && <span className="bridge-tag">bridge</span>}
                      </div>
                      <p className="source-text">{s.text}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab]                 = useState("upload");
  const [pipelineReady, setPipelineReady] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [docCount, setDocCount]       = useState(0);

  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${API}/api/health`);
        const d = await r.json();
        setPipelineReady(d.pipeline_ready);
        setDocCount(d.documents_count ?? 0);
        setPipelineRunning(d.pipeline_running ?? false);
      } catch { /* backend not up */ }
    };
    check();
    const id = setInterval(check, 10_000);
    return () => clearInterval(id);
  }, []);

  const handlePipelineReady = () => {
    setPipelineReady(true);
    setTimeout(() => setTab("chat"), 1500);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <div className="logo-dot" />
          <span className="app-name">DocChat</span>
        </div>
        <nav className="tabs">
          <button className={`tab${tab === "upload" ? " tab-active" : ""}`}
            onClick={() => setTab("upload")}>
            Upload
            {docCount > 0 && <span className="tab-badge">{docCount}</span>}
          </button>
          <button className={`tab${tab === "chat" ? " tab-active" : ""}`}
            onClick={() => setTab("chat")}>
            Chat
            {pipelineReady && <span className="ready-dot" />}
          </button>
        </nav>
      </header>

      <main className="main">
        {tab === "upload"
          ? <UploadTab onPipelineReady={handlePipelineReady} pipelineRunning={pipelineRunning} />
          : <ChatTab pipelineReady={pipelineReady} onGoToUpload={() => setTab("upload")} />
        }
      </main>
    </div>
  );
}