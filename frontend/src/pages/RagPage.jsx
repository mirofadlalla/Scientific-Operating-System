import { useState, useEffect, useRef } from 'react';
import { BACKEND_URL } from '../config';

const STEP_KEYS = ['upload', 'chunk', 'embed', 'index', 'reload'];
const STEP_LABELS = {
  upload: '📤  Uploading file to server',
  chunk:  '✂️  Chunking document',
  embed:  '🔢  Generating embeddings',
  index:  '📦  Indexing into vector store',
  reload: '🔄  Reloading query engine',
};

const STATUS_TO_STEP = {
  pending:   null,
  reading:   'upload',
  chunking:  'chunk',
  embedding: 'embed',
  indexing:  'index',
  reloading: 'reload',
  completed: 'done',
  failed:    'failed',
};

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Step pipeline UI ─────────────────────────────────────────────────────────
function StepPipeline({ steps }) {
  return (
    <div className="kb-steps">
      {STEP_KEYS.map(key => {
        const state = steps[key] || 'idle';
        return (
          <div key={key} className={`kb-step ${state}`}>
            <div className="kb-step-icon">
              {state === 'active' && <span className="spin">⟳</span>}
              {state === 'done'  && '✓'}
              {state === 'error' && '✗'}
              {state === 'idle'  && '○'}
            </div>
            <div className="kb-step-label">{STEP_LABELS[key]}</div>
          </div>
        );
      })}
    </div>
  );
}

// ── KB Status panel ──────────────────────────────────────────────────────────
function KBStatus({ status }) {
  const online = status?.weaviate_connected === true;
  const ready  = status?.engine_ready === true;
  const nodes  = typeof status?.node_count === 'number' ? status.node_count : '–';

  const dotClass = !online ? 'offline' : ready ? 'online' : 'loading';

  return (
    <div>
      <div className="kb-stat-grid">
        <div className="kb-stat">
          <div className="val">{nodes}</div>
          <div className="lbl">Nodes</div>
        </div>
        <div className="kb-stat">
          <div className="val" style={{ fontSize: 18 }}>{ready ? '✅' : online ? '…' : '❌'}</div>
          <div className="lbl">Engine</div>
        </div>
        <div className="kb-stat">
          <div className="val" style={{ fontSize: 14, color: 'var(--text-sec)' }}>
            {status?.search_mode?.split(' ')[0] || '–'}
          </div>
          <div className="lbl">Mode</div>
        </div>
      </div>

      <div className="kb-status-row">
        <div className={`kb-dot ${dotClass}`} />
        <span style={{ color: 'var(--text-sec)', fontSize: 13 }}>
          {online
            ? `RAG: ${ready ? 'Ready' : 'Initialising'} · ${nodes} nodes indexed`
            : 'RAG: Offline — Weaviate not connected'}
        </span>
      </div>
    </div>
  );
}

// ── Ingestion log entry ──────────────────────────────────────────────────────
function LogEntry({ entry }) {
  return (
    <div className={`log-entry ${entry.type}`}>
      <div>{entry.message}</div>
      <div className="log-time">{entry.time}</div>
    </div>
  );
}

// ── Main RAG Page ─────────────────────────────────────────────────────────────
export default function RagPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [strategy, setStrategy]         = useState('markdown');
  const [uploading, setUploading]        = useState(false);
  const [steps, setSteps]               = useState({});
  const [kbStatus, setKbStatus]         = useState(null);
  const [log, setLog]                   = useState([]);
  const fileInputRef                    = useRef(null);
  const dropRef                         = useRef(null);

  // Fetch KB status on mount and every 30s
  useEffect(() => {
    fetchKBStatus();
    const id = setInterval(fetchKBStatus, 30_000);
    return () => clearInterval(id);
  }, []);

  const fetchKBStatus = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/rag/status`);
      if (res.ok) setKbStatus(await res.json());
    } catch { /* silent */ }
  };

  const addLog = (type, message) => {
    const time = new Date().toLocaleTimeString();
    setLog(prev => [{ type, message, time }, ...prev].slice(0, 20));
  };

  // ── Drag & drop ─────────────────────────────────────────────────────────
  const onDragOver  = (e) => { e.preventDefault(); dropRef.current?.classList.add('drag-over'); };
  const onDragLeave = ()  => dropRef.current?.classList.remove('drag-over');
  const onDrop      = (e) => {
    e.preventDefault();
    dropRef.current?.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) selectFile(f);
  };

  const selectFile = (file) => {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['md', 'txt'].includes(ext)) {
      addLog('error', `❌ Unsupported file type .${ext} — only .md and .txt allowed`);
      return;
    }
    setSelectedFile(file);
    setSteps({});
  };

  // ── Upload & poll ─────────────────────────────────────────────────────────
  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setSteps({ upload: 'active' });
    addLog('', `📤 Starting ingestion: ${selectedFile.name}`);

    const form = new FormData();
    form.append('file', selectedFile, selectedFile.name);
    form.append('strategy', strategy);

    try {
      const res    = await fetch(`${BACKEND_URL}/rag/ingest`, { method: 'POST', body: form });
      const result = await res.json();

      if (!res.ok || result.status !== 'success') {
        setSteps({ upload: 'error' });
        addLog('error', `❌ ${result.detail || result.message || 'Upload failed'}`);
        return;
      }

      const jobId = result.job_id;
      let done = false;

      while (!done) {
        await sleep(600);
        const sr   = await fetch(`${BACKEND_URL}/rag/ingest/status/${jobId}`);
        const data = await sr.json();
        const step = STATUS_TO_STEP[data.status];

        if (data.status === 'pending' || data.status === 'reading') {
          setSteps({ upload: 'active' });
        } else if (data.status === 'chunking') {
          setSteps({ upload: 'done', chunk: 'active' });
        } else if (data.status === 'embedding') {
          setSteps({ upload: 'done', chunk: 'done', embed: 'active' });
        } else if (data.status === 'indexing') {
          setSteps({ upload: 'done', chunk: 'done', embed: 'done', index: 'active' });
        } else if (data.status === 'reloading') {
          setSteps({ upload: 'done', chunk: 'done', embed: 'done', index: 'done', reload: 'active' });
        } else if (data.status === 'completed') {
          setSteps({ upload: 'done', chunk: 'done', embed: 'done', index: 'done', reload: 'done' });
          addLog('success', `✅ ${data.filename} — ${data.nodes_created} nodes · strategy: ${data.strategy}`);
          done = true;
          await fetchKBStatus();
        } else if (data.status === 'failed') {
          setSteps(prev => {
            const upd = { ...prev };
            const active = Object.keys(upd).find(k => upd[k] === 'active');
            if (active) upd[active] = 'error';
            return upd;
          });
          addLog('error', `❌ Ingestion failed: ${data.error_message || 'unknown error'}`);
          done = true;
        }
      }
    } catch (err) {
      setSteps({ upload: 'error' });
      addLog('error', `❌ Network error: ${err.message}`);
    } finally {
      setUploading(false);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <div className="rag-page">
      <div className="rag-header">
        <h2>🗄️ Knowledge Base</h2>
        <p>Upload documents to expand the RAG knowledge base. Supports Markdown (.md) and plain text (.txt).</p>
      </div>

      <div className="rag-grid">
        {/* ── Left: Upload ── */}
        <div>
          <div className="rag-card">
            <div className="rag-card-title">📤 Ingest Document</div>

            {/* Dropzone */}
            <div
              className="dropzone"
              ref={dropRef}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="dz-icon">📄</div>
              <div className="dz-label">
                Drag & drop a file here<br />or click to browse
              </div>
              <div className="dz-sub">.md · .txt · max 10 MB</div>
            </div>
            <input
              type="file"
              accept=".md,.txt"
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={e => e.target.files[0] && selectFile(e.target.files[0])}
            />

            {selectedFile && (
              <div className="selected-file">
                <span className="file-icon">📝</span>
                <span>{selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)</span>
                <span className="file-remove" onClick={() => setSelectedFile(null)}>✕</span>
              </div>
            )}

            {/* Strategy */}
            <select
              className="strategy-select"
              value={strategy}
              onChange={e => setStrategy(e.target.value)}
            >
              <option value="markdown">Markdown (semantic sections)</option>
              <option value="sentence">Sentence (natural boundaries)</option>
              <option value="token">Token (fixed-size chunks)</option>
            </select>

            <button
              className="upload-btn"
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
            >
              {uploading ? '⟳ Ingesting…' : '🚀 Start Ingestion'}
            </button>

            {/* Pipeline steps */}
            {Object.keys(steps).length > 0 && (
              <>
                <div style={{ marginTop: 20, marginBottom: 8, fontSize: 12, color: 'var(--text-sec)', fontFamily: 'var(--mono)', textTransform: 'uppercase', letterSpacing: '.5px' }}>
                  Pipeline Progress
                </div>
                <StepPipeline steps={steps} />
              </>
            )}
          </div>

          {/* Ingestion log */}
          {log.length > 0 && (
            <div className="rag-card" style={{ marginTop: 20 }}>
              <div className="rag-card-title">📋 Ingestion Log</div>
              <div className="ingest-log">
                {log.map((entry, i) => <LogEntry key={i} entry={entry} />)}
              </div>
            </div>
          )}
        </div>

        {/* ── Right: KB Status ── */}
        <div>
          <div className="rag-card">
            <div className="rag-card-title">📊 Knowledge Base Status</div>
            <KBStatus status={kbStatus} />
          </div>

          {/* Index info */}
          {kbStatus?.index_name && (
            <div className="rag-card" style={{ marginTop: 20 }}>
              <div className="rag-card-title">🗃️ Index Info</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 4 }}>
                {[
                  ['Index Name', kbStatus.index_name],
                  ['Search Mode', kbStatus.search_mode || '–'],
                  ['Weaviate', kbStatus.weaviate_connected ? '✅ Connected' : '❌ Offline'],
                  ['Engine Ready', kbStatus.engine_ready ? '✅ Ready' : '⏳ Loading'],
                ].map(([label, val]) => (
                  <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
                    <span style={{ color: 'var(--text-sec)' }}>{label}</span>
                    <span style={{ fontFamily: 'var(--mono)', color: 'var(--text-prim)' }}>{val}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tips */}
          <div className="rag-card" style={{ marginTop: 20 }}>
            <div className="rag-card-title">💡 Chunking Strategy Guide</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 8 }}>
              {[
                { name: 'Markdown', desc: 'Best for documentation, README files, and structured content with headers.' },
                { name: 'Sentence', desc: 'Best for research papers, articles, and natural language content.' },
                { name: 'Token', desc: 'Best for code, logs, or content where consistent chunk sizes matter.' },
              ].map(s => (
                <div key={s.name} style={{ padding: '10px 14px', background: 'rgba(255,255,255,.02)', border: '1px solid var(--border)', borderRadius: 10 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--accent)', marginBottom: 4 }}>{s.name}</div>
                  <div style={{ fontSize: 12, color: 'var(--text-sec)', lineHeight: 1.6 }}>{s.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
