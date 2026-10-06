import { useState, useEffect, useRef } from 'react';
import {
  LoginGate,
  StepPipeline,
  KBStatus,
  LogEntry,
} from '../components/RagComponents';
import {
  getRAGStatus,
  ingestDocument,
  getIngestionStatus,
} from '../services/api';

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

export default function RagPage() {
  const [token, setToken] = useState(null);
  const [loggedInUser, setLoggedInUser] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [strategy, setStrategy] = useState('markdown');
  const [uploading, setUploading] = useState(false);
  const [steps, setSteps] = useState({});
  const [kbStatus, setKbStatus] = useState(null);
  const [log, setLog] = useState([]);
  const fileInputRef = useRef(null);
  const dropRef = useRef(null);

  const handleLogin = (accessToken, username) => {
    setToken(accessToken);
    setLoggedInUser(username);
  };

  const handleLogout = () => {
    setToken(null);
    setLoggedInUser(null);
  };

  const fetchKBStatus = async () => {
    try {
      const data = await getRAGStatus();
      setKbStatus(data);
    } catch {
      /* silent */
    }
  };

  useEffect(() => {
    if (!token) return undefined;
    fetchKBStatus();
    const id = setInterval(fetchKBStatus, 30_000);
    return () => clearInterval(id);
  }, [token]);

  // Show login gate if not authenticated
  if (!token) return <LoginGate onLogin={handleLogin} />;

  const addLog = (type, message) => {
    const time = new Date().toLocaleTimeString();
    setLog(prev => [{ type, message, time }, ...prev].slice(0, 20));
  };

  // Drag & drop
  const onDragOver = (e) => {
    e.preventDefault();
    dropRef.current?.classList.add('drag-over');
  };

  const onDragLeave = () => {
    dropRef.current?.classList.remove('drag-over');
  };

  const onDrop = (e) => {
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

  // Upload & poll
  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setSteps({ upload: 'active' });
    addLog('', `📤 Starting ingestion: ${selectedFile.name}`);

    try {
      const result = await ingestDocument(selectedFile, strategy, token);
      const jobId = result.job_id;
      let done = false;

      while (!done) {
        await sleep(600);
        const data = await getIngestionStatus(jobId);

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
      addLog('error', `❌ ${err.message || 'Network error'}`);
    } finally {
      setUploading(false);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <div className="rag-page">
      <div className="rag-header">
        <div className="rag-header-top">
          <h2>🗄️ Knowledge Base</h2>
          <div className="rag-auth-badge">
            <span className="auth-user">👤 {loggedInUser}</span>
            <button className="logout-btn" onClick={handleLogout}>Sign Out</button>
          </div>
        </div>
        <p>Upload documents to expand the RAG knowledge base. Supports Markdown (.md) and plain text (.txt).</p>
      </div>

      <div className="rag-grid">
        {/* Left: Upload */}
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
                <div style={{
                  marginTop: 20, marginBottom: 8, fontSize: 12,
                  color: 'var(--text-sec)', fontFamily: 'var(--mono)',
                  textTransform: 'uppercase', letterSpacing: '.5px'
                }}>
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

        {/* Right: KB Status */}
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
                  <div key={label} style={{
                    display: 'flex', justifyContent: 'space-between', fontSize: 13,
                    padding: '8px 0', borderBottom: '1px solid var(--border)'
                  }}>
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
                <div key={s.name} style={{
                  padding: '10px 14px', background: 'rgba(255,255,255,.02)',
                  border: '1px solid var(--border)', borderRadius: 10
                }}>
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
