import { useState } from 'react';
import { loginUser } from '../services/api';

export function LoginGate({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username || !password) return;
    setLoading(true);
    setError('');
    try {
      const data = await loginUser(username, password);
      onLogin(data.access_token, username);
    } catch (err) {
      setError(err.message || 'Invalid credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-gate">
      <div className="login-card">
        <div className="login-icon">🔐</div>
        <h2>Knowledge Base Access</h2>
        <p>Sign in to ingest documents into the RAG knowledge base.</p>
        <form onSubmit={handleSubmit} className="login-form">
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={e => setUsername(e.target.value)}
            autoComplete="username"
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />
          {error && <div className="login-error">⚠️ {error}</div>}
          <button type="submit" className="upload-btn" disabled={loading}>
            {loading ? '⟳ Signing in…' : '🔑 Sign In'}
          </button>
        </form>
      </div>
    </div>
  );
}

const STEP_KEYS = ['upload', 'chunk', 'embed', 'index', 'reload'];
const STEP_LABELS = {
  upload: '📤  Uploading file to server',
  chunk:  '✂️  Chunking document',
  embed:  '🔢  Generating embeddings',
  index:  '📦  Indexing into vector store',
  reload: '🔄  Reloading query engine',
};

export function StepPipeline({ steps }) {
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

export function KBStatus({ status }) {
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

export function LogEntry({ entry }) {
  return (
    <div className={`log-entry ${entry.type}`}>
      <div>{entry.message}</div>
      <div className="log-time">{entry.time}</div>
    </div>
  );
}
