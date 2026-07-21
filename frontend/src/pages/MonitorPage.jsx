import { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadialBarChart, RadialBar, Cell, Legend
} from 'recharts';
import { BACKEND_URL } from '../config';

const REFRESH_MS = 10_000; // auto-refresh every 10 seconds

// ── Custom Tooltip ────────────────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#111927', border: '1px solid rgba(99,179,237,.2)',
      borderRadius: 8, padding: '8px 12px', fontSize: 12,
      fontFamily: 'JetBrains Mono, monospace', color: '#e2e8f0'
    }}>
      {label && <div style={{ color: '#8898aa', marginBottom: 4 }}>{label}</div>}
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color }}>{p.name}: {
          typeof p.value === 'number' ? p.value.toLocaleString() : p.value
        }</div>
      ))}
    </div>
  );
};

// ── Agent calls bar chart ────────────────────────────────────────────────────
function AgentChart({ data }) {
  const agents = ['CHEMICAL_AGENT', 'MEDICAL_AGENT', 'RAG_AGENT', 'APP_AGENT'];
  const colors = ['#3ecfcf', '#6366f1', '#f59e0b', '#22c55e'];
  const chartData = agents.map((a, i) => ({
    name: a.replace('_AGENT', ''),
    calls: data?.[a]?.total_calls || 0,
    fill: colors[i],
  }));
  return (
    <div className="chart-card">
      <div className="chart-title">Agent Distribution</div>
      <div className="chart-sub">Total calls per agent type</div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,.07)" />
          <XAxis dataKey="name" tick={{ fill: '#8898aa', fontSize: 11 }} />
          <YAxis tick={{ fill: '#8898aa', fontSize: 11 }} />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="calls" radius={[4, 4, 0, 0]}>
            {chartData.map((d, i) => <Cell key={i} fill={d.fill} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Requests over time ────────────────────────────────────────────────────────
function RequestsChart({ recentRequests }) {
  // bucket last 60 requests into time windows
  const grouped = [];
  const windowMs = 30_000;
  const now = Date.now();
  for (let i = 5; i >= 0; i--) {
    const from = now - (i + 1) * windowMs;
    const to   = now - i * windowMs;
    const count = recentRequests.filter(r => {
      const ts = (r.timestamp || 0) * 1000;
      return ts >= from && ts < to;
    }).length;
    grouped.push({ label: `-${(i + 1) * 30}s`, count });
  }
  return (
    <div className="chart-card">
      <div className="chart-title">Request Volume</div>
      <div className="chart-sub">Requests per 30-second window</div>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={grouped} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
          <defs>
            <linearGradient id="reqGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3ecfcf" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#3ecfcf" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,.07)" />
          <XAxis dataKey="label" tick={{ fill: '#8898aa', fontSize: 11 }} />
          <YAxis tick={{ fill: '#8898aa', fontSize: 11 }} allowDecimals={false} />
          <Tooltip content={<CustomTooltip />} />
          <Area type="monotone" dataKey="count" name="Requests" stroke="#3ecfcf" fill="url(#reqGrad)" strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Latency chart ─────────────────────────────────────────────────────────────
function LatencyChart({ recentRequests }) {
  const points = recentRequests.slice(-30).map((r, i) => ({
    i,
    latency: Math.round(r.latency_ms || 0),
    endpoint: (r.path || '').replace(/^\//, '').slice(0, 14),
  }));
  return (
    <div className="chart-card">
      <div className="chart-title">Response Latency</div>
      <div className="chart-sub">Last 30 requests (ms)</div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={points} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,.07)" />
          <XAxis dataKey="endpoint" tick={{ fill: '#8898aa', fontSize: 10 }} interval="preserveStartEnd" />
          <YAxis tick={{ fill: '#8898aa', fontSize: 11 }} unit="ms" />
          <Tooltip content={<CustomTooltip />} />
          <Line type="monotone" dataKey="latency" name="Latency (ms)" stroke="#f59e0b" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Status code distribution ─────────────────────────────────────────────────
function StatusChart({ recentRequests }) {
  const counts = {};
  recentRequests.forEach(r => {
    const s = String(r.status_code || '?');
    counts[s] = (counts[s] || 0) + 1;
  });
  const data = Object.entries(counts).map(([code, count]) => ({ code, count }));
  const colorFor = (code) => code.startsWith('2') ? '#22c55e' : code.startsWith('4') ? '#f59e0b' : '#ef4444';
  return (
    <div className="chart-card">
      <div className="chart-title">Status Codes</div>
      <div className="chart-sub">Recent request outcomes</div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,.07)" />
          <XAxis dataKey="code" tick={{ fill: '#8898aa', fontSize: 12 }} />
          <YAxis tick={{ fill: '#8898aa', fontSize: 11 }} allowDecimals={false} />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="count" name="Count" radius={[4,4,0,0]}>
            {data.map((d, i) => <Cell key={i} fill={colorFor(d.code)} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Token usage ───────────────────────────────────────────────────────────────
function TokenChart({ snapshot }) {
  const models = Object.entries(snapshot?.token_usage || {}).map(([model, v]) => ({
    model: model.split('-')[0],
    prompt: v.prompt_tokens || 0,
    completion: v.completion_tokens || 0,
  }));
  return (
    <div className="chart-card">
      <div className="chart-title">Token Usage</div>
      <div className="chart-sub">Prompt vs completion tokens by model</div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={models} margin={{ top: 4, right: 4, bottom: 4, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,.07)" />
          <XAxis dataKey="model" tick={{ fill: '#8898aa', fontSize: 11 }} />
          <YAxis tick={{ fill: '#8898aa', fontSize: 11 }} />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="prompt" name="Prompt" stackId="a" fill="#6366f1" radius={[0,0,0,0]} />
          <Bar dataKey="completion" name="Completion" stackId="a" fill="#3ecfcf" radius={[4,4,0,0]} />
          <Legend wrapperStyle={{ fontSize: 11, color: '#8898aa' }} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Error rate radial ─────────────────────────────────────────────────────────
function ErrorRateRadial({ snapshot }) {
  const total  = snapshot?.requests?.total || 1;
  const errors = snapshot?.requests?.errors || 0;
  const rate   = Math.round((errors / total) * 100);
  const data   = [{ name: 'Errors', value: rate, fill: '#ef4444' }, { name: 'OK', value: 100 - rate, fill: 'rgba(34,197,94,.2)' }];
  return (
    <div className="chart-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div className="chart-title">Error Rate</div>
      <div className="chart-sub">% of failed requests</div>
      <ResponsiveContainer width="100%" height={200}>
        <RadialBarChart innerRadius="60%" outerRadius="100%" data={data} startAngle={180} endAngle={0}>
          <RadialBar dataKey="value" />
          <text x="50%" y="54%" textAnchor="middle" dominantBaseline="middle"
            style={{ fontSize: 28, fontWeight: 700, fill: rate > 10 ? '#ef4444' : '#22c55e', fontFamily: 'JetBrains Mono' }}>
            {rate}%
          </text>
        </RadialBarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Main MonitorPage ──────────────────────────────────────────────────────────
export default function MonitorPage() {
  const [snapshot, setSnapshot]         = useState(null);
  const [recentReqs, setRecentReqs]     = useState([]);
  const [loading, setLoading]           = useState(true);
  const [lastUpdated, setLastUpdated]   = useState('');
  const [uptime, setUptime]             = useState('–');

  const fetchData = useCallback(async () => {
    try {
      const [snap, reqs] = await Promise.all([
        fetch(`${BACKEND_URL}/metrics`).then(r => r.json()),
        fetch(`${BACKEND_URL}/metrics/requests?limit=100`).then(r => r.json()),
      ]);
      setSnapshot(snap);
      setRecentReqs(Array.isArray(reqs) ? reqs : []);
      setLastUpdated(new Date().toLocaleTimeString());
      if (snap?.server?.uptime_seconds != null) {
        const s = Math.round(snap.server.uptime_seconds);
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / 60);
        setUptime(`${h}h ${m}m`);
      }
    } catch (e) {
      console.warn('Monitor fetch error', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, REFRESH_MS);
    return () => clearInterval(id);
  }, [fetchData]);

  const totalReq  = snapshot?.requests?.total || 0;
  const totalErr  = snapshot?.requests?.errors || 0;
  const successRate = totalReq ? Math.round(((totalReq - totalErr) / totalReq) * 100) : 100;
  const avgLatency  = snapshot?.requests?.avg_latency_ms != null
    ? Math.round(snapshot.requests.avg_latency_ms) + ' ms'
    : '– ms';

  return (
    <div className="monitor-page">
      <div className="monitor-header">
        <div>
          <h2>System Monitor</h2>
          <p>Live metrics · auto-refresh every {REFRESH_MS / 1000}s · Last: {lastUpdated || '–'}</p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--text-muted)' }}>Uptime: {uptime}</span>
          <button className="refresh-btn" onClick={fetchData}>↻ Refresh</button>
        </div>
      </div>

      {loading ? (
        <div style={{ textAlign: 'center', padding: '60px', color: 'var(--text-sec)', fontFamily: 'var(--mono)' }}>
          Loading metrics…
        </div>
      ) : (
        <>
          {/* Stat Cards */}
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-label">Total Requests</div>
              <div className="stat-value accent">{totalReq.toLocaleString()}</div>
              <div className="stat-sub">All time</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Success Rate</div>
              <div className={`stat-value ${successRate >= 95 ? 'green' : 'orange'}`}>{successRate}%</div>
              <div className="stat-sub">{totalReq - totalErr} successful</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Avg Latency</div>
              <div className="stat-value accent">{avgLatency}</div>
              <div className="stat-sub">Response time</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Errors</div>
              <div className={`stat-value ${totalErr === 0 ? 'green' : 'red'}`}>{totalErr}</div>
              <div className="stat-sub">Total failures</div>
            </div>
          </div>

          {/* Row 1: 2 charts */}
          <div className="charts-grid">
            <RequestsChart recentRequests={recentReqs} />
            <LatencyChart recentRequests={recentReqs} />
          </div>

          {/* Row 2: 3 charts */}
          <div className="charts-grid-3">
            <AgentChart data={snapshot?.agents} />
            <StatusChart recentRequests={recentReqs} />
            <ErrorRateRadial snapshot={snapshot} />
          </div>

          {/* Token usage */}
          <div className="charts-grid" style={{ marginBottom: 24 }}>
            <TokenChart snapshot={snapshot} />
            {/* Out-of-domain stats */}
            <div className="chart-card">
              <div className="chart-title">Out-of-Domain</div>
              <div className="chart-sub">Rejected queries</div>
              <div style={{ padding: '16px 0', display: 'flex', flexDirection: 'column', gap: 12 }}>
                {(() => {
                  const ood = snapshot?.out_of_domain || {};
                  return Object.entries(ood).slice(0, 8).map(([reason, count]) => (
                    <div key={reason} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13 }}>
                      <span style={{ color: 'var(--text-sec)', fontFamily: 'var(--mono)', fontSize: 11 }} title={reason}>
                        {reason.slice(0, 40) || '(unknown)'}
                      </span>
                      <span style={{ color: 'var(--accent)', fontFamily: 'var(--mono)', fontWeight: 700 }}>{count}</span>
                    </div>
                  ));
                })()}
                {!Object.keys(snapshot?.out_of_domain || {}).length && (
                  <div style={{ color: 'var(--green)', fontFamily: 'var(--mono)', fontSize: 12 }}>✅ No out-of-domain queries</div>
                )}
              </div>
            </div>
          </div>

          {/* Recent requests table */}
          <div className="chart-card">
            <div className="chart-title" style={{ marginBottom: 12 }}>Recent Requests</div>
            <div style={{ overflowX: 'auto' }}>
              <table className="requests-table">
                <thead>
                  <tr>
                    <th>Endpoint</th>
                    <th>Method</th>
                    <th>Status</th>
                    <th>Latency</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {recentReqs.slice(-20).reverse().map((r, i) => (
                    <tr key={i}>
                      <td style={{ fontFamily: 'var(--mono)' }}>{r.path || '–'}</td>
                      <td>{r.method || '–'}</td>
                      <td className={r.status_code < 400 ? 'status-ok' : 'status-err'}>
                        {r.status_code}
                      </td>
                      <td style={{ fontFamily: 'var(--mono)' }}>{Math.round(r.latency_ms || 0)}ms</td>
                      <td style={{ color: 'var(--text-muted)', fontFamily: 'var(--mono)', fontSize: 11 }}>
                        {r.timestamp ? new Date(r.timestamp * 1000).toLocaleTimeString() : '–'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
