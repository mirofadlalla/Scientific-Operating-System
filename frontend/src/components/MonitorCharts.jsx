import {
  LineChart, Line, BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadialBarChart, RadialBar, Cell
} from 'recharts';

export const CustomTooltip = ({ active, payload, label }) => {
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

export function AgentChart({ data }) {
  const agents = ['CHEMICAL_AGENT', 'MEDICAL_AGENT', 'RAG_AGENT', 'APP_AGENT'];
  const colors = ['#3ecfcf', '#6366f1', '#f59e0b', '#22c55e'];
  const chartData = agents.map((a, i) => ({
    name: a.replace('_AGENT', ''),
    calls: data?.distribution?.[a]?.count || 0,
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

export function RequestsChart({ recentRequests }) {
  const grouped = [];
  const windowMs = 30_000;
  const now = Date.now();
  for (let i = 5; i >= 0; i--) {
    const from = now - (i + 1) * windowMs;
    const to   = now - i * windowMs;
    const count = recentRequests.filter(r => {
      const ts = r.ts ? Date.parse(r.ts) : 0;
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

export function LatencyChart({ recentRequests }) {
  const points = recentRequests.slice(-30).map((r, i) => ({
    i,
    latency: Math.round(r.latency_ms || 0),
    endpoint: (r.endpoint || '').replace(/^\//, '').slice(0, 14),
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

export function StatusChart({ recentRequests }) {
  const counts = {};
  recentRequests.forEach(r => {
    const s = String(r.status || '?');
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
          <Bar dataKey="count" name="Count" radius={[4, 4, 0, 0]}>
            {data.map((d, i) => <Cell key={i} fill={colorFor(d.code)} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TokenChart({ snapshot }) {
  const tokenStats = snapshot?.tokens || {};
  const entries = Object.entries(tokenStats).filter(([k]) => k !== '__all__');
  const allStats = tokenStats['__all__'] || {};

  return (
    <div className="chart-card">
      <div className="chart-title">LLM Metrics & Cost</div>
      <div className="chart-sub">Tokens, TTFT, TPS & Estimated Cost (USD)</div>

      <div style={{
        margin: '12px 0', padding: '10px 14px', background: 'rgba(99,102,241,0.1)',
        border: '1px solid rgba(99,102,241,0.2)', borderRadius: 8,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center'
      }}>
        <div>
          <div style={{ fontSize: 11, color: '#8898aa', textTransform: 'uppercase' }}>Total Tokens / Cost</div>
          <div style={{ fontSize: 16, fontWeight: 700, color: '#e2e8f0', fontFamily: 'var(--mono)' }}>
            {(allStats.total || 0).toLocaleString()} <span style={{ fontSize: 12, color: '#3ecfcf' }}>(${(allStats.cost_usd || 0).toFixed(6)})</span>
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 11, color: '#8898aa', textTransform: 'uppercase' }}>Avg TTFT / TPS</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#f59e0b', fontFamily: 'var(--mono)' }}>
            {allStats.avg_ttft_ms || 0}ms | {allStats.avg_tps || 0} tps
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 150, overflowY: 'auto' }}>
        {entries.map(([model, v]) => (
          <div key={model} style={{ padding: '6px 10px', background: 'var(--bg)', borderRadius: 6, display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12, border: '1px solid var(--border)' }}>
            <div>
              <div style={{ fontWeight: 600, color: '#e2e8f0', fontFamily: 'var(--mono)' }}>{model}</div>
              <div style={{ fontSize: 10, color: '#8898aa' }}>Prompt: {v.prompt} | Comp: {v.completion}</div>
            </div>
            <div style={{ textAlign: 'right', fontFamily: 'var(--mono)' }}>
              <div style={{ color: '#3ecfcf', fontWeight: 600 }}>${(v.cost_usd || 0).toFixed(6)}</div>
              <div style={{ fontSize: 10, color: '#8898aa' }}>{v.avg_ttft_ms}ms TTFT • {v.avg_tps} TPS</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ErrorRateRadial({ snapshot }) {
  const rate = Math.min(100, Math.round(snapshot?.requests?.error_rate || 0));
  const data = [
    { name: 'Errors', value: rate, fill: '#ef4444' },
    { name: 'OK', value: 100 - rate, fill: 'rgba(34,197,94,.2)' }
  ];
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
