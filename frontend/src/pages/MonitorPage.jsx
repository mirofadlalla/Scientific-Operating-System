import { useState, useEffect, useCallback } from 'react';
import {
  AgentChart,
  RequestsChart,
  LatencyChart,
  StatusChart,
  TokenChart,
  ErrorRateRadial,
} from '../components/MonitorCharts';
import { getMonitoringData } from '../services/api';

const REFRESH_MS = 10_000; // auto-refresh every 10 seconds

export default function MonitorPage() {
  const [snapshot, setSnapshot]         = useState(null);
  const [recentReqs, setRecentReqs]     = useState([]);
  const [loading, setLoading]           = useState(true);
  const [lastUpdated, setLastUpdated]   = useState('');
  const [uptime, setUptime]             = useState('–');

  const fetchData = useCallback(async () => {
    try {
      const { snapshot: snap, requests: reqs } = await getMonitoringData(100);
      setSnapshot(snap);
      setRecentReqs(reqs);
      setLastUpdated(new Date().toLocaleTimeString());
      if (snap?.uptime?.seconds != null) {
        const s = Math.round(snap.uptime.seconds);
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
  const totalErr  = snapshot?.errors?.total || 0;
  const successRate = totalReq ? Math.round(((totalReq - totalErr) / totalReq) * 100) : 100;
  const avgLatency  = snapshot?.latency?.count
    ? Math.round(snapshot.latency.avg) + ' ms'
    : '– ms';

  return (
    <div className="monitor-page">
      <div className="monitor-header">
        <div>
          <h2>System Monitor</h2>
          <p>Live metrics · auto-refresh every {REFRESH_MS / 1000}s · Last: {lastUpdated || '–'}</p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--text-muted)' }}>
            Uptime: {uptime}
          </span>
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
                {(snapshot?.requests?.out_of_domain || 0) > 0 ? (
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13 }}>
                    <span style={{ color: 'var(--text-sec)', fontFamily: 'var(--mono)', fontSize: 11 }}>Rejected queries</span>
                    <span style={{ color: 'var(--accent)', fontFamily: 'var(--mono)', fontWeight: 700 }}>
                      {snapshot.requests.out_of_domain}
                    </span>
                  </div>
                ) : (
                  <div style={{ color: 'var(--green)', fontFamily: 'var(--mono)', fontSize: 12 }}>
                    ✅ No out-of-domain queries
                  </div>
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
                      <td style={{ fontFamily: 'var(--mono)' }}>{r.endpoint || '–'}</td>
                      <td>{r.method || '–'}</td>
                      <td className={r.status < 400 ? 'status-ok' : 'status-err'}>
                        {r.status}
                      </td>
                      <td style={{ fontFamily: 'var(--mono)' }}>{Math.round(r.latency_ms || 0)}ms</td>
                      <td style={{ color: 'var(--text-muted)', fontFamily: 'var(--mono)', fontSize: 11 }}>
                        {r.ts ? new Date(r.ts).toLocaleTimeString() : '–'}
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
