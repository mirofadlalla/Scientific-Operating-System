/**
 * API service for Chat, Authentication, Knowledge Base (RAG), and Monitoring.
 */

import { API_BASE } from '../config';

/**
 * Stream orchestrator reply token-by-token.
 * @param {string} text
 * @param {string} sessionId
 * @param {string} userId
 * @param {(token: string) => void} onToken
 * @param {AbortSignal} [signal]
 */
export async function streamOrchestrate(text, sessionId, userId, onToken, signal) {
  const res = await fetch(`${API_BASE}/orchestrate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, user_id: userId, text_input: text }),
    signal,
  });

  if (!res.ok) {
    const errorBody = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status}: ${errorBody || res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let fullText = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const token = decoder.decode(value, { stream: true });
    fullText += token;
    onToken(token);
  }

  return fullText;
}

/**
 * Authenticate user with username and password.
 */
export async function loginUser(username, password) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || 'Invalid credentials.');
  }
  return data;
}

/**
 * Fetch knowledge base RAG status.
 */
export async function getRAGStatus() {
  const res = await fetch(`${API_BASE}/rag/status`);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: Failed to fetch RAG status`);
  }
  return res.json();
}

/**
 * Upload document to RAG ingestion pipeline.
 */
export async function ingestDocument(file, strategy, token) {
  const form = new FormData();
  form.append('file', file, file.name);
  form.append('strategy', strategy);

  const res = await fetch(`${API_BASE}/rag/ingest`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });

  const result = await res.json();
  if (!res.ok || result.status !== 'success') {
    throw new Error(result.detail || result.message || 'Upload failed');
  }
  return result;
}

/**
 * Poll ingestion job status.
 */
export async function getIngestionStatus(jobId) {
  const res = await fetch(`${API_BASE}/rag/ingest/status/${jobId}`);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: Failed to fetch ingestion job status`);
  }
  return res.json();
}

/**
 * Fetch system metrics snapshot and recent requests.
 */
export async function getMonitoringData(limit = 100) {
  const [snapshot, requests] = await Promise.all([
    fetch(`${API_BASE}/metrics`).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    }),
    fetch(`${API_BASE}/metrics/requests?limit=${limit}`).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    }),
  ]);
  return { snapshot, requests: Array.isArray(requests) ? requests : [] };
}
