import { useRef, useEffect, useState, useCallback } from 'react';
import { WS_URL } from '../config';

const MAX_QUEUED_FRAMES = 32;

/**
 * Hook for managing full-duplex WebSocket connection to the voice channel.
 * Includes automated keepalive pings and tablet wake-up resync.
 *
 * Outbound JSON is queued while the socket is down so VAD events
 * (client_info, audio_chunk, audio_end) are not silently dropped.
 */
export function useVoiceSocket({ sessionId, onMessage, onBinaryChunk, onOpen }) {
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const onMessageRef = useRef(onMessage);
  const onBinaryChunkRef = useRef(onBinaryChunk);
  const onOpenRef = useRef(onOpen);
  const outboundQueueRef = useRef([]);
  const lastClientInfoRef = useRef(null);
  const unmountedRef = useRef(false);

  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    onBinaryChunkRef.current = onBinaryChunk;
  }, [onBinaryChunk]);

  useEffect(() => {
    onOpenRef.current = onOpen;
  }, [onOpen]);

  const flushQueue = useCallback((ws) => {
    const q = outboundQueueRef.current;
    while (q.length && ws.readyState === WebSocket.OPEN) {
      ws.send(q.shift());
    }
  }, []);

  const connectWS = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    const old = wsRef.current;
    if (old) {
      // Detach handlers first: a late onclose from the stale socket must not
      // schedule a reconnect that kills the healthy replacement.
      old.onopen = old.onclose = old.onerror = old.onmessage = null;
      try { old.close(); } catch {}
    }

    const ws = new WebSocket(`${WS_URL}?session_id=${sessionId}`);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WS] Connected', WS_URL);
      setWsConnected(true);
      flushQueue(ws);
      if (lastClientInfoRef.current && ws.readyState === WebSocket.OPEN) {
        ws.send(lastClientInfoRef.current);
      }
      if (onOpenRef.current) onOpenRef.current();
    };

    ws.onclose = () => {
      if (wsRef.current !== ws) return; // stale socket
      console.log('[WS] Disconnected, will reconnect in 3s');
      setWsConnected(false);
      if (!unmountedRef.current) {
        reconnectTimerRef.current = setTimeout(connectWS, 3000);
      }
    };

    ws.onerror = () => {
      if (wsRef.current !== ws) return;
      setWsConnected(false);
    };

    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        if (onBinaryChunkRef.current) {
          onBinaryChunkRef.current(ev.data);
        }
        return;
      }

      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }

      if (onMessageRef.current) {
        onMessageRef.current(msg);
      }
    };
  }, [sessionId, flushQueue]);

  useEffect(() => {
    unmountedRef.current = false;
    connectWS();
    return () => {
      unmountedRef.current = true;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      const cur = wsRef.current;
      if (cur) {
        cur.onopen = cur.onclose = cur.onerror = cur.onmessage = null;
        try { cur.close(); } catch {}
      }
    };
  }, [connectWS]);

  // Keepalive pings (10s interval + visibilitychange trigger for tablets)
  useEffect(() => {
    const sendPing = () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    };

    const iv = setInterval(sendPing, 10000);

    const onVisible = () => {
      if (document.visibilityState === 'visible') {
        console.log('[WS] Tab became visible — sending keepalive ping');
        sendPing();
      }
    };
    document.addEventListener('visibilitychange', onVisible);

    return () => {
      clearInterval(iv);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, []);

  const sendJson = useCallback((payload) => {
    let data;
    try {
      data = JSON.stringify(payload);
    } catch (err) {
      console.error('[WS] Failed to serialize payload', payload?.type, err);
      return false;
    }

    if (payload?.type === 'client_info') {
      lastClientInfoRef.current = data;
    }

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
      return true;
    }

    // Never queue keepalive — it is meaningless after a reconnect.
    if (payload?.type === 'ping') return false;

    const q = outboundQueueRef.current;
    q.push(data);
    if (q.length > MAX_QUEUED_FRAMES) q.splice(0, q.length - MAX_QUEUED_FRAMES);
    console.warn(`[WS] Queued ${payload?.type} (${q.length} pending) — socket not open`);
    return true;
  }, []);

  return { wsConnected, sendJson, wsRef };
}
