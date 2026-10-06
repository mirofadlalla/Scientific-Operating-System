import { useRef, useEffect, useState, useCallback } from 'react';
import { WS_URL } from '../config';

/**
 * Hook for managing full-duplex WebSocket connection to the voice channel.
 * Includes automated keepalive pings and tablet wake-up resync.
 */
export function useVoiceSocket({ sessionId, onMessage, onBinaryChunk }) {
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const onMessageRef = useRef(onMessage);
  const onBinaryChunkRef = useRef(onBinaryChunk);

  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    onBinaryChunkRef.current = onBinaryChunk;
  }, [onBinaryChunk]);

  const connectWS = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      try { wsRef.current.close(); } catch {}
    }

    const ws = new WebSocket(`${WS_URL}?session_id=${sessionId}`);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WS] Connected');
      setWsConnected(true);
    };

    ws.onclose = () => {
      console.log('[WS] Disconnected, will reconnect in 3s');
      setWsConnected(false);
      reconnectTimerRef.current = setTimeout(connectWS, 3000);
    };

    ws.onerror = () => {
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
  }, [sessionId]);

  useEffect(() => {
    connectWS();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) {
        try { wsRef.current.close(); } catch {}
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
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
      return true;
    }
    return false;
  }, []);

  return { wsConnected, sendJson, wsRef };
}
