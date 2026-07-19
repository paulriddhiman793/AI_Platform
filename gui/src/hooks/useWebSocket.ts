import { useEffect, useRef, useCallback } from 'react';

interface WebSocketOptions {
  url: string;
  onMessage: (msg: any) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (err: any) => void;
  reconnect?: boolean;
  reconnectInterval?: number;
}

export function useWebSocket({
  url,
  onMessage,
  onOpen,
  onClose,
  onError,
  reconnect = true,
  reconnectInterval = 3000,
}: WebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const shouldReconnectRef = useRef(true);
  const onMessageRef = useRef(onMessage);
  const onOpenRef = useRef(onOpen);
  const onCloseRef = useRef(onClose);
  const onErrorRef = useRef(onError);

  // Keep callbacks up to date
  useEffect(() => { onMessageRef.current = onMessage; }, [onMessage]);
  useEffect(() => { onOpenRef.current = onOpen; }, [onOpen]);
  useEffect(() => { onCloseRef.current = onClose; }, [onClose]);
  useEffect(() => { onErrorRef.current = onError; }, [onError]);

  const connect = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;
      ws.onopen = () => {
        onOpenRef.current?.();
      };
      ws.onclose = () => {
        onCloseRef.current?.();
        wsRef.current = null;
        if (shouldReconnectRef.current && reconnect) {
          reconnectTimeoutRef.current = setTimeout(connect, reconnectInterval);
        }
      };
      ws.onerror = (err) => {
        onErrorRef.current?.(err);
      };
      ws.onmessage = (evt) => {
        try {
          onMessageRef.current(JSON.parse(evt.data));
        } catch (e) {
          console.error('WS message parse error:', e);
        }
      };
    } catch {
      onErrorRef.current?.(new Error('WebSocket connection failed'));
    }
  }, [url, reconnect, reconnectInterval]);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  const send = useCallback((msg: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  useEffect(() => {
    shouldReconnectRef.current = true;
    connect();
    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { send, disconnect, wsRef, get readyState() { return wsRef.current?.readyState; } };
}

// Hook for deduplication
export function useDeduplication<T>(windowMs: number = 1500) {
  const cacheRef = useRef<Map<string, number>>(new Map());

  const isDuplicate = useCallback((key: T) => {
    const keyStr = JSON.stringify(key);
    const now = Date.now();
    const last = cacheRef.current.get(keyStr);
    if (last && now - last < windowMs) {
      return true;
    }
    cacheRef.current.set(keyStr, now);
    // Cleanup old entries
    if (cacheRef.current.size > 500) {
      const firstKey = cacheRef.current.keys().next().value;
      if (firstKey) cacheRef.current.delete(firstKey);
    }
    return false;
  }, [windowMs]);

  return { isDuplicate };
}

// Hook for auto-scroll
export function useAutoScroll(dependencies: any[]) {
  const refsRef = useRef<Record<string, HTMLDivElement | null>>({});

  const registerRef = useCallback((key: string, el: HTMLDivElement | null) => {
    refsRef.current[key] = el;
  }, []);

  useEffect(() => {
    Object.values(refsRef.current).forEach(el => {
      if (el) el.scrollTop = el.scrollHeight;
    });
  }, dependencies);

  return { registerRef };
}