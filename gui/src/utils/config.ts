export const API_URL = (import.meta as any).env?.VITE_API_URL || "http://localhost:8000";
export const WS_URL = (import.meta as any).env?.VITE_WS_URL || `${API_URL.replace("http://", "ws://").replace("https://", "wss://")}/ws`;

export const safeLocalStorageGet = (key: string): string => {
  try { return localStorage.getItem(key) || ""; } catch { return ""; }
};

export const safeLocalStorageSet = (key: string, value: string): void => {
  try { localStorage.setItem(key, value); } catch {}
};

export const safeLocalStorageRemove = (key: string): void => {
  try { localStorage.removeItem(key); } catch {}
};
