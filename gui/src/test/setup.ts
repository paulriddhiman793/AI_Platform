/// <reference types="vitest/globals" />
/// <reference types="@testing-library/jest-dom" />

import { vi } from 'vitest';
import '@testing-library/jest-dom';

// Mock the AGENTS global
beforeEach(() => {
  vi.resetModules();
  (window as any).AGENTS = {
    ml_engineer: {
      id: 'ml_engineer',
      name: 'ML Engineer',
      shortName: 'ML',
      icon: 'ML',
      color: '#10b981',
      bgColor: '#022c22',
      role: 'Builds ML models',
      status: 'idle',
    },
  };
  (window as any).TAG_STYLES = {
    STATUS: { bg: '#0d1f0d', color: '#4ade80', border: '#1a3a1a' },
  };
});

// Mock ResizeObserver
globalThis.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
};
Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

// Mock WebSocket
globalThis.WebSocket = vi.fn().mockImplementation(() => ({
  send: vi.fn(),
  close: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  readyState: WebSocket.OPEN,
})) as any;