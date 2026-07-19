// Global type declarations
import type { Agent, TagStyle } from './types';

declare global {
  interface Window {
    AGENTS: Record<string, Agent>;
    TAG_STYLES: Record<string, TagStyle>;
  }
}

export {};