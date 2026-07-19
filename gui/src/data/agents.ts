import type { Agent } from '../types';

export const AGENTS: Record<string, Agent> = {
  orchestrator: {
    id: 'orchestrator',
    name: 'Orchestrator',
    shortName: 'ORCH',
    icon: 'OR',
    color: '#6366f1',
    bgColor: '#1e1b4b',
    role: 'Routes tasks, coordinates the team, speaks to you',
    status: 'idle',
  },
  ml_engineer: {
    id: 'ml_engineer',
    name: 'ML Engineer',
    shortName: 'ML',
    icon: 'ML',
    color: '#10b981',
    bgColor: '#022c22',
    role: 'Builds, trains, deploys ML models. Self-healing debug loop.',
    status: 'idle',
  },
  data_scientist: {
    id: 'data_scientist',
    name: 'Data Scientist',
    shortName: 'DS',
    icon: 'DS',
    color: '#3b82f6',
    bgColor: '#0c1a3a',
    role: 'EDA, hypothesis testing, feature engineering, experiments.',
    status: 'idle',
  },
  data_analyst: {
    id: 'data_analyst',
    name: 'Data Analyst',
    shortName: 'DA',
    icon: 'DA',
    color: '#f59e0b',
    bgColor: '#1c1000',
    role: 'Business insights, dashboards, twice-daily health reports.',
    status: 'idle',
  },
  github: {
    id: 'github',
    name: 'GitHub Agent',
    shortName: 'GH',
    icon: 'GH',
    color: '#93c5fd',
    bgColor: '#0b1022',
    role: 'Pushes repo, per-agent branches, and merges to main.',
    status: 'idle',
  },
};

export const TAG_STYLES = {
  STATUS: { bg: '#0d1f0d', color: '#4ade80', border: '#1a3a1a' },
  REPORT: { bg: '#0a1628', color: '#60a5fa', border: '#1a3060' },
  ALERT: { bg: '#2a0808', color: '#f87171', border: '#5a1818' },
  DONE: { bg: '#16113a', color: '#a78bfa', border: '#302060' },
} as const;

export type TagType = keyof typeof TAG_STYLES;