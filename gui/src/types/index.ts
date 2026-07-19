export interface Agent {
  id: string;
  name: string;
  shortName: string;
  icon: string;
  color: string;
  bgColor: string;
  role: string;
  status: 'idle' | 'working' | 'error';
}

export interface TagStyle {
  bg: string;
  color: string;
  border: string;
}

export interface Message {
  id: string;
  from: string;
  content: string;
  tag?: 'STATUS' | 'REPORT' | 'ALERT' | 'DONE' | null;
  timestamp: number;
}

export interface Chat {
  id: string;
  type: 'team' | 'direct' | 'group';
  title?: string;
  reason?: string;
  members?: string[];
  messages: Message[];
  unread: number;
  isNew?: boolean;
}

export interface P2PEntry {
  id: string;
  from: string;
  to: string;
  content: string;
  timestamp: number;
}

export interface FileEntry {
  id: string;
  agent_id: string;
  filename: string;
  full_path: string;
  timestamp: number;
  version: number;
}

export interface Project {
  id: string;
  name: string;
  root: string;
  owner: string;
  created: string | null;
  modified: string | null;
  target_col?: string;
}

export interface AgentStatus {
  [agentId: string]: 'idle' | 'working' | 'error';
}

export interface TypingState {
  [chatId: string]: string | null;
}

export interface AuthState {
  token: string;
  email: string;
  isLoading: boolean;
  error: string;
}

export interface ConnectionState {
  status: 'connecting' | 'connected' | 'disconnected';
  ws: WebSocket | null;
}

export interface DatasetInfo {
  filename: string;
  path: string;
}

export interface ProjectSettings {
  project_root: string;
  target_col: string;
}