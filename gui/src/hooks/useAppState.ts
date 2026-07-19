import { useState, useCallback } from 'react';
import type { Message, Chat, P2PEntry, FileEntry, Project } from '@/types';
import { AGENTS, TAG_STYLES } from '@/data/agents';
import { safeLocalStorageGet, safeLocalStorageRemove, safeLocalStorageSet } from '@/utils/config';

const fmt = (ts: number) =>
  new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

let _id = 0;
const uid = () => `${Date.now().toString(36)}_${(++_id).toString(36)}_${Math.random().toString(36).slice(2, 8)}`;

function createInitialChatList(): Chat[] {
  const list: Chat[] = [
    {
      id: 'team',
      type: 'team',
      messages: [
        {
          id: uid(),
          from: 'orchestrator',
          tag: 'STATUS',
          content:
            'AI Engineering Platform online. All agents standing by.\n\nBackend running → messages go to real Python agents.\nBackend offline → full mock mode with simulated responses.',
          timestamp: Date.now(),
        },
      ],
      unread: 0,
    },
  ];
  for (const a of Object.values(AGENTS)) {
    list.push({
      id: a.id,
      type: 'direct',
      messages: [
        {
          id: uid(),
          from: a.id,
          tag: 'STATUS',
          content: `Hi, I'm ${a.name}. ${a.role}\n\nGive me a direct instruction. If I need another agent, a group thread auto-spawns.`,
          timestamp: Date.now(),
        },
      ],
      unread: 0,
    });
  }
  return list;
}

export function useAppState() {
  const [agents] = useState(() =>
    Object.fromEntries(Object.entries(AGENTS).map(([k, v]) => [k, { ...v }]))
  );
  const [chatList, setChatList] = useState<Chat[]>(() => createInitialChatList());
  const [activeChat, setActiveChat] = useState('team');
  const [typingIn, setTypingIn] = useState<Record<string, string | null>>({});
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [isBusy, setIsBusy] = useState(false);
  const [connStatus, setConnStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [projectName, setProjectName] = useState<string | null>(null);
  const [projectRoot, setProjectRoot] = useState<string | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectsLoaded, setProjectsLoaded] = useState(false);
  const [p2pLog, setP2pLog] = useState<P2PEntry[]>([]);
  const [fileLog, setFileLog] = useState<FileEntry[]>([]);
  const [uploading, setUploading] = useState(false);
  const [datasetReady, setDatasetReady] = useState(false);
  const [datasetInfo, setDatasetInfo] = useState<{ filename: string; path: string } | null>(null);
  const [projectTargetColumn, setProjectTargetColumn] = useState('');
  const [targetColumnDraft, setTargetColumnDraft] = useState('');
  const [showFilesPanel, setShowFilesPanel] = useState(false);

  // Auth state
  // Restore an existing authenticated session on reload.  The server still
  // validates the token for every protected request.
  const [authToken, setAuthTokenState] = useState(() => safeLocalStorageGet('auth_token'));
  const [authEmail, setAuthEmailState] = useState(() => safeLocalStorageGet('auth_email'));
  const [authPassword, setAuthPassword] = useState('');
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState('');

  // GitHub connection state
  const [showGithubConnect, setShowGithubConnect] = useState(false);
  const [githubToken, setGithubToken] = useState('');
  const [githubRepo, setGithubRepo] = useState('');
  const [githubOwner, setGithubOwner] = useState('');
  const [githubVisibility, setGithubVisibility] = useState<'private' | 'public'>('private');

  // ML Engineer Worker state
  const [showMLWorkerHelp, setShowMLWorkerHelp] = useState(false);
  const [workerToken, setWorkerToken] = useState('');
  const [workerStatus, setWorkerStatus] = useState<'connected' | 'disconnected' | 'error' | 'unknown'>('unknown');
  const [workerBusy, setWorkerBusy] = useState(false);
  const [workerError, setWorkerError] = useState('');
  const [workerProjectPath, setWorkerProjectPath] = useState(() => safeLocalStorageGet('worker_project_path'));

  // Files panel index & viewer state
  const [filesIndex, setFilesIndex] = useState<any[]>([]);
  const [filesLoading, setFilesLoading] = useState(false);
  const [filesError, setFilesError] = useState('');
  const [fileSearch, setFileSearch] = useState('');
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState('');
  const [fileBinary, setFileBinary] = useState(false);

  const setAuthToken = useCallback((token: string) => {
    const value = token || '';
    setAuthTokenState(value);
    if (value) safeLocalStorageSet('auth_token', value);
    else safeLocalStorageRemove('auth_token');
  }, []);

  const setAuthEmail = useCallback((email: string) => {
    const value = (email || '').trim();
    setAuthEmailState(value);
    if (value) safeLocalStorageSet('auth_email', value);
    else safeLocalStorageRemove('auth_email');
  }, []);

  const addMsg = useCallback(
    (chatId: string, from: string, content: string, tag?: Message['tag']) => {
      setChatList((prev) =>
        prev.map((c) => {
          if (c.id !== chatId) return c;
          const isVisible = activeChat === chatId;
          return {
            ...c,
            messages: [
              ...c.messages,
              { id: uid(), from, content, tag: tag as any, timestamp: Date.now() },
            ],
            unread: isVisible ? 0 : (c.unread || 0) + (from !== 'user' ? 1 : 0),
          };
        })
      );
    },
    [activeChat]
  );

  const setTyping = useCallback((chatId: string, agentId: string | null) => {
    setTypingIn((prev) => ({ ...prev, [chatId]: agentId }));
  }, []);

  const setAgentStatus = useCallback(() => {
    // Agent status is managed externally
  }, []);

  const clearTimers = () => {};

  const upsertFile = useCallback((agent_id: string, filename: string, full_path: string) => {
    setFileLog((prev) => {
      const idx = prev.findIndex((f) => f.filename === filename && f.agent_id === agent_id);
      const entry: FileEntry = {
        id: idx >= 0 ? prev[idx].id : uid(),
        agent_id,
        filename,
        full_path,
        timestamp: Date.now(),
        version: idx >= 0 ? (prev[idx].version || 1) + 1 : 1,
      };
      if (idx >= 0) {
        const a = [...prev];
        a[idx] = entry;
        return a;
      }
      return [...prev, entry];
    });
  }, []);

  const seedFile = useCallback((agent_id: string, filename: string, full_path: string) => {
    setFileLog((prev) => {
      if (prev.some((f) => f.filename === filename && f.agent_id === agent_id)) return prev;
      return [
        ...prev,
        {
          id: uid(),
          agent_id,
          filename,
          full_path,
          timestamp: Date.now(),
          version: 1,
        },
      ];
    });
  }, []);

  return {
    agents,
    chatList,
    setChatList,
    activeChat,
    setActiveChat,
    typingIn,
    setTypingIn,
    inputs,
    setInputs,
    isBusy,
    setIsBusy,
    connStatus,
    setConnStatus,
    projectName,
    setProjectName,
    projectRoot,
    setProjectRoot,
    projects,
    setProjects,
    projectsLoaded,
    setProjectsLoaded,
    p2pLog,
    setP2pLog,
    fileLog,
    setFileLog,
    uploading,
    setUploading,
    datasetReady,
    setDatasetReady,
    datasetInfo,
    setDatasetInfo,
    projectTargetColumn,
    setProjectTargetColumn,
    targetColumnDraft,
    setTargetColumnDraft,
    showFilesPanel,
    setShowFilesPanel,
    authToken,
    setAuthToken,
    authEmail,
    setAuthEmail,
    authPassword,
    setAuthPassword,
    authMode,
    setAuthMode,
    authLoading,
    setAuthLoading,
    authError,
    setAuthError,
    showGithubConnect,
    setShowGithubConnect,
    githubToken,
    setGithubToken,
    githubRepo,
    setGithubRepo,
    githubOwner,
    setGithubOwner,
    githubVisibility,
    setGithubVisibility,
    showMLWorkerHelp,
    setShowMLWorkerHelp,
    workerToken,
    setWorkerToken,
    workerStatus,
    setWorkerStatus,
    workerBusy,
    setWorkerBusy,
    workerError,
    setWorkerError,
    workerProjectPath,
    setWorkerProjectPath,
    filesIndex,
    setFilesIndex,
    filesLoading,
    setFilesLoading,
    filesError,
    setFilesError,
    fileSearch,
    setFileSearch,
    selectedFile,
    setSelectedFile,
    fileContent,
    setFileContent,
    fileBinary,
    setFileBinary,
    addMsg,
    setTyping,
    setAgentStatus,
    clearTimers,
    upsertFile,
    seedFile,
    TAG_STYLES,
    fmt,
    AGENTS,
  };
}
