"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAppState } from "@/hooks/useAppState";
import { useWebSocket } from "@/hooks/useWebSocket";
import { Sidebar, ChatHeader, Message, RightPanel } from "@/components";
import { ParticleBackground, Aurora, MeshGradient, Vignette } from "@/components/Background";
import { Button, Textarea, Input } from "@/components/UI";
import { AuthPage } from "@/components/AuthPage";
import { GithubConnectModal } from "@/components/Modals/GithubConnectModal";
import { MLEngineerModal } from "@/components/Modals/MLEngineerModal";
import { FilesPanelModal } from "@/components/Modals/FilesPanelModal";
import { API_URL, safeLocalStorageGet, safeLocalStorageRemove, safeLocalStorageSet } from "@/utils/config";
import type { Chat, Project } from "@/types";

export default function App() {
  const appState = useAppState();
  const {
    agents,
    chatList,
    setChatList,
    activeChat,
    setActiveChat,
    typingIn,
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
    p2pLog,
    setP2pLog,
    fileLog,
    setFileLog,
    addMsg,
    setTyping,
    upsertFile,
    projectTargetColumn,
    setProjectTargetColumn,
    targetColumnDraft,
    setTargetColumnDraft,
    // Auth state
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
    // GitHub connection modal state
    showGithubConnect,
    setShowGithubConnect,
    githubToken,
    setGithubToken,
    githubOwner,
    setGithubOwner,
    githubRepo,
    setGithubRepo,
    githubVisibility,
    setGithubVisibility,
    // ML Engineer local worker state
    showMLWorkerHelp,
    setShowMLWorkerHelp,
    workerToken,
    setWorkerToken,
    workerStatus,
    setWorkerStatus,
    workerProjectPath,
    setWorkerProjectPath,
    // Access Files modal state
    showFilesPanel,
    setShowFilesPanel,
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
    // Upload & Dataset state
    uploading,
    setUploading,
    datasetReady,
    setDatasetReady,
    datasetInfo,
    setDatasetInfo,
  } = appState;

  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectRoot, setNewProjectRoot] = useState("");
  const [showNewProjectModal, setShowNewProjectModal] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const uploadRef = useRef<HTMLInputElement>(null);
  const restoredChatKeyRef = useRef<string | null>(null);
  const suppressChatSaveKeyRef = useRef<string | null>(null);
  const [isTargetColReadOnly, setIsTargetColReadOnly] = useState(true);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  const hydrateProjectFiles = useCallback(async (root: string) => {
    if (!authToken || !root) return;
    try {
      const res = await fetch(`${API_URL}/files`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auth_token: authToken, project_root: root }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && Array.isArray(data.files)) {
        const restoredFiles = data.files.flatMap((file: any, index: number) => {
          const relativePath = String(file.path || file.name || file.filename || "").replace(/\\/g, "/");
          const [agentId, ...filenameParts] = relativePath.split("/");
          if (!agentId || filenameParts.length === 0) return [];
          return [{
            id: `restored_${index}_${relativePath}`,
            agent_id: agentId,
            filename: filenameParts.join("/"),
            full_path: file.full_path || `${root.replace(/[\\/]$/, "")}/${relativePath}`,
            timestamp: Date.now(),
            version: 1,
          }];
        });
        setFileLog(restoredFiles);
        const dataset = data.files.find((file: any) =>
          String(file.path || file.name || file.filename || "").startsWith("shared/datasets/")
        );
        if (dataset) {
          setDatasetReady(true);
          setDatasetInfo({
            filename: String(dataset.path || dataset.name || dataset.filename).split(/[\\/]/).pop() || "dataset",
            path: dataset.full_path || dataset.absolute_path || dataset.path || "",
          });
        }
      }
    } catch (error) {
      console.warn("Failed to restore project files:", error);
    }
  }, [authToken, setDatasetInfo, setDatasetReady, setFileLog]);

  const fetchProjects = useCallback(async () => {
    if (!authToken) return;
    try {
      const res = await fetch(`${API_URL}/projects`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auth_token: authToken }),
      });
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data.projects)) {
          setProjects(data.projects);
          if (data.projects.length > 0 && !projectRoot) {
            const first = data.projects[0];
            setProjectName(first.name);
            setProjectRoot(first.root);
            if (first.target_col) {
              setProjectTargetColumn(first.target_col);
              setTargetColumnDraft(first.target_col);
            }
            const selected = await fetch(`${API_URL}/projects/select`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                auth_token: authToken,
                project_id: first.id || first.root,
                project_root: first.root,
              }),
            });
            if (selected.ok) await hydrateProjectFiles(first.root);
          }
        }
      }
    } catch (e) {
      console.warn("Failed to fetch projects:", e);
    }
  }, [authToken, hydrateProjectFiles, projectRoot, setProjects, setProjectName, setProjectRoot, setProjectTargetColumn, setTargetColumnDraft]);

  useEffect(() => {
    if (!authToken) return;
    fetchProjects();
    if (targetColumnDraft && (targetColumnDraft.includes("@") || targetColumnDraft === authEmail)) {
      setTargetColumnDraft("");
      setProjectTargetColumn("");
    }
  }, [authToken, fetchProjects, authEmail, targetColumnDraft, setTargetColumnDraft, setProjectTargetColumn]);

  const handleLogout = useCallback(() => {
    setAuthToken("");
    setAuthEmail("");
    setProjectName(null);
    setProjectRoot(null);
    setProjects([]);
    setFileLog([]);
    setTargetColumnDraft("");
    setProjectTargetColumn("");
    safeLocalStorageRemove("auth_token");
    safeLocalStorageRemove("auth_email");
  }, [setAuthToken, setAuthEmail, setFileLog, setProjectName, setProjectRoot, setProjects, setTargetColumnDraft, setProjectTargetColumn]);

  const selectProject = useCallback(
    async (project: Project) => {
      setProjectName(project.name);
      setProjectRoot(project.root);
      if (project.target_col) {
        setProjectTargetColumn(project.target_col);
        setTargetColumnDraft(project.target_col);
      } else {
        setProjectTargetColumn("");
        setTargetColumnDraft("");
      }
      try {
        const res = await fetch(`${API_URL}/projects/select`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            auth_token: authToken,
            project_id: project.id || project.root,
            project_root: project.root,
          }),
        });
        if (res.ok) await hydrateProjectFiles(project.root);
      } catch (e) {
        console.warn("Failed to select project on backend:", e);
      }
      addMsg(
        "team",
        "orchestrator",
        `Switched active project to: **${project.name}** (${project.root})`,
        "STATUS"
      );
    },
    [authToken, hydrateProjectFiles, setProjectName, setProjectRoot, setProjectTargetColumn, setTargetColumnDraft, addMsg]
  );

  const chatHistoryKey = authEmail && projectRoot
    ? `ai_platform_chat_history:${authEmail.toLowerCase()}:${projectRoot}`
    : "";

  useEffect(() => {
    if (!chatHistoryKey || restoredChatKeyRef.current === chatHistoryKey) return;
    const raw = safeLocalStorageGet(chatHistoryKey);
    restoredChatKeyRef.current = chatHistoryKey;
    if (!raw) return;
    try {
      const saved = JSON.parse(raw);
      if (Array.isArray(saved) && saved.length > 0) {
        suppressChatSaveKeyRef.current = chatHistoryKey;
        setChatList(saved);
        queueMicrotask(() => {
          if (suppressChatSaveKeyRef.current === chatHistoryKey) {
            suppressChatSaveKeyRef.current = null;
          }
        });
      }
    } catch {
      safeLocalStorageRemove(chatHistoryKey);
    }
  }, [chatHistoryKey, setChatList]);

  useEffect(() => {
    if (!chatHistoryKey || suppressChatSaveKeyRef.current === chatHistoryKey) return;
    safeLocalStorageSet(chatHistoryKey, JSON.stringify(chatList));
  }, [chatHistoryKey, chatList]);

  const createProject = useCallback(async () => {
    if (!newProjectName.trim()) return;
    setIsBusy(true);
    try {
      const res = await fetch(`${API_URL}/projects/open`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newProjectName,
          root: newProjectRoot,
          owner: authEmail,
          auth_token: authToken,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        await fetchProjects();
        setProjectName(data.name || newProjectName);
        setProjectRoot(data.root || newProjectRoot);
        setShowNewProjectModal(false);
        setNewProjectName("");
        setNewProjectRoot("");
        addMsg(
          "team",
          "orchestrator",
          `Initialized new workspace: **${data.name || newProjectName}** at \`${data.root || newProjectRoot}\``,
          "STATUS"
        );
      } else {
        const err = await res.json().catch(() => ({}));
        addMsg("team", "orchestrator", `Failed to create/open project: ${err?.detail || "unknown"}`, "ALERT");
      }
    } catch (e) {
      console.error("Failed to create project:", e);
      addMsg("team", "orchestrator", `Error creating workspace: ${String(e)}`, "ALERT");
    } finally {
      setIsBusy(false);
    }
  }, [
    newProjectName,
    newProjectRoot,
    authEmail,
    authToken,
    fetchProjects,
    setProjectName,
    setProjectRoot,
    addMsg,
    setIsBusy,
  ]);

  const _getOrCreateGroupChat = useCallback(
    (members: string[], title?: string): string => {
      const existing = chatList.find(
        (c) =>
          c.type === "group" &&
          c.members?.slice().sort().join(",") === members.slice().sort().join(",")
      );
      if (existing) return existing.id;

      const newId = `group_${Date.now()}`;
      const newChat: Chat = {
        id: newId,
        type: "group",
        title: title || `Group (${members.map((m) => agents[m]?.shortName || m).join(", ")})`,
        members,
        messages: [],
        unread: 0,
        isNew: true,
      };
      setChatList((prev) => [...prev, newChat]);
      return newId;
    },
    [chatList, agents, setChatList]
  );

  const handleWsMessage = useCallback(
    (evt: any) => {
      if (!evt || typeof evt !== "object") return;
      const { type, payload, ...rest } = evt;

      if (type === "status") {
        if (payload?.agent_id && payload?.status) {
          if (agents[payload.agent_id]) {
            agents[payload.agent_id].status = payload.status;
          }
        }
      } else if (type === "message") {
        const { chat_id, from, content, tag } = payload || rest || {};
        if (chat_id && from && content) {
          addMsg(chat_id, from, content, tag);
          setTyping(chat_id, null);
        }
      } else if (type === "p2p_message") {
        const { id, from, to, content, timestamp } = payload || rest || {};
        if (from && to && content) {
          setP2pLog((prev) => [
            ...prev,
            {
              id: id || `${Date.now()}_${Math.random()}`,
              from,
              to,
              content,
              timestamp: timestamp || Date.now(),
            },
          ]);
        }
      } else if (type === "file_log" || type === "file_written") {
        const extra = evt.extra || payload || rest.extra || rest || {};
        const agent_id = extra.agent_id || payload?.agent_id || rest.agent_id;
        const filename = extra.filename || payload?.filename || rest.filename;
        const full_path = extra.full_path || extra.path || payload?.full_path || payload?.path || rest.full_path || rest.path;
        if (agent_id && filename) {
          upsertFile(agent_id, filename, full_path || filename);
        }
      } else if (type === "files_snapshot") {
        const extra = evt.extra || payload || rest.extra || rest || {};
        const files = extra.files || payload?.files || rest.files || [];
        if (Array.isArray(files)) {
          files.forEach((f: any) => {
            if (f.agent_id && f.filename) {
              upsertFile(f.agent_id, f.filename, f.full_path || f.filename);
            }
          });
        }
      } else if (type === "typing") {
        const { chat_id, agent_id } = payload || rest || {};
        if (chat_id) {
          setTyping(chat_id, agent_id || null);
        }
      } else if (type === "group_chat") {
        const { members, title } = payload || rest || {};
        if (Array.isArray(members) && members.length > 0) {
          _getOrCreateGroupChat(members, title);
        }
      } else if (type === "dataset_uploaded") {
        const extra = evt.extra || payload || rest.extra || rest || {};
        const filename = extra.filename || payload?.filename || rest.filename || evt.filename || "dataset.csv";
        const fullPath = extra.path || payload?.path || rest.path || evt.path || "";
        setDatasetReady(true);
        setDatasetInfo({ filename, path: fullPath });
        setUploading(false);
        addMsg(
          "team",
          "orchestrator",
          `Dataset uploaded: **${filename}**\nPATH: \`${fullPath}\``,
          "STATUS"
        );
      } else if (type === "project_settings") {
        const targetCol = payload?.target_col || rest?.target_col || evt.target_col || "";
        if (targetCol) {
          setProjectTargetColumn(targetCol);
          setTargetColumnDraft(targetCol);
        }
      } else if (type === "github_connected" || type === "github_error") {
        const msg =
          (payload || rest)?.message ||
          (type === "github_connected" ? "GitHub connected successfully" : "Failed to connect GitHub");
        addMsg("team", "orchestrator", msg, type === "github_connected" ? "STATUS" : "ALERT");
      }
    },
    [
      agents,
      addMsg,
      setTyping,
      setP2pLog,
      upsertFile,
      _getOrCreateGroupChat,
      setDatasetReady,
      setDatasetInfo,
      setUploading,
      setProjectTargetColumn,
      setTargetColumnDraft,
    ]
  );

  const wsUrl =
    typeof window !== "undefined"
      ? `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws`
      : "/ws";

  const { send, readyState, wsRef } = useWebSocket({
    url: wsUrl,
    onMessage: handleWsMessage,
    onOpen: () => setConnStatus("connected"),
    onClose: () => setConnStatus("disconnected"),
    onError: () => setConnStatus("disconnected"),
  });

  const currentChat = chatList.find((c) => c.id === activeChat) || chatList[0] || null;
  const currentInput = inputs[activeChat] || "";

  const handleSend = useCallback(() => {
    const text = currentInput.trim();
    if (!text || isBusy) return;

    setInputs((prev) => ({ ...prev, [activeChat]: "" }));
    addMsg(activeChat, "user", text);

    if (readyState === WebSocket.OPEN && send) {
      send({
        type: "user_message",
        payload: {
          chat_id: activeChat,
          content: text,
          project_root: projectRoot,
          target_col: projectTargetColumn,
          auth_token: authToken,
        },
      });
    } else {
      setTyping(activeChat, activeChat === "team" ? "orchestrator" : activeChat);
      setTimeout(() => {
        const responder = activeChat === "team" ? "orchestrator" : activeChat;
        addMsg(
          activeChat,
          responder,
          `[Offline Mode] Received task: "${text.slice(0, 50)}...". Processing locally.`,
          "REPORT"
        );
        setTyping(activeChat, null);
      }, 1400);
    }
  }, [
    currentInput,
    isBusy,
    activeChat,
    setInputs,
    addMsg,
    readyState,
    send,
    projectRoot,
    projectTargetColumn,
    authToken,
    setTyping,
  ]);

  const handleDatasetUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      setUploading(true);
      const reader = new FileReader();
      reader.onload = () => {
        const buffer = reader.result as ArrayBuffer;
        const bytes = new Uint8Array(buffer);
        let binary = "";
        const chunk = 0x8000;
        for (let i = 0; i < bytes.length; i += chunk) {
          binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
        }
        wsRef.current?.send(
          JSON.stringify({
            type: "dataset_upload",
            filename: file.name,
            content_b64: btoa(binary),
            to: "orchestrator",
            auth_token: authToken,
          })
        );
      };
      reader.onerror = () => {
        setUploading(false);
        addMsg("team", "orchestrator", "Error reading file for upload.", "ALERT");
      };
      reader.readAsArrayBuffer(file);
      e.target.value = "";
    },
    [wsRef, authToken, addMsg, setUploading]
  );

  const handlePhase3Check = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    setActiveChat("team");
    addMsg("team", "user", "Run readiness checks on uploaded dataset");
    wsRef.current.send(
      JSON.stringify({
        type: "phase3_check",
        task_id: `check_${Date.now()}`,
        auth_token: authToken,
        project_root: projectRoot,
      })
    );
  }, [wsRef, authToken, projectRoot, setActiveChat, addMsg]);

  const handleRunDirect = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    setActiveChat("team");
    addMsg("team", "user", "Run direct analysis on dataset without readiness checks");
    wsRef.current.send(
      JSON.stringify({
        type: "user_message",
        payload: {
          chat_id: "team",
          content: "Run direct analysis and modeling on dataset without readiness checks.",
          project_root: projectRoot,
          target_col: projectTargetColumn,
          auth_token: authToken,
        },
      })
    );
  }, [wsRef, authToken, projectRoot, projectTargetColumn, setActiveChat, addMsg]);

  const handleGithubConnectAction = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    setActiveChat("team");
    addMsg("team", "user", `Connecting GitHub repo: **${githubOwner}/${githubRepo}** (${githubVisibility})`);
    wsRef.current.send(
      JSON.stringify({
        type: "github_connect",
        token: githubToken.trim(),
        owner: githubOwner.trim(),
        repo: githubRepo.trim(),
        visibility: githubVisibility,
        task_id: `gh_${Date.now()}`,
        auth_token: authToken,
      })
    );
  }, [wsRef, githubToken, githubOwner, githubRepo, githubVisibility, authToken, setActiveChat, addMsg]);

  useEffect(() => {
    scrollToBottom();
  }, [currentChat?.messages?.length, typingIn[activeChat], scrollToBottom]);

  if (!authToken) {
    return (
      <AuthPage
        authMode={authMode}
        setAuthMode={setAuthMode}
        authEmail={authEmail}
        setAuthEmail={setAuthEmail}
        authPassword={authPassword}
        setAuthPassword={setAuthPassword}
        authLoading={authLoading}
        setAuthLoading={setAuthLoading}
        authError={authError}
        setAuthError={setAuthError}
        setAuthToken={setAuthToken}
        onSuccessReset={fetchProjects}
      />
    );
  }

  return (
    <div className="relative flex flex-col h-screen w-screen bg-[#050508] text-zinc-100 overflow-hidden font-sans select-none">
      {/* Background Layers */}
      <ParticleBackground className="opacity-80" />
      <Aurora className="opacity-40" />
      <MeshGradient className="opacity-30" />
      <Vignette />

      {/* Top Application Header */}
      <header className="relative z-20 flex flex-wrap items-center justify-between gap-4 h-auto min-h-[56px] px-5 py-2.5 bg-[#09090f]/95 backdrop-blur-md border-b border-zinc-800/80 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-indigo-700 flex items-center justify-center font-mono font-bold text-white shadow-lg shadow-indigo-500/20 text-xs tracking-wider flex-shrink-0">
            AI
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-zinc-100 flex items-center gap-2">
              Autonomous Engineering Platform
              <span className="text-[10px] uppercase tracking-widest px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono">
                v2.4
              </span>
            </h1>
          </div>
        </div>

        {/* Project Indicator & Connection Status */}
        <div className="flex flex-wrap items-center gap-4 text-xs font-mono ml-auto">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900/90 border border-zinc-800/90 shadow-sm">
            <span className="text-zinc-500 font-bold">PROJECT:</span>
            <span className="text-zinc-200 font-medium truncate max-w-[200px]">
              {projectName || "No Project Active"}
            </span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900/90 border border-zinc-800/90 shadow-sm">
            <span
              className={`w-2 h-2 rounded-full ${
                connStatus === "connected"
                  ? "bg-emerald-400 shadow-[0_0_8px_#34d399]"
                  : connStatus === "connecting"
                  ? "bg-amber-400 animate-pulse"
                  : "bg-rose-500"
              }`}
            />
            <span className="text-zinc-300 capitalize text-xs font-semibold tracking-wide">{connStatus}</span>
          </div>
        </div>
      </header>

      {/* Main 3-Column Layout */}
      <div className="relative z-10 flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <aside className="w-[240px] xl:w-[260px] flex-shrink-0 flex flex-col bg-[#07070c]/95 border-r border-zinc-800/80 z-10 overflow-hidden">
          <Sidebar
            agents={agents}
            activeChat={activeChat}
            onSelectChat={setActiveChat}
            chatList={chatList}
            projects={projects}
            activeProjectRoot={projectRoot}
            onSelectProject={selectProject}
            onRefreshProjects={fetchProjects}
          />
        </aside>

        {/* Center Chat View */}
        <main className="flex-1 flex flex-col min-w-0 bg-[#06060a]/60 backdrop-blur-sm relative overflow-hidden">
          <ChatHeader
            chat={currentChat}
            agents={agents}
            backendLive={connStatus === "connected"}
            projectRoot={projectRoot}
            authEmail={authEmail}
            onNewChat={() => setShowNewProjectModal(true)}
            onConnectGithub={() => setShowGithubConnect(true)}
            onGetMLEngineer={() => setShowMLWorkerHelp(true)}
            onAccessFiles={() => setShowFilesPanel(true)}
            onLogout={handleLogout}
          />

          {/* Target Column & Dataset Toolbar */}
          <div className="px-5 py-2.5 bg-zinc-900/80 border-b border-zinc-800/80 flex flex-wrap items-center justify-between gap-3 text-xs flex-shrink-0 min-h-[48px]">
            <div className="flex flex-wrap items-center gap-2 min-w-0">
              <span className="text-zinc-400 font-mono font-bold tracking-wider flex-shrink-0">TARGET COL:</span>
              <input
                type="text"
                name="dataset_target_variable_v3"
                id="dataset_target_variable_v3"
                readOnly={isTargetColReadOnly}
                onFocus={() => setIsTargetColReadOnly(false)}
                autoComplete="off"
                autoCorrect="off"
                autoCapitalize="none"
                spellCheck="false"
                data-lpignore="true"
                placeholder="e.g. churn, price, label..."
                value={targetColumnDraft}
                onChange={(e) => setTargetColumnDraft(e.target.value)}
                className="bg-zinc-950/90 border border-zinc-800 px-3 py-1.5 rounded-lg text-zinc-100 font-mono text-xs focus:outline-none focus:border-indigo-500 transition-colors w-40 shadow-inner"
              />
              <Button
                size="xs"
                variant="outline"
                onClick={() => setProjectTargetColumn(targetColumnDraft)}
                className="text-xs px-3 py-1.5 font-mono bg-zinc-900 hover:bg-zinc-800 flex-shrink-0"
              >
                Save
              </Button>
              {projectTargetColumn && (
                <span className="text-emerald-400 font-mono text-xs bg-emerald-950/50 px-2.5 py-1 rounded-lg border border-emerald-800/60 shadow-sm">
                  Active: {projectTargetColumn}
                </span>
              )}
            </div>

            {/* Dataset status bar */}
            {datasetReady && datasetInfo && (
              <div className="flex items-center gap-2 bg-emerald-950/40 border border-emerald-800/60 px-3 py-1.5 rounded-lg text-xs font-mono text-emerald-300 shadow-sm ml-auto">
                <span>CSV Ready: {datasetInfo.filename || "dataset.csv"}</span>
              </div>
            )}
          </div>

          {/* Messages Feed */}
          <div className="flex-1 overflow-y-auto px-6 py-5 flex flex-col gap-4 custom-scrollbar min-w-0">
            {currentChat?.messages.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center text-zinc-500 text-sm gap-2">
                <div className="w-12 h-12 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center text-zinc-400 font-mono text-lg shadow-sm">
                  #
                </div>
                <p className="font-semibold text-zinc-300">No messages yet in this thread.</p>
                <p className="text-xs text-zinc-500 font-mono">Start the conversation or upload a dataset below.</p>
              </div>
            ) : (
              currentChat?.messages.map((msg) => <Message key={msg.id} msg={msg} />)
            )}

            {/* Typing Indicator */}
            {typingIn[activeChat] && (
              <motion.div
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center gap-3 text-zinc-400 text-xs px-2 py-1"
              >
                <div className="flex gap-1.5 items-center bg-zinc-900/90 border border-zinc-800 px-3.5 py-2 rounded-full shadow-sm">
                  <span
                    className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0ms" }}
                  />
                  <span
                    className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  />
                  <span
                    className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  />
                  <span className="ml-2 font-mono text-xs text-zinc-300">
                    {agents[typingIn[activeChat]!]?.name || typingIn[activeChat]} is working...
                  </span>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Footer Area */}
          <div className="p-4 bg-[#08080f]/95 border-t border-zinc-800/80 backdrop-blur-md flex-shrink-0">
            <div className="max-w-4xl mx-auto flex flex-col gap-3">
              {/* File Upload & Readiness Actions Row */}
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-2.5">
                  <input
                    ref={uploadRef}
                    type="file"
                    accept=".csv,text/csv"
                    onChange={handleDatasetUpload}
                    className="hidden"
                  />
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => uploadRef.current?.click()}
                    disabled={uploading}
                    className="h-10 px-4 text-xs font-bold bg-gradient-to-r from-indigo-950/80 via-purple-950/80 to-zinc-900 hover:from-indigo-900 hover:to-purple-900 border border-indigo-500/50 hover:border-indigo-400 text-indigo-200 flex items-center gap-2 shadow-lg shadow-indigo-950/40 transition-all rounded-xl"
                  >
                    <span className="text-base">📁</span>
                    <span>{uploading ? "⏳ Uploading..." : "Upload CSV Dataset"}</span>
                  </Button>

                  {datasetReady && (
                    <div className="flex items-center gap-2">
                      <Button
                        size="sm"
                        onClick={handlePhase3Check}
                        className="h-10 text-xs font-bold bg-emerald-700 hover:bg-emerald-600 text-white border border-emerald-500/40 shadow-lg shadow-emerald-950/50 rounded-xl px-3 transition-all"
                      >
                        ⚡ Check Agents Readiness
                      </Button>
                      <Button
                        size="sm"
                        onClick={handleRunDirect}
                        className="h-10 text-xs font-bold bg-indigo-700 hover:bg-indigo-600 text-white border border-indigo-500/40 shadow-lg shadow-indigo-950/50 rounded-xl px-3 transition-all"
                      >
                        ▶ Run Without Check
                      </Button>
                    </div>
                  )}
                </div>

                <div className="text-xs text-zinc-400 font-mono ml-auto">
                  {datasetReady ? "CSV Dataset Attached" : "Supported: Tabular CSV datasets"}
                </div>
              </div>

              {/* Message Input Box - Separated Textarea and Send Button for Zero Overlap */}
              <div className="flex items-end gap-3 bg-zinc-900/90 border border-zinc-800 rounded-2xl p-3 focus-within:border-indigo-500/80 focus-within:ring-1 focus-within:ring-indigo-500/30 transition-all shadow-lg">
                <Textarea
                  value={currentInput}
                  onChange={(e) => setInputs((prev) => ({ ...prev, [activeChat]: e.target.value }))}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder={`Send instruction to ${
                    activeChat === "team" ? "the engineering team" : agents[activeChat]?.name || activeChat
                  }... (Shift+Enter for newline)`}
                  rows={2}
                  className="flex-1 min-w-0 bg-transparent border-none text-sm text-zinc-100 placeholder-zinc-500 focus:ring-0 resize-none px-2 py-1 font-sans leading-relaxed"
                />
                <Button
                  size="sm"
                  onClick={handleSend}
                  disabled={!currentInput.trim() || isBusy}
                  className="h-11 px-5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-semibold shadow-md shadow-indigo-600/20 flex items-center gap-2 transition-all flex-shrink-0"
                >
                  <span>Send</span>
                  <span className="font-mono text-sm">↵</span>
                </Button>
              </div>
            </div>
          </div>
        </main>

        {/* Right Activity Panel */}
        <aside className="w-[300px] xl:w-[320px] flex-shrink-0 flex flex-col bg-[#07070c]/95 border-l border-zinc-800/80 z-10 overflow-hidden">
          <RightPanel
            p2pLog={p2pLog}
            fileLog={fileLog}
            projectRoot={projectRoot}
            onSelectFile={(entry) => {
              setSelectedFile(`${entry.agent_id}/${entry.filename}`);
              setShowFilesPanel(true);
            }}
          />
        </aside>
      </div>

      {/* Modals */}
      <GithubConnectModal
        isOpen={showGithubConnect}
        onClose={() => setShowGithubConnect(false)}
        githubToken={githubToken}
        setGithubToken={setGithubToken}
        githubOwner={githubOwner}
        setGithubOwner={setGithubOwner}
        githubRepo={githubRepo}
        setGithubRepo={setGithubRepo}
        githubVisibility={githubVisibility}
        setGithubVisibility={setGithubVisibility}
        onConnect={handleGithubConnectAction}
      />

      <MLEngineerModal
        isOpen={showMLWorkerHelp}
        onClose={() => setShowMLWorkerHelp(false)}
        authToken={authToken}
        workerStatus={workerStatus}
        setWorkerStatus={setWorkerStatus}
        workerToken={workerToken}
        setWorkerToken={setWorkerToken}
        workerProjectPath={workerProjectPath}
        setWorkerProjectPath={setWorkerProjectPath}
      />

      <FilesPanelModal
        isOpen={showFilesPanel}
        onClose={() => setShowFilesPanel(false)}
        authToken={authToken}
        projectRoot={projectRoot}
        filesIndex={filesIndex}
        setFilesIndex={setFilesIndex}
        filesLoading={filesLoading}
        setFilesLoading={setFilesLoading}
        filesError={filesError}
        setFilesError={setFilesError}
        fileSearch={fileSearch}
        setFileSearch={setFileSearch}
        selectedFile={selectedFile}
        setSelectedFile={setSelectedFile}
        fileContent={fileContent}
        setFileContent={setFileContent}
        fileBinary={fileBinary}
        setFileBinary={setFileBinary}
      />

      {/* New Project Modal */}
      <AnimatePresence>
        {showNewProjectModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-md p-4 font-mono"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-md bg-[#0c0c14] border border-zinc-800 rounded-2xl shadow-2xl p-6 flex flex-col gap-4"
            >
              <div className="flex justify-between items-center border-b border-zinc-800 pb-3">
                <h3 className="text-base font-bold text-white font-sans">Initialize Workspace Project</h3>
                <button
                  onClick={() => setShowNewProjectModal(false)}
                  className="text-zinc-400 hover:text-zinc-200 text-lg"
                >
                  ×
                </button>
              </div>

              <div className="flex flex-col gap-4">
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-zinc-400 uppercase tracking-wider">Project Name</label>
                  <Input
                    placeholder="e.g. Customer Churn Analysis"
                    value={newProjectName}
                    onChange={(e) => setNewProjectName(e.target.value)}
                    className="bg-zinc-900 border-zinc-800 text-sm"
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-zinc-400 uppercase tracking-wider">
                    Local Data Root / Directory Path (Optional)
                  </label>
                  <Input
                    placeholder="e.g. ./datasets/churn (or leave blank for auto-created folder)"
                    value={newProjectRoot}
                    onChange={(e) => setNewProjectRoot(e.target.value)}
                    className="bg-zinc-900 border-zinc-800 text-sm font-mono"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-2.5 pt-3 border-t border-zinc-800 mt-2 font-sans">
                <Button variant="ghost" size="sm" onClick={() => setShowNewProjectModal(false)}>
                  Cancel
                </Button>
                <Button
                  size="sm"
                  onClick={createProject}
                  disabled={!newProjectName.trim() || isBusy}
                  className="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold"
                >
                  Create Workspace
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
