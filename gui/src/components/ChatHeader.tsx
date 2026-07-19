"use client";

import { Avatar } from "./Avatar";
import { StatusDot } from "./StatusDot";
import type { Chat, Agent } from "@/types";

interface ChatHeaderProps {
  chat: Chat | null;
  agents: Record<string, Agent>;
  backendLive?: boolean;
  projectRoot?: string | null;
  authEmail?: string;
  onNewChat?: () => void;
  onConnectGithub?: () => void;
  onGetMLEngineer?: () => void;
  onAccessFiles?: () => void;
  onLogout?: () => void;
}

export function ChatHeader({
  chat,
  agents,
  backendLive = true,
  projectRoot,
  authEmail,
  onNewChat,
  onConnectGithub,
  onGetMLEngineer,
  onAccessFiles,
  onLogout,
}: ChatHeaderProps) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3 bg-[#07070a]/95 border-b border-zinc-800/80 backdrop-blur-md select-none flex-shrink-0 min-h-[64px]">
      {/* Left side: Active Chat info */}
      <div className="flex items-center gap-3.5 min-w-[200px] flex-1 mr-2 overflow-hidden">
        {!chat ? (
          <div className="text-xs text-zinc-500">Select a chat</div>
        ) : chat.id === "team" ? (
          <>
            <div className="w-9 h-9 rounded-xl bg-indigo-950/80 border border-indigo-500/40 flex items-center justify-center text-[11px] font-bold text-indigo-300 shadow-sm flex-shrink-0">
              TEAM
            </div>
            <div className="min-w-0 flex-1 truncate">
              <div className="text-sm font-bold text-indigo-300 tracking-wide font-sans truncate">
                Team Chat
              </div>
              <div className="text-[11px] text-zinc-500 font-mono truncate">
                Real-time multi-agent orchestrator channel
              </div>
            </div>
          </>
        ) : chat.type === "group" ? (
          <>
            <div className="w-9 h-9 rounded-xl bg-purple-950/80 border border-purple-500/40 flex items-center justify-center text-[11px] font-bold text-purple-300 shadow-sm flex-shrink-0">
              GRP
            </div>
            <div className="min-w-0 flex-1 truncate">
              <div className="text-sm font-bold text-purple-300 tracking-wide font-sans truncate">
                {chat.title}
              </div>
              <div className="flex items-center gap-2 mt-0.5 font-mono text-[11px] truncate">
                {(chat.members || []).map((m) => {
                  const a = agents[m];
                  if (!a) return null;
                  return (
                    <span key={a.id} style={{ color: a.color }} className="flex items-center gap-1 flex-shrink-0">
                      <span>{a.icon}</span>
                      <span>{a.shortName}</span>
                    </span>
                  );
                })}
              </div>
            </div>
          </>
        ) : (
          (() => {
            const a = agents[chat.id];
            return (
              <>
                <div className="flex-shrink-0">
                  <Avatar agentId={chat.id} size={36} />
                </div>
                <div className="min-w-0 flex-1 truncate">
                  <div
                    style={{ color: a?.color }}
                    className="text-sm font-bold tracking-wide font-sans flex items-center gap-2 truncate"
                  >
                    <span className="truncate">{a?.name || chat.id}</span>
                    <StatusDot status={a?.status || "idle"} />
                  </div>
                  <div className="text-[11px] text-zinc-500 font-mono truncate">
                    {a?.role || "Direct Agent Channel"}
                  </div>
                </div>
              </>
            );
          })()
        )}
      </div>

      {/* Right side: Actions Toolbar & User Auth info */}
      <div className="flex flex-wrap items-center gap-2 font-mono flex-shrink-0 ml-auto">
        <div
          className={`px-2.5 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wider border flex items-center gap-1.5 ${
            backendLive
              ? "bg-emerald-950/60 text-emerald-300 border-emerald-500/40 shadow-sm"
              : "bg-amber-950/60 text-amber-300 border-amber-500/40 shadow-sm"
          }`}
          title={backendLive ? "Real Python backend connected" : "Simulated mock mode active"}
        >
          <span className={`w-1.5 h-1.5 rounded-full ${backendLive ? "bg-emerald-400 animate-pulse" : "bg-amber-400"}`} />
          <span>{backendLive ? "Live Agents" : "Mock Mode"}</span>
        </div>
        {onNewChat && (
          <button
            onClick={onNewChat}
            className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-zinc-900/90 hover:bg-zinc-800 text-zinc-200 border border-zinc-700/80 hover:border-zinc-500 transition-all shadow-sm flex items-center gap-1.5 flex-shrink-0"
            title="Start a new chat project"
          >
            <span>✨ New Chat</span>
          </button>
        )}
        {onConnectGithub && (
          <button
            onClick={onConnectGithub}
            className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-indigo-950/60 hover:bg-indigo-900/80 text-indigo-300 border border-indigo-500/40 hover:border-indigo-400/80 transition-all shadow-sm flex items-center gap-1.5 flex-shrink-0"
            title="Connect GitHub repository"
          >
            <svg className="w-3.5 h-3.5 fill-current text-indigo-400 flex-shrink-0" viewBox="0 0 24 24">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
            </svg>
            <span>Connect GitHub</span>
          </button>
        )}
        {onGetMLEngineer && (
          <button
            onClick={onGetMLEngineer}
            className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-amber-950/60 hover:bg-amber-900/80 text-amber-300 border border-amber-500/40 hover:border-amber-400/80 transition-all shadow-sm flex items-center gap-1.5 flex-shrink-0"
            title="Pair local ML Engineer worker"
          >
            <span>⚡ Get ML Engineer</span>
          </button>
        )}
        {onAccessFiles && (
          <button
            onClick={onAccessFiles}
            disabled={!projectRoot}
            className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-emerald-950/60 hover:bg-emerald-900/80 disabled:opacity-40 disabled:hover:bg-emerald-950/60 text-emerald-300 border border-emerald-500/40 hover:border-emerald-400/80 transition-all shadow-sm flex items-center gap-1.5 flex-shrink-0"
            title={projectRoot ? "View & download project files" : "Start a project to access files"}
          >
            <span>📁 Access Files</span>
          </button>
        )}

        {/* User Pill & Logout */}
        {authEmail && (
          <div className="flex items-center gap-2.5 ml-2 pl-3 border-l border-zinc-800 flex-shrink-0">
            <span className="text-xs text-zinc-300 bg-zinc-950 px-2.5 py-1.5 rounded-lg border border-zinc-800/80 truncate max-w-[150px]">
              {authEmail}
            </span>
            {onLogout && (
              <button
                onClick={onLogout}
                className="px-2.5 py-1.5 text-xs font-semibold text-red-400/90 hover:text-red-300 hover:bg-red-950/40 rounded-lg border border-transparent hover:border-red-900/50 transition-colors flex-shrink-0"
                title="Log out of workspace"
              >
                Logout
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}