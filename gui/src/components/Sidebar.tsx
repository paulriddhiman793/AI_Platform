"use client";

import { Avatar } from "./Avatar";
import { StatusDot } from "./StatusDot";
import type { Project, Chat } from "@/types";

interface SidebarProps {
  agents: Record<string, any>;
  activeChat: string;
  onSelectChat: (chatId: string) => void;
  chatList: Chat[];
  projects: Project[];
  activeProjectRoot: string | null;
  onSelectProject: (project: Project) => void;
  onRefreshProjects: () => void;
}

export function Sidebar({
  agents,
  activeChat,
  onSelectChat,
  chatList,
  projects,
  activeProjectRoot,
  onSelectProject,
  onRefreshProjects,
}: SidebarProps) {
  const groupChats = chatList.filter((c) => c.type === "group");

  return (
    <div className="w-full h-full bg-[#050508] flex flex-col font-mono select-none overflow-hidden flex-shrink-0">
      {/* Projects Section */}
      <div className="p-3 border-b border-zinc-800/80 flex flex-col gap-2 max-h-48 overflow-hidden flex-shrink-0">
        <div className="flex items-center justify-between text-[11px] font-bold text-zinc-400 uppercase tracking-wider px-1">
          <span>Workspace Projects</span>
          <button
            onClick={onRefreshProjects}
            className="text-[10px] text-indigo-400 hover:text-indigo-300 transition-colors"
            title="Refresh projects"
          >
            🔄 Refresh
          </button>
        </div>
        <div className="flex-1 overflow-y-auto space-y-1 pr-1 custom-scrollbar">
          {projects.length === 0 ? (
            <div className="text-[11px] text-zinc-600 px-1 py-3 text-center italic">
              No previous projects. Click "New Chat" to create one.
            </div>
          ) : (
            projects.map((p) => {
              const isActive = activeProjectRoot && p.root === activeProjectRoot;
              return (
                <button
                  key={p.id}
                  onClick={() => onSelectProject(p)}
                  className={`w-full text-left px-2.5 py-2 rounded-xl text-xs transition-all flex flex-col gap-0.5 border ${
                    isActive
                      ? "bg-indigo-950/60 border-indigo-500/40 text-indigo-200 shadow-sm"
                      : "bg-zinc-900/40 border-transparent text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"
                  }`}
                  title={p.root}
                >
                  <div className="font-semibold truncate font-sans tracking-wide">
                    {p.name || p.id}
                  </div>
                  {p.created || p.modified ? (
                    <div className="text-[10px] text-zinc-500 truncate font-mono">
                      {p.created || p.modified}
                    </div>
                  ) : null}
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* Main Chats Section */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4 custom-scrollbar">
        {/* Team Chat Button */}
        <div>
          <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider px-2 mb-1.5 font-sans">
            Orchestrator
          </div>
          <button
            onClick={() => onSelectChat("team")}
            className={`w-full text-left px-3 py-2.5 rounded-xl transition-all flex items-center gap-3 border ${
              activeChat === "team"
                ? "bg-indigo-950/70 border-indigo-500/50 text-indigo-200 shadow-md shadow-indigo-950/40"
                : "bg-transparent border-transparent text-zinc-400 hover:bg-zinc-900/60 hover:text-zinc-200"
            }`}
          >
            <div className="w-8 h-8 rounded-xl bg-indigo-900/60 border border-indigo-500/40 flex items-center justify-center text-[10px] font-bold text-indigo-300 flex-shrink-0">
              TEAM
            </div>
            <div className="flex-1 min-w-0 truncate">
              <div className="text-xs font-bold font-sans tracking-wide truncate">Team Chat</div>
              <div className="text-[10px] text-zinc-500 font-mono truncate">
                All agent reports & updates
              </div>
            </div>
            {activeChat === "team" && (
              <div className="w-2 h-2 rounded-full bg-indigo-400 shadow-[0_0_8px_rgba(129,140,248,0.8)] flex-shrink-0" />
            )}
          </button>
        </div>

        {/* Group Chats */}
        {groupChats.length > 0 && (
          <div>
            <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider px-2 mb-1.5 font-sans">
              Group Threads
            </div>
            <div className="space-y-1">
              {groupChats.map((c) => (
                <button
                  key={c.id}
                  onClick={() => onSelectChat(c.id)}
                  className={`w-full text-left px-3 py-2.5 rounded-xl transition-all flex items-center gap-3 border ${
                    activeChat === c.id
                      ? "bg-purple-950/70 border-purple-500/50 text-purple-200 shadow-md shadow-purple-950/40"
                      : "bg-transparent border-transparent text-zinc-400 hover:bg-zinc-900/60 hover:text-zinc-200"
                  }`}
                >
                  <div className="w-8 h-8 rounded-xl bg-purple-900/60 border border-purple-500/40 flex items-center justify-center text-[10px] font-bold text-purple-300 flex-shrink-0">
                    GRP
                  </div>
                  <div className="flex-1 min-w-0 truncate">
                    <div className="text-xs font-bold font-sans tracking-wide truncate">
                      {c.title || "Group Thread"}
                    </div>
                    <div className="text-[10px] text-zinc-500 font-mono truncate">
                      {(c.members || []).length} agents
                    </div>
                  </div>
                  {c.unread ? (
                    <span className="px-1.5 py-0.5 text-[10px] rounded-full bg-purple-600 text-white font-bold flex-shrink-0">
                      {c.unread}
                    </span>
                  ) : activeChat === c.id ? (
                    <div className="w-2 h-2 rounded-full bg-purple-400 shadow-[0_0_8px_rgba(192,132,252,0.8)] flex-shrink-0" />
                  ) : null}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Direct Agents */}
        <div>
          <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider px-2 mb-1.5 font-sans">
            Direct Agents
          </div>
          <div className="space-y-1">
            {Object.values(agents).map((a) => {
              const isActive = activeChat === a.id;
              return (
                <button
                  key={a.id}
                  onClick={() => onSelectChat(a.id)}
                  className={`w-full text-left px-3 py-2.5 rounded-xl transition-all flex items-center gap-3 border ${
                    isActive
                      ? "bg-zinc-900/90 border-zinc-700 text-white shadow-md"
                      : "bg-transparent border-transparent text-zinc-400 hover:bg-zinc-900/60 hover:text-zinc-200"
                  }`}
                >
                  <div className="relative flex-shrink-0">
                    <Avatar agentId={a.id} size={32} />
                    <div className="absolute -bottom-0.5 -right-0.5">
                      <StatusDot status={a.status || "idle"} showLabel={false} />
                    </div>
                  </div>
                  <div className="flex-1 min-w-0 truncate">
                    <div
                      style={{ color: isActive ? "#fff" : a.color }}
                      className="text-xs font-bold font-sans tracking-wide truncate"
                    >
                      {a.name}
                    </div>
                    <div className="text-[10px] text-zinc-500 font-mono truncate">
                      {a.role}
                    </div>
                  </div>
                  {isActive && (
                    <div
                      style={{ backgroundColor: a.color, boxShadow: `0 0 8px ${a.color}` }}
                      className="w-2 h-2 rounded-full flex-shrink-0"
                    />
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}