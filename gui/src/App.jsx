import { useState, useEffect, useRef, useCallback } from "react";
import { AGENTS, detectScenario, detectDirectResponse } from "./data/agents.js";

// â”€â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS_URL = import.meta.env.VITE_WS_URL || `${API_URL.replace("http://", "ws://").replace("https://", "wss://")}/ws`;

const safeLocalStorageGet = (key) => {
  try { return localStorage.getItem(key) || ""; } catch { return ""; }
};
const safeLocalStorageSet = (key, value) => {
  try { localStorage.setItem(key, value); } catch {}
};
const safeLocalStorageRemove = (key) => {
  try { localStorage.removeItem(key); } catch {}
};

let _id = 0;
const uid = () => ++_id;
const fmt = (ts) => new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

const TAG = {
  STATUS: { bg: "#0d1f0d", color: "#4ade80", border: "#1a3a1a" },
  REPORT: { bg: "#0a1628", color: "#60a5fa", border: "#1a3060" },
  ALERT:  { bg: "#2a0808", color: "#f87171", border: "#5a1818" },
  DONE:   { bg: "#16113a", color: "#a78bfa", border: "#302060" },
};

const createInitialChatList = () => {
  const list = [
    { id: "team", type: "team", messages: [{ id: uid(), from: "orchestrator", tag: "STATUS", content: "AI Engineering Platform online. All agents standing by.\n\nBackend running -> messages go to real Python agents.\nBackend offline -> full mock mode with simulated responses.", timestamp: Date.now() }], unread: 0 },
  ];
  for (const a of Object.values(AGENTS)) {
    list.push({ id: a.id, type: "direct", messages: [{ id: uid(), from: a.id, tag: "STATUS", content: `Hi, I'm ${a.name}. ${a.role}\n\nGive me a direct instruction. If I need another agent, a group thread auto-spawns.`, timestamp: Date.now() }], unread: 0 });
  }
  return list;
};

// â”€â”€â”€ UI Components â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function Avatar({ agentId, size = 32 }) {
  const a = AGENTS[agentId];
  const isUser = agentId === "user";
  return (
    <div style={{ width: size, height: size, borderRadius: "50%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: Math.floor(size * 0.44), background: isUser ? "#14122e" : (a?.bgColor || "#111"), border: `2px solid ${isUser ? "#6366f1" : (a?.color || "#333")}` }}>
      {isUser ? "U" : a?.icon}
    </div>
  );
}

function GettingStartedCard({ compact = false }) {
  const steps = [
    "Log in and create a new chat",
    "Upload a CSV dataset",
    "Optional: click “Check Agents” to run readiness checks",
    "Click “Run Without Check” to start analysis",
    "Ask for “train model” when analysis completes",
    "Use Access Files to view or download outputs",
    "Example after training: “Show training process for XGB on engineered dataset”",
    "Optional: click “Get ML Engineer” to run models locally on your machine",
  ];
  return (
    <div style={{ padding: compact ? "10px 12px" : "14px 16px", borderRadius: 10, background: "#090909", border: "1px solid #141414" }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: "#e8e8f0", marginBottom: 6 }}>Getting Started</div>
      <div style={{ fontSize: 10, color: "#3a3a3a", marginBottom: 8 }}>
        Currently supports tabular CSV datasets. CNN, NLP, and hybrid datasets are in progress.
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {steps.map((s, i) => (
          <div key={i} style={{ fontSize: 10.5, color: "#9ca3af", lineHeight: 1.4 }}>
            {i + 1}. {s}
          </div>
        ))}
      </div>
    </div>
  );
}

function StatusDot({ status }) {
  const c = { idle: "#4ade80", working: "#facc15", error: "#f87171" }[status] || "#4ade80";
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
      <span style={{ width: 7, height: 7, borderRadius: "50%", background: c, boxShadow: `0 0 5px ${c}`, display: "inline-block", animation: status === "working" ? "sPulse 1.1s ease-in-out infinite" : "none" }} />
      <span style={{ fontSize: 10, color: c }}>{{ idle: "Idle", working: "Working", error: "Error" }[status] || "Idle"}</span>
    </span>
  );
}

function Message({ msg, compact = false }) {
  const isUser = msg.from === "user";
  const a = AGENTS[msg.from];
  const ts = TAG[msg.tag];
  return (
    <div style={{ display: "flex", flexDirection: isUser ? "row-reverse" : "row", gap: 12, alignItems: "flex-start", animation: "mIn .28s ease-out" }}>
      <Avatar agentId={msg.from} size={compact ? 28 : 32} />
      <div style={{ maxWidth: "80%", display: "flex", flexDirection: "column", gap: 5 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7, flexDirection: isUser ? "row-reverse" : "row" }}>
          <span style={{ fontSize: compact ? 10 : 11, fontWeight: 700, textTransform: "uppercase", color: isUser ? "#818cf8" : (a?.color || "#888") }}>
            {isUser ? "You" : a?.name || msg.from}
          </span>
          {msg.tag && !isUser && ts && (
            <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 4, background: ts.bg, color: ts.color, border: `1px solid ${ts.border}`, letterSpacing: "0.07em" }}>{msg.tag}</span>
          )}
          <span style={{ fontSize: 9, color: "#2a2a2a" }}>{fmt(msg.timestamp)}</span>
        </div>
        <div style={{ padding: "11px 16px", borderRadius: isUser ? "14px 3px 14px 14px" : "3px 14px 14px 14px", background: isUser ? "#1a1740" : "#0b0b0b", border: `1px solid ${isUser ? "#6366f122" : (ts ? ts.border + "44" : "#181818")}`, color: "#c8c8d0", fontSize: 13.5, lineHeight: 1.7, whiteSpace: "pre-wrap" }}>
          {msg.content}
        </div>
      </div>
    </div>
  );
}

function Typing({ agentId }) {
  const a = AGENTS[agentId];
  if (!a) return null;
  return (
    <div style={{ display: "flex", gap: 10, alignItems: "center", opacity: .65, animation: "mIn .2s ease-out" }}>
      <Avatar agentId={agentId} size={28} />
      <div style={{ padding: "9px 14px", borderRadius: "3px 13px 13px 13px", background: "#0b0b0b", border: "1px solid #181818", display: "flex", gap: 4, alignItems: "center" }}>
        <span style={{ fontSize: 10, color: a.color, marginRight: 5 }}>{a.name}</span>
        {[0,1,2].map(i => <span key={i} style={{ width: 5, height: 5, borderRadius: "50%", background: a.color, display: "inline-block", animation: `tDot 1.2s ease-in-out ${i*.2}s infinite` }} />)}
      </div>
    </div>
  );
}

function ActivityCard({ entry }) {
  const fa = AGENTS[entry.from];
  const ta = AGENTS[entry.to];
  if (!fa || !ta) return null;
  return (
    <div style={{ padding: "12px 14px", borderRadius: 8, background: "#090909", border: `1px solid ${fa.color}28`, marginBottom: 6, animation: "mIn .22s ease-out" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <Avatar agentId={entry.from} size={24} />
        <span style={{ fontSize: 10, color: fa.color, fontWeight: 700 }}>{fa.name}</span>
        <svg width="14" height="10" viewBox="0 0 14 10" fill="none"><path d="M1 5h10M8 2l3 3-3 3" stroke="#333" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
        <Avatar agentId={entry.to} size={24} />
        <span style={{ fontSize: 10, color: ta.color, fontWeight: 700 }}>{ta.name}</span>
        <span style={{ fontSize: 9, color: "#252525", marginLeft: "auto" }}>{fmt(entry.timestamp)}</span>
      </div>
      <div style={{ fontSize: 12, color: "#666", lineHeight: 1.6, padding: "8px 10px", borderRadius: 6, background: "#060606", border: "1px solid #141414" }}>{entry.content}</div>
    </div>
  );
}

function FileCard({ entry }) {
  const a = AGENTS[entry.agent_id];
  const isUpdate = (entry.version || 1) > 1;
  return (
    <div style={{ padding: "10px 12px", borderRadius: 8, background: isUpdate ? "#060a10" : "#060f06", border: `1px solid ${isUpdate ? "#1a2a3a" : "#1a3a1a"}`, marginBottom: 6, animation: "mIn .22s ease-out" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
        <span style={{ fontSize: 11 }}>{isUpdate ? "EDIT" : "FILE"}</span>
        <span style={{ fontSize: 10, color: a?.color || "#4ade80", fontWeight: 700 }}>{entry.agent_id}</span>
        {isUpdate && (
          <span style={{ fontSize: 9, color: "#60a5fa", background: "#0a1628", border: "1px solid #1a3060", padding: "1px 6px", borderRadius: 4 }}>v{entry.version} overwritten</span>
        )}
        <span style={{ fontSize: 9, color: "#252525", marginLeft: "auto" }}>{fmt(entry.timestamp)}</span>
      </div>
      <div style={{ fontSize: 11, color: isUpdate ? "#60a5fa" : "#4ade80", fontFamily: "monospace" }}>{entry.filename}</div>
      {entry.full_path && (
        <div style={{ fontSize: 10, color: "#2a3a4a", marginTop: 2, fontFamily: "monospace", wordBreak: "break-all" }}>{entry.full_path}</div>
      )}
    </div>
  );
}

function ConnBadge({ status, project }) {
  const cfg = {
    connected:    { color: "#4ade80", bg: "#0a2a0a", border: "#1a4a1a", label: "Live" },
    connecting:   { color: "#facc15", bg: "#1a1400", border: "#3a3000", label: "Connecting..." },
    disconnected: { color: "#f87171", bg: "#2a0a0a", border: "#4a1a1a", label: "Offline - mock mode" },
  }[status] || {};
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      {project && <div style={{ fontSize: 10, color: "#383838", background: "#0a0a0a", border: "1px solid #181818", padding: "3px 8px", borderRadius: 20 }}>Current Chat: {project}</div>}
      <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: cfg.color, background: cfg.bg, border: `1px solid ${cfg.border}`, padding: "3px 8px", borderRadius: 20 }}>
        <span style={{ width: 5, height: 5, borderRadius: "50%", background: cfg.color, display: "inline-block", animation: status === "connecting" ? "sPulse 1s infinite" : "none" }} />
        {cfg.label}
      </div>
    </div>
  );
}

function ProjectHistory({ projects, activeProjectRoot, onSelectProject, onRefresh }) {
  return (
    <div style={{ padding: "10px 12px", borderBottom: "1px solid #0f0f0f" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 9, color: "#3a3a3a", fontWeight: 700, letterSpacing: ".12em" }}>PROJECTS</span>
        <button
          onClick={onRefresh}
          style={{ fontSize: 9, padding: "2px 6px", borderRadius: 6, background: "#0b0b0b", border: "1px solid #1a1a1a", color: "#9ca3af", cursor: "pointer" }}
        >
          Refresh
        </button>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: 140, overflowY: "auto" }}>
        {projects.map(p => {
          const isActive = activeProjectRoot && p.root === activeProjectRoot;
          return (
            <button
              key={p.id}
              onClick={() => onSelectProject(p)}
              style={{
                width: "100%",
                textAlign: "left",
                padding: "6px 8px",
                borderRadius: 6,
                background: isActive ? "#12103a" : "#080808",
                border: `1px solid ${isActive ? "#6366f155" : "#121212"}`,
                color: "#cbd5f5",
                fontSize: 10,
                cursor: "pointer",
              }}
              title={p.root}
            >
              <div style={{ fontWeight: 700 }}>{p.name || p.id}</div>
              <div style={{ fontSize: 9, color: "#3a3a3a" }}>{p.created || p.modified || ""}</div>
            </button>
          );
        })}
        {projects.length === 0 && (
          <div style={{ fontSize: 10, color: "#2a2a2a", textAlign: "center", padding: "6px 0" }}>
            No previous projects.
          </div>
        )}
      </div>
    </div>
  );
}

function Sidebar({ agents, activeChat, onSelectChat, chatList, projects, activeProjectRoot, onSelectProject, onRefreshProjects }) {
  const groupChats = chatList.filter(c => c.type === "group");
  return (
    <div style={S.sidebar}>
      <ProjectHistory
        projects={projects}
        activeProjectRoot={activeProjectRoot}
        onSelectProject={onSelectProject}
        onRefresh={onRefreshProjects}
      />
      <div style={{ padding: "12px 12px 6px", borderBottom: "1px solid #0f0f0f", flexShrink: 0 }}>
        <span style={{ fontSize: 9, color: "#3a3a3a", fontWeight: 700, letterSpacing: ".12em" }}>CHATS</span>
      </div>
      <div style={{ padding: "8px 8px 2px" }}>
        <SidebarBtn id="team" isActive={activeChat === "team"} icon="TEAM" iconBg="#12103a" iconBorder="#6366f1" color="#a5b4fc" name="Team" sub="All agent reports" unread={chatList.find(c=>c.id==="team")?.unread||0} onClick={() => onSelectChat("team")} />
      </div>
      <div style={{ padding: "8px 12px 4px" }}>
        <span style={{ fontSize: 9, color: "#252525", letterSpacing: ".1em" }}>DIRECT</span>
      </div>
      <div style={{ padding: "0 8px 4px", display: "flex", flexDirection: "column", gap: 1 }}>
        {Object.values(agents).map(a => (
          <SidebarBtn key={a.id} id={a.id} isActive={activeChat === a.id}
            icon={a.icon} iconBg={a.bgColor} iconBorder={a.color} color={a.color}
            name={a.name} sub={<StatusDot status={a.status} />}
            unread={chatList.find(c=>c.id===a.id)?.unread||0} onClick={() => onSelectChat(a.id)} />
        ))}
      </div>
      {groupChats.length > 0 && (
        <>
          <div style={{ padding: "8px 12px 4px", borderTop: "1px solid #0f0f0f" }}>
            <span style={{ fontSize: 9, color: "#252525", letterSpacing: ".1em" }}>GROUP THREADS</span>
          </div>
          <div style={{ padding: "0 8px 8px", display: "flex", flexDirection: "column", gap: 1, flex: 1, overflowY: "auto" }}>
            {groupChats.map(g => (
              <SidebarBtn key={g.id} id={g.id} isActive={activeChat === g.id}
                icon="GRP" iconBg="#1a1a2e" iconBorder="#6366f1" color="#a78bfa"
                name={g.title} sub={g.members.map(m => AGENTS[m]?.icon).join(" ")}
                unread={g.unread||0} onClick={() => onSelectChat(g.id)} isNew={g.isNew} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function SidebarBtn({ isActive, icon, iconBg, iconBorder, color, name, sub, unread, onClick, isNew }) {
  const iconText = String(icon || "");
  const iconSize = iconText.length >= 4 ? 9 : iconText.length === 3 ? 10 : 14;
  return (
    <button onClick={onClick} style={{ width: "100%", padding: "8px 9px", borderRadius: 8, cursor: "pointer", display: "flex", alignItems: "center", gap: 9, marginBottom: 1, background: isActive ? iconBg + "cc" : "transparent", border: `1px solid ${isActive ? iconBorder + "aa" : "transparent"}`, transition: "all .15s" }}>
      <div style={{ position: "relative", flexShrink: 0 }}>
        <div style={{ width: 34, height: 34, borderRadius: "50%", background: iconBg, border: `2px solid ${iconBorder}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: iconSize, fontWeight: 700 }}>{icon}</div>
        {unread > 0 && <span style={{ position: "absolute", top: -2, right: -2, width: 14, height: 14, borderRadius: "50%", background: "#ef4444", border: "2px solid #050505", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 7, color: "#fff", fontWeight: 700 }}>{unread}</span>}
        {isNew && !unread && <span style={{ position: "absolute", top: -2, right: -2, width: 10, height: 10, borderRadius: "50%", background: "#4ade80", border: "2px solid #050505", animation: "sPulse 1.5s infinite" }} />}
      </div>
      <div style={{ flex: 1, textAlign: "left", minWidth: 0 }}>
        <div style={{ fontSize: 11, fontWeight: 700, color, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{name}</div>
        <div style={{ fontSize: 10, color: "#3a3a3a", marginTop: 1 }}>{typeof sub === "string" ? sub : sub}</div>
      </div>
      {isActive && <span style={{ width: 6, height: 6, borderRadius: "50%", background: color, boxShadow: `0 0 5px ${color}`, flexShrink: 0 }} />}
    </button>
  );
}

function ChatHeader({ chat, agents }) {
  if (!chat) return null;
  if (chat.id === "team") return (
    <div style={{ ...S.chatHdr, background: "#070707", borderBottom: "1px solid #111" }}>
      <div style={{ width: 34, height: 34, borderRadius: "50%", background: "#12103a", border: "2px solid #6366f1", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700 }}>TEAM</div>
      <div>
        <div style={{ fontSize: 13, fontWeight: 700, color: "#a5b4fc" }}>Team Chat</div>
        <div style={{ fontSize: 10, color: "#383838" }}>All agent reports stream here in real time</div>
      </div>
    </div>
  );
  if (chat.type === "group") {
    const memberAgents = chat.members.map(m => AGENTS[m]).filter(Boolean);
    return (
      <div style={{ ...S.chatHdr, background: "#0a091f", borderBottom: "1px solid #6366f122" }}>
        <div style={{ width: 34, height: 34, borderRadius: "50%", background: "#1a1a3a", border: "2px solid #6366f1", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700 }}>GRP</div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#a78bfa" }}>{chat.title}</div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 2 }}>
            {memberAgents.map(a => <span key={a.id} style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: a.color }}><span>{a.icon}</span>{a.shortName}</span>)}
          </div>
        </div>
        <div style={{ fontSize: 10, color: "#252525", maxWidth: 180, textAlign: "right", lineHeight: 1.4 }}>{chat.reason}</div>
      </div>
    );
  }
  const a = agents[chat.id];
  return (
    <div style={{ ...S.chatHdr, background: a?.bgColor + "44", borderBottom: `1px solid ${a?.color}22` }}>
      <Avatar agentId={chat.id} size={34} />
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: a?.color }}>{a?.name}</div>
        <div style={{ fontSize: 10, color: "#383838" }}>{a?.role}</div>
      </div>
      <StatusDot status={a?.status || "idle"} />
    </div>
  );
}

function RightPanel({ p2pLog, fileLog, projectRoot }) {
  const [tab, setTab] = useState("p2p");
  return (
    <div style={S.actPanel}>
      <div style={{ padding: "10px", borderBottom: "1px solid #0f0f0f" }}>
        <GettingStartedCard compact />
      </div>
      <div style={{ display: "flex", borderBottom: "1px solid #0f0f0f", flexShrink: 0 }}>
        {[["p2p", "P2P MESSAGES"], ["files", "FILES"]].map(([id, label]) => (
          <button key={id} onClick={() => setTab(id)} style={{ flex: 1, padding: "10px 0", fontSize: 9, fontWeight: 700, letterSpacing: ".1em", cursor: "pointer", border: "none", color: tab === id ? "#c8c8d0" : "#2a2a2a", background: tab === id ? "#0d0d0d" : "transparent", borderBottom: tab === id ? "2px solid #6366f1" : "2px solid transparent" }}>
            {label} {id === "files" && fileLog.length > 0 && <span style={{ color: "#4ade80" }}>({fileLog.length})</span>}
          </button>
        ))}
      </div>
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 10px" }}>
        {tab === "p2p" && (
          <>
            {!p2pLog.length && <div style={{ color: "#1e1e1e", fontSize: 11, textAlign: "center", marginTop: 20 }}>Agent-to-agent messages appear here...</div>}
            {p2pLog.map(e => <ActivityCard key={e.id} entry={e} />)}
          </>
        )}
        {tab === "files" && (
          <>
            {projectRoot && <div style={{ padding: "8px 10px", marginBottom: 8, borderRadius: 6, background: "#060f06", border: "1px solid #1a3a1a", fontSize: 10, color: "#2a4a2a", fontFamily: "monospace", wordBreak: "break-all" }}>DIR {projectRoot}</div>}
            {!fileLog.length && <div style={{ color: "#1e1e1e", fontSize: 11, textAlign: "center", marginTop: 20 }}>Files written by agents appear here...</div>}
            {fileLog.map(e => <FileCard key={e.id} entry={e} />)}
          </>
        )}
      </div>
    </div>
  );
}

// â”€â”€â”€ Main App â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export default function App() {
  const [agents, setAgents] = useState(() =>
    Object.fromEntries(Object.entries(AGENTS).map(([k,v]) => [k, { ...v }]))
  );
  const [chatList, setChatList] = useState(() => createInitialChatList());

  const [activeChat, setActiveChat]     = useState("team");
  const [typingIn, setTypingIn]         = useState({});
  const [inputs, setInputs]             = useState({});
  const [isBusy, setIsBusy]             = useState(false);
  const [connStatus, setConnStatus]     = useState("connecting");
  const [projectName, setProjectName]   = useState(null);
  const [projectRoot, setProjectRoot]   = useState(null);
  const [projects, setProjects]         = useState([]);
  const [projectsLoaded, setProjectsLoaded] = useState(false);
  const [p2pLog, setP2pLog]             = useState([]);
  const [fileLog, setFileLog]           = useState([]);
  const [uploading, setUploading]       = useState(false);
  const [datasetReady, setDatasetReady] = useState(false);
  const [datasetInfo, setDatasetInfo]   = useState(null);
  const [projectInitBusy, setProjectInitBusy] = useState(false);
  const [showGithubConnect, setShowGithubConnect] = useState(false);
  const [githubToken, setGithubToken] = useState("");
  const [githubRepo, setGithubRepo] = useState("");
  const [githubOwner, setGithubOwner] = useState("");
  const [githubVisibility, setGithubVisibility] = useState("private");
  const [authToken, setAuthToken] = useState(() => safeLocalStorageGet("auth_token"));
  const [authEmail, setAuthEmail] = useState(() => safeLocalStorageGet("auth_email"));
  const [authPassword, setAuthPassword] = useState("");
  const [authMode, setAuthMode] = useState("login");
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState("");
  const [showFilesPanel, setShowFilesPanel] = useState(false);
  const [showMLWorkerHelp, setShowMLWorkerHelp] = useState(false);
  const [filesIndex, setFilesIndex] = useState([]);
  const [filesLoading, setFilesLoading] = useState(false);
  const [filesError, setFilesError] = useState("");
  const [fileSearch, setFileSearch] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileContent, setFileContent] = useState("");
  const [fileBinary, setFileBinary] = useState(false);
  const [fileMime, setFileMime] = useState("");
  const [fileTruncated, setFileTruncated] = useState(false);

  const wsRef         = useRef(null);
  const uploadRef     = useRef(null);
  const timers        = useRef([]);
  const msgsEndRef    = useRef({});
  const reconnTimer   = useRef(null);
  const shouldReconnectRef = useRef(true);
  const activeChatRef = useRef(activeChat);
  const taskRouteRef  = useRef({});
  const groupMapRef   = useRef({});
  const recentMsgRef  = useRef(new Map());
  const projectStateRef = useRef({});

  // Track whether backend is truly live (not just WS connected)
  const backendLive = connStatus === "connected";

  const getOrCreateGroupChat = useCallback((members, reason = null) => {
    const normalized = [...new Set(members)].filter(Boolean).sort();
    if (normalized.length < 2) return null;
    const key = normalized.join("|");
    const existing = groupMapRef.current[key];
    if (existing) return existing;

    const groupId = `group_${key.replace(/\|/g, "_")}`;
    const title = normalized.map(m => AGENTS[m]?.shortName || m).join(" + ");
    setChatList(prev => {
      const already = prev.find(c => c.id === groupId);
      if (already) return prev;
      return [
        ...prev,
        {
          id: groupId,
          type: "group",
          title: `Group: ${title}`,
          reason: reason || "Auto-created from private multi-agent collaboration.",
          members: normalized,
          messages: [],
          unread: 0,
          isNew: true,
        },
      ];
    });
    groupMapRef.current[key] = groupId;
    return groupId;
  }, []);

  useEffect(() => { activeChatRef.current = activeChat; }, [activeChat]);

  const handleAuth = useCallback(async () => {
    if (!authEmail.trim() || !authPassword.trim()) return;
    setAuthLoading(true);
    setAuthError("");
    try {
      const endpoint = authMode === "register" ? "/auth/register" : "/auth/login";
      const resp = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: authEmail.trim(), password: authPassword.trim() }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setAuthError(data?.detail || "Authentication failed.");
        return;
      }
      if (authMode === "register") {
        setAuthMode("login");
        setAuthError("Account created. Please log in.");
        return;
      }
      setAuthToken(data.token || "");
      setAuthEmail(data.email || authEmail.trim());
      safeLocalStorageSet("auth_token", data.token || "");
      safeLocalStorageSet("auth_email", data.email || authEmail.trim());
      setAuthPassword("");
      // Clear prior user state when logging in as a different account
      setProjectName(null);
      setProjectRoot(null);
      setProjects([]);
      setProjectsLoaded(false);
      setChatList(createInitialChatList());
      setActiveChat("team");
      setP2pLog([]);
      setFileLog([]);
      setDatasetReady(false);
      setDatasetInfo(null);
      projectStateRef.current = {};
      groupMapRef.current = {};
      taskRouteRef.current = {};
      recentMsgRef.current = new Map();
    } catch (_) {
      setAuthError("Auth service unavailable.");
    } finally {
      setAuthLoading(false);
    }
  }, [authEmail, authPassword, authMode]);

  const handleLogout = useCallback(() => {
    setAuthToken("");
    setAuthEmail("");
    safeLocalStorageRemove("auth_token");
    safeLocalStorageRemove("auth_email");
    setProjectName(null);
    setProjectRoot(null);
    setProjects([]);
    setProjectsLoaded(false);
    projectStateRef.current = {};
  }, []);

  useEffect(() => {
    if (!authToken) return;
    let cancelled = false;
    (async () => {
      try {
        const resp = await fetch(`${API_URL}/auth/verify`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ auth_token: authToken }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data?.detail || "Session invalid.");
        if (!cancelled && data?.email) {
          setAuthEmail(data.email);
          safeLocalStorageSet("auth_email", data.email);
        }
      } catch (_) {
        if (!cancelled) {
          handleLogout();
          setAuthError("Session expired. Please log in again.");
        }
      }
    })();
    return () => { cancelled = true; };
  }, [authToken, handleLogout]);

  // Auto-scroll
  useEffect(() => {
    const el = msgsEndRef.current[activeChat];
    if (el) el.scrollTop = el.scrollHeight;
  }, [chatList, typingIn, activeChat, p2pLog, fileLog]);

  // Clear unread on switch
  useEffect(() => {
    setChatList(prev => prev.map(c => c.id === activeChat ? { ...c, unread: 0, isNew: false } : c));
  }, [activeChat]);

  // â”€â”€ WebSocket â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const connectWS = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }
    setConnStatus("connecting");
    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen    = () => { setConnStatus("connected"); };
      ws.onclose   = () => {
        setConnStatus("disconnected");
        wsRef.current = null;
        if (!shouldReconnectRef.current) return;
        if (reconnTimer.current) clearTimeout(reconnTimer.current);
        reconnTimer.current = setTimeout(connectWS, 3000);
      };
      ws.onerror   = () => { setConnStatus("disconnected"); };
      ws.onmessage = (evt) => {
        try { handleWsMessage(JSON.parse(evt.data)); } catch(e) {}
      };
    } catch {
      setConnStatus("disconnected");
      if (reconnTimer.current) clearTimeout(reconnTimer.current);
      reconnTimer.current = setTimeout(connectWS, 3000);
    }
  }, []);

  useEffect(() => {
    shouldReconnectRef.current = true;
    connectWS();
    return () => {
      shouldReconnectRef.current = false;
      clearTimeout(reconnTimer.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connectWS]);

  // â”€â”€ File log helper â€” updates existing entry or adds new â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const upsertFile = useCallback((agent_id, filename, full_path) => {
    setFileLog(prev => {
      const idx = prev.findIndex(f => f.filename === filename && f.agent_id === agent_id);
      const entry = {
        id:        idx >= 0 ? prev[idx].id : uid(),
        agent_id, filename, full_path,
        timestamp: Date.now(),
        version:   idx >= 0 ? (prev[idx].version || 1) + 1 : 1,
      };
      if (idx >= 0) { const a = [...prev]; a[idx] = entry; return a; }
      return [...prev, entry];
    });
  }, []);

  // â”€â”€ File log seeding â€” add only if missing (for snapshot) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const seedFile = useCallback((agent_id, filename, full_path) => {
    setFileLog(prev => {
      const idx = prev.findIndex(f => f.filename === filename && f.agent_id === agent_id);
      if (idx >= 0) return prev;
      return [...prev, {
        id: uid(),
        agent_id, filename, full_path,
        timestamp: Date.now(),
        version: 1,
      }];
    });
  }, []);

  const fetchProjects = useCallback(async () => {
    if (!authToken) return;
    try {
      const res = await fetch(`${API_URL}/projects`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auth_token: authToken }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Failed to load projects.");
      setProjects(data.projects || []);
      setProjectsLoaded(true);
    } catch (_) {
      setProjectsLoaded(true);
    }
  }, [authToken]);

  useEffect(() => {
    if (!authToken) return;
    fetchProjects();
  }, [authToken, fetchProjects]);

  const switchToProjectState = useCallback((next) => {
    if (projectRoot) {
      projectStateRef.current[projectRoot] = {
        chatList,
        p2pLog,
        activeChat,
      };
    }
    const saved = next?.project_root ? projectStateRef.current[next.project_root] : null;
    setChatList(saved?.chatList || createInitialChatList());
    setP2pLog(saved?.p2pLog || []);
    setActiveChat(saved?.activeChat || "team");
    setProjectName(next?.project_name || null);
    setProjectRoot(next?.project_root || null);
    setDatasetReady(false);
    setDatasetInfo(null);
    setFileLog([]);
    groupMapRef.current = {};
    taskRouteRef.current = {};
    recentMsgRef.current = new Map();
  }, [projectRoot, chatList, p2pLog, activeChat]);

  // â”€â”€ Handle incoming backend messages â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const handleWsMessage = useCallback((msg) => {
    const { type, from, to, content, tag, extra, task_id } = msg;
    const pushMsg = (chatId, fromId, text, messageTag) => {
      const dedupKey = `${chatId}|${fromId}|${messageTag || ""}|${text}`;
      const now = Date.now();
      const last = recentMsgRef.current.get(dedupKey);
      if (last && now - last < 1500) return;
      recentMsgRef.current.set(dedupKey, now);

      setChatList(prev => prev.map(c => {
        if (c.id !== chatId) return c;
        const isVisible = activeChatRef.current === chatId;
        return {
          ...c,
          messages: [...c.messages, { id: uid(), from: fromId, content: text, tag: messageTag, timestamp: Date.now() }],
          unread: isVisible ? 0 : (c.unread || 0) + (fromId !== "user" ? 1 : 0),
        };
      }));
    };

    if (type === "project_info") {
      switchToProjectState({ project_name: extra?.project_name, project_root: extra?.project_root });
      setProjectInitBusy(false);
      pushMsg("team", "orchestrator",
        `Project: ${extra?.project_name}\nDIR ${extra?.project_root}`, "STATUS");
      fetchProjects();
      return;
    }

    if (type === "files_snapshot") {
      const files = extra?.files || [];
      files.forEach(f => seedFile(f.agent_id, f.filename, f.full_path));
      return;
    }

    if (type === "team_message") {
      const route = task_id ? taskRouteRef.current[task_id] : null;
      const targetChatId = route?.groupChatId || route?.chatId || "team";

      setTypingIn(prev => ({ ...prev, [targetChatId]: from }));
      const t = setTimeout(() => {
        setTypingIn(prev => ({ ...prev, [targetChatId]: null }));
        pushMsg(targetChatId, from, content, tag || detectTag(content));
        if (typeof content === "string" && content.toLowerCase().includes("project init failed")) {
          setProjectInitBusy(false);
        }
      }, 500);
      timers.current.push(t);
      return;
    }

    if (type === "p2p_message") {
      setP2pLog(prev => [...prev, { id: uid(), from, to, content, timestamp: Date.now() }]);
      const route = task_id ? taskRouteRef.current[task_id] : null;
      if (route?.mode === "direct") {
        const fromAgent = AGENTS[from] ? from : null;
        const toAgent = AGENTS[to] ? to : null;
        if (fromAgent) route.members.add(fromAgent);
        if (toAgent) route.members.add(toAgent);

        if (route.members.size > 1) {
          const groupId = getOrCreateGroupChat(
            Array.from(route.members),
            "Auto-created from private multi-agent collaboration."
          );
          if (groupId) route.groupChatId = groupId;
        }
      }
      return;
    }

    if (type === "agent_status") {
      setAgents(prev => ({ ...prev, [from]: { ...prev[from], status: content } }));
      return;
    }

    if (type === "file_written") {
      // Backend confirms a real file was written to disk
      upsertFile(extra?.agent_id || from, extra?.filename, extra?.full_path);
      return;
    }

    if (type === "dataset_uploaded") {
      setDatasetReady(true);
      setDatasetInfo(extra || null);
      pushMsg("team", "orchestrator",
        `Dataset uploaded: ${extra?.filename || "dataset"}\nDIR ${extra?.path || ""}`, "STATUS");
      return;
    }
  }, [upsertFile, seedFile, getOrCreateGroupChat, switchToProjectState, fetchProjects]);

  useEffect(() => {
    if (wsRef.current) {
      wsRef.current.onmessage = (evt) => {
        try { handleWsMessage(JSON.parse(evt.data)); } catch(e) {}
      };
    }
  }, [handleWsMessage]);

  // â”€â”€ Message helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const addMsgDirect = (chatId, from, content, tag) => {
    setChatList(prev => prev.map(c => {
      if (c.id !== chatId) return c;
      const isVisible = activeChatRef.current === chatId;
      return {
        ...c,
        messages: [...c.messages, { id: uid(), from, content, tag, timestamp: Date.now() }],
        unread: isVisible ? 0 : (c.unread || 0) + (from !== "user" ? 1 : 0),
      };
    }));
  };
  const addMsg = useCallback(addMsgDirect, []);
  const setTyping = useCallback((chatId, agentId) => setTypingIn(prev => ({ ...prev, [chatId]: agentId || null })), []);
  const setAgentStatus = useCallback((id, status) => setAgents(prev => ({ ...prev, [id]: { ...prev[id], status } })), []);
  const clearTimers = () => { timers.current.forEach(clearTimeout); timers.current = []; };
  // â”€â”€ Fire file write to backend (local/mock-only when backend offline) â”€â”€â”€â”€â”€
  const fireFileWrite = useCallback((fileWrite, fromAgent) => {
    if (!fileWrite) return;
    const { agent, filename, content } = fileWrite;

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      // Send to backend â†’ Python workspace.write() â†’ real file on disk
      // Backend will confirm back via "file_written" message â†’ upsertFile()
        wsRef.current.send(JSON.stringify({
          type: "file_write", agent_id: agent, filename, content, from: fromAgent,
          auth_token: authToken,
        }));
    } else {
      // Backend offline â€” update UI only (no disk write possible)
      upsertFile(agent, filename, `[offline]/${agent}/${filename}`);
    }
    }, [upsertFile, authToken]);

  // â”€â”€ Group chat spawn (always local â€” direct/group chats don't touch backend)
  const spawnGroupChat = useCallback((groupDef, triggerChatId) => {
    const groupId = "group_" + uid();
    setChatList(prev => [...prev, { id: groupId, type: "group", title: groupDef.title, reason: groupDef.reason, members: groupDef.members, messages: [], unread: 0, isNew: true }]);
    addMsg(triggerChatId, groupDef.members[0], `Spawning group thread: "${groupDef.title}" - ${groupDef.reason}`, "STATUS");

    const t = setTimeout(() => {
      setActiveChat(groupId);
      groupDef.members.forEach(id => setAgentStatus(id, "working"));

      groupDef.groupFlow.forEach((step, idx) => {
        const isLast = idx === groupDef.groupFlow.length - 1;
        const t1 = setTimeout(() => setTyping(groupId, step.from), step.delay - 400);
        const t2 = setTimeout(() => {
          setTyping(groupId, null);
          addMsg(groupId, step.from, step.content, step.tag);

          if (step.fileWrite) {
            // âœ… This fires the real disk write via backend
            fireFileWrite(step.fileWrite, step.from);
          }

          if (idx > 0) {
            setP2pLog(prev => [...prev, { id: uid(), from: step.from, to: groupDef.members.find(m => m !== step.from) || groupDef.members[0], content: step.content, timestamp: Date.now() }]);
          }
          if (isLast) groupDef.members.forEach(id => setAgentStatus(id, "idle"));
        }, step.delay);
        timers.current.push(t1, t2);
      });
    }, 800);
    timers.current.push(t);
  }, [addMsg, setAgentStatus, setTyping, fireFileWrite]);

  // â”€â”€ Team scenario (mock â€” only runs when backend is OFFLINE) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const runTeamScenario = useCallback((scenario) => {
    clearTimers();
    setIsBusy(true);
    const inv = new Set(scenario.flow.filter(s => s.from !== "user").map(s => s.from));
    inv.forEach(id => setAgentStatus(id, "working"));
    scenario.flow.forEach((step, idx) => {
      const isLast = idx === scenario.flow.length - 1;
      const t1 = setTimeout(() => { if (step.type === "team") setTyping("team", step.from); }, step.delay - 400);
      const t2 = setTimeout(() => {
        setTyping("team", null);
        if (step.type === "team") {
          addMsg("team", step.from, step.content, step.tag);
          if (step.fileWrite) fireFileWrite(step.fileWrite, step.from);
        } else if (step.type === "p2p") {
          setP2pLog(prev => [...prev, { id: uid(), from: step.from, to: step.to, content: step.content, timestamp: Date.now() }]);
        }
        if (isLast) { inv.forEach(id => setAgentStatus(id, "idle")); setIsBusy(false); }
      }, step.delay);
      timers.current.push(t1, t2);
    });
  }, [addMsg, setAgentStatus, setTyping, fireFileWrite]);

  // â”€â”€ Send handler â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const handleSend = useCallback((chatId) => {
    const text = (inputs[chatId] || "").trim();
    if (!text) return;
    setInputs(prev => ({ ...prev, [chatId]: "" }));
    const chat = chatList.find(c => c.id === chatId);
    if (!chat) return;

    addMsg(chatId, "user", text, null);

    // â”€â”€ Team chat â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if (chatId === "team") {
      if (isBusy) return;

      if (backendLive && wsRef.current?.readyState === WebSocket.OPEN) {
        const taskId = Math.random().toString(36).slice(2, 10);
        taskRouteRef.current[taskId] = {
          mode: "team",
          chatId: "team",
          members: new Set(),
          groupChatId: null,
        };
        // âœ… FIX 1: Backend live â€” send ONLY to backend, do NOT run mock scenario
        // Real Python agents respond â†’ messages stream back via WebSocket
          wsRef.current.send(JSON.stringify({
            type: "user_message", content: text, to: "team",
            task_id: taskId,
            auth_token: authToken,
          }));
      } else {
        // Backend offline â€” run mock scenario (it handles everything locally)
        const scenario = detectScenario(text);
        setTimeout(() => runTeamScenario(scenario), 350);
      }
      return;
    }

    // â”€â”€ Direct chat (always local â€” private) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if (chat.type === "direct") {
      const agentId = chatId;

      if (backendLive && wsRef.current?.readyState === WebSocket.OPEN) {
        const taskId = Math.random().toString(36).slice(2, 10);
        taskRouteRef.current[taskId] = {
          mode: "direct",
          chatId: agentId,
          members: new Set([agentId]),
          groupChatId: null,
        };
        setTyping(chatId, agentId);
          wsRef.current.send(JSON.stringify({
            type: "user_message",
            content: text,
            to: agentId,
            task_id: taskId,
            auth_token: authToken,
          }));
        return;
      }

      setAgentStatus(agentId, "working");
      const { steps, spawnGroup } = detectDirectResponse(agentId, text);

      steps.forEach((step, idx) => {
        const isLast = idx === steps.length - 1;
        const willSpawn = isLast && spawnGroup;
        const t1 = setTimeout(() => setTyping(chatId, agentId), step.delay - 400);
        const t2 = setTimeout(() => {
          setTyping(chatId, null);
          addMsg(chatId, agentId, step.content, step.tag);
          // âœ… FIX 2: fire real file write for direct chat steps
          if (step.fileWrite) fireFileWrite(step.fileWrite, agentId);
          if (isLast) {
            setAgentStatus(agentId, "idle");
            setTyping(chatId, null);
            if (willSpawn) spawnGroupChat(spawnGroup, chatId);
          }
        }, step.delay);
        timers.current.push(t1, t2);
      });
      return;
    }

    // â”€â”€ Group chat â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if (chat.type === "group") {
      const responder = chat.members[0];
      setAgentStatus(responder, "working");
      const t1 = setTimeout(() => setTyping(chatId, responder), 700);
      const t2 = setTimeout(() => {
        setTyping(chatId, null);
        setAgentStatus(responder, "idle");
        addMsg(chatId, responder, "Noted. Incorporating that and updating the thread.", "STATUS");
      }, 1500);
      timers.current.push(t1, t2);
    }
  }, [inputs, chatList, isBusy, backendLive, addMsg, setAgentStatus, setTyping, runTeamScenario, spawnGroupChat, fireFileWrite, authToken]);

  const handleDatasetUpload = useCallback(async (file) => {
    if (!file || !backendLive || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    if (!/\.csv$/i.test(file.name)) return;
    setUploading(true);
    setDatasetReady(false);
    setDatasetInfo(null);
    try {
      const ab = await file.arrayBuffer();
      let binary = "";
      const bytes = new Uint8Array(ab);
      const chunk = 0x8000;
      for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
      }
      wsRef.current.send(JSON.stringify({
        type: "dataset_upload",
        filename: file.name,
        content_b64: btoa(binary),
        to: "orchestrator",
        auth_token: authToken,
      }));
    } catch (_) {
      // Silent by design.
    } finally {
      setUploading(false);
      if (uploadRef.current) uploadRef.current.value = "";
    }
  }, [backendLive, authToken]);

  const handlePhase3Check = useCallback(() => {
    if (!backendLive || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    setActiveChat("team");
    addMsg("team", "user", "phase 3 check", null);
    wsRef.current.send(JSON.stringify({
      type: "phase3_check",
      task_id: `phase3_${Date.now()}`,
      auth_token: authToken,
    }));
    setDatasetReady(false);
  }, [backendLive, addMsg, authToken]);

  const handleRunDirect = useCallback(() => {
    if (!backendLive || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    const taskId = `run_${Date.now()}`;
    taskRouteRef.current[taskId] = {
      mode: "team",
      chatId: "team",
      members: new Set(),
      groupChatId: null,
    };
    setActiveChat("team");
    addMsg("team", "user", "analyse data", null);
      wsRef.current.send(JSON.stringify({
        type: "user_message",
        content: "analyse data",
        to: "team",
        task_id: taskId,
        auth_token: authToken,
      }));
    setDatasetReady(false);
  }, [backendLive, addMsg, authToken]);

  const handleGithubConnect = useCallback(() => {
    if (!backendLive || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    if (!githubToken.trim() || !githubRepo.trim()) return;
    const taskId = `gh_${Date.now()}`;
    setActiveChat("team");
    addMsg("team", "user", "connect github", null);
    wsRef.current.send(JSON.stringify({
      type: "github_connect",
      token: githubToken.trim(),
      owner: githubOwner.trim(),
      repo: githubRepo.trim(),
      visibility: githubVisibility,
      task_id: taskId,
      auth_token: authToken,
    }));
    setShowGithubConnect(false);
  }, [backendLive, addMsg, githubToken, githubRepo, githubOwner, githubVisibility, authToken]);

  const fetchFilesIndex = useCallback(async () => {
    if (!authToken) return;
    setFilesLoading(true);
    setFilesError("");
    try {
      const res = await fetch(`${API_URL}/files`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auth_token: authToken }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Failed to load file list.");
      const list = (data.files || []).slice().sort((a, b) => a.path.localeCompare(b.path));
      setFilesIndex(list);
      if (!projectRoot && data.project_root) setProjectRoot(data.project_root);
      setFileLog(list.map(f => {
        const parts = String(f.path || "").split("/");
        const agent_id = parts[0] || "shared";
        const filename = parts.slice(1).join("/") || parts[0] || f.path;
        return {
          id: uid(),
          agent_id,
          filename,
          full_path: (data.project_root ? `${data.project_root}\\${f.path}` : f.path),
          timestamp: Date.now(),
          version: 1,
        };
      }));
    } catch (err) {
      setFilesError(err?.message || String(err));
    } finally {
      setFilesLoading(false);
    }
  }, [authToken, projectRoot]);

  const openFile = useCallback(async (path) => {
    if (!authToken || !path) return;
    setSelectedFile(path);
    setFileContent("");
    setFileBinary(false);
    setFileMime("");
    setFileTruncated(false);
    try {
      const res = await fetch(`${API_URL}/file`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auth_token: authToken, path }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Failed to load file.");
      setFileBinary(Boolean(data.binary));
      setFileMime(data.mime || "");
      setFileTruncated(Boolean(data.truncated));
      if (data.binary && data.content_b64) {
        setFileContent(`data:${data.mime};base64,${data.content_b64}`);
      } else if (!data.binary) {
        setFileContent(data.content || "");
      } else {
        setFileContent("");
      }
    } catch (err) {
      setFileContent(`Error: ${err?.message || err}`);
    }
  }, [authToken]);

  const handleOpenFilesPanel = useCallback(() => {
    if (!authToken) return;
    setShowFilesPanel(true);
    fetchFilesIndex();
  }, [authToken, fetchFilesIndex]);

  const handleSelectProject = useCallback(async (proj) => {
    if (!authToken || !proj?.id) return;
    try {
      const res = await fetch(`${API_URL}/projects/select`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auth_token: authToken, project_id: proj.id }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Failed to switch project.");
      switchToProjectState({ project_name: data.project_name, project_root: data.project_root });
      fetchFilesIndex();
    } catch (err) {
      setFilesError(err?.message || String(err));
    }
  }, [authToken, switchToProjectState, fetchFilesIndex]);

  const handleDownloadZip = useCallback(async () => {
    if (!authToken) return;
    try {
      const res = await fetch(`${API_URL}/project_zip`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auth_token: authToken }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Failed to download ZIP.");
      }
      const blob = await res.blob();
      const name = `${projectName || "project"}.zip`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setFilesError(err?.message || String(err));
    }
  }, [authToken, projectName]);

  const sendInitProject = useCallback((name) => {
    if (!backendLive || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    if (!name || !name.trim()) return;
    setProjectInitBusy(true);
    wsRef.current.send(JSON.stringify({
      type: "init_project",
      project_name: name.trim(),
      task_id: `init_${Date.now()}`,
      auth_token: authToken,
    }));
  }, [backendLive, authToken]);

  const handleNewChat = useCallback(() => {
    if (!backendLive || !authToken || projectInitBusy) return;
    const name = `chat_${new Date().toISOString().replace(/[-:]/g, "").slice(0, 15)}`;
    sendInitProject(name);
  }, [backendLive, authToken, projectInitBusy, sendInitProject]);

  useEffect(() => {
    if (!backendLive || !authToken || projectRoot || projectInitBusy) return;
    if (!projectsLoaded) return;
    if (projects.length > 0) return;
    const name = `chat_${new Date().toISOString().replace(/[-:]/g, "").slice(0, 15)}`;
    sendInitProject(name);
  }, [backendLive, authToken, projectRoot, projectInitBusy, projectsLoaded, projects.length, sendInitProject]);

  const currentChat = chatList.find(c => c.id === activeChat);

  if (!authToken) {
    return (
      <div style={S.root}>
        <style>{CSS}</style>
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 24, gap: 20, flexWrap: "wrap" }}>
          <div style={{ width: 420, background: "#070707", border: "1px solid #141414", borderRadius: 14, padding: 22 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
              <div style={S.logo}>AI</div>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, color: "#e8e8f0" }}>AI Engineering Platform</div>
                <div style={{ fontSize: 10, color: "#3a3a3a", letterSpacing: ".08em", textTransform: "uppercase" }}>
                  Secure Login
                </div>
              </div>
            </div>
            <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
              {["login", "register"].map(m => (
                <button
                  key={m}
                  onClick={() => setAuthMode(m)}
                  style={{
                    flex: 1, padding: "6px 8px", borderRadius: 8, border: "1px solid #1a1a1a",
                    background: authMode === m ? "#12103a" : "#0b0b0b",
                    color: authMode === m ? "#a5b4fc" : "#707070", fontSize: 11, fontWeight: 700,
                  }}
                >
                  {m === "login" ? "Log In" : "Sign Up"}
                </button>
              ))}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <input
                value={authEmail}
                onChange={e => setAuthEmail(e.target.value)}
                placeholder="Email"
                style={{ background: "#0b0b0b", border: "1px solid #1a1a1a", color: "#cbd5f5", borderRadius: 8, padding: "10px 12px", fontSize: 12 }}
              />
              <input
                value={authPassword}
                onChange={e => setAuthPassword(e.target.value)}
                placeholder="Password"
                type="password"
                style={{ background: "#0b0b0b", border: "1px solid #1a1a1a", color: "#cbd5f5", borderRadius: 8, padding: "10px 12px", fontSize: 12 }}
              />
              {authError && <div style={{ fontSize: 10, color: "#f87171" }}>{authError}</div>}
              <button
                onClick={handleAuth}
                disabled={authLoading || !authEmail.trim() || !authPassword.trim()}
                style={{ padding: "10px 12px", borderRadius: 8, background: "#1f2937", border: "1px solid #374151", color: "#e5e7eb", cursor: "pointer", opacity: (authLoading || !authEmail.trim() || !authPassword.trim()) ? 0.5 : 1 }}
              >
                {authLoading ? "Please wait..." : (authMode === "login" ? "Log In" : "Create Account")}
              </button>
            </div>
          </div>
          <div style={{ width: 420 }}>
            <GettingStartedCard />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={S.root}>
      <style>{CSS}</style>

      <header style={S.header}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={S.logo}>AI</div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, letterSpacing: ".05em", color: "#e8e8f0" }}>AI Engineering Platform</div>
            <div style={{ fontSize: 9, color: "#3a3a3a", letterSpacing: ".1em", textTransform: "uppercase" }}>Multi-Agent Autonomous System | v2.1</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {backendLive && (
            <button
              onClick={handleNewChat}
              style={{ fontSize: 10, padding: "6px 10px", borderRadius: 6, background: "#0b0b0b", border: "1px solid #1a1a1a", color: "#9ca3af", cursor: "pointer" }}
            >
              New Chat
            </button>
          )}
          {backendLive && (
            <button
              onClick={() => setShowGithubConnect(v => !v)}
              style={{ fontSize: 10, padding: "6px 10px", borderRadius: 6, background: "#0b0b0b", border: "1px solid #1a1a1a", color: "#9ca3af", cursor: "pointer" }}
            >
              Connect GitHub
            </button>
          )}
          {backendLive && (
            <button
              onClick={() => setShowMLWorkerHelp(true)}
              style={{ fontSize: 10, padding: "6px 10px", borderRadius: 6, background: "#0b0b0b", border: "1px solid #1a1a1a", color: "#9ca3af", cursor: "pointer" }}
            >
              Get ML Engineer
            </button>
          )}
          {backendLive && projectRoot && (
            <button
              onClick={handleOpenFilesPanel}
              style={{ fontSize: 10, padding: "6px 10px", borderRadius: 6, background: "#0b0b0b", border: "1px solid #1a1a1a", color: "#9ca3af", cursor: "pointer" }}
            >
              Access Files
            </button>
          )}
          {authEmail && (
            <div style={{ fontSize: 10, color: "#3a3a3a", background: "#0a0a0a", border: "1px solid #181818", padding: "3px 8px", borderRadius: 20 }}>
              {authEmail}
            </div>
          )}
          {authToken && (
            <button
              onClick={handleLogout}
              style={{ fontSize: 10, padding: "6px 10px", borderRadius: 6, background: "#120606", border: "1px solid #3a1414", color: "#fca5a5", cursor: "pointer" }}
            >
              Log Out
            </button>
          )}
          <ConnBadge status={connStatus} project={projectName} />
        </div>
      </header>

      {showFilesPanel && (
        <div style={S.filesOverlay}>
          <div style={S.filesPanel}>
            <div style={S.filesHeader}>
              <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#e8e8f0" }}>Project Files</div>
                <div style={{ fontSize: 10, color: "#3a3a3a" }}>{projectRoot || "No project initialized"}</div>
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <button
                  onClick={fetchFilesIndex}
                  disabled={filesLoading}
                  style={{ fontSize: 10, padding: "6px 10px", borderRadius: 6, background: "#0b0b0b", border: "1px solid #1a1a1a", color: "#9ca3af", cursor: "pointer", opacity: filesLoading ? 0.5 : 1 }}
                >
                  {filesLoading ? "Refreshing..." : "Refresh"}
                </button>
                <button
                  onClick={handleDownloadZip}
                  disabled={filesLoading}
                  style={{ fontSize: 10, padding: "6px 10px", borderRadius: 6, background: "#0b0b0b", border: "1px solid #1a1a1a", color: "#9ca3af", cursor: "pointer", opacity: filesLoading ? 0.5 : 1 }}
                >
                  Download ZIP
                </button>
                <button
                  onClick={() => setShowFilesPanel(false)}
                  style={{ fontSize: 10, padding: "6px 10px", borderRadius: 6, background: "#120606", border: "1px solid #3a1414", color: "#fca5a5", cursor: "pointer" }}
                >
                  Close
                </button>
              </div>
            </div>
            <div style={S.filesBody}>
              <div style={S.filesList}>
                <input
                  value={fileSearch}
                  onChange={e => setFileSearch(e.target.value)}
                  placeholder="Search files..."
                  style={{ background: "#0b0b0b", border: "1px solid #1a1a1a", color: "#cbd5f5", borderRadius: 6, padding: "8px 10px", fontSize: 11, marginBottom: 8 }}
                />
                {filesError && <div style={{ fontSize: 10, color: "#f87171", marginBottom: 8 }}>{filesError}</div>}
                <div style={{ overflowY: "auto", flex: 1 }}>
                  {(filesIndex || [])
                    .filter(f => !fileSearch || f.path.toLowerCase().includes(fileSearch.toLowerCase()))
                    .map(f => (
                      <button
                        key={f.path}
                        onClick={() => openFile(f.path)}
                        style={{
                          width: "100%",
                          textAlign: "left",
                          padding: "8px 10px",
                          marginBottom: 4,
                          borderRadius: 6,
                          background: selectedFile === f.path ? "#12103a" : "#080808",
                          border: `1px solid ${selectedFile === f.path ? "#6366f155" : "#121212"}`,
                          color: "#cbd5f5",
                          fontSize: 11,
                          cursor: "pointer",
                        }}
                      >
                        <div style={{ fontWeight: 600 }}>{f.path}</div>
                        <div style={{ fontSize: 9, color: "#3a3a3a" }}>{(f.size || 0).toLocaleString()} bytes</div>
                      </button>
                    ))}
                  {(!filesLoading && (!filesIndex || filesIndex.length === 0)) && (
                    <div style={{ fontSize: 10, color: "#3a3a3a", textAlign: "center", marginTop: 12 }}>
                      No files found.
                    </div>
                  )}
                </div>
              </div>
              <div style={S.filesPreview}>
                {!selectedFile && (
                  <div style={{ fontSize: 11, color: "#2a2a2a" }}>Select a file to preview.</div>
                )}
                {selectedFile && fileBinary && fileContent && (
                  <img alt={selectedFile} src={fileContent} style={{ maxWidth: "100%", maxHeight: "100%", borderRadius: 8 }} />
                )}
                {selectedFile && fileBinary && !fileContent && (
                  <div style={{ fontSize: 11, color: "#2a2a2a" }}>Binary file preview not available.</div>
                )}
                {selectedFile && !fileBinary && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 8, height: "100%" }}>
                    <div style={{ fontSize: 10, color: "#3a3a3a" }}>
                      {selectedFile} {fileTruncated ? " (truncated)" : ""}
                    </div>
                    <pre style={{ flex: 1, overflow: "auto", background: "#060606", border: "1px solid #141414", borderRadius: 8, padding: 12, fontSize: 11, color: "#cbd5f5", whiteSpace: "pre-wrap" }}>
                      {fileContent}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {showMLWorkerHelp && (
        <div style={S.filesOverlay}>
          <div style={{ ...S.filesPanel, width: "720px", height: "auto", maxHeight: "80vh" }}>
            <div style={S.filesHeader}>
              <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#e8e8f0" }}>Get ML Engineer (Local Worker)</div>
                <div style={{ fontSize: 10, color: "#3a3a3a" }}>
                  Enables ML Engineer to use your local terminal and deploy trained models on your machine.
                </div>
              </div>
              <button
                onClick={() => setShowMLWorkerHelp(false)}
                style={{ fontSize: 10, padding: "6px 10px", borderRadius: 6, background: "#120606", border: "1px solid #3a1414", color: "#fca5a5", cursor: "pointer" }}
              >
                Close
              </button>
            </div>
            <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ fontSize: 12, color: "#9ca3af", lineHeight: 1.6 }}>
                The local worker runs on your computer and connects back to the platform. This is required if you want
                ML Engineer to execute commands and deploy models locally.
              </div>
              <div style={{ fontSize: 12, color: "#c8c8d0", lineHeight: 1.8, background: "#0b0b0b", border: "1px solid #151515", borderRadius: 8, padding: 12 }}>
                <div style={{ fontWeight: 700, marginBottom: 6, color: "#e8e8f0" }}>How to start</div>
                <div>1. Download the Local ML Worker (provided by your admin).</div>
                <div>2. Run it and enter the pairing token from this platform.</div>
                <div>3. Keep it running in the background.</div>
                <div>4. Ask: “deploy the trained model locally” or “start the model server”.</div>
              </div>
              <div style={{ fontSize: 11, color: "#4b5563" }}>
                Note: This does not run automatically. A user action is required to download and launch the worker.
              </div>
            </div>
          </div>
        </div>
      )}

      {(!projectRoot || projectInitBusy) && (
        <div style={{ padding: "10px 20px", borderBottom: "1px solid #111", background: "#080808", display: "flex", gap: 10, alignItems: "center" }}>
          <div style={{ fontSize: 11, color: "#3a3a3a" }}>
            {projectInitBusy ? "Creating new chat..." : "No active chat yet."}
          </div>
        </div>
      )}

      {showGithubConnect && (
        <div style={{ padding: "10px 20px", borderBottom: "1px solid #111", background: "#080808", display: "flex", gap: 10, alignItems: "center" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, flex: 1 }}>
            <input
              value={githubToken}
              onChange={e => setGithubToken(e.target.value)}
              placeholder="GitHub token (PAT)"
              type="password"
              style={{ background: "#0b0b0b", border: "1px solid #1a1a1a", color: "#cbd5f5", borderRadius: 6, padding: "8px 10px", fontSize: 11 }}
            />
              <div style={{ fontSize: 10, color: "#8b95a7", lineHeight: 1.35 }}>
                Fine-grained PAT (recommended). Required permissions:<br />
                Repository: Contents (Read &amp; Write), Pull requests (Read &amp; Write), Metadata (Read-only),
                Administration (Read &amp; Write), Workflows (Read &amp; Write if triggering Actions),
                Commit statuses (Read &amp; Write if setting CI statuses).<br />
                Account: No extra permissions needed — repo creation is covered by Administration above.
                {" "}
                <a
                  href="https://github.com/settings/personal-access-tokens"
                  target="_blank"
                  rel="noreferrer"
                  style={{ color: "#9fb4ff", textDecoration: "underline" }}
                >
                  Create token
                </a>
              </div>
          </div>
          <input
            value={githubOwner}
            onChange={e => setGithubOwner(e.target.value)}
            placeholder="Owner (optional org/user)"
            style={{ width: 200, background: "#0b0b0b", border: "1px solid #1a1a1a", color: "#cbd5f5", borderRadius: 6, padding: "8px 10px", fontSize: 11 }}
          />
          <input
            value={githubRepo}
            onChange={e => setGithubRepo(e.target.value)}
            placeholder="Repo name"
            style={{ width: 200, background: "#0b0b0b", border: "1px solid #1a1a1a", color: "#cbd5f5", borderRadius: 6, padding: "8px 10px", fontSize: 11 }}
          />
          <select
            value={githubVisibility}
            onChange={e => setGithubVisibility(e.target.value)}
            style={{ width: 120, background: "#0b0b0b", border: "1px solid #1a1a1a", color: "#cbd5f5", borderRadius: 6, padding: "8px 10px", fontSize: 11 }}
          >
            <option value="private">Private</option>
            <option value="public">Public</option>
          </select>
          <button
            onClick={handleGithubConnect}
            disabled={!backendLive || !githubToken.trim() || !githubRepo.trim()}
            style={{ fontSize: 11, padding: "8px 12px", borderRadius: 6, background: "#1f2937", border: "1px solid #374151", color: "#e5e7eb", cursor: "pointer", opacity: (!backendLive || !githubToken.trim() || !githubRepo.trim()) ? 0.4 : 1 }}
          >
            Save
          </button>
        </div>
      )}

      <div style={S.body}>
        <Sidebar
          agents={agents}
          activeChat={activeChat}
          onSelectChat={setActiveChat}
          chatList={chatList}
          projects={projects}
          activeProjectRoot={projectRoot}
          onSelectProject={handleSelectProject}
          onRefreshProjects={fetchProjects}
        />

        <div style={S.chatArea}>
          <ChatHeader chat={currentChat} agents={agents} />

          {currentChat?.type === "direct" && (
            <div style={{ padding: "8px 18px", background: "#090909", borderBottom: "1px solid #111", fontSize: 11, color: "#2a2a2a" }}>
              <span style={{ color: AGENTS[activeChat]?.color, fontWeight: 700 }}>Private channel</span>
              {backendLive
                ? " - routed to real backend agent."
                : " - backend offline, using local mock flow."}
            </div>
          )}
          {currentChat?.type === "group" && (
            <div style={{ padding: "8px 18px", background: "#0a091f", borderBottom: "1px solid #6366f122", fontSize: 11, color: "#2a2a2a" }}>
              <span style={{ color: "#a78bfa", fontWeight: 700 }}>Group thread</span>
              {" - auto-created. All members coordinate here. Files are rewritten as changes are made."}
            </div>
          )}
          {!backendLive && chatList.find(c=>c.id==="team")?.messages.length <= 1 && (
            <div style={{ padding: "8px 18px", background: "#1a0f00", borderBottom: "1px solid #3a2000", fontSize: 11, color: "#f59e0b" }}>
              Backend offline - running in mock mode. Run <code style={{ background: "#111", padding: "1px 5px", borderRadius: 3 }}>python -m api.main</code> to connect real agents.
            </div>
          )}

          <div ref={el => { msgsEndRef.current[activeChat] = el; }} style={S.messages}>
            {currentChat?.messages.map(m => <Message key={m.id} msg={m} compact={currentChat.type === "group"} />)}
            {typingIn[activeChat] && <Typing agentId={typingIn[activeChat]} />}
          </div>

          <div style={S.inputArea}>
            <input
              ref={uploadRef}
              type="file"
              accept=".csv,text/csv"
              style={{ display: "none" }}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleDatasetUpload(f);
              }}
            />
            <div style={S.inputRow}>
              <button
                onClick={() => uploadRef.current?.click()}
                disabled={!backendLive || uploading}
                title="Upload dataset (.csv)"
                style={{
                  width: 42,
                  height: 42,
                  borderRadius: 9,
                  border: "1px solid #1f2937",
                  background: "#0a0f1a",
                  color: "#93c5fd",
                  fontSize: 15,
                  cursor: "pointer",
                  opacity: (!backendLive || uploading) ? 0.35 : 1,
                  flexShrink: 0,
                }}
              >
                {uploading ? "..." : "Upload"}
              </button>
              {datasetReady && (
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <button
                    onClick={handlePhase3Check}
                    style={{ fontSize: 11, padding: "8px 10px", borderRadius: 6, background: "#0a1a12", border: "1px solid #1f3a2a", color: "#6ee7b7", cursor: "pointer" }}
                  >
                    Check Agents
                  </button>
                  <button
                    onClick={handleRunDirect}
                    style={{ fontSize: 11, padding: "8px 10px", borderRadius: 6, background: "#111827", border: "1px solid #374151", color: "#cbd5f5", cursor: "pointer" }}
                  >
                    Run Without Check
                  </button>
                </div>
              )}
              <textarea
                value={inputs[activeChat] || ""}
                onChange={e => setInputs(prev => ({ ...prev, [activeChat]: e.target.value }))}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(activeChat); } }}
                disabled={activeChat === "team" && isBusy}
                rows={2}
                placeholder={
                  activeChat === "team" && isBusy ? "Agents working..."
                  : activeChat === "team" ? "Give the team an instruction... e.g. 'create a fraud detection model'"
                  : currentChat?.type === "group" ? "Add instruction to this thread..."
                  : `Message ${AGENTS[activeChat]?.name}... e.g. 'fix the bug' | 'retrain' | 'deploy'`
                }
                style={{ ...S.textarea, borderColor: activeChat === "team" ? "#1a1a1a" : currentChat?.type === "group" ? "#6366f133" : (AGENTS[activeChat]?.color + "33"), opacity: activeChat === "team" && isBusy ? 0.5 : 1 }}
              />
              <button
                onClick={() => handleSend(activeChat)}
                disabled={(activeChat === "team" && isBusy) || !(inputs[activeChat]||"").trim()}
                style={{ ...S.sendBtn, background: activeChat === "team" ? "#6366f1" : currentChat?.type === "group" ? "#6366f1" : (AGENTS[activeChat]?.color || "#6366f1"), opacity: ((activeChat === "team" && isBusy) || !(inputs[activeChat]||"").trim()) ? 0.3 : 1 }}
              >Send</button>
            </div>
            <div style={{ fontSize: 9, color: "#1e1e1e", marginTop: 5 }}>
              ENTER to send | SHIFT+ENTER for newline
              {activeChat !== "team" && currentChat?.type !== "group" && <span style={{ color: "#2a2a2a" }}> | Group threads auto-spawn when more agents are needed</span>}
            </div>
          </div>
        </div>

        <RightPanel p2pLog={p2pLog} fileLog={fileLog} projectRoot={projectRoot} />
      </div>
    </div>
  );
}

// â”€â”€â”€ Styles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const S = {
  root:     { width: "100vw", height: "100vh", background: "#050505", display: "flex", flexDirection: "column", fontFamily: "'IBM Plex Mono',monospace", color: "#c8c8d0", overflow: "hidden" },
  header:   { padding: "10px 20px", borderBottom: "1px solid #111", display: "flex", alignItems: "center", justifyContent: "space-between", background: "#070707", flexShrink: 0 },
  logo:     { width: 36, height: 36, borderRadius: 9, background: "linear-gradient(135deg,#1e1b4b,#0c1a3a)", border: "1px solid #6366f133", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18 },
  body:     { flex: 1, display: "grid", gridTemplateColumns: "220px 1fr 310px", overflow: "hidden", minHeight: 0 },
  sidebar:  { borderRight: "1px solid #0f0f0f", background: "#060606", display: "flex", flexDirection: "column", overflow: "hidden" },
  chatArea: { display: "flex", flexDirection: "column", overflow: "hidden", minHeight: 0 },
  chatHdr:  { padding: "11px 18px", display: "flex", alignItems: "center", gap: 12, flexShrink: 0, minHeight: 58 },
  messages: { flex: 1, overflowY: "auto", padding: "18px 22px", display: "flex", flexDirection: "column", gap: 16, minHeight: 0 },
  inputArea:{ padding: "12px 18px 14px", borderTop: "1px solid #111", background: "#070707", flexShrink: 0 },
  inputRow: { display: "flex", gap: 8, alignItems: "flex-end" },
  textarea: { flex: 1, background: "#090909", border: "1px solid #1a1a1a", borderRadius: 9, padding: "10px 14px", color: "#c8c8d0", fontSize: 13, resize: "none", fontFamily: "inherit", lineHeight: 1.5, outline: "none", transition: "border-color .2s" },
  sendBtn:  { width: 42, height: 42, borderRadius: 9, border: "none", color: "#fff", fontSize: 18, cursor: "pointer", fontWeight: 700, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", transition: "opacity .15s" },
  actPanel: { borderLeft: "1px solid #0f0f0f", background: "#060606", display: "flex", flexDirection: "column", overflow: "hidden" },
  filesOverlay: { position: "fixed", inset: 0, background: "rgba(0,0,0,0.65)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 50, padding: 20 },
  filesPanel: { width: "90vw", height: "80vh", background: "#070707", border: "1px solid #141414", borderRadius: 12, display: "flex", flexDirection: "column", overflow: "hidden" },
  filesHeader: { padding: "12px 16px", borderBottom: "1px solid #111", display: "flex", alignItems: "center", justifyContent: "space-between" },
  filesBody: { flex: 1, display: "grid", gridTemplateColumns: "320px 1fr", minHeight: 0 },
  filesList: { borderRight: "1px solid #111", padding: 12, display: "flex", flexDirection: "column", minHeight: 0 },
  filesPreview: { padding: 12, minHeight: 0, overflow: "hidden" },
};

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #050505; }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #181818; border-radius: 4px; }
  textarea::placeholder { color: #222; }
  textarea:focus { outline: none; }
  button:focus { outline: none; }
  @keyframes mIn { from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:translateY(0);} }
  @keyframes sPulse { 0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(.8);} }
  @keyframes tDot { 0%,80%,100%{opacity:.2;transform:scale(.8);}40%{opacity:1;transform:scale(1.2);} }
`;

function detectTag(content) {
  const c = content.toLowerCase();
  if (["complete","done","deployed","approved","passed","success"].some(k => c.includes(k))) return "DONE";
  if (["critical","exploit","blocked","error","failed","alert"].some(k => c.includes(k))) return "ALERT";
  if (["found","result","report","eda","accuracy","score"].some(k => c.includes(k))) return "REPORT";
  return "STATUS";
}



