import { useState, useEffect, useRef, useCallback } from "react";
import { AGENTS, detectScenario, detectDirectResponse } from "./data/agents.js";

let _id = 0;
const uid = () => ++_id;
const fmt = (ts) => new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

const TAG = {
  STATUS: { bg: "#0d1f0d", color: "#4ade80", border: "#1a3a1a" },
  REPORT: { bg: "#0a1628", color: "#60a5fa", border: "#1a3060" },
  ALERT:  { bg: "#2a0808", color: "#f87171", border: "#5a1818" },
  DONE:   { bg: "#16113a", color: "#a78bfa", border: "#302060" },
};

// ─── Avatar ──────────────────────────────────────────────────────────────────
function Avatar({ agentId, size = 32 }) {
  const a = AGENTS[agentId];
  const isUser = agentId === "user";
  return (
    <div style={{
      width: size, height: size, borderRadius: "50%", flexShrink: 0,
      display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: Math.floor(size * 0.44),
      background: isUser ? "#14122e" : (a?.bgColor || "#111"),
      border: `2px solid ${isUser ? "#6366f1" : (a?.color || "#333")}`,
    }}>
      {isUser ? "👤" : a?.icon}
    </div>
  );
}

function StatusDot({ status }) {
  const c = { idle: "#4ade80", working: "#facc15", error: "#f87171" }[status] || "#4ade80";
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
      <span style={{
        width: 7, height: 7, borderRadius: "50%", background: c,
        boxShadow: `0 0 5px ${c}`, display: "inline-block",
        animation: status === "working" ? "sPulse 1.1s ease-in-out infinite" : "none",
      }} />
      <span style={{ fontSize: 10, color: c }}>
        {{ idle: "Idle", working: "Working", error: "Error" }[status] || "Idle"}
      </span>
    </span>
  );
}

// ─── Message bubble (team + direct) ─────────────────────────────────────────
function Message({ msg, compact = false }) {
  const isUser = msg.from === "user";
  const a = AGENTS[msg.from];
  const ts = TAG[msg.tag];
  const bubbleSize = compact ? 28 : 32;

  return (
    <div style={{
      display: "flex",
      flexDirection: isUser ? "row-reverse" : "row",
      gap: 12, alignItems: "flex-start",
      animation: "mIn .28s ease-out",
    }}>
      <Avatar agentId={msg.from} size={bubbleSize} />
      <div style={{ maxWidth: "80%", display: "flex", flexDirection: "column", gap: 5 }}>
        <div style={{
          display: "flex", alignItems: "center", gap: 7,
          flexDirection: isUser ? "row-reverse" : "row",
        }}>
          <span style={{
            fontSize: compact ? 10 : 11, fontWeight: 700, textTransform: "uppercase",
            color: isUser ? "#818cf8" : (a?.color || "#888"),
          }}>
            {isUser ? "You" : a?.name || msg.from}
          </span>
          {msg.tag && !isUser && ts && (
            <span style={{
              fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 4,
              background: ts.bg, color: ts.color, border: `1px solid ${ts.border}`,
              letterSpacing: "0.07em",
            }}>{msg.tag}</span>
          )}
          <span style={{ fontSize: 9, color: "#2a2a2a" }}>{fmt(msg.timestamp)}</span>
        </div>
        <div style={{
          padding: "11px 16px",
          borderRadius: isUser ? "14px 3px 14px 14px" : "3px 14px 14px 14px",
          background: isUser ? "#1a1740" : "#0b0b0b",
          border: `1px solid ${isUser ? "#6366f122" : (ts ? ts.border + "44" : "#181818")}`,
          color: "#c8c8d0", fontSize: 13.5, lineHeight: 1.7,
        }}>
          {msg.content}
        </div>
      </div>
    </div>
  );
}

// ─── Typing indicator ─────────────────────────────────────────────────────────
function Typing({ agentId }) {
  const a = AGENTS[agentId];
  if (!a) return null;
  return (
    <div style={{ display: "flex", gap: 10, alignItems: "center", opacity: .65, animation: "mIn .2s ease-out" }}>
      <Avatar agentId={agentId} size={28} />
      <div style={{ padding: "9px 14px", borderRadius: "3px 13px 13px 13px", background: "#0b0b0b", border: "1px solid #181818", display: "flex", gap: 4, alignItems: "center" }}>
        <span style={{ fontSize: 10, color: a.color, marginRight: 5 }}>{a.name}</span>
        {[0,1,2].map(i => (
          <span key={i} style={{ width: 5, height: 5, borderRadius: "50%", background: a.color, display: "inline-block", animation: `tDot 1.2s ease-in-out ${i*.2}s infinite` }} />
        ))}
      </div>
    </div>
  );
}

// ─── Activity feed card (wider & clearer) ────────────────────────────────────
function ActivityCard({ entry }) {
  const fa = AGENTS[entry.from];
  const ta = AGENTS[entry.to];
  if (!fa || !ta) return null;
  return (
    <div style={{
      padding: "12px 14px", borderRadius: 8, background: "#090909",
      border: `1px solid ${fa.color}28`, marginBottom: 6,
      animation: "mIn .22s ease-out",
    }}>
      {/* Header row */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <Avatar agentId={entry.from} size={24} />
        <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
          <span style={{ fontSize: 10, color: fa.color, fontWeight: 700 }}>{fa.name}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 4 }}>
          <svg width="14" height="10" viewBox="0 0 14 10" fill="none">
            <path d="M1 5h10M8 2l3 3-3 3" stroke="#333" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <Avatar agentId={entry.to} size={24} />
          <span style={{ fontSize: 10, color: ta.color, fontWeight: 700 }}>{ta.name}</span>
        </div>
        <span style={{ fontSize: 9, color: "#252525", marginLeft: "auto" }}>{fmt(entry.timestamp)}</span>
      </div>
      {/* Message content */}
      <div style={{
        fontSize: 12, color: "#666", lineHeight: 1.6,
        padding: "8px 10px", borderRadius: 6,
        background: "#060606", border: "1px solid #141414",
      }}>
        {entry.content}
      </div>
    </div>
  );
}

// ─── Sidebar ─────────────────────────────────────────────────────────────────
function Sidebar({ agents, activeChat, onSelectChat, chatList }) {
  return (
    <div style={S.sidebar}>
      <div style={{ padding: "12px 12px 6px", borderBottom: "1px solid #0f0f0f", flexShrink: 0 }}>
        <span style={{ fontSize: 9, color: "#3a3a3a", fontWeight: 700, letterSpacing: ".12em" }}>CHATS</span>
      </div>

      {/* Team */}
      <div style={{ padding: "8px 8px 2px" }}>
        <SidebarBtn
          id="team"
          isActive={activeChat === "team"}
          icon="🤝"
          iconBg="#12103a"
          iconBorder="#6366f1"
          color="#a5b4fc"
          name="Team"
          sub="All agents · main channel"
          unread={0}
          onClick={() => onSelectChat("team")}
        />
      </div>

      <div style={{ padding: "8px 12px 4px" }}>
        <span style={{ fontSize: 9, color: "#252525", letterSpacing: ".1em" }}>DIRECT</span>
      </div>

      {/* Individual agents */}
      <div style={{ padding: "0 8px 8px", display: "flex", flexDirection: "column", gap: 1 }}>
        {Object.values(agents).map(a => {
          const chat = chatList.find(c => c.id === a.id);
          const unread = chat?.unread || 0;
          return (
            <SidebarBtn
              key={a.id}
              id={a.id}
              isActive={activeChat === a.id}
              icon={a.icon}
              iconBg={a.bgColor}
              iconBorder={a.color}
              color={a.color}
              name={a.name}
              sub={<StatusDot status={a.status} />}
              unread={unread}
              onClick={() => onSelectChat(a.id)}
            />
          );
        })}
      </div>

      {/* Group chats */}
      {chatList.filter(c => c.type === "group").length > 0 && (
        <>
          <div style={{ padding: "6px 12px 4px", borderTop: "1px solid #0f0f0f" }}>
            <span style={{ fontSize: 9, color: "#252525", letterSpacing: ".1em" }}>GROUP THREADS</span>
          </div>
          <div style={{ padding: "0 8px 8px", display: "flex", flexDirection: "column", gap: 1, flex: 1, overflowY: "auto" }}>
            {chatList.filter(c => c.type === "group").map(g => (
              <SidebarBtn
                key={g.id}
                id={g.id}
                isActive={activeChat === g.id}
                icon="👥"
                iconBg="#1a1a2e"
                iconBorder="#6366f1"
                color="#a78bfa"
                name={g.title}
                sub={g.members.map(m => AGENTS[m]?.icon).join(" ")}
                unread={g.unread || 0}
                onClick={() => onSelectChat(g.id)}
                isNew={g.isNew}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function SidebarBtn({ id, isActive, icon, iconBg, iconBorder, color, name, sub, unread, onClick, isNew }) {
  return (
    <button onClick={onClick} style={{
      width: "100%", padding: "8px 9px", borderRadius: 8, cursor: "pointer",
      display: "flex", alignItems: "center", gap: 9, marginBottom: 1,
      background: isActive ? iconBg + "cc" : "transparent",
      border: `1px solid ${isActive ? iconBorder + "aa" : "transparent"}`,
      transition: "all .15s",
    }}>
      <div style={{ position: "relative", flexShrink: 0 }}>
        <div style={{
          width: 34, height: 34, borderRadius: "50%", background: iconBg,
          border: `2px solid ${iconBorder}`, display: "flex",
          alignItems: "center", justifyContent: "center", fontSize: 16,
        }}>{icon}</div>
        {unread > 0 && (
          <span style={{
            position: "absolute", top: -2, right: -2,
            width: 14, height: 14, borderRadius: "50%", background: "#ef4444",
            border: "2px solid #050505", display: "flex", alignItems: "center",
            justifyContent: "center", fontSize: 7, color: "#fff", fontWeight: 700,
          }}>{unread}</span>
        )}
        {isNew && (
          <span style={{
            position: "absolute", top: -2, right: -2,
            width: 10, height: 10, borderRadius: "50%", background: "#4ade80",
            border: "2px solid #050505", animation: "sPulse 1.5s ease-in-out infinite",
          }} />
        )}
      </div>
      <div style={{ flex: 1, textAlign: "left", minWidth: 0 }}>
        <div style={{ fontSize: 11, fontWeight: 700, color, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {name}
        </div>
        <div style={{ fontSize: 10, color: "#3a3a3a", marginTop: 1 }}>
          {typeof sub === "string" ? sub : sub}
        </div>
      </div>
      {isActive && <span style={{ width: 6, height: 6, borderRadius: "50%", background: color, boxShadow: `0 0 5px ${color}`, flexShrink: 0 }} />}
    </button>
  );
}

// ─── Chat header ─────────────────────────────────────────────────────────────
function ChatHeader({ chat, agents }) {
  if (chat.id === "team") return (
    <div style={{ ...S.chatHdr, background: "#070707", borderBottom: "1px solid #111" }}>
      <div style={{ width: 34, height: 34, borderRadius: "50%", background: "#12103a", border: "2px solid #6366f1", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 17 }}>🤝</div>
      <div>
        <div style={{ fontSize: 13, fontWeight: 700, color: "#a5b4fc" }}>Team Chat</div>
        <div style={{ fontSize: 10, color: "#383838" }}>All agent reports and updates appear here</div>
      </div>
    </div>
  );

  if (chat.type === "group") {
    const memberAgents = chat.members.map(m => AGENTS[m]).filter(Boolean);
    return (
      <div style={{ ...S.chatHdr, background: "#0a091f", borderBottom: "1px solid #6366f122" }}>
        <div style={{ width: 34, height: 34, borderRadius: "50%", background: "#1a1a3a", border: "2px solid #6366f1", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16 }}>👥</div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#a78bfa" }}>{chat.title}</div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 2 }}>
            {memberAgents.map(a => (
              <span key={a.id} style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: a.color }}>
                <span>{a.icon}</span>{a.shortName}
              </span>
            ))}
          </div>
        </div>
        <div style={{ fontSize: 10, color: "#252525", maxWidth: 200, textAlign: "right", lineHeight: 1.4 }}>
          {chat.reason}
        </div>
      </div>
    );
  }

  // Direct chat
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

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [agents, setAgents] = useState(() =>
    Object.fromEntries(Object.entries(AGENTS).map(([k,v]) => [k, { ...v }]))
  );

  // chatList: array of chat objects
  // type: "team" | "direct" | "group"
  const [chatList, setChatList] = useState(() => {
    const list = [
      { id: "team", type: "team", messages: [{
        id: uid(), from: "orchestrator", tag: "STATUS",
        content: "AI Engineering Platform online. All 6 agents standing by. What would you like to build?",
        timestamp: Date.now(),
      }], unread: 0 },
    ];
    for (const a of Object.values(AGENTS)) {
      list.push({
        id: a.id, type: "direct",
        messages: [{
          id: uid(), from: a.id, tag: "STATUS",
          content: `Hi, I'm ${a.name}. ${a.role} Give me a direct instruction and I'll handle it here.`,
          timestamp: Date.now(),
        }],
        unread: 0,
      });
    }
    return list;
  });

  const [activeChat, setActiveChat] = useState("team");
  const [typingIn, setTypingIn] = useState({}); // { chatId: agentId | null }
  const [inputs, setInputs] = useState({});
  const [isBusy, setIsBusy] = useState(false);

  const chatEndRefs = useRef({});
  const timers = useRef([]);

  // Auto-scroll active chat
  useEffect(() => {
    const el = chatEndRefs.current[activeChat];
    if (el) el.scrollTop = el.scrollHeight;
  }, [chatList, typingIn, activeChat]);

  // Clear unread on chat switch
  useEffect(() => {
    setChatList(prev => prev.map(c => c.id === activeChat ? { ...c, unread: 0, isNew: false } : c));
  }, [activeChat]);

  const setAgentStatus = useCallback((id, s) => {
    setAgents(prev => ({ ...prev, [id]: { ...prev[id], status: s } }));
  }, []);

  const addMsg = useCallback((chatId, from, content, tag) => {
    setChatList(prev => prev.map(c => {
      if (c.id !== chatId) return c;
      const isVisible = activeChat === chatId;
      return {
        ...c,
        messages: [...c.messages, { id: uid(), from, content, tag, timestamp: Date.now() }],
        unread: isVisible ? 0 : (c.unread || 0) + (from !== "user" ? 1 : 0),
      };
    }));
  }, [activeChat]);

  const addActivity = useCallback((from, to, content) => {
    // Add to activity log stored in team chat's activityLog array
    setChatList(prev => prev.map(c => {
      if (c.id !== "team") return c;
      return { ...c, activityLog: [...(c.activityLog || []), { id: uid(), from, to, content, timestamp: Date.now() }] };
    }));
  }, []);

  const setTyping = useCallback((chatId, agentId) => {
    setTypingIn(prev => ({ ...prev, [chatId]: agentId || null }));
  }, []);

  const clearTimers = () => { timers.current.forEach(clearTimeout); timers.current = []; };

  // ── Spawn a group chat ────────────────────────────────────────────────────
  const spawnGroupChat = useCallback((groupDef, triggerChatId) => {
    const groupId = "group_" + uid();
    const newGroup = {
      id: groupId,
      type: "group",
      title: groupDef.title,
      reason: groupDef.reason,
      members: groupDef.members,
      messages: [],
      unread: 0,
      isNew: true,
    };

    setChatList(prev => [...prev, newGroup]);

    // Post a notice in the originating chat
    const originAgent = groupDef.members[0];
    addMsg(triggerChatId, originAgent, `↗ Spawning group thread: "${groupDef.title}" — ${groupDef.reason}`, "STATUS");

    // After short delay, switch to group chat and run its flow
    const t = setTimeout(() => {
      setActiveChat(groupId);

      // Run group flow
      groupDef.members.forEach(id => setAgentStatus(id, "working"));

      groupDef.groupFlow.forEach((step, idx) => {
        const isLast = idx === groupDef.groupFlow.length - 1;

        const t1 = setTimeout(() => {
          setTyping(groupId, step.from);
        }, step.delay - 400);

        const t2 = setTimeout(() => {
          setTyping(groupId, null);
          addMsg(groupId, step.from, step.content, step.tag);
          if (isLast) {
            groupDef.members.forEach(id => setAgentStatus(id, "idle"));
          }
        }, step.delay);

        timers.current.push(t1, t2);
      });
    }, 800);
    timers.current.push(t);
  }, [addMsg, setAgentStatus, setTyping]);

  // ── Run team scenario ─────────────────────────────────────────────────────
  const runTeamScenario = useCallback((scenario) => {
    clearTimers();
    setIsBusy(true);

    const inv = new Set(scenario.flow.filter(s => s.from !== "user").map(s => s.from));
    inv.forEach(id => setAgentStatus(id, "working"));

    scenario.flow.forEach((step, idx) => {
      const isLast = idx === scenario.flow.length - 1;

      const t1 = setTimeout(() => {
        if (step.type === "team") setTyping("team", step.from);
      }, step.delay - 400);

      const t2 = setTimeout(() => {
        setTyping("team", null);
        if (step.type === "team") addMsg("team", step.from, step.content, step.tag);
        else if (step.type === "p2p") addActivity(step.from, step.to, step.content);
        if (isLast) { inv.forEach(id => setAgentStatus(id, "idle")); setIsBusy(false); }
      }, step.delay);

      timers.current.push(t1, t2);
    });
  }, [addMsg, addActivity, setAgentStatus, setTyping]);

  // ── Handle team send ──────────────────────────────────────────────────────
  const handleSend = useCallback((chatId) => {
    const text = (inputs[chatId] || "").trim();
    if (!text) return;
    setInputs(prev => ({ ...prev, [chatId]: "" }));

    const chat = chatList.find(c => c.id === chatId);
    if (!chat) return;

    addMsg(chatId, "user", text, null);

    if (chatId === "team") {
      if (isBusy) return;
      const scenario = detectScenario(text);
      const t = setTimeout(() => runTeamScenario(scenario), 350);
      timers.current.push(t);
      return;
    }

    if (chat.type === "direct") {
      const agentId = chatId;
      setAgentStatus(agentId, "working");

      const { steps, spawnGroup } = detectDirectResponse(agentId, text);

      steps.forEach((step, idx) => {
        const isLastStep = idx === steps.length - 1;
        const willSpawn = isLastStep && spawnGroup;

        const t1 = setTimeout(() => setTyping(chatId, agentId), step.delay - 400);
        const t2 = setTimeout(() => {
          setTyping(chatId, null);
          addMsg(chatId, agentId, step.content, step.tag);
          if (isLastStep) {
            setAgentStatus(agentId, "idle");
            if (willSpawn) {
              spawnGroupChat(spawnGroup, chatId);
            }
          }
        }, step.delay);

        timers.current.push(t1, t2);
      });
      return;
    }

    if (chat.type === "group") {
      // In group chats, the first member responds
      const responder = chat.members[0];
      setAgentStatus(responder, "working");

      const t1 = setTimeout(() => setTyping(chatId, responder), 800);
      const t2 = setTimeout(() => {
        setTyping(chatId, null);
        setAgentStatus(responder, "idle");
        addMsg(chatId, responder, "Noted. I'll incorporate that into the current workflow and update everyone here.", "STATUS");
      }, 1600);
      timers.current.push(t1, t2);
    }
  }, [inputs, chatList, isBusy, addMsg, addActivity, setAgentStatus, setTyping, runTeamScenario, spawnGroupChat]);

  const currentChat = chatList.find(c => c.id === activeChat);
  const teamChat = chatList.find(c => c.id === "team");

  return (
    <div style={S.root}>
      <style>{CSS}</style>

      {/* Header */}
      <header style={S.header}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={S.logo}>🤖</div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, letterSpacing: ".05em", color: "#e8e8f0" }}>AI Engineering Platform</div>
            <div style={{ fontSize: 9, color: "#3a3a3a", letterSpacing: ".1em", textTransform: "uppercase" }}>Multi-Agent Autonomous System · v1.0</div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          {["Phase 1 Active", "6 Agents", "Mock Mode"].map((t, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: "#4ade80", background: "#0a2a0a", border: "1px solid #1a4a1a", padding: "3px 8px", borderRadius: 20 }}>
              <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#4ade80", display: "inline-block" }} />
              {t}
            </div>
          ))}
        </div>
      </header>

      <div style={S.body}>
        {/* Sidebar */}
        <Sidebar
          agents={agents}
          activeChat={activeChat}
          onSelectChat={setActiveChat}
          chatList={chatList}
        />

        {/* Main chat */}
        <div style={S.chatArea}>
          {currentChat && <ChatHeader chat={currentChat} agents={agents} />}

          {/* Messages */}
          <div
            ref={el => { chatEndRefs.current[activeChat] = el; }}
            style={S.messages}
          >
            {/* Group chat context banner */}
            {currentChat?.type === "group" && (
              <div style={{
                padding: "10px 14px", borderRadius: 8, marginBottom: 4,
                background: "#0a091f", border: "1px solid #6366f122",
                fontSize: 12, color: "#404060", lineHeight: 1.6,
              }}>
                <span style={{ color: "#a78bfa", fontWeight: 700 }}>Group thread</span> — auto-created because additional agents were needed.
                You can give instructions here and all members will coordinate and respond.
              </div>
            )}

            {/* Direct chat context banner */}
            {currentChat?.type === "direct" && (
              <div style={{
                padding: "10px 14px", borderRadius: 8, marginBottom: 4,
                background: "#090909", border: `1px solid ${AGENTS[activeChat]?.color}18`,
                fontSize: 12, color: "#383838", lineHeight: 1.6,
              }}>
                <span style={{ color: AGENTS[activeChat]?.color, fontWeight: 700 }}>Private channel</span> — instructions here are handled directly. If another agent is needed, a group thread will be created automatically.
              </div>
            )}

            {currentChat?.messages.map(m => <Message key={m.id} msg={m} compact={currentChat.type === "group"} />)}
            {typingIn[activeChat] && <Typing agentId={typingIn[activeChat]} />}
          </div>

          {/* Input */}
          <div style={S.inputArea}>
            <div style={S.inputRow}>
              <textarea
                value={inputs[activeChat] || ""}
                onChange={e => setInputs(prev => ({ ...prev, [activeChat]: e.target.value }))}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(activeChat); } }}
                disabled={activeChat === "team" && isBusy}
                rows={2}
                placeholder={
                  activeChat === "team"
                    ? (isBusy ? "Agents working…" : "Give the team an instruction… try: 'train a churn model' · 'security audit' · 'build a dashboard' · 'run EDA'")
                    : currentChat?.type === "group"
                    ? `Add instruction to this thread…`
                    : `Message ${AGENTS[activeChat]?.name} directly… try: 'fix the bug' · 'retrain' · 'deploy' · 'run a full audit'`
                }
                style={{
                  ...S.textarea,
                  borderColor: activeChat === "team" ? "#1a1a1a" :
                    currentChat?.type === "group" ? "#6366f122" :
                    (AGENTS[activeChat]?.color + "22"),
                  opacity: (activeChat === "team" && isBusy) ? 0.5 : 1,
                }}
              />
              <button
                onClick={() => handleSend(activeChat)}
                disabled={activeChat === "team" && isBusy}
                style={{
                  ...S.sendBtn,
                  background: activeChat === "team" ? "#6366f1" :
                    currentChat?.type === "group" ? "#6366f1" :
                    (AGENTS[activeChat]?.color || "#6366f1"),
                  opacity: ((activeChat === "team" && isBusy) || !(inputs[activeChat] || "").trim()) ? 0.3 : 1,
                }}
              >↑</button>
            </div>
            <div style={{ fontSize: 9, color: "#1e1e1e", marginTop: 5, letterSpacing: ".05em" }}>
              ENTER to send · SHIFT+ENTER for newline
              {activeChat !== "team" && currentChat?.type !== "group" && (
                <span style={{ color: "#2a2a2a" }}> · Group threads auto-spawn when more agents are needed</span>
              )}
            </div>
          </div>
        </div>

        {/* Activity feed */}
        <div style={S.actPanel}>
          <div style={{ padding: "12px 12px 8px", borderBottom: "1px solid #0f0f0f", display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
            <span style={{ fontSize: 9, color: "#3a3a3a", fontWeight: 700, letterSpacing: ".12em" }}>P2P MESSAGES</span>
            <span style={{ fontSize: 9, color: "#4ade80", background: "#0a2a0a", border: "1px solid #1a4a1a", padding: "2px 6px", borderRadius: 4 }}>LIVE</span>
          </div>
          <div style={{ flex: 1, overflowY: "auto", padding: "8px 10px" }}>
            {!(teamChat?.activityLog?.length) && (
              <div style={{ color: "#1e1e1e", fontSize: 11, textAlign: "center", marginTop: 20 }}>Agent-to-agent messages appear here…</div>
            )}
            {(teamChat?.activityLog || []).map(e => <ActivityCard key={e.id} entry={e} />)}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Styles ──────────────────────────────────────────────────────────────────
const S = {
  root: { width: "100vw", height: "100vh", background: "#050505", display: "flex", flexDirection: "column", fontFamily: "'IBM Plex Mono',monospace", color: "#c8c8d0", overflow: "hidden" },
  header: { padding: "10px 20px", borderBottom: "1px solid #111", display: "flex", alignItems: "center", justifyContent: "space-between", background: "#070707", flexShrink: 0 },
  logo: { width: 36, height: 36, borderRadius: 9, background: "linear-gradient(135deg,#1e1b4b,#0c1a3a)", border: "1px solid #6366f133", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18 },
  body: { flex: 1, display: "grid", gridTemplateColumns: "220px 1fr 310px", overflow: "hidden", minHeight: 0 },
  sidebar: { borderRight: "1px solid #0f0f0f", background: "#060606", display: "flex", flexDirection: "column", overflow: "hidden" },
  chatArea: { display: "flex", flexDirection: "column", overflow: "hidden", minHeight: 0 },
  chatHdr: { padding: "11px 18px", display: "flex", alignItems: "center", gap: 12, flexShrink: 0, minHeight: 58 },
  messages: { flex: 1, overflowY: "auto", padding: "18px 22px", display: "flex", flexDirection: "column", gap: 16, minHeight: 0 },
  inputArea: { padding: "12px 18px 14px", borderTop: "1px solid #111", background: "#070707", flexShrink: 0 },
  inputRow: { display: "flex", gap: 8, alignItems: "flex-end" },
  textarea: { flex: 1, background: "#090909", border: "1px solid #1a1a1a", borderRadius: 9, padding: "10px 14px", color: "#c8c8d0", fontSize: 13, resize: "none", fontFamily: "inherit", lineHeight: 1.5, transition: "border-color .2s", outline: "none" },
  sendBtn: { width: 42, height: 42, borderRadius: 9, border: "none", color: "#fff", fontSize: 18, cursor: "pointer", fontWeight: 700, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", transition: "opacity .15s" },
  actPanel: { borderLeft: "1px solid #0f0f0f", background: "#060606", display: "flex", flexDirection: "column", overflow: "hidden" },
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
  @keyframes mIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
  @keyframes sPulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:.4; transform:scale(.8); } }
  @keyframes tDot { 0%,80%,100% { opacity:.2; transform:scale(.8); } 40% { opacity:1; transform:scale(1.2); } }
`;