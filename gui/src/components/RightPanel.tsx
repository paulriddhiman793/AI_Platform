"use client";

import { useState } from "react";
import { ActivityCard } from "./ActivityCard";
import { FileCard } from "./FileCard";
import { GettingStartedCard } from "./GettingStartedCard";
import type { P2PEntry, FileEntry } from "@/types";

interface RightPanelProps {
  p2pLog: P2PEntry[];
  fileLog: FileEntry[];
  projectRoot: string | null;
  onSelectFile?: (entry: FileEntry) => void;
}

export function RightPanel({ p2pLog, fileLog, projectRoot, onSelectFile }: RightPanelProps) {
  const [tab, setTab] = useState<"p2p" | "files">("p2p");

  return (
    <div className="w-full h-full bg-[#050508] flex flex-col font-mono select-none overflow-hidden">
      <div className="p-3 border-b border-zinc-800/80 flex-shrink-0">
        <GettingStartedCard compact />
      </div>

      {/* Tabs Header */}
      <div className="flex border-b border-zinc-800/80 bg-zinc-950/60 flex-shrink-0">
        {[
          ["p2p", "P2P MESSAGES"],
          ["files", "FILES"],
        ].map(([id, label]) => (
          <button
            key={id}
            onClick={() => setTab(id as "p2p" | "files")}
            className={`flex-1 py-2.5 text-[10px] font-bold tracking-widest uppercase transition-all border-b-2 ${
              tab === id
                ? "border-indigo-500 text-indigo-300 bg-zinc-900/60"
                : "border-transparent text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/30"
            }`}
          >
            <span>{label}</span>
            {id === "files" && fileLog.length > 0 && (
              <span className="ml-1 px-1.5 py-0.2 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800 text-[9px]">
                {fileLog.length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab Content Area */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2.5 custom-scrollbar font-sans">
        {tab === "p2p" && (
          <>
            {!p2pLog.length && (
              <div className="text-zinc-500 text-xs text-center py-10 font-mono italic">
                Agent-to-agent messages appear here...
              </div>
            )}
            {p2pLog.map((e) => (
              <ActivityCard key={e.id} entry={e} />
            ))}
          </>
        )}

        {tab === "files" && (
          <>
            {projectRoot && (
              <div className="p-2 mb-2 rounded-lg bg-zinc-900/90 border border-zinc-800 text-[11px] text-zinc-300 font-mono break-all">
                <span className="text-zinc-500 mr-1.5">DIR:</span>
                <span className="text-indigo-300">{projectRoot}</span>
              </div>
            )}
            {!fileLog.length && (
              <div className="text-zinc-500 text-xs text-center py-10 font-mono italic">
                Files written by agents appear here...
              </div>
            )}
            {fileLog.map((e) => (
              <FileCard key={e.id} entry={e} onSelectFile={onSelectFile} />
            ))}
          </>
        )}
      </div>
    </div>
  );
}