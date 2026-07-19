"use client";

interface GettingStartedCardProps {
  compact?: boolean;
}

export function GettingStartedCard({ compact = false }: GettingStartedCardProps) {
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
    <div className={`rounded-xl bg-zinc-900/80 border border-zinc-800/80 ${compact ? "p-3" : "p-4"}`}>
      <div className="text-xs font-bold text-zinc-100 mb-1.5 font-sans tracking-wide flex items-center gap-1.5">
        <span>🚀 Getting Started</span>
      </div>
      <div className="text-[11px] text-zinc-400 mb-2.5 leading-relaxed font-sans">
        Currently supports tabular CSV datasets. CNN, NLP, and hybrid datasets are in progress.
      </div>
      <div className="flex flex-col gap-1.5 max-h-[180px] overflow-y-auto pr-1 custom-scrollbar font-sans">
        {steps.map((s, i) => (
          <div key={i} className="text-[11px] text-zinc-300 leading-snug flex gap-2">
            <span className="text-indigo-400 font-mono font-semibold">{i + 1}.</span>
            <span className="flex-1">{s}</span>
          </div>
        ))}
      </div>
    </div>
  );
}