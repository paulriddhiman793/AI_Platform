"use client";

interface StatusDotProps {
  status: "idle" | "working" | "error";
  showLabel?: boolean;
}

export function StatusDot({ status, showLabel = true }: StatusDotProps) {
  const colors = {
    idle: "#4ade80",
    working: "#facc15",
    error: "#f87171",
  };
  const color = colors[status] || colors.idle;

  if (!showLabel) {
    return (
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          backgroundColor: color,
          boxShadow: `0 0 6px ${color}`,
          border: "1.5px solid #050508",
          display: "inline-block",
          flexShrink: 0,
          animation: status === "working" ? "sPulse 1.1s ease-in-out infinite" : "none",
        }}
        title={status === "idle" ? "Idle" : status === "working" ? "Working" : "Error"}
      />
    );
  }

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
      }}
    >
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: "50%",
          backgroundColor: color,
          boxShadow: `0 0 5px ${color}`,
          display: "inline-block",
          flexShrink: 0,
          animation: status === "working" ? "sPulse 1.1s ease-in-out infinite" : "none",
        }}
      />
      <span
        style={{
          fontSize: 11,
          fontWeight: 600,
          color: color,
          fontFamily: "monospace",
          letterSpacing: "0.02em",
        }}
      >
        {status === "idle" ? "Idle" : status === "working" ? "Working" : "Error"}
      </span>
    </span>
  );
}