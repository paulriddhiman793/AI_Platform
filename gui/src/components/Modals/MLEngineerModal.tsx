"use client";

import React, { useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { API_URL, safeLocalStorageSet } from "@/utils/config";

interface MLEngineerModalProps {
  isOpen: boolean;
  onClose: () => void;
  authToken: string;
  workerStatus: "connected" | "disconnected" | "error" | "unknown";
  setWorkerStatus: (s: "connected" | "disconnected" | "error" | "unknown") => void;
  workerToken: string;
  setWorkerToken: (t: string) => void;
  workerProjectPath: string;
  setWorkerProjectPath: (p: string) => void;
}

export const MLEngineerModal: React.FC<MLEngineerModalProps> = ({
  isOpen,
  onClose,
  authToken,
  workerStatus,
  setWorkerStatus,
  workerToken,
  setWorkerToken,
  workerProjectPath,
  setWorkerProjectPath,
}) => {
  const [loadingPair, setLoadingPair] = React.useState(false);
  const [copied, setCopied] = React.useState(false);

  const fetchStatus = useCallback(async () => {
    if (!authToken) return;
    try {
      const res = await fetch(`${API_URL}/worker/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auth_token: authToken }),
      });
      const data = await res.json().catch(() => ({ status: "disconnected" }));
      if (res.ok && data.status) {
        setWorkerStatus(data.status);
      } else {
        setWorkerStatus("disconnected");
      }
    } catch {
      setWorkerStatus("disconnected");
    }
  }, [authToken, setWorkerStatus]);

  React.useEffect(() => {
    if (isOpen) fetchStatus();
  }, [isOpen, fetchStatus]);

  const handleGeneratePairToken = async () => {
    if (!authToken) return;
    setLoadingPair(true);
    try {
      const res = await fetch(`${API_URL}/worker/pair`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auth_token: authToken }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.token) {
        setWorkerToken(data.token);
      }
    } catch {
      // ignore
    } finally {
      setLoadingPair(false);
    }
  };

  const handleCopy = () => {
    if (!workerToken) return;
    navigator.clipboard.writeText(workerToken);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!isOpen) return null;

  const statusColor = {
    connected: "bg-emerald-500 text-emerald-400 border-emerald-500/30",
    disconnected: "bg-red-500 text-red-400 border-red-500/30",
    error: "bg-amber-500 text-amber-400 border-amber-500/30",
    unknown: "bg-zinc-500 text-zinc-400 border-zinc-500/30",
  }[workerStatus] || "bg-zinc-500 text-zinc-400 border-zinc-500/30";

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 font-mono"
      >
        <motion.div
          initial={{ scale: 0.95, y: 15, opacity: 0 }}
          animate={{ scale: 1, y: 0, opacity: 1 }}
          exit={{ scale: 0.95, y: 15, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
          className="w-full max-w-xl bg-[#0b0b10] border border-zinc-800 rounded-2xl p-6 shadow-2xl overflow-hidden"
        >
          <div className="flex items-center justify-between pb-4 border-b border-zinc-800/80 mb-5">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
                ⚡
              </div>
              <div>
                <h2 className="text-base font-bold text-white font-sans">Get ML Engineer (Local Worker)</h2>
                <p className="text-xs text-zinc-400">Execute terminal commands & deploy locally on your machine</p>
              </div>
            </div>
            <button onClick={onClose} className="text-zinc-500 hover:text-white transition-colors p-1">
              ✕
            </button>
          </div>

          <div className="flex flex-col gap-5">
            {/* Status box */}
            <div className="flex items-center justify-between p-3.5 bg-zinc-900/60 rounded-xl border border-zinc-800">
              <div className="flex items-center gap-2.5">
                <div className="text-xs text-zinc-400">Worker Connection Status:</div>
                <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-zinc-950 border border-zinc-800">
                  <div className={`w-2 h-2 rounded-full ${statusColor.split(" ")[0]}`} />
                  <span className="text-xs font-semibold uppercase tracking-wider text-zinc-200">
                    {workerStatus}
                  </span>
                </div>
              </div>
              <button
                onClick={fetchStatus}
                className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
              >
                Refresh Status
              </button>
            </div>

            {/* Local project path */}
            <div>
              <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">
                Local Project Folder (used for deployments)
              </label>
              <input
                type="text"
                value={workerProjectPath}
                onChange={(e) => {
                  setWorkerProjectPath(e.target.value);
                  safeLocalStorageSet("worker_project_path", e.target.value);
                }}
                placeholder="C:\Users\riddh\Projects\MyModel"
                className="w-full bg-zinc-900/80 border border-zinc-800 rounded-xl px-3.5 py-2.5 text-xs text-white placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500/60"
              />
            </div>

            {/* Pairing Token */}
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                  Pairing Token
                </label>
                <button
                  type="button"
                  onClick={handleGeneratePairToken}
                  disabled={loadingPair}
                  className="text-xs text-indigo-400 hover:text-indigo-300 font-semibold"
                >
                  {loadingPair ? "Generating..." : "Generate New Pairing Token"}
                </button>
              </div>
              <div className="flex gap-2">
                <input
                  type="text"
                  readOnly
                  value={workerToken || "Click 'Generate New Pairing Token' to start"}
                  className="flex-1 bg-zinc-950 border border-zinc-800 rounded-xl px-3.5 py-2.5 text-xs text-zinc-300 font-mono select-all focus:outline-none"
                />
                <button
                  type="button"
                  disabled={!workerToken}
                  onClick={handleCopy}
                  className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40 text-xs font-semibold text-white rounded-xl border border-zinc-700 transition-all flex items-center gap-1.5"
                >
                  {copied ? "Copied!" : "Copy"}
                </button>
              </div>
            </div>

            {/* Step-by-step instructions */}
            <div className="bg-zinc-900/40 border border-zinc-800/80 rounded-xl p-4 text-xs text-zinc-400 leading-relaxed space-y-2">
              <div className="font-semibold text-zinc-300 uppercase text-[11px] tracking-wider">
                Setup Instructions:
              </div>
              <div>1. Download the local worker archive below and extract it on your computer.</div>
              <div>2. Run <code className="text-indigo-300 bg-zinc-950 px-1.5 py-0.5 rounded">pip install -r requirements.txt</code> inside the folder.</div>
              <div>
                3. Start the worker by running: <br />
                <code className="text-emerald-400 bg-zinc-950 px-2 py-1 rounded block mt-1">
                  python worker.py --token {workerToken || "<YOUR_PAIRING_TOKEN>"} --url {API_URL}
                </code>
              </div>
            </div>
          </div>

          <div className="flex justify-between items-center mt-6 pt-4 border-t border-zinc-800/80">
            <a
              href={`${API_URL}/worker/download`}
              target="_blank"
              rel="noreferrer"
              className="px-4 py-2 text-xs font-semibold rounded-xl bg-zinc-800 hover:bg-zinc-700 text-indigo-300 border border-zinc-700 transition-colors flex items-center gap-2"
            >
              <span>📥 Download local_ml_worker.zip</span>
            </a>
            <button
              type="button"
              onClick={onClose}
              className="px-5 py-2 text-xs font-semibold rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white transition-colors"
            >
              Done
            </button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};
