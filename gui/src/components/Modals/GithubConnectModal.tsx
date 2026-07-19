"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";

interface GithubConnectModalProps {
  isOpen: boolean;
  onClose: () => void;
  githubToken: string;
  setGithubToken: (v: string) => void;
  githubOwner: string;
  setGithubOwner: (v: string) => void;
  githubRepo: string;
  setGithubRepo: (v: string) => void;
  githubVisibility: "private" | "public";
  setGithubVisibility: (v: "private" | "public") => void;
  onConnect: () => void;
}

export const GithubConnectModal: React.FC<GithubConnectModalProps> = ({
  isOpen,
  onClose,
  githubToken,
  setGithubToken,
  githubOwner,
  setGithubOwner,
  githubRepo,
  setGithubRepo,
  githubVisibility,
  setGithubVisibility,
  onConnect,
}) => {
  if (!isOpen) return null;

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
          className="w-full max-w-lg bg-[#0b0b10] border border-zinc-800 rounded-2xl p-6 shadow-2xl overflow-hidden"
        >
          <div className="flex items-center justify-between pb-4 border-b border-zinc-800/80 mb-5">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                </svg>
              </div>
              <div>
                <h2 className="text-base font-bold text-white font-sans">Connect GitHub Repository</h2>
                <p className="text-xs text-zinc-400">Link your workspace for branch creation & PRs</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-zinc-500 hover:text-white transition-colors p-1"
            >
              ✕
            </button>
          </div>

          <div className="flex flex-col gap-4">
            <div>
              <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">
                Personal Access Token (PAT)
              </label>
              <input
                type="password"
                value={githubToken}
                onChange={(e) => setGithubToken(e.target.value)}
                placeholder="github_pat_..."
                className="w-full bg-zinc-900/80 border border-zinc-800 rounded-xl px-3.5 py-2.5 text-xs text-white placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500/60"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">
                  Owner / Org
                </label>
                <input
                  type="text"
                  value={githubOwner}
                  onChange={(e) => setGithubOwner(e.target.value)}
                  placeholder="e.g. torvalds"
                  className="w-full bg-zinc-900/80 border border-zinc-800 rounded-xl px-3.5 py-2.5 text-xs text-white placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500/60"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">
                  Repository Name
                </label>
                <input
                  type="text"
                  value={githubRepo}
                  onChange={(e) => setGithubRepo(e.target.value)}
                  placeholder="e.g. linux"
                  className="w-full bg-zinc-900/80 border border-zinc-800 rounded-xl px-3.5 py-2.5 text-xs text-white placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500/60"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">
                Visibility
              </label>
              <div className="grid grid-cols-2 gap-2 p-1 bg-zinc-900/60 rounded-xl border border-zinc-800">
                {(["private", "public"] as const).map((v) => (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setGithubVisibility(v)}
                    className={`py-1.5 text-xs font-semibold rounded-lg capitalize transition-all ${
                      githubVisibility === v
                        ? "bg-zinc-800 text-indigo-300 border border-zinc-700/60"
                        : "text-zinc-500 hover:text-zinc-300"
                    }`}
                  >
                    {v}
                  </button>
                ))}
              </div>
            </div>

            <div className="bg-zinc-900/40 border border-zinc-800/80 rounded-xl p-3 text-[11px] text-zinc-400 leading-relaxed">
              Fine-grained PAT required with permissions: <span className="text-zinc-300 font-semibold">Contents (Read & Write)</span>,{" "}
              <span className="text-zinc-300 font-semibold">Pull requests (Read & Write)</span>, and Administration.{" "}
              <a
                href="https://github.com/settings/personal-access-tokens"
                target="_blank"
                rel="noreferrer"
                className="text-indigo-400 hover:underline inline-block mt-1"
              >
                Create token on GitHub →
              </a>
            </div>
          </div>

          <div className="flex justify-end gap-3 mt-6 pt-4 border-t border-zinc-800/80">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-xs font-semibold rounded-xl border border-zinc-800 hover:bg-zinc-900 text-zinc-400 transition-colors"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={!githubToken.trim() || !githubRepo.trim()}
              onClick={() => {
                onConnect();
                onClose();
              }}
              className="px-5 py-2 text-xs font-semibold rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white transition-colors shadow-lg shadow-indigo-600/20"
            >
              Connect & Sync
            </button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};
