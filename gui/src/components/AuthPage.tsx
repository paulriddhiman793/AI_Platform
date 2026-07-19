"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { API_URL } from "@/utils/config";

interface AuthPageProps {
  authMode: "login" | "register";
  setAuthMode: (mode: "login" | "register") => void;
  authEmail: string;
  setAuthEmail: (email: string) => void;
  authPassword: string;
  setAuthPassword: (password: string) => void;
  authLoading: boolean;
  setAuthLoading: (loading: boolean) => void;
  authError: string;
  setAuthError: (err: string) => void;
  setAuthToken: (token: string) => void;
  onSuccessReset: () => void;
}

export const AuthPage: React.FC<AuthPageProps> = ({
  authMode,
  setAuthMode,
  authEmail,
  setAuthEmail,
  authPassword,
  setAuthPassword,
  authLoading,
  setAuthLoading,
  authError,
  setAuthError,
  setAuthToken,
  onSuccessReset,
}) => {
  const [isReadOnly, setIsReadOnly] = useState(true);

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
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
      const data = await resp.json().catch(() => ({}));

      if (!resp.ok) {
        setAuthError(data?.detail || "Authentication failed. Please check credentials.");
        return;
      }

      if (authMode === "register") {
        setAuthMode("login");
        setAuthError("Account created successfully! Please log in.");
        return;
      }

      const token = data.token || "";
      const email = data.email || authEmail.trim();
      setAuthToken(token);
      setAuthEmail(email);
      setAuthPassword("");
      onSuccessReset();
    } catch (err: any) {
      setAuthError("Auth service unavailable or network error.");
    } finally {
      setAuthLoading(false);
    }
  };

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-[#030306] flex items-center justify-center font-mono text-zinc-300">
      {/* Background ambient lighting */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-[20%] -left-[10%] w-[650px] h-[650px] rounded-full bg-indigo-600/15 blur-[160px] animate-pulse" style={{ animationDuration: '8s' }} />
        <div className="absolute -bottom-[20%] -right-[10%] w-[650px] h-[650px] rounded-full bg-emerald-600/15 blur-[160px] animate-pulse" style={{ animationDuration: '10s' }} />
      </div>

      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 15 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        className="relative z-10 w-full max-w-md mx-4 bg-[#08080d]/95 border border-zinc-800/90 rounded-2xl p-8 shadow-[0_0_50px_rgba(99,102,241,0.15)] backdrop-blur-2xl"
      >
        <div className="flex items-center gap-4 mb-8">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500/20 to-emerald-500/20 border border-indigo-500/40 flex items-center justify-center text-indigo-400 font-bold text-xl tracking-wider shadow-inner">
            AI
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-wide font-sans">
              AI Engineering Platform
            </h1>
            <p className="text-xs text-zinc-500 uppercase tracking-widest mt-0.5">
              Autonomous Team Workspace
            </p>
          </div>
        </div>

        {/* Mode Switcher */}
        <div className="grid grid-cols-2 gap-2 p-1 bg-zinc-900/80 rounded-xl border border-zinc-800 mb-6">
          {(["login", "register"] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => {
                setAuthMode(mode);
                setAuthError("");
              }}
              className={`py-2.5 text-xs font-semibold rounded-lg transition-all ${
                authMode === mode
                  ? "bg-zinc-800 text-indigo-300 shadow border border-zinc-700/60"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {mode === "login" ? "Log In" : "Sign Up"}
            </button>
          ))}
        </div>

        <form onSubmit={handleAuth} autoComplete="off" className="flex flex-col gap-4">
          <div>
            <label className="block text-xs font-semibold text-zinc-400 mb-1.5 uppercase tracking-wider">
              Email Address
            </label>
            <input
              type="text"
              name="ai_platform_workspace_user_id_v2"
              required
              readOnly={isReadOnly}
              onFocus={() => setIsReadOnly(false)}
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="none"
              spellCheck="false"
              value={authEmail}
              onChange={(e) => setAuthEmail(e.target.value)}
              placeholder="you@domain.com"
              className="w-full bg-zinc-900/60 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-white placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500/60 focus:ring-2 focus:ring-indigo-500/10 transition-all"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-zinc-400 mb-1.5 uppercase tracking-wider">
              Password {authMode === "register" && <span className="text-[10px] text-zinc-500 normal-case">(min 12 chars)</span>}
            </label>
            <input
              type="password"
              name="ai_platform_workspace_user_pass_v2"
              required
              readOnly={isReadOnly}
              onFocus={() => setIsReadOnly(false)}
              autoComplete="new-password"
              value={authPassword}
              onChange={(e) => setAuthPassword(e.target.value)}
              placeholder="••••••••••••"
              className="w-full bg-zinc-900/60 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-white placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500/60 focus:ring-2 focus:ring-indigo-500/10 transition-all"
            />
          </div>

          <AnimatePresence>
            {authError && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className={`text-xs px-3 py-2 rounded-lg border ${
                  authError.includes("created")
                    ? "bg-emerald-950/40 text-emerald-400 border-emerald-800/60"
                    : "bg-red-950/40 text-red-400 border-red-800/60"
                }`}
              >
                {authError}
              </motion.div>
            )}
          </AnimatePresence>

          <button
            type="submit"
            disabled={authLoading || !authEmail.trim() || !authPassword.trim()}
            className="w-full mt-2 bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded-xl text-sm transition-all shadow-lg shadow-indigo-600/20 flex items-center justify-center gap-2"
          >
            {authLoading && (
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            )}
            {authMode === "login" ? "Enter Workspace" : "Create Account"}
          </button>
        </form>
      </motion.div>
    </div>
  );
};
