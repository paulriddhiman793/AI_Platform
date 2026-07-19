"use client";

import React, { useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { API_URL } from "@/utils/config";

interface FilesPanelModalProps {
  isOpen: boolean;
  onClose: () => void;
  authToken: string;
  projectRoot: string | null;
  filesIndex: any[];
  setFilesIndex: (list: any[]) => void;
  filesLoading: boolean;
  setFilesLoading: (l: boolean) => void;
  filesError: string;
  setFilesError: (e: string) => void;
  fileSearch: string;
  setFileSearch: (s: string) => void;
  selectedFile: string | null;
  setSelectedFile: (f: string | null) => void;
  fileContent: string;
  setFileContent: (c: string) => void;
  fileBinary: boolean;
  setFileBinary: (b: boolean) => void;
}

export const FilesPanelModal: React.FC<FilesPanelModalProps> = ({
  isOpen,
  onClose,
  authToken,
  projectRoot,
  filesIndex,
  setFilesIndex,
  filesLoading,
  setFilesLoading,
  filesError,
  setFilesError,
  fileSearch,
  setFileSearch,
  selectedFile,
  setSelectedFile,
  fileContent,
  setFileContent,
  fileBinary,
  setFileBinary,
}) => {
  const fetchIndex = useCallback(async () => {
    if (!authToken || !projectRoot) return;
    setFilesLoading(true);
    setFilesError("");
    try {
      const res = await fetch(`${API_URL}/files`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auth_token: authToken, project_root: projectRoot }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && Array.isArray(data.files)) {
        setFilesIndex(data.files);
      } else {
        setFilesError(data?.detail || "Could not list project files.");
      }
    } catch {
      setFilesError("Network error accessing files.");
    } finally {
      setFilesLoading(false);
    }
  }, [authToken, projectRoot, setFilesIndex, setFilesLoading, setFilesError]);

  const handleSelectFile = useCallback(
    async (filePath: string) => {
      setSelectedFile(filePath);
      setFileBinary(false);
      if (!authToken || !projectRoot || !filePath) return;
      try {
        const res = await fetch(`${API_URL}/file/read`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ auth_token: authToken, path: filePath, project_root: projectRoot }),
        });
        const data = await res.json().catch(() => ({}));
        if (res.ok) {
          setFileContent(data.content || "");
          setFileBinary(!!data.is_binary);
        } else {
          setFileContent(`[Error loading file: ${data?.detail || "unknown"}]`);
        }
      } catch {
        setFileContent("[Network error loading file content]");
      }
    },
    [authToken, projectRoot, setSelectedFile, setFileContent, setFileBinary]
  );

  useEffect(() => {
    if (isOpen && projectRoot) {
      fetchIndex();
    }
  }, [isOpen, projectRoot, fetchIndex]);

  useEffect(() => {
    if (isOpen && projectRoot && selectedFile) {
      handleSelectFile(selectedFile);
    }
  }, [isOpen, projectRoot, selectedFile, handleSelectFile]);

  useEffect(() => {
    if (isOpen && !selectedFile && filesIndex && filesIndex.length > 0) {
      const firstPath = filesIndex[0].path || filesIndex[0].name;
      if (firstPath) {
        handleSelectFile(firstPath);
      }
    }
  }, [isOpen, selectedFile, filesIndex, handleSelectFile]);

  if (!isOpen) return null;

  const filteredFiles = (filesIndex || []).filter(
    (f: any) =>
      !fileSearch.trim() ||
      String(f?.path || f?.name || "")
        .toLowerCase()
        .includes(fileSearch.trim().toLowerCase())
  );

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
          className="w-full max-w-5xl h-[80vh] bg-[#0b0b10] border border-zinc-800 rounded-2xl p-6 shadow-2xl flex flex-col overflow-hidden"
        >
          {/* Header */}
          <div className="flex items-center justify-between pb-4 border-b border-zinc-800/80 mb-4 flex-shrink-0">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
                📁
              </div>
              <div>
                <h2 className="text-base font-bold text-white font-sans">Project Files Explorer</h2>
                <p className="text-xs text-zinc-400">
                  {projectRoot ? `Root: ${projectRoot}` : "No active project selected"}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={fetchIndex}
                disabled={filesLoading || !projectRoot}
                className="px-3 py-1.5 text-xs font-semibold rounded-xl bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 text-zinc-300 transition-colors flex items-center gap-1.5"
              >
                <span>🔄 Refresh</span>
              </button>
              <form action={`${API_URL}/project_zip`} method="POST" target="_blank">
                <input type="hidden" name="auth_token" value={authToken} />
                <input type="hidden" name="project_root" value={projectRoot || ""} />
                <button
                  type="submit"
                  disabled={!projectRoot}
                  className="px-3 py-1.5 text-xs font-semibold rounded-xl bg-indigo-600/20 hover:bg-indigo-600/30 border border-indigo-500/30 text-indigo-300 disabled:opacity-40 transition-colors"
                >
                  📥 Download ZIP
                </button>
              </form>
              <button onClick={onClose} className="text-zinc-500 hover:text-white transition-colors p-1.5 ml-2">
                ✕
              </button>
            </div>
          </div>

          {/* Body: Left sidebar (file tree) & Right view (content) */}
          <div className="flex-1 flex gap-4 overflow-hidden">
            {/* Left Column: File search + List */}
            <div className="w-72 flex flex-col gap-2 bg-zinc-950/60 border border-zinc-800/80 rounded-xl p-3 flex-shrink-0 overflow-hidden">
              <input
                type="text"
                value={fileSearch}
                onChange={(e) => setFileSearch(e.target.value)}
                placeholder="Search files..."
                className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs text-white placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500/60"
              />
              <div className="flex-1 overflow-y-auto space-y-1 pr-1">
                {filesLoading && (
                  <div className="text-xs text-zinc-500 py-4 text-center">Loading files...</div>
                )}
                {filesError && (
                  <div className="text-xs text-red-400 py-2 text-center">{filesError}</div>
                )}
                {!filesLoading && !filesError && filteredFiles.length === 0 && (
                  <div className="text-xs text-zinc-600 py-6 text-center">No files found.</div>
                )}
                {filteredFiles.map((f: any, idx: number) => {
                  const p = f?.path || f?.name || `file_${idx}`;
                  const isSelected = selectedFile === p;
                  return (
                    <button
                      key={idx}
                      onClick={() => handleSelectFile(p)}
                      className={`w-full text-left px-2.5 py-2 rounded-lg text-xs truncate transition-all flex items-center justify-between ${
                        isSelected
                          ? "bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 font-semibold"
                          : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"
                      }`}
                      title={p}
                    >
                      <span className="truncate">{f?.name || p}</span>
                      {f?.size !== undefined && (
                        <span className="text-[10px] text-zinc-600 ml-2">
                          {(f.size / 1024).toFixed(1)} KB
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Right Column: File Content View */}
            <div className="flex-1 bg-zinc-950/80 border border-zinc-800/80 rounded-xl p-4 flex flex-col overflow-hidden relative">
              {selectedFile ? (
                <>
                  <div className="flex items-center justify-between pb-3 border-b border-zinc-900 mb-3 text-xs text-zinc-400 flex-shrink-0">
                    <span className="font-semibold text-zinc-200 truncate pr-4">{selectedFile}</span>
                    <div className="flex items-center gap-2">
                      <a
                        href={`${API_URL}/workspace/files/${selectedFile}?project_root=${encodeURIComponent(projectRoot || "")}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-2.5 py-1 rounded bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold no-underline flex items-center gap-1 shadow transition-all"
                      >
                        📥 Download / Open in New Tab
                      </a>
                      {fileBinary && (
                        <span className="px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 text-[10px]">
                          Binary File
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex-1 overflow-y-auto font-mono text-xs text-zinc-300 select-text leading-relaxed pr-2">
                    {fileBinary ? (
                      /\.(png|jpg|jpeg|gif|webp)$/i.test(selectedFile || "") ? (
                        <div className="py-4 flex flex-col items-center justify-center">
                          <img
                            src={fileContent && !fileContent.startsWith("http") ? `data:image/png;base64,${fileContent}` : `${API_URL}/workspace/files/${selectedFile}?project_root=${encodeURIComponent(projectRoot || "")}`}
                            alt={selectedFile}
                            className="max-h-[65vh] object-contain rounded-lg border border-zinc-800 shadow-md"
                          />
                        </div>
                      ) : (
                        <div className="text-zinc-500 italic py-10 text-center flex flex-col items-center gap-3">
                          <span>This is a binary file. Inline text preview not available.</span>
                          <a
                            href={`${API_URL}/workspace/files/${selectedFile}?project_root=${encodeURIComponent(projectRoot || "")}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-indigo-400 hover:underline not-italic font-semibold"
                          >
                            Click here to download
                          </a>
                        </div>
                      )
                    ) : (
                      <pre className="whitespace-pre-wrap break-words">{fileContent || "Empty file content..."}</pre>
                    )}
                  </div>
                </>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-zinc-600 text-xs gap-2">
                  <div className="text-2xl opacity-40">📄</div>
                  <div>Select a file from the explorer to preview content</div>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};
