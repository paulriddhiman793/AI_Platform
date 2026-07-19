import { createRoot } from "react-dom/client";
import { StrictMode } from "react";
import { AnimatePresence } from "framer-motion";
import App from "./App";
import "./index.css";
import { AGENTS, TAG_STYLES } from "./data/agents";

// Initialize global variables required by existing component references
if (typeof window !== "undefined") {
  window.AGENTS = AGENTS;
  window.TAG_STYLES = TAG_STYLES as any;
}

const rootElement = document.getElementById("root");
if (rootElement) {
  createRoot(rootElement).render(
    <StrictMode>
      <AnimatePresence mode="wait">
        <App />
      </AnimatePresence>
    </StrictMode>
  );
}
