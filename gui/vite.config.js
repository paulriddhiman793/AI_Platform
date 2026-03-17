import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiUrl = (env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");
  const wsTarget = apiUrl.replace(/^http:/, "ws:").replace(/^https:/, "wss:");

  return {
    plugins: [react()],
    server: {
      proxy: {
        "/ws": {
          target: wsTarget,
          ws: true,
        },
        "/health": apiUrl,
        "/files": apiUrl,
        "/auth": apiUrl,
        "/projects": apiUrl,
        "/project_zip": apiUrl,
        "/worker": apiUrl,
        "/file": apiUrl,
      },
    },
  };
});
