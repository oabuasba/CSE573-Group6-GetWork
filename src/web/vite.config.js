import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { mockApiPlugin } from "./dev-mock.js";

export function apiProxyTarget(env = process.env) {
  return {
    protocol: "http:",
    host: env.GETWORK_API_HOST || "127.0.0.1",
    port: Number(env.GETWORK_API_PORT || 8000),
  };
}

export default defineConfig(({ command }) => {
  const useMock = command === "serve" && !process.env.VITEST && process.env.GETWORK_REAL_API !== "1";
  return {
    plugins: [react(), ...(useMock ? [mockApiPlugin()] : [])],
    server: { proxy: { "/api": { target: apiProxyTarget() } } },
    test: {
      environment: "jsdom",
      setupFiles: ["./src/test/setup.js"],
      globals: true,
    },
  };
});
