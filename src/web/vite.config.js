import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { mockApiPlugin } from "./dev-mock.js";

export default defineConfig(({ command }) => {
  const useMock = command === "serve" && !process.env.VITEST && process.env.GETWORK_REAL_API !== "1";
  return {
    plugins: [react(), ...(useMock ? [mockApiPlugin()] : [])],
    server: { proxy: { "/api": "http://127.0.0.1:8000" } },
    test: {
      environment: "jsdom",
      setupFiles: ["./src/test/setup.js"],
      globals: true,
    },
  };
});
