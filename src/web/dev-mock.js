// Dev-only stand-in for the GetWork API, used while the backend does not exist yet.
// It answers /api/health, /api/examples, /api/match and /api/explain from
// src/fixtures/contract.json so `npm run dev` shows the UI. It is registered only for
// `vite` dev serve (see vite.config.js), never for tests or `vite build`.
// Set GETWORK_REAL_API=1 to turn it off and use the real server through the /api proxy.
import { readFileSync } from "node:fs";

const contract = JSON.parse(readFileSync(new URL("./src/fixtures/contract.json", import.meta.url), "utf8"));

const ROUTES = {
  "GET /api/health": contract.health,
  "GET /api/examples": contract.examples,
  "POST /api/match": contract.match,
  "POST /api/explain": contract.explain,
};

export function mockResponse(method, url) {
  const path = url.split("?")[0];
  const body = ROUTES[`${method} ${path}`];
  if (body !== undefined) return { status: 200, body };
  const message = `No mock for ${method} ${path}.`;
  return { status: 404, body: { error: { code: "not_found", message, hint: null, field: null } } };
}

export function mockApiPlugin() {
  return {
    name: "getwork-dev-mock-api",
    apply: "serve",
    configureServer(server) {
      console.log("[getwork] dev mock API active (sample data, set GETWORK_REAL_API=1 to disable)");
      server.middlewares.use("/api", (req, res) => {
        const { status, body } = mockResponse(req.method, `/api${req.url}`);
        res.statusCode = status;
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify(body));
      });
    },
  };
}
