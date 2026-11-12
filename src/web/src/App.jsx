import { useCallback, useEffect, useState } from "react";
import Layout from "./Layout.jsx";
import Notice from "./components/Notice.jsx";
import { describeError, getHealth } from "./api.js";
import AboutPage from "./pages/AboutPage.jsx";
import ChatPage from "./pages/ChatPage.jsx";
import FindPage from "./pages/FindPage.jsx";
import { chatIdFromHash, DIRECTION_BY_ROUTE, parseRoute } from "./routes.js";

const ROUTE_TITLES = {
  jobs: "Find jobs", candidates: "Find candidates", about: "About GetWork",
  chat: "Chat about this suggestion",
};

function useHash() {
  const [hash, setHash] = useState(() => window.location.hash);
  useEffect(() => {
    const onChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return hash;
}

function useHealth(fetchImpl) {
  const [state, setState] = useState({ status: "loading", health: null, error: null });
  const load = useCallback(() => {
    setState({ status: "loading", health: null, error: null });
    getHealth(fetchImpl).then(
      (health) => setState({ status: "ready", health, error: null }),
      (error) => setState({ status: "error", health: null, error }),
    );
  }, [fetchImpl]);
  useEffect(load, [load]);
  return [state, load];
}

function UnreachableAlert({ error, onRetry }) {
  const { title, body, hint } = describeError(error);
  return (
    <Notice kind="error" role="alert">
      <h2>{title}</h2>
      <p>{body}</p>
      {hint && <p>{hint}</p>}
      <p>
        Run <code>uv run webapp serve</code>, then try again.
      </p>
      <button type="button" className="btn btn-primary" onClick={onRetry}>
        Retry
      </button>
    </Notice>
  );
}

const defaultFetch = (...args) => fetch(...args);

export default function App({ fetchImpl = defaultFetch }) {
  const hash = useHash();
  const route = parseRoute(hash);
  const chatId = chatIdFromHash(hash);
  const [{ status, health, error }, retry] = useHealth(fetchImpl);

  return (
    <Layout route={route} health={health}>
      {status === "ready" && route === "about" && <AboutPage health={health} />}
      {status === "ready" && route === "chat" && (
        <ChatPage key={chatId} matchId={chatId} health={health} fetchImpl={fetchImpl} />
      )}
      {status === "ready" && (route === "jobs" || route === "candidates") && (
        <FindPage key={route} direction={DIRECTION_BY_ROUTE[route]} health={health} fetchImpl={fetchImpl} />
      )}
      {status !== "ready" && (
        <>
          <h1>{ROUTE_TITLES[route]}</h1>
          {status === "loading" && <p role="status">Loading…</p>}
          {status === "error" && <UnreachableAlert error={error} onRetry={retry} />}
        </>
      )}
    </Layout>
  );
}
