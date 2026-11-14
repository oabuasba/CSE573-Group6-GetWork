import Notice from "./components/Notice.jsx";
import { hrefFor } from "./routes.js";

const NAV_LINKS = [
  { route: "home", label: "Home" },
  { route: "jobs", label: "Find jobs" },
  { route: "candidates", label: "Find candidates" },
  { route: "about", label: "About" },
];

export const DISCLAIMER =
  "GetWork is a CSE 573 course project at Arizona State University (Group 6). It is not an official ASU website.";

export default function Layout({ route, health, children }) {
  const mainClass = route === "chat" ? "container app-main app-main-chat" : "container app-main";
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to main content
      </a>
      <header className="site-header">
        <div className="header-top">
          <div className="container">Course project: CSE 573, Group 6</div>
        </div>
        <div className="header-main">
          <div className="container">
            <a className="wordmark" href={hrefFor("home")}>
              GetWork
            </a>
            <nav className="site-nav" aria-label="Main">
              {NAV_LINKS.map(({ route: target, label }) => (
                <a key={target} href={hrefFor(target)} aria-current={target === route ? "page" : undefined}>
                  {label}
                </a>
              ))}
            </nav>
          </div>
        </div>
      </header>
      {health?.mode === "demo" && (
        <Notice kind="warning" label="Demo mode." className="banner-demo" role="status">
          Results come from a small built-in sample, not the full data set.
        </Notice>
      )}
      <main id="main" tabIndex={-1} className={mainClass}>
        {children}
      </main>
      <footer className="site-footer">
        <div className="container">
          <p>{DISCLAIMER}</p>
          <p>
            <a href={hrefFor("about")}>Contact the team</a>
          </p>
        </div>
      </footer>
    </div>
  );
}
