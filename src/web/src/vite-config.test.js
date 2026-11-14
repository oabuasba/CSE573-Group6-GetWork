// @vitest-environment node
import { expect, test } from "vitest";
import makeConfig, { apiProxyTarget } from "../vite.config.js";

test("the /api dev proxy has a target (defaults and overrides)", () => {
  const proxy = makeConfig({ command: "build" }).server.proxy["/api"];
  expect(proxy.target).toEqual(apiProxyTarget());
  expect(apiProxyTarget({})).toEqual({ protocol: "http:", host: "127.0.0.1", port: 8000 });
  const custom = { GETWORK_API_HOST: "10.0.0.5", GETWORK_API_PORT: "9100" };
  expect(apiProxyTarget(custom)).toEqual({ protocol: "http:", host: "10.0.0.5", port: 9100 });
});
