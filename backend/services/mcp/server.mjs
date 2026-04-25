import { randomUUID } from "node:crypto";
import { execSync } from "node:child_process";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";
import { requireBearerAuth } from "@modelcontextprotocol/sdk/server/auth/middleware/bearerAuth.js";
import { mcpAuthRouter, createOAuthMetadata, mcpAuthMetadataRouter, getOAuthProtectedResourceMetadataUrl } from "@modelcontextprotocol/sdk/server/auth/router.js";
import { resourceUrlFromServerUrl } from "@modelcontextprotocol/sdk/shared/auth-utils.js";
import express from "express";
import { z } from "zod";

const PORT = 3101;
const BASE_URL = new URL("https://coin.tantaliden.com/mcp");

// === In-memory OAuth provider ===
class ClientsStore {
  constructor() { this.clients = new Map(); }
  async getClient(id) { return this.clients.get(id); }
  async registerClient(meta) { this.clients.set(meta.client_id, meta); return meta; }
}

class AuthProvider {
  constructor() {
    this.clientsStore = new ClientsStore();
    this.codes = new Map();
    this.tokens = new Map();
  }
  async authorize(client, params, res) {
    const code = randomUUID();
    const sp = new URLSearchParams({ code });
    if (params.state) sp.set("state", params.state);
    this.codes.set(code, { client, params });
    if (!client.redirect_uris.includes(params.redirectUri)) {
      res.status(400).json({ error: "invalid redirect_uri" }); return;
    }
    const target = new URL(params.redirectUri);
    target.search = sp.toString();
    res.redirect(target.toString());
  }
  async challengeForAuthorizationCode(_client, code) {
    const d = this.codes.get(code);
    if (!d) throw new Error("Invalid code");
    return d.params.codeChallenge;
  }
  async exchangeAuthorizationCode(client, code, _verifier) {
    const d = this.codes.get(code);
    if (!d) throw new Error("Invalid code");
    if (d.client.client_id !== client.client_id) throw new Error("Client mismatch");
    this.codes.delete(code);
    const token = randomUUID();
    this.tokens.set(token, {
      token, clientId: client.client_id,
      scopes: d.params.scopes || [], expiresAt: Date.now() + 86400000,
      resource: d.params.resource, type: "access"
    });
    return { access_token: token, token_type: "bearer", expires_in: 86400, scope: (d.params.scopes || []).join(" ") };
  }
  async exchangeRefreshToken() { throw new Error("Not implemented"); }
  async verifyAccessToken(token) {
    const t = this.tokens.get(token);
    if (!t || t.expiresAt < Date.now()) throw new Error("Invalid token");
    return { token, clientId: t.clientId, scopes: t.scopes, expiresAt: Math.floor(t.expiresAt / 1000) };
  }
}

const provider = new AuthProvider();

// === MCP Server with tools ===
function getServer() {
  const server = new McpServer({ name: "analyser-remote", version: "1.0.0" });
  server.tool("exec", "Execute a shell command", {
    command: z.string().describe("Shell command to execute"),
    description: z.string().optional().describe("What this command does"),
  }, async ({ command }) => {
    try {
      const out = execSync(command, { timeout: 120000, maxBuffer: 10*1024*1024, encoding: "utf-8", cwd: "/home/claude" });
      return { content: [{ type: "text", text: out || "(no output)" }] };
    } catch (e) { return { content: [{ type: "text", text: `Error: ${e.stderr || e.message}` }] }; }
  });
  server.tool("sudo-exec", "Execute a shell command with sudo", {
    command: z.string().describe("Shell command to execute with sudo"),
    description: z.string().optional().describe("What this command does"),
  }, async ({ command }) => {
    try {
      const out = execSync(`sudo ${command}`, { timeout: 120000, maxBuffer: 10*1024*1024, encoding: "utf-8", cwd: "/home/claude" });
      return { content: [{ type: "text", text: out || "(no output)" }] };
    } catch (e) { return { content: [{ type: "text", text: `Error: ${e.stderr || e.message}` }] }; }
  });
  return server;
}

// === Express app ===
const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// OAuth metadata and auth routes
const oauthMetadata = createOAuthMetadata({
  provider,
  issuerUrl: BASE_URL,
  scopesSupported: ["mcp:tools"],
});

// Auth routes (authorize, token, register endpoints)
app.use(mcpAuthRouter({ provider, issuerUrl: BASE_URL, scopesSupported: ["mcp:tools"] }));

// Protected resource metadata
app.use(mcpAuthMetadataRouter({
  oauthMetadata,
  resourceServerUrl: BASE_URL,
  scopesSupported: ["mcp:tools"],
  resourceName: "Analyser MCP Server",
}));

// Token verifier
const tokenVerifier = {
  verifyAccessToken: async (token) => provider.verifyAccessToken(token),
};

const authMiddleware = requireBearerAuth({
  verifier: tokenVerifier,
  requiredScopes: [],
  resourceMetadataUrl: getOAuthProtectedResourceMetadataUrl(BASE_URL),
});

// MCP transport handling
const transports = {};

app.post("/mcp", authMiddleware, async (req, res) => {
  const sessionId = req.headers["mcp-session-id"];
  try {
    let transport;
    if (sessionId && transports[sessionId]) {
      transport = transports[sessionId];
    } else if (!sessionId && isInitializeRequest(req.body)) {
      transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        onsessioninitialized: (sid) => { transports[sid] = transport; },
      });
      transport.onclose = () => { const s = transport.sessionId; if (s) delete transports[s]; };
      await getServer().connect(transport);
      await transport.handleRequest(req, res, req.body);
      return;
    } else {
      res.status(400).json({ error: "Bad request" }); return;
    }
    await transport.handleRequest(req, res, req.body);
  } catch (e) {
    if (!res.headersSent) res.status(500).json({ error: e.message });
  }
});

app.get("/mcp", authMiddleware, async (req, res) => {
  const sid = req.headers["mcp-session-id"];
  if (!sid || !transports[sid]) { res.status(400).send("Invalid session"); return; }
  await transports[sid].handleRequest(req, res);
});

app.delete("/mcp", authMiddleware, async (req, res) => {
  const sid = req.headers["mcp-session-id"];
  if (!sid || !transports[sid]) { res.status(400).send("Invalid session"); return; }
  await transports[sid].handleRequest(req, res);
});

app.get("/health", (_req, res) => res.json({ status: "ok" }));

app.listen(PORT, "127.0.0.1", () => {
  console.log(`MCP Remote Server on port ${PORT}`);
});
