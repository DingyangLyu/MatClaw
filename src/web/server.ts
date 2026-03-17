/**
 * MatClaw Monitoring Dashboard
 * Lightweight HTTP + WebSocket server for real-time agent monitoring.
 */
import http from 'http';
import fs from 'fs';
import path from 'path';
import { WebSocketServer, WebSocket } from 'ws';

import { agentEvents, type AgentEvent } from './events.js';
import { parseTranscript, formatLogEntries } from './transcript-parser.js';
import { logger } from '../logger.js';
import { DATA_DIR, GROUPS_DIR } from '../config.js';
import { getRecentMessages, type WebChatThread } from '../db.js';
import { handleWebChatMessage, setWebBroadcast } from '../channels/web.js';
import type { ChannelOpts } from '../channels/registry.js';

const PORT = parseInt(process.env.DASHBOARD_PORT || '3210', 10);

/** Recursively search for a file by basename within a directory (max 3 levels deep). */
function findFileRecursive(
  dir: string,
  basename: string,
  root: string,
  depth = 0,
): string | null {
  if (depth > 3) return null;
  try {
    for (const entry of fs.readdirSync(dir)) {
      const full = path.join(dir, entry);
      if (entry === basename && fs.statSync(full).isFile()) return full;
      if (
        depth < 3 &&
        fs.statSync(full).isDirectory() &&
        !entry.startsWith('.')
      ) {
        const found = findFileRecursive(full, basename, root, depth + 1);
        if (found) return found;
      }
    }
  } catch {
    /* permission or read error, skip */
  }
  return null;
} // v4: compact md, no breaks, hide br

let wss: WebSocketServer;

interface WebChatController {
  listSessions(): WebChatThread[];
  getActiveThreadId(): string;
  createSession(): WebChatThread;
  switchSession(threadId: string): WebChatThread | undefined;
  renameSession(threadId: string, title: string): WebChatThread | undefined;
  deleteSession(threadId: string):
    | {
        deletedThreadId: string;
        activeThreadId: string;
        activeSession: WebChatThread;
      }
    | undefined;
}

interface ChatArtifact {
  path: string;
  name: string;
  kind: 'image' | 'file';
  source: 'conversation' | 'workspace';
  size?: number;
  modified?: string;
}

function detectArtifactKind(filePath: string): 'image' | 'file' {
  return /\.(png|jpe?g|gif|webp|svg)$/i.test(filePath) ? 'image' : 'file';
}

function extractArtifactRefs(
  text: string,
): Array<{ path: string; kind: 'image' | 'file' }> {
  const seen = new Set<string>();
  const results: Array<{ path: string; kind: 'image' | 'file' }> = [];
  const add = (filePath: string, kind?: 'image' | 'file') => {
    const clean = String(filePath || '').trim();
    if (!clean || seen.has(clean)) return;
    seen.add(clean);
    results.push({ path: clean, kind: kind || detectArtifactKind(clean) });
  };

  for (const match of text.matchAll(/!\[[^\]]*\]\(([^)]+)\)/g)) {
    add(match[1], 'image');
  }
  for (const match of text.matchAll(/\[file:([^\]]+)\]/g)) {
    add(match[1], 'file');
  }
  for (const match of text.matchAll(/\[Attached (image|file): ([^\]]+)\]/g)) {
    add(match[2], match[1] === 'image' ? 'image' : 'file');
  }

  return results;
}

function listWorkspaceArtifacts(groupFolder: string): ChatArtifact[] {
  const root = path.join(GROUPS_DIR, groupFolder);
  if (!fs.existsSync(root)) return [];

  const allowedExt = new Set([
    '.png',
    '.jpg',
    '.jpeg',
    '.gif',
    '.webp',
    '.svg',
    '.pdf',
    '.csv',
    '.tsv',
    '.xlsx',
    '.xls',
    '.json',
    '.md',
    '.txt',
    '.html',
    '.zip',
  ]);
  const ignoredDirs = new Set(['logs', '.claude', 'conversations']);
  const items: ChatArtifact[] = [];

  const walk = (dir: string, depth: number) => {
    if (depth > 4 || items.length >= 120) return;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.name.startsWith('.') && entry.name !== '.well-known') continue;
      if (entry.isDirectory()) {
        if (ignoredDirs.has(entry.name)) continue;
        walk(path.join(dir, entry.name), depth + 1);
        continue;
      }
      const ext = path.extname(entry.name).toLowerCase();
      if (!allowedExt.has(ext)) continue;
      const full = path.join(dir, entry.name);
      const stat = fs.statSync(full);
      items.push({
        path: path.relative(root, full),
        name: entry.name,
        kind: detectArtifactKind(entry.name),
        source: 'workspace',
        size: stat.size,
        modified: stat.mtime.toISOString(),
      });
    }
  };

  walk(root, 0);
  return items.sort((a, b) =>
    (b.modified || '').localeCompare(a.modified || ''),
  );
}

function getChatArtifacts(threadId?: string): {
  conversation: ChatArtifact[];
  workspace: ChatArtifact[];
} {
  const conversation = new Map<string, ChatArtifact>();
  const messages = getRecentMessages('web:chat', 200, threadId);
  for (const msg of messages) {
    for (const artifact of extractArtifactRefs(msg.content || '')) {
      if (!conversation.has(artifact.path)) {
        conversation.set(artifact.path, {
          path: artifact.path,
          name: path.basename(artifact.path),
          kind: artifact.kind,
          source: 'conversation',
        });
      }
    }
  }
  return {
    conversation: [...conversation.values()],
    workspace: listWorkspaceArtifacts('web_chat'),
  };
}

function broadcast(event: AgentEvent) {
  const msg = JSON.stringify(event);
  for (const client of wss.clients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(msg);
    }
  }
}

/** Broadcast any JSON to all WS clients (used for chat:message events) */
function broadcastRaw(data: unknown): void {
  const msg = JSON.stringify(data);
  for (const client of wss.clients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(msg);
    }
  }
}

/** List available groups with their log files and transcript count */
function getGroupsInfo(): {
  folder: string;
  logCount: number;
  transcriptCount: number;
  lastLog: string | null;
}[] {
  const results: {
    folder: string;
    logCount: number;
    transcriptCount: number;
    lastLog: string | null;
  }[] = [];
  if (!fs.existsSync(GROUPS_DIR)) return results;

  for (const entry of fs.readdirSync(GROUPS_DIR)) {
    const logsDir = path.join(GROUPS_DIR, entry, 'logs');
    if (!fs.existsSync(logsDir)) continue;
    const logs = fs
      .readdirSync(logsDir)
      .filter(
        (f) =>
          f.startsWith('container-') &&
          f.endsWith('.log') &&
          f !== 'container-live.log',
      )
      .sort()
      .reverse();

    // Count transcripts
    const transcriptDir = path.join(
      DATA_DIR,
      'sessions',
      entry,
      '.claude',
      'projects',
      '-workspace-group',
    );
    let transcriptCount = 0;
    if (fs.existsSync(transcriptDir)) {
      transcriptCount = fs
        .readdirSync(transcriptDir)
        .filter((f) => f.endsWith('.jsonl')).length;
    }

    results.push({
      folder: entry,
      logCount: logs.length,
      transcriptCount,
      lastLog: logs[0] || null,
    });
  }
  return results;
}

/** Read a specific log file */
function readLogFile(groupFolder: string, filename: string): string | null {
  if (filename.includes('..') || groupFolder.includes('..')) return null;
  const logPath = path.join(GROUPS_DIR, groupFolder, 'logs', filename);
  if (!fs.existsSync(logPath)) return null;
  return fs.readFileSync(logPath, 'utf-8');
}

/** List log files for a group */
function listLogs(groupFolder: string): string[] {
  if (groupFolder.includes('..')) return [];
  const logsDir = path.join(GROUPS_DIR, groupFolder, 'logs');
  if (!fs.existsSync(logsDir)) return [];
  return fs
    .readdirSync(logsDir)
    .filter((f) => f.startsWith('container-') && f.endsWith('.log'))
    .sort()
    .reverse();
}

/** List transcript files for a group */
function listTranscripts(
  groupFolder: string,
): { file: string; size: number; modified: string }[] {
  if (groupFolder.includes('..')) return [];
  const dir = path.join(
    DATA_DIR,
    'sessions',
    groupFolder,
    '.claude',
    'projects',
    '-workspace-group',
  );
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir)
    .filter((f) => f.endsWith('.jsonl'))
    .map((f) => {
      const stat = fs.statSync(path.join(dir, f));
      return {
        file: f,
        size: stat.size,
        modified: stat.mtime.toISOString(),
      };
    })
    .sort((a, b) => b.modified.localeCompare(a.modified));
}

/** Parse a transcript file and return formatted log */
function getTranscriptLog(
  groupFolder: string,
  filename: string,
): string | null {
  if (filename.includes('..') || groupFolder.includes('..')) return null;
  const filePath = path.join(
    DATA_DIR,
    'sessions',
    groupFolder,
    '.claude',
    'projects',
    '-workspace-group',
    filename,
  );
  if (!fs.existsSync(filePath)) return null;
  const content = fs.readFileSync(filePath, 'utf-8');
  const entries = parseTranscript(content);
  return formatLogEntries(entries);
}

/** Parse a transcript file and return structured JSON entries */
function getTranscriptEntries(
  groupFolder: string,
  filename: string,
): ReturnType<typeof parseTranscript> | null {
  if (filename.includes('..') || groupFolder.includes('..')) return null;
  const filePath = path.join(
    DATA_DIR,
    'sessions',
    groupFolder,
    '.claude',
    'projects',
    '-workspace-group',
    filename,
  );
  if (!fs.existsSync(filePath)) return null;
  const content = fs.readFileSync(filePath, 'utf-8');
  return parseTranscript(content);
}

export function startDashboard(
  channelOpts?: ChannelOpts,
  webChatController?: WebChatController,
) {
  const dashboardPath = path.join(import.meta.dirname, 'dashboard.html');

  const server = http.createServer((req, res) => {
    const url = new URL(req.url || '/', `http://${req.headers.host}`);

    if (url.pathname === '/' || url.pathname === '/dashboard') {
      // Re-read on each request so HTML changes take effect without restart
      const html = fs.readFileSync(dashboardPath, 'utf-8');
      res.writeHead(200, {
        'Content-Type': 'text/html; charset=utf-8',
        'Cache-Control': 'no-cache',
      });
      res.end(html);
      return;
    }

    // ── API: Groups ──
    if (url.pathname === '/api/groups') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(getGroupsInfo()));
      return;
    }

    // ── API: Container logs list ──
    if (url.pathname === '/api/logs') {
      const group = url.searchParams.get('group');
      if (!group) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Missing group parameter' }));
        return;
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(listLogs(group)));
      return;
    }

    // ── API: Container log content ──
    if (url.pathname === '/api/log') {
      const group = url.searchParams.get('group');
      const file = url.searchParams.get('file');
      if (!group || !file) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Missing group or file parameter' }));
        return;
      }
      const content = readLogFile(group, file);
      if (content === null) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Log not found' }));
        return;
      }
      res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end(content);
      return;
    }

    // ── API: Transcripts list ──
    if (url.pathname === '/api/transcripts') {
      const group = url.searchParams.get('group');
      if (!group) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Missing group parameter' }));
        return;
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(listTranscripts(group)));
      return;
    }

    // ── API: Transcript parsed as text ──
    if (url.pathname === '/api/transcript') {
      const group = url.searchParams.get('group');
      const file = url.searchParams.get('file');
      if (!group || !file) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Missing group or file parameter' }));
        return;
      }
      const content = getTranscriptLog(group, file);
      if (content === null) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Transcript not found' }));
        return;
      }
      res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end(content);
      return;
    }

    // ── API: Transcript parsed as JSON (structured) ──
    if (url.pathname === '/api/transcript/json') {
      const group = url.searchParams.get('group');
      const file = url.searchParams.get('file');
      if (!group || !file) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Missing group or file parameter' }));
        return;
      }
      const entries = getTranscriptEntries(group, file);
      if (entries === null) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Transcript not found' }));
        return;
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(entries));
      return;
    }

    // ── API: Serve files from group workspace ──
    // Supports: /api/file?group=feishu&path=band_structure.png
    // Container path /workspace/group/X maps to groups/{folder}/X on host
    if (url.pathname === '/api/file') {
      const group = url.searchParams.get('group');
      const filePath = url.searchParams.get('path');
      if (!group || !filePath) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Missing group or path parameter' }));
        return;
      }
      // Security: reject path traversal
      if (group.includes('..') || filePath.includes('..')) {
        res.writeHead(403);
        res.end('Forbidden');
        return;
      }
      // Strip container prefix if present
      const cleanPath = filePath
        .replace(/^\/workspace\/group\//, '')
        .replace(/^\/home\/node\//, '');
      const resolved = path.join(GROUPS_DIR, group, cleanPath);
      // Ensure resolved path stays inside GROUPS_DIR
      if (!resolved.startsWith(path.resolve(GROUPS_DIR))) {
        res.writeHead(403);
        res.end('Forbidden');
        return;
      }
      if (!fs.existsSync(resolved)) {
        // Fallback: agent may output wrong relative path (e.g. dir_combined/file.png
        // when actual path is dir/file.png). Search by filename in group directory.
        const basename = path.basename(cleanPath);
        const groupDir = path.join(GROUPS_DIR, group);
        const fallback = findFileRecursive(groupDir, basename, groupDir);
        if (fallback && fallback.startsWith(path.resolve(GROUPS_DIR))) {
          const ext2 = path.extname(fallback).toLowerCase();
          const mimeTypes2: Record<string, string> = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.webp': 'image/webp',
            '.pdf': 'application/pdf',
            '.csv': 'text/csv',
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.html': 'text/html',
          };
          const ct = mimeTypes2[ext2] || 'application/octet-stream';
          const st = fs.statSync(fallback);
          res.writeHead(200, {
            'Content-Type': ct,
            'Content-Length': st.size,
            'Cache-Control': 'public, max-age=300',
          });
          fs.createReadStream(fallback).pipe(res);
          return;
        }
        res.writeHead(404);
        res.end('File not found');
        return;
      }
      const ext = path.extname(resolved).toLowerCase();
      const mimeTypes: Record<string, string> = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.webp': 'image/webp',
        '.pdf': 'application/pdf',
        '.csv': 'text/csv',
        '.txt': 'text/plain',
        '.json': 'application/json',
        '.html': 'text/html',
      };
      const contentType = mimeTypes[ext] || 'application/octet-stream';
      const stat = fs.statSync(resolved);
      res.writeHead(200, {
        'Content-Type': contentType,
        'Content-Length': stat.size,
        'Cache-Control': 'public, max-age=300',
      });
      fs.createReadStream(resolved).pipe(res);
      return;
    }

    // ── API: Chat history ──
    if (url.pathname === '/api/chat/sessions') {
      if (!webChatController) {
        res.writeHead(503, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Web chat controller unavailable' }));
        return;
      }

      if (req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(
          JSON.stringify({
            sessions: webChatController.listSessions(),
            activeThreadId: webChatController.getActiveThreadId(),
          }),
        );
        return;
      }

      if (req.method === 'POST') {
        const thread = webChatController.createSession();
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(
          JSON.stringify({
            session: thread,
            activeThreadId: thread.id,
          }),
        );
        return;
      }
    }

    if (url.pathname === '/api/chat/switch' && req.method === 'POST') {
      if (!webChatController) {
        res.writeHead(503, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Web chat controller unavailable' }));
        return;
      }

      let body = '';
      req.on('data', (chunk) => {
        body += chunk;
      });
      req.on('end', () => {
        try {
          const { threadId } = JSON.parse(body);
          if (!threadId || typeof threadId !== 'string') {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Missing threadId field' }));
            return;
          }
          const thread = webChatController.switchSession(threadId);
          if (!thread) {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Session not found' }));
            return;
          }
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(
            JSON.stringify({
              session: thread,
              activeThreadId: thread.id,
            }),
          );
        } catch {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Invalid JSON' }));
        }
      });
      return;
    }

    if (url.pathname === '/api/chat/rename' && req.method === 'POST') {
      if (!webChatController) {
        res.writeHead(503, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Web chat controller unavailable' }));
        return;
      }

      let body = '';
      req.on('data', (chunk) => {
        body += chunk;
      });
      req.on('end', () => {
        try {
          const { threadId, title } = JSON.parse(body);
          if (
            !threadId ||
            typeof threadId !== 'string' ||
            !title ||
            typeof title !== 'string'
          ) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(
              JSON.stringify({ error: 'Missing threadId or title field' }),
            );
            return;
          }
          const thread = webChatController.renameSession(threadId, title);
          if (!thread) {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Session not found' }));
            return;
          }
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ session: thread }));
        } catch {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Invalid JSON' }));
        }
      });
      return;
    }

    if (url.pathname === '/api/chat/delete' && req.method === 'POST') {
      if (!webChatController) {
        res.writeHead(503, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Web chat controller unavailable' }));
        return;
      }

      let body = '';
      req.on('data', (chunk) => {
        body += chunk;
      });
      req.on('end', () => {
        try {
          const { threadId } = JSON.parse(body);
          if (!threadId || typeof threadId !== 'string') {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Missing threadId field' }));
            return;
          }
          const result = webChatController.deleteSession(threadId);
          if (!result) {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Session not found' }));
            return;
          }
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(result));
        } catch {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Invalid JSON' }));
        }
      });
      return;
    }

    if (url.pathname === '/api/chat/history') {
      const threadId =
        url.searchParams.get('threadId') ||
        webChatController?.getActiveThreadId() ||
        undefined;
      const messages = getRecentMessages('web:chat', 100, threadId);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(messages));
      return;
    }

    if (url.pathname === '/api/chat/artifacts') {
      const threadId =
        url.searchParams.get('threadId') ||
        webChatController?.getActiveThreadId() ||
        undefined;
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(getChatArtifacts(threadId)));
      return;
    }

    // ── API: Chat file upload ──
    if (url.pathname === '/api/chat/upload' && req.method === 'POST') {
      const filename = url.searchParams.get('filename');
      if (!filename) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Missing filename parameter' }));
        return;
      }
      // Sanitize filename
      const safeName = path
        .basename(filename)
        .replace(/[^a-zA-Z0-9._\-\u4e00-\u9fff]/g, '_');
      const uniqueName = `${Date.now()}_${safeName}`;
      const uploadDir = path.join(GROUPS_DIR, 'web_chat', 'uploads');
      fs.mkdirSync(uploadDir, { recursive: true });
      const filePath = path.join(uploadDir, uniqueName);

      const chunks: Buffer[] = [];
      let totalSize = 0;
      const MAX_SIZE = 50 * 1024 * 1024; // 50MB
      let aborted = false;

      req.on('data', (chunk: Buffer) => {
        totalSize += chunk.length;
        if (totalSize > MAX_SIZE && !aborted) {
          aborted = true;
          res.writeHead(413, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'File too large (max 50MB)' }));
          req.destroy();
          return;
        }
        chunks.push(chunk);
      });

      req.on('end', () => {
        if (aborted) return;
        fs.writeFileSync(filePath, Buffer.concat(chunks));
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(
          JSON.stringify({ path: `uploads/${uniqueName}`, filename: safeName }),
        );
      });
      return;
    }

    // ── API: Chat send (POST fallback for non-WS clients) ──
    if (url.pathname === '/api/chat/send' && req.method === 'POST') {
      let body = '';
      req.on('data', (chunk) => {
        body += chunk;
      });
      req.on('end', () => {
        try {
          const { text, threadId } = JSON.parse(body);
          if (!text || typeof text !== 'string') {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Missing text field' }));
            return;
          }
          if (threadId && typeof threadId === 'string' && webChatController) {
            webChatController.switchSession(threadId);
          }
          if (channelOpts) {
            handleWebChatMessage(text, channelOpts, threadId);
          }
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: true }));
        } catch {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Invalid JSON' }));
        }
      });
      return;
    }

    // ── Static: bundled assets ──
    if (url.pathname.startsWith('/assets/')) {
      const assetName = path.basename(url.pathname);
      const assetPath = path.join(process.cwd(), 'assets', assetName);
      const resolved = path.resolve(assetPath);
      const assetsRoot = path.resolve(path.join(process.cwd(), 'assets'));
      if (!resolved.startsWith(assetsRoot)) {
        res.writeHead(403);
        res.end('Forbidden');
        return;
      }
      if (!fs.existsSync(resolved)) {
        res.writeHead(404);
        res.end('Not found');
        return;
      }
      const ext = path.extname(resolved).toLowerCase();
      const mimeTypes: Record<string, string> = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.webp': 'image/webp',
        '.ico': 'image/x-icon',
      };
      res.writeHead(200, {
        'Content-Type': mimeTypes[ext] || 'application/octet-stream',
        'Cache-Control': 'public, max-age=86400',
      });
      fs.createReadStream(resolved).pipe(res);
      return;
    }

    res.writeHead(404);
    res.end('Not found');
  });

  wss = new WebSocketServer({ server });

  wss.on('connection', (ws) => {
    logger.debug('Dashboard WebSocket client connected');

    ws.on('message', (raw) => {
      try {
        const msg = JSON.parse(String(raw));
        if (
          msg.type === 'chat:send' &&
          typeof msg.text === 'string' &&
          channelOpts
        ) {
          if (typeof msg.threadId === 'string' && webChatController) {
            webChatController.switchSession(msg.threadId);
          }
          handleWebChatMessage(
            msg.text,
            channelOpts,
            typeof msg.threadId === 'string' ? msg.threadId : undefined,
          );
        }
      } catch {
        // Ignore malformed messages
      }
    });

    ws.on('close', () => {
      logger.debug('Dashboard WebSocket client disconnected');
    });
  });

  // Wire up web channel broadcast
  setWebBroadcast(broadcastRaw);

  // Forward all agent events to WebSocket clients
  agentEvents.on('agent', broadcast);

  let currentPort = PORT;
  const MAX_PORT_ATTEMPTS = 10;

  const tryListen = (attempt: number) => {
    server.once('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'EADDRINUSE' && attempt < MAX_PORT_ATTEMPTS) {
        currentPort++;
        logger.info(
          { port: currentPort },
          `Port ${currentPort - 1} in use, trying ${currentPort}`,
        );
        tryListen(attempt + 1);
      } else if (err.code === 'EADDRINUSE') {
        logger.warn(
          `All ports ${PORT}-${currentPort} in use — dashboard disabled`,
        );
      } else {
        logger.error({ err }, 'Dashboard server error');
      }
    });

    server.listen(currentPort, () => {
      logger.info(
        { port: currentPort },
        `Dashboard running at http://localhost:${currentPort}`,
      );
    });
  };

  tryListen(0);
}
