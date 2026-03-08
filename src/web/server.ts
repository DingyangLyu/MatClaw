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

const PORT = parseInt(process.env.DASHBOARD_PORT || '3210', 10);

let wss: WebSocketServer;

function broadcast(event: AgentEvent) {
  const msg = JSON.stringify(event);
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

export function startDashboard() {
  const dashboardHtml = fs.readFileSync(
    path.join(import.meta.dirname, 'dashboard.html'),
    'utf-8',
  );

  const server = http.createServer((req, res) => {
    const url = new URL(req.url || '/', `http://${req.headers.host}`);

    if (url.pathname === '/' || url.pathname === '/dashboard') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(dashboardHtml);
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

    // ── Static: Logo ──
    if (url.pathname === '/assets/logo.png') {
      const logoPath = path.join(
        process.cwd(),
        'assets',
        'matclaw-icon-square.png',
      );
      if (fs.existsSync(logoPath)) {
        res.writeHead(200, {
          'Content-Type': 'image/png',
          'Cache-Control': 'public, max-age=86400',
        });
        fs.createReadStream(logoPath).pipe(res);
        return;
      }
    }

    res.writeHead(404);
    res.end('Not found');
  });

  wss = new WebSocketServer({ server });

  wss.on('connection', (ws) => {
    logger.debug('Dashboard WebSocket client connected');

    ws.on('close', () => {
      logger.debug('Dashboard WebSocket client disconnected');
    });
  });

  // Forward all agent events to WebSocket clients
  agentEvents.on('agent', broadcast);

  server.listen(PORT, () => {
    logger.info(
      { port: PORT },
      `Dashboard running at http://localhost:${PORT}`,
    );
  });
}
