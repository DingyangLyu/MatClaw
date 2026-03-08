/**
 * MatClaw Monitoring Dashboard
 * Lightweight HTTP + WebSocket server for real-time agent monitoring.
 */
import http from 'http';
import fs from 'fs';
import path from 'path';
import { WebSocketServer, WebSocket } from 'ws';

import { agentEvents, type AgentEvent } from './events.js';
import { logger } from '../logger.js';
import { GROUPS_DIR } from '../config.js';

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

/** List available groups with their log files */
function getGroupsInfo(): {
  folder: string;
  logCount: number;
  lastLog: string | null;
}[] {
  const results: {
    folder: string;
    logCount: number;
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
    results.push({
      folder: entry,
      logCount: logs.length,
      lastLog: logs[0] || null,
    });
  }
  return results;
}

/** Read a specific log file */
function readLogFile(groupFolder: string, filename: string): string | null {
  // Prevent path traversal
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

    if (url.pathname === '/api/groups') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(getGroupsInfo()));
      return;
    }

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
