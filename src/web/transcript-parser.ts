/**
 * Parse Claude Agent SDK .jsonl transcripts into structured, human-readable logs.
 * Each line in the jsonl is a message object with type, timestamp, and content.
 */
import fs from 'fs';

export interface LogEntry {
  time: string; // HH:mm:ss.SSS
  type: 'user' | 'assistant' | 'tool-call' | 'tool-result' | 'thinking' | 'system' | 'result';
  content: string;
  meta?: string; // tool name, tool id, etc.
}

interface TranscriptLine {
  type: string;
  subtype?: string;
  timestamp?: string;
  message?: {
    role: string;
    content: string | Array<{
      type: string;
      text?: string;
      thinking?: string;
      name?: string;
      id?: string;
      input?: unknown;
      tool_use_id?: string;
      content?: string | unknown;
    }>;
  };
  toolUseResult?: {
    stdout?: string;
    stderr?: string;
  } | Array<{ type: string; text?: string }>;
  result?: string;
  session_id?: string;
}

function formatTime(ts?: string): string {
  if (!ts) return '??:??:??.???';
  try {
    return new Date(ts).toISOString().slice(11, 23);
  } catch {
    return '??:??:??.???';
  }
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, max) + `... (${s.length} chars total)`;
}

export function parseTranscript(jsonlContent: string): LogEntry[] {
  const entries: LogEntry[] = [];
  const lines = jsonlContent.split('\n').filter(l => l.trim());

  for (const line of lines) {
    let obj: TranscriptLine;
    try {
      obj = JSON.parse(line);
    } catch {
      continue;
    }

    const time = formatTime(obj.timestamp);

    // System messages
    if (obj.type === 'system') {
      if (obj.subtype === 'init') {
        entries.push({ time, type: 'system', content: `Session initialized: ${obj.session_id || ''}` });
      } else if (obj.subtype === 'task_started') {
        entries.push({ time, type: 'system', content: `Task started` });
      } else if (obj.subtype === 'task_notification') {
        const tn = obj as unknown as { task_id: string; status: string; summary: string };
        entries.push({ time, type: 'system', content: `Task ${tn.task_id}: ${tn.status} - ${tn.summary}` });
      } else {
        entries.push({ time, type: 'system', content: `${obj.subtype || 'system event'}` });
      }
      continue;
    }

    // Queue operations
    if (obj.type === 'queue-operation') {
      continue; // Skip internal queue ops
    }

    // Rate limit events
    if (obj.type === 'rate_limit_event') {
      entries.push({ time, type: 'system', content: 'Rate limit hit, waiting...' });
      continue;
    }

    // Result messages
    if (obj.type === 'result') {
      const resultText = obj.result || (obj as unknown as { result?: string }).result || '';
      entries.push({
        time,
        type: 'result',
        content: truncate(resultText, 2000),
        meta: `subtype=${obj.subtype || 'unknown'}`,
      });
      continue;
    }

    if (!obj.message?.content) continue;
    const content = obj.message.content;

    // User messages
    if (obj.type === 'user') {
      if (typeof content === 'string') {
        entries.push({ time, type: 'user', content: truncate(content, 1000) });
        continue;
      }

      // Array content — could be tool results or user text
      if (Array.isArray(content)) {
        for (const block of content) {
          if (block.type === 'tool_result') {
            let resultText: string;
            if (typeof block.content === 'string') {
              resultText = block.content;
            } else if (Array.isArray(block.content)) {
              resultText = (block.content as Array<{ text?: string }>)
                .map(c => c.text || '')
                .join('');
            } else {
              resultText = JSON.stringify(block.content || '');
            }
            entries.push({
              time,
              type: 'tool-result',
              content: truncate(resultText, 3000),
              meta: block.tool_use_id || undefined,
            });
          } else if (block.type === 'text' && block.text) {
            entries.push({ time, type: 'user', content: truncate(block.text, 1000) });
          }
        }
      }
      continue;
    }

    // Assistant messages
    if (obj.type === 'assistant') {
      if (typeof content === 'string') {
        entries.push({ time, type: 'assistant', content: truncate(content, 2000) });
        continue;
      }

      if (Array.isArray(content)) {
        for (const block of content) {
          if (block.type === 'thinking' && block.thinking) {
            entries.push({
              time,
              type: 'thinking',
              content: truncate(block.thinking, 2000),
            });
          } else if (block.type === 'text' && block.text) {
            entries.push({
              time,
              type: 'assistant',
              content: truncate(block.text, 2000),
            });
          } else if (block.type === 'tool_use') {
            const inputStr = JSON.stringify(block.input || {});
            entries.push({
              time,
              type: 'tool-call',
              content: truncate(inputStr, 3000),
              meta: `${block.name || 'unknown'} (${block.id || ''})`,
            });
          }
        }
      }
      continue;
    }
  }

  return entries;
}

/** Format log entries into a readable text block */
export function formatLogEntries(entries: LogEntry[]): string {
  const lines: string[] = [];

  for (const e of entries) {
    const prefix = `[${e.time}]`;

    switch (e.type) {
      case 'system':
        lines.push(`${prefix} [SYS] ${e.content}`);
        break;
      case 'user':
        lines.push(`${prefix} [USR] ${e.content}`);
        break;
      case 'thinking':
        lines.push(`${prefix} [THK] ${e.content}`);
        break;
      case 'assistant':
        lines.push(`${prefix} [AST] ${e.content}`);
        break;
      case 'tool-call':
        lines.push(`${prefix} [CALL] ${e.meta}`);
        lines.push(`${prefix}        ${e.content}`);
        break;
      case 'tool-result': {
        lines.push(`${prefix} [RES] ${e.meta || ''}`);
        const resultLines = e.content.split('\n');
        if (resultLines.length <= 10) {
          lines.push(`${prefix}       ${e.content}`);
        } else {
          for (const rl of resultLines.slice(0, 8)) {
            lines.push(`${prefix}       ${rl}`);
          }
          lines.push(`${prefix}       ... (${resultLines.length} lines total)`);
        }
        break;
      }
      case 'result':
        lines.push(`${prefix} [OUT] ${e.meta}: ${e.content}`);
        break;
    }
  }

  return lines.join('\n');
}

/** Parse a jsonl file and return formatted text */
export function parseTranscriptFile(filePath: string): string {
  if (!fs.existsSync(filePath)) return 'File not found';
  const content = fs.readFileSync(filePath, 'utf-8');
  const entries = parseTranscript(content);
  return formatLogEntries(entries);
}
