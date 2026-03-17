/**
 * Claude Agent SDK engine for MatClaw.
 * Wraps the @anthropic-ai/claude-agent-sdk query() function.
 */

import fs from 'fs';
import path from 'path';
import { query, HookCallback, PreCompactHookInput, PreToolUseHookInput } from '@anthropic-ai/claude-agent-sdk';
import { AgentEngine, EngineContext, QueryResult } from './interface.js';

// ── SDK message type for MessageStream ──

interface SDKUserMessage {
  type: 'user';
  message: { role: 'user'; content: string };
  parent_tool_use_id: null;
  session_id: string;
}

// ── Push-based async iterable for streaming user messages to the SDK ──

class MessageStream {
  private queue: SDKUserMessage[] = [];
  private waiting: (() => void) | null = null;
  private done = false;

  push(text: string): void {
    this.queue.push({
      type: 'user',
      message: { role: 'user', content: text },
      parent_tool_use_id: null,
      session_id: '',
    });
    this.waiting?.();
  }

  end(): void {
    this.done = true;
    this.waiting?.();
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<SDKUserMessage> {
    while (true) {
      while (this.queue.length > 0) {
        yield this.queue.shift()!;
      }
      if (this.done) return;
      await new Promise<void>(r => { this.waiting = r; });
      this.waiting = null;
    }
  }
}

// ── Session archiving helpers ──

interface SessionEntry {
  sessionId: string;
  fullPath: string;
  summary: string;
  firstPrompt: string;
}

interface SessionsIndex {
  entries: SessionEntry[];
}

interface ParsedMessage {
  role: 'user' | 'assistant';
  content: string;
}

function getSessionSummary(sessionId: string, transcriptPath: string): string | null {
  const projectDir = path.dirname(transcriptPath);
  const indexPath = path.join(projectDir, 'sessions-index.json');
  if (!fs.existsSync(indexPath)) return null;
  try {
    const index: SessionsIndex = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
    return index.entries.find(e => e.sessionId === sessionId)?.summary ?? null;
  } catch {
    return null;
  }
}

function sanitizeFilename(summary: string): string {
  return summary.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 50);
}

function generateFallbackName(): string {
  const t = new Date();
  return `conversation-${t.getHours().toString().padStart(2, '0')}${t.getMinutes().toString().padStart(2, '0')}`;
}

function parseTranscript(content: string): ParsedMessage[] {
  const messages: ParsedMessage[] = [];
  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      if (entry.type === 'user' && entry.message?.content) {
        const text = typeof entry.message.content === 'string'
          ? entry.message.content
          : entry.message.content.map((c: { text?: string }) => c.text || '').join('');
        if (text) messages.push({ role: 'user', content: text });
      } else if (entry.type === 'assistant' && entry.message?.content) {
        const textParts = entry.message.content
          .filter((c: { type: string }) => c.type === 'text')
          .map((c: { text: string }) => c.text);
        if (textParts.join('')) messages.push({ role: 'assistant', content: textParts.join('') });
      }
    } catch { /* skip malformed lines */ }
  }
  return messages;
}

function formatTranscriptMarkdown(messages: ParsedMessage[], title?: string | null, assistantName?: string): string {
  const now = new Date();
  const fmt = (d: Date) => d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true });
  const lines: string[] = [
    `# ${title || 'Conversation'}`, '',
    `Archived: ${fmt(now)}`, '', '---', '',
  ];
  for (const msg of messages) {
    const sender = msg.role === 'user' ? 'User' : (assistantName || 'Assistant');
    const content = msg.content.length > 2000 ? msg.content.slice(0, 2000) + '...' : msg.content;
    lines.push(`**${sender}**: ${content}`, '');
  }
  return lines.join('\n');
}

// ── Hooks ──

const SECRET_ENV_VARS = ['ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL', 'CLAUDE_CODE_OAUTH_TOKEN'];

function createPreCompactHook(assistantName?: string): HookCallback {
  return async (input) => {
    const preCompact = input as PreCompactHookInput;
    const transcriptPath = preCompact.transcript_path;
    const sessionId = preCompact.session_id;
    if (!transcriptPath || !fs.existsSync(transcriptPath)) return {};
    try {
      const content = fs.readFileSync(transcriptPath, 'utf-8');
      const messages = parseTranscript(content);
      if (messages.length === 0) return {};
      const summary = getSessionSummary(sessionId, transcriptPath);
      const name = summary ? sanitizeFilename(summary) : generateFallbackName();
      const conversationsDir = '/workspace/group/conversations';
      fs.mkdirSync(conversationsDir, { recursive: true });
      const date = new Date().toISOString().split('T')[0];
      fs.writeFileSync(
        path.join(conversationsDir, `${date}-${name}.md`),
        formatTranscriptMarkdown(messages, summary, assistantName),
      );
    } catch { /* non-fatal */ }
    return {};
  };
}

function createSanitizeBashHook(): HookCallback {
  return async (input) => {
    const preInput = input as PreToolUseHookInput;
    const command = (preInput.tool_input as { command?: string })?.command;
    if (!command) return {};
    const unsetPrefix = `unset ${SECRET_ENV_VARS.join(' ')} 2>/dev/null; `;
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse',
        updatedInput: {
          ...(preInput.tool_input as Record<string, unknown>),
          command: unsetPrefix + command,
        },
      },
    };
  };
}

// ── Claude Engine ──

const IPC_POLL_MS = 500;

export class ClaudeEngine implements AgentEngine {
  readonly name = 'claude';

  async runQuery(
    prompt: string,
    sessionId: string | undefined,
    ctx: EngineContext,
    resumeAt?: string,
  ): Promise<QueryResult> {
    const stream = new MessageStream();
    stream.push(prompt);

    // Poll IPC for follow-up messages and _close sentinel during the query
    let ipcPolling = true;
    let closedDuringQuery = false;
    const pollIpc = () => {
      if (!ipcPolling) return;
      if (ctx.shouldClose()) {
        ctx.log('Close sentinel detected during query, ending stream');
        closedDuringQuery = true;
        stream.end();
        ipcPolling = false;
        return;
      }
      ctx.refreshSdkEnv(ctx.sdkEnv);
      const messages = ctx.drainIpcInput();
      for (const text of messages) {
        ctx.log(`Piping IPC message into active query (${text.length} chars)`);
        stream.push(text);
      }
      setTimeout(pollIpc, IPC_POLL_MS);
    };
    setTimeout(pollIpc, IPC_POLL_MS);

    let newSessionId: string | undefined;
    let lastAssistantUuid: string | undefined;
    let messageCount = 0;
    let resultCount = 0;

    // Load global CLAUDE.md
    const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
    let globalClaudeMd: string | undefined;
    if (!ctx.isMain && fs.existsSync(globalClaudeMdPath)) {
      globalClaudeMd = fs.readFileSync(globalClaudeMdPath, 'utf-8');
    }

    // Discover additional directories
    const extraDirs: string[] = [];
    const extraBase = '/workspace/extra';
    if (fs.existsSync(extraBase)) {
      for (const entry of fs.readdirSync(extraBase)) {
        const fullPath = path.join(extraBase, entry);
        if (fs.statSync(fullPath).isDirectory()) extraDirs.push(fullPath);
      }
    }
    if (extraDirs.length > 0) ctx.log(`Additional directories: ${extraDirs.join(', ')}`);

    // Set model if AGENT_MODEL is configured (maps to CLAUDE_CODE_MODEL for SDK)
    const agentModel = process.env['AGENT_MODEL'];
    if (agentModel) {
      ctx.sdkEnv['CLAUDE_CODE_MODEL'] = agentModel;
      ctx.log(`Using model: ${agentModel}`);
    }

    for await (const message of query({
      prompt: stream,
      options: {
        cwd: '/workspace/group',
        additionalDirectories: extraDirs.length > 0 ? extraDirs : undefined,
        resume: sessionId,
        resumeSessionAt: resumeAt,
        systemPrompt: globalClaudeMd
          ? { type: 'preset' as const, preset: 'claude_code' as const, append: globalClaudeMd }
          : undefined,
        allowedTools: [
          'Bash', 'Read', 'Write', 'Edit', 'Glob', 'Grep',
          'WebSearch', 'WebFetch',
          'Task', 'TaskOutput', 'TaskStop',
          'TeamCreate', 'TeamDelete', 'SendMessage',
          'TodoWrite', 'ToolSearch', 'Skill',
          'NotebookEdit',
          'mcp__matclaw__*', 'mcp__gmail__*',
        ],
        env: ctx.sdkEnv,
        permissionMode: 'bypassPermissions',
        allowDangerouslySkipPermissions: true,
        settingSources: ['project', 'user'],
        mcpServers: {
          matclaw: {
            command: 'node',
            args: [ctx.mcpServerPath],
            env: {
              MATCLAW_CHAT_JID: ctx.chatJid,
              MATCLAW_GROUP_FOLDER: ctx.groupFolder,
              MATCLAW_IS_MAIN: ctx.isMain ? '1' : '0',
            },
          },
          gmail: {
            command: 'npx',
            args: ['-y', '@gongrzhe/server-gmail-autoauth-mcp'],
          },
        },
        hooks: {
          PreCompact: [{ hooks: [createPreCompactHook(ctx.assistantName)] }],
          PreToolUse: [{ matcher: 'Bash', hooks: [createSanitizeBashHook()] }],
        },
      }
    })) {
      messageCount++;
      const msgType = message.type === 'system' ? `system/${(message as { subtype?: string }).subtype}` : message.type;
      ctx.log(`[msg #${messageCount}] type=${msgType}`);

      if (message.type === 'assistant') {
        if ('uuid' in message) lastAssistantUuid = (message as { uuid: string }).uuid;
        const msg = message as { message?: { content?: Array<{ type: string; text?: string; thinking?: string; name?: string; input?: unknown; id?: string }> } };
        if (msg.message?.content) {
          for (const block of msg.message.content) {
            if (block.type === 'thinking' && block.thinking) ctx.log(`[Thinking] ${block.thinking.slice(0, 2000)}`);
            else if (block.type === 'text' && block.text) ctx.log(`[Assistant] ${block.text.slice(0, 2000)}`);
            else if (block.type === 'tool_use') {
              ctx.log(`[ToolCall] ${block.name} (id: ${block.id})`);
              ctx.log(`[ToolInput] ${JSON.stringify(block.input || {}).slice(0, 3000)}`);
            }
          }
        }
      }

      if (message.type === 'user') {
        const msg = message as { message?: { content?: Array<{ type: string; tool_use_id?: string; content?: unknown }> } };
        if (msg.message?.content) {
          for (const block of msg.message.content) {
            if (block.type === 'tool_result') {
              const resultStr = typeof block.content === 'string' ? block.content : JSON.stringify(block.content || '');
              ctx.log(`[ToolResult] (id: ${block.tool_use_id}) ${resultStr.slice(0, 3000)}`);
            }
          }
        }
      }

      if (message.type === 'system' && message.subtype === 'init') {
        newSessionId = message.session_id;
        ctx.log(`Session initialized: ${newSessionId}`);
      }

      if (message.type === 'system' && (message as { subtype?: string }).subtype === 'task_notification') {
        const tn = message as { task_id: string; status: string; summary: string };
        ctx.log(`Task notification: task=${tn.task_id} status=${tn.status} summary=${tn.summary}`);
      }

      if (message.type === 'result') {
        resultCount++;
        const textResult = 'result' in message ? (message as { result?: string }).result : null;
        ctx.log(`[Result #${resultCount}] subtype=${message.subtype}${textResult ? ` text=${textResult.slice(0, 500)}` : ''}`);
        ctx.writeOutput({ status: 'success', result: textResult || null, newSessionId });
      }
    }

    ipcPolling = false;
    ctx.log(`Query done. Messages: ${messageCount}, results: ${resultCount}, lastAssistantUuid: ${lastAssistantUuid || 'none'}, closedDuringQuery: ${closedDuringQuery}`);
    return { newSessionId, lastAssistantUuid, closedDuringQuery };
  }
}
