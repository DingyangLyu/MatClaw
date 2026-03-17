/**
 * Codex SDK engine for MatClaw.
 * Wraps @openai/codex-sdk to support any OpenAI-compatible API provider.
 */

import fs from 'fs';
import path from 'path';
import {
  Codex,
  type ThreadEvent,
  type ThreadItem,
  type AgentMessageItem,
  type CommandExecutionItem,
  type FileChangeItem,
  type McpToolCallItem,
  type ReasoningItem,
  type ThreadStartedEvent,
  type TurnCompletedEvent,
  type TurnFailedEvent,
  type ThreadErrorEvent,
  type ItemStartedEvent,
  type ItemCompletedEvent,
  type ItemUpdatedEvent,
} from '@openai/codex-sdk';
import { AgentEngine, EngineContext, QueryResult } from './interface.js';

const IPC_POLL_MS = 500;

function summarizeItem(item: ThreadItem): string {
  switch (item.type) {
    case 'agent_message':
      return item.text.slice(0, 200);
    case 'reasoning':
      return item.text.slice(0, 200);
    case 'command_execution':
      return item.command.slice(0, 200);
    case 'file_change':
      return item.changes.map(c => c.path).join(', ');
    case 'mcp_tool_call':
      return `${item.server}.${item.tool}`;
    default:
      return item.type;
  }
}

/** Conversation entry for archiving. */
interface ConversationEntry {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export class CodexEngine implements AgentEngine {
  readonly name = 'codex';
  private threadId?: string;
  private conversationLog: ConversationEntry[] = [];

  async runQuery(
    prompt: string,
    sessionId: string | undefined,
    ctx: EngineContext,
    _resumeAt?: string,
  ): Promise<QueryResult> {
    // Use saved threadId for session continuity
    const effectiveThreadId = sessionId || this.threadId;

    // Write Codex config with MCP servers
    this.writeCodexConfig(ctx);

    // Write system prompt as AGENTS.md (Codex reads from working directory)
    this.writeSystemPrompt(ctx);

    const apiKey = ctx.sdkEnv['CODEX_API_KEY'] || ctx.sdkEnv['OPENAI_API_KEY'];
    const baseUrl = ctx.sdkEnv['OPENAI_BASE_URL'];
    const model = ctx.sdkEnv['CODEX_MODEL'] || process.env['AGENT_MODEL'] || 'o4-mini';

    // If no API key is set, Codex CLI falls back to OAuth tokens in ~/.codex/auth.json
    // (obtained via `codex login` on the host, mounted by container-runner)
    const codexOpts: { apiKey?: string; baseUrl?: string } = {};
    if (apiKey) codexOpts.apiKey = apiKey;
    if (baseUrl) codexOpts.baseUrl = baseUrl;

    if (!apiKey) {
      const authPath = path.join(process.env['HOME'] || '/home/node', '.codex', 'auth.json');
      if (fs.existsSync(authPath)) {
        ctx.log('Using Codex OAuth authentication from auth.json');
      } else {
        ctx.log('Warning: No CODEX_API_KEY/OPENAI_API_KEY and no auth.json found — Codex may fail to authenticate');
      }
    }

    const codex = new Codex(codexOpts);

    // Discover additional directories
    const additionalDirs: string[] = [];
    const extraBase = '/workspace/extra';
    if (fs.existsSync(extraBase)) {
      for (const entry of fs.readdirSync(extraBase)) {
        const fullPath = path.join(extraBase, entry);
        if (fs.statSync(fullPath).isDirectory()) additionalDirs.push(fullPath);
      }
    }

    const threadOpts = {
      workingDirectory: '/workspace/group',
      sandboxMode: 'danger-full-access' as const,
      model,
      skipGitRepoCheck: true,
      additionalDirectories: additionalDirs.length > 0 ? additionalDirs : undefined,
    };

    const thread = effectiveThreadId
      ? codex.resumeThread(effectiveThreadId, threadOpts)
      : codex.startThread(threadOpts);

    // Set up abort controller for _close sentinel
    const abortController = new AbortController();

    // Poll IPC for _close sentinel during query
    let ipcPolling = true;
    let closedDuringQuery = false;
    const pollIpc = () => {
      if (!ipcPolling) return;
      if (ctx.shouldClose()) {
        ctx.log('Close sentinel detected during Codex query, aborting');
        closedDuringQuery = true;
        abortController.abort();
        ipcPolling = false;
        return;
      }
      ctx.refreshSdkEnv(ctx.sdkEnv);
      // Codex doesn't support mid-turn message injection like Claude's MessageStream.
      // IPC messages sent during a query are queued and handled by the main loop
      // after this turn completes.
      setTimeout(pollIpc, IPC_POLL_MS);
    };
    setTimeout(pollIpc, IPC_POLL_MS);

    // Record user prompt for conversation archive (cap at 200 entries to prevent memory bloat)
    this.conversationLog.push({
      role: 'user',
      content: prompt,
      timestamp: new Date().toISOString(),
    });
    if (this.conversationLog.length > 200) {
      // Archive before trimming so we don't lose history
      this.archiveConversation(ctx);
      this.conversationLog = this.conversationLog.slice(-50);
    }

    let newSessionId: string | undefined;
    let resultText: string | null = null;
    let eventCount = 0;

    try {
      const { events } = await thread.runStreamed(prompt, {
        signal: abortController.signal,
      });

      for await (const event of events) {
        eventCount++;
        ctx.log(`[codex #${eventCount}] ${event.type}`);

        switch (event.type) {
          case 'thread.started': {
            const e = event as ThreadStartedEvent;
            this.threadId = e.thread_id;
            newSessionId = e.thread_id;
            ctx.log(`Codex thread started: ${newSessionId}`);
            break;
          }

          case 'item.started':
          case 'item.updated': {
            const e = event as ItemStartedEvent | ItemUpdatedEvent;
            ctx.log(`[${e.item.type}] ${summarizeItem(e.item)}`);
            break;
          }

          case 'item.completed': {
            const e = event as ItemCompletedEvent;
            switch (e.item.type) {
              case 'agent_message':
                resultText = e.item.text;
                ctx.log(`[Agent] ${resultText.slice(0, 500)}`);
                break;
              case 'command_execution':
                ctx.log(`[Command] ${e.item.command} → exit ${e.item.exit_code ?? '?'}`);
                if (e.item.aggregated_output) ctx.log(`[Output] ${e.item.aggregated_output.slice(0, 1000)}`);
                break;
              case 'file_change':
                ctx.log(`[FileChange] ${e.item.changes.map(c => c.path).join(', ')}`);
                break;
              case 'mcp_tool_call':
                ctx.log(`[MCP] ${e.item.server}.${e.item.tool}`);
                break;
              default:
                ctx.log(`[${e.item.type}] completed`);
            }
            break;
          }

          case 'turn.completed': {
            const e = event as TurnCompletedEvent;
            ctx.log(`Turn completed. Tokens: ${e.usage.input_tokens} in / ${e.usage.output_tokens} out`);
            break;
          }

          case 'turn.failed': {
            const e = event as TurnFailedEvent;
            ctx.log(`Turn failed: ${e.error.message}`);
            resultText = `Error: ${e.error.message}`;
            break;
          }

          case 'error': {
            const e = event as ThreadErrorEvent;
            ctx.log(`Codex error: ${e.message}`);
            break;
          }
        }
      }
    } catch (err) {
      if (abortController.signal.aborted) {
        ctx.log('Codex query aborted by _close sentinel');
      } else {
        throw err;
      }
    }

    ipcPolling = false;

    // Record assistant response for conversation archive
    if (resultText !== null) {
      this.conversationLog.push({
        role: 'assistant',
        content: resultText,
        timestamp: new Date().toISOString(),
      });
    }

    // Emit the result
    if (resultText !== null) {
      ctx.writeOutput({
        status: 'success',
        result: resultText,
        newSessionId,
      });
    }

    // Archive conversation to conversations/ (equivalent of Claude's PreCompact hook)
    this.archiveConversation(ctx);

    ctx.log(`Codex query done. Events: ${eventCount}, threadId: ${this.threadId || 'none'}, closedDuringQuery: ${closedDuringQuery}`);
    return {
      newSessionId,
      lastAssistantUuid: this.threadId, // Codex uses threadId for resume
      closedDuringQuery,
    };
  }

  /**
   * Write Codex config.toml with MCP server definitions.
   * Codex reads MCP config from ~/.codex/config.toml.
   */
  private writeCodexConfig(ctx: EngineContext): void {
    const configDir = path.join(process.env['HOME'] || '/home/node', '.codex');
    fs.mkdirSync(configDir, { recursive: true });

    const mcpServerPath = ctx.mcpServerPath;
    const toml = [
      '# Auto-generated by MatClaw agent-runner',
      '',
      '[mcp_servers.matclaw]',
      `command = "node"`,
      `args = ["${mcpServerPath}"]`,
      '',
      '[mcp_servers.matclaw.env]',
      `MATCLAW_CHAT_JID = "${ctx.chatJid}"`,
      `MATCLAW_GROUP_FOLDER = "${ctx.groupFolder}"`,
      `MATCLAW_IS_MAIN = "${ctx.isMain ? '1' : '0'}"`,
      '',
      '[mcp_servers.gmail]',
      `command = "npx"`,
      `args = ["-y", "@gongrzhe/server-gmail-autoauth-mcp"]`,
      '',
    ].join('\n');

    fs.writeFileSync(path.join(configDir, 'config.toml'), toml);
  }

  /**
   * Write system prompt as AGENTS.md in the working directory.
   * Codex reads AGENTS.md for project-level instructions (similar to CLAUDE.md).
   *
   * Skills are loaded natively by Codex from ~/.codex/skills/ (symlinked from
   * ~/.claude/skills/ by the entrypoint script). This method only handles
   * the global CLAUDE.md → AGENTS.md conversion.
   */
  private writeSystemPrompt(ctx: EngineContext): void {
    const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
    if (!ctx.isMain && fs.existsSync(globalClaudeMdPath)) {
      const content = fs.readFileSync(globalClaudeMdPath, 'utf-8');
      // Write as AGENTS.md so Codex picks it up as project context
      fs.writeFileSync('/workspace/group/AGENTS.md', content);
    }

    // Ensure skills are accessible at ~/.codex/skills/
    // The entrypoint symlinks .claude/skills/ → .codex/skills/, but if
    // skills were mounted after entrypoint ran, we need to re-sync here.
    const claudeSkills = '/home/node/.claude/skills';
    const codexSkills = '/home/node/.codex/skills';
    if (fs.existsSync(claudeSkills) && !fs.existsSync(codexSkills)) {
      fs.mkdirSync(path.dirname(codexSkills), { recursive: true });
      for (const entry of fs.readdirSync(claudeSkills)) {
        const src = path.join(claudeSkills, entry);
        const dst = path.join(codexSkills, entry);
        if (fs.statSync(src).isDirectory() && !fs.existsSync(dst)) {
          fs.symlinkSync(src, dst);
        }
      }
      ctx.log(`Synced ${fs.readdirSync(codexSkills).length} skills to ${codexSkills}`);
    }
  }

  /**
   * Archive conversation to conversations/ directory.
   * Equivalent of Claude's PreCompact hook — saves searchable history
   * so future sessions can recall context from past conversations.
   */
  private archiveConversation(ctx: EngineContext): void {
    if (this.conversationLog.length === 0) return;

    try {
      const conversationsDir = '/workspace/group/conversations';
      fs.mkdirSync(conversationsDir, { recursive: true });

      const date = new Date().toISOString().split('T')[0];
      // Use first user message as title (sanitized)
      const firstPrompt = this.conversationLog.find(e => e.role === 'user')?.content || '';
      const titleSlug = firstPrompt
        .slice(0, 60)
        .toLowerCase()
        .replace(/[^a-z0-9\u4e00-\u9fff]+/g, '-')
        .replace(/^-+|-+$/g, '')
        || 'conversation';
      const filename = `${date}-${titleSlug}.md`;
      const filePath = path.join(conversationsDir, filename);

      const now = new Date();
      const fmt = (d: Date) => d.toLocaleString('en-US', {
        month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true,
      });

      const lines: string[] = [
        `# ${firstPrompt.slice(0, 100) || 'Conversation'}`,
        '',
        `Archived: ${fmt(now)} | Engine: codex | Thread: ${this.threadId || 'unknown'}`,
        '',
        '---',
        '',
      ];

      for (const entry of this.conversationLog) {
        const sender = entry.role === 'user' ? 'User' : (ctx.assistantName || 'Assistant');
        const content = entry.content.length > 2000
          ? entry.content.slice(0, 2000) + '...'
          : entry.content;
        lines.push(`**${sender}**: ${content}`, '');
      }

      fs.writeFileSync(filePath, lines.join('\n'));
      ctx.log(`Archived conversation to ${filePath} (${this.conversationLog.length} messages)`);
    } catch (err) {
      ctx.log(`Failed to archive conversation: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
}
