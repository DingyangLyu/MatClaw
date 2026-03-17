import fs from 'fs';
import path from 'path';

import { readEnvFile } from './env.js';

// Apply proxy settings from .env before any network requests
const proxyVars = readEnvFile(['https_proxy', 'http_proxy', 'no_proxy']);
for (const [key, value] of Object.entries(proxyVars)) {
  if (!process.env[key]) process.env[key] = value;
  // Also set uppercase variants
  if (!process.env[key.toUpperCase()]) process.env[key.toUpperCase()] = value;
}

import {
  ASSISTANT_NAME,
  DATA_DIR,
  IDLE_TIMEOUT,
  POLL_INTERVAL,
  TRIGGER_PATTERN,
} from './config.js';
import './channels/index.js';
import {
  getChannelFactory,
  getRegisteredChannelNames,
} from './channels/registry.js';
import {
  ContainerOutput,
  runContainerAgent,
  writeGroupsSnapshot,
  writeTasksSnapshot,
} from './container-runner.js';
import { startDashboard } from './web/server.js';
import {
  cleanupOrphans,
  ensureContainerRuntimeRunning,
  ensureImageAvailable,
  killContainer,
} from './container-runtime.js';
import {
  assignWebChatThreadSession,
  createWebChatThread,
  deleteWebChatThread,
  getAllChats,
  getAllRegisteredGroups,
  deleteSession,
  getAllSessions,
  getAllTasks,
  getMessagesSince,
  getNewMessages,
  getActiveWebChatThreadId,
  getRouterState,
  getWebChatThread,
  initDatabase,
  listWebChatThreads,
  renameWebChatThread,
  setRegisteredGroup,
  setRouterState,
  setSession,
  setActiveWebChatThreadId,
  storeChatMetadata,
  storeMessage,
} from './db.js';
import { GroupQueue } from './group-queue.js';
import { resolveGroupFolderPath } from './group-folder.js';
import { startIpcWatcher } from './ipc.js';
import { findChannel, formatMessages, formatOutbound } from './router.js';
import {
  isSenderAllowed,
  isTriggerAllowed,
  loadSenderAllowlist,
  shouldDropMessage,
} from './sender-allowlist.js';
import { startSchedulerLoop } from './task-scheduler.js';
import { Channel, NewMessage, RegisteredGroup } from './types.js';
import { logger } from './logger.js';
import { setWebChatActiveThread } from './channels/web.js';

// Re-export for backwards compatibility during refactor
export { escapeXml, formatMessages } from './router.js';

let lastTimestamp = '';
let sessions: Record<string, string> = {};
let previousSessions: Record<string, string> = {}; // saved before /new, restored by /resume
let registeredGroups: Record<string, RegisteredGroup> = {};
let lastAgentTimestamp: Record<string, string> = {};
let messageLoopRunning = false;

const channels: Channel[] = [];
const queue = new GroupQueue();
const WEB_CHAT_FOLDER = 'web_chat';
const WEB_CHAT_JID = 'web:chat';

function loadState(): void {
  lastTimestamp = getRouterState('last_timestamp') || '';
  const agentTs = getRouterState('last_agent_timestamp');
  try {
    lastAgentTimestamp = agentTs ? JSON.parse(agentTs) : {};
  } catch {
    logger.warn('Corrupted last_agent_timestamp in DB, resetting');
    lastAgentTimestamp = {};
  }
  sessions = getAllSessions();
  registeredGroups = getAllRegisteredGroups();
  logger.info(
    { groupCount: Object.keys(registeredGroups).length },
    'State loaded',
  );
}

function saveState(): void {
  setRouterState('last_timestamp', lastTimestamp);
  setRouterState('last_agent_timestamp', JSON.stringify(lastAgentTimestamp));
}

function syncWebChatRuntimeSession(
  threadId: string,
  syncOutputThread = true,
): void {
  const thread = getWebChatThread(threadId);
  if (!thread) return;
  setActiveWebChatThreadId(threadId);
  if (syncOutputThread) {
    setWebChatActiveThread(threadId);
  }

  if (thread.agent_session_id) {
    sessions[WEB_CHAT_FOLDER] = thread.agent_session_id;
    setSession(WEB_CHAT_FOLDER, thread.agent_session_id);
  } else {
    delete sessions[WEB_CHAT_FOLDER];
    deleteSession(WEB_CHAT_FOLDER);
  }
}

function activateWebChatThread(threadId: string): void {
  const currentThreadId = getActiveWebChatThreadId();
  if (currentThreadId === threadId) {
    syncWebChatRuntimeSession(threadId);
    return;
  }
  const state = queue.getState(WEB_CHAT_JID);
  const hasRunningWebAgent = !!(state?.active && state.process);
  if (state?.active && state.process) {
    queue.markSessionCleared(WEB_CHAT_JID);
    state.process.kill('SIGTERM');
    if (state.containerName) killContainer(state.containerName);
  }
  syncWebChatRuntimeSession(threadId, !hasRunningWebAgent);
}

function getAgentCursorKey(chatJid: string, threadId?: string): string {
  if (chatJid === WEB_CHAT_JID) {
    return `${chatJid}:${threadId || getActiveWebChatThreadId()}`;
  }
  return chatJid;
}

function deleteAndReplaceWebChatThread(threadId: string):
  | {
      deletedThreadId: string;
      activeThreadId: string;
      activeSession: NonNullable<ReturnType<typeof getWebChatThread>>;
    }
  | undefined {
  const target = getWebChatThread(threadId);
  if (!target) return undefined;

  const existing = listWebChatThreads().filter(
    (thread) => thread.id !== threadId,
  );
  let fallback = existing[0];
  if (!fallback) {
    fallback = createWebChatThread('New chat');
  }

  const activeThreadId = getActiveWebChatThreadId();
  deleteWebChatThread(threadId);

  if (activeThreadId === threadId) {
    activateWebChatThread(fallback.id);
  } else if (!getWebChatThread(activeThreadId)) {
    activateWebChatThread(fallback.id);
  }

  const nextActive = getWebChatThread(getActiveWebChatThreadId());
  if (!nextActive) return undefined;

  return {
    deletedThreadId: threadId,
    activeThreadId: getActiveWebChatThreadId(),
    activeSession: nextActive,
  };
}

function registerGroup(jid: string, group: RegisteredGroup): void {
  let groupDir: string;
  try {
    groupDir = resolveGroupFolderPath(group.folder);
  } catch (err) {
    logger.warn(
      { jid, folder: group.folder, err },
      'Rejecting group registration with invalid folder',
    );
    return;
  }

  registeredGroups[jid] = group;
  setRegisteredGroup(jid, group);

  // Create group folder
  fs.mkdirSync(path.join(groupDir, 'logs'), { recursive: true });

  logger.info(
    { jid, name: group.name, folder: group.folder },
    'Group registered',
  );
}

/**
 * Get available groups list for the agent.
 * Returns groups ordered by most recent activity.
 */
export function getAvailableGroups(): import('./container-runner.js').AvailableGroup[] {
  const chats = getAllChats();
  const registeredJids = new Set(Object.keys(registeredGroups));

  return chats
    .filter((c) => c.jid !== '__group_sync__' && c.is_group)
    .map((c) => ({
      jid: c.jid,
      name: c.name,
      lastActivity: c.last_message_time,
      isRegistered: registeredJids.has(c.jid),
    }));
}

/** @internal - exported for testing */
export function _setRegisteredGroups(
  groups: Record<string, RegisteredGroup>,
): void {
  registeredGroups = groups;
}

/**
 * Process all pending messages for a group.
 * Called by the GroupQueue when it's this group's turn.
 */
async function processGroupMessages(chatJid: string): Promise<boolean> {
  const group = registeredGroups[chatJid];
  if (!group) return true;

  const channel = findChannel(channels, chatJid);
  if (!channel) {
    logger.warn({ chatJid }, 'No channel owns JID, skipping messages');
    return true;
  }

  const isMainGroup = group.isMain === true;
  const activeThreadId =
    group.folder === WEB_CHAT_FOLDER ? getActiveWebChatThreadId() : undefined;
  const cursorKey = getAgentCursorKey(chatJid, activeThreadId);
  const sinceTimestamp = lastAgentTimestamp[cursorKey] || '';
  const missedMessages = getMessagesSince(
    chatJid,
    sinceTimestamp,
    ASSISTANT_NAME,
    activeThreadId,
  );

  if (missedMessages.length === 0) return true;

  // ── Session control commands ──
  const lastMsg = missedMessages[missedMessages.length - 1].content.trim();

  if (/^\/watch\b/i.test(lastMsg)) {
    lastAgentTimestamp[cursorKey] =
      missedMessages[missedMessages.length - 1].timestamp;
    saveState();
    // Read tail of container-live.log and extract recent activity
    const liveLogPath = path.join(
      DATA_DIR,
      'groups',
      group.folder,
      'logs',
      'container-live.log',
    );
    if (!fs.existsSync(liveLogPath)) {
      if (channel) await channel.sendMessage(chatJid, 'No agent activity yet.');
      return true;
    }
    const raw = fs.readFileSync(liveLogPath, 'utf-8');
    const allLines = raw.split('\n');
    // Take last 200 lines, extract meaningful entries
    const tail = allLines.slice(-200);
    const activities: string[] = [];
    for (const line of tail) {
      const m = line.match(/\[stderr\]\s*\[[\d:.]+\]\s*(.+)/);
      if (!m) continue;
      const content = m[1];
      if (content.startsWith('[ToolCall]')) {
        activities.push(content.replace('[ToolCall] ', ''));
      } else if (content.startsWith('[Assistant]')) {
        activities.push(
          'Assistant: ' + content.replace('[Assistant] ', '').slice(0, 100),
        );
      } else if (content.startsWith('[Thinking]')) {
        activities.push('Thinking...');
      } else if (content.startsWith('[Result')) {
        activities.push(content.slice(0, 120));
      }
    }
    const state = queue.getState(chatJid);
    const status = state?.active ? 'Running' : 'Idle';
    const recent = activities.slice(-10);
    if (channel) {
      const msg =
        recent.length > 0
          ? `Agent: ${status}\n\nRecent activity:\n${recent.join('\n')}`
          : `Agent: ${status}\n\nNo recent tool activity.`;
      await channel.sendMessage(chatJid, msg);
    }
    return true;
  }

  if (/^\/help\b/i.test(lastMsg)) {
    lastAgentTimestamp[cursorKey] =
      missedMessages[missedMessages.length - 1].timestamp;
    saveState();
    if (channel) {
      await channel.sendMessage(
        chatJid,
        [
          'Commands:',
          '/watch — see what agent is doing right now',
          '/status — agent status (running/idle, session, queue)',
          '/stop — force stop running agent',
          '/sessions — list all sessions',
          '/new — start fresh conversation',
          '/resume [id] — restore previous or specific session',
          '/compact [focus] — compress agent memory',
          '/help — this message',
        ].join('\n'),
      );
    }
    return true;
  }

  if (/^\/stop\b/i.test(lastMsg)) {
    lastAgentTimestamp[cursorKey] =
      missedMessages[missedMessages.length - 1].timestamp;
    saveState();
    const state = queue.getState(chatJid);
    if (state?.active && state.process) {
      state.process.kill('SIGTERM');
      if (state.containerName) killContainer(state.containerName);
      logger.info({ group: group.name }, 'Agent stopped via /stop command');
      if (channel) {
        await channel.sendMessage(chatJid, 'Agent stopped.');
      }
    } else {
      if (channel) {
        await channel.sendMessage(chatJid, 'No agent running.');
      }
    }
    return true;
  }

  if (/^\/status\b/i.test(lastMsg)) {
    lastAgentTimestamp[cursorKey] =
      missedMessages[missedMessages.length - 1].timestamp;
    saveState();
    const state = queue.getState(chatJid);
    const sid = sessions[group.folder];
    const parts: string[] = [];
    parts.push(`Group: ${group.name} (${group.folder})`);
    parts.push(
      `Agent: ${state?.active ? 'running' : 'idle'}${state?.containerName ? ` (${state.containerName})` : ''}`,
    );
    parts.push(`Session: ${sid ? sid.slice(0, 8) + '...' : 'none'}`);
    if (state?.pendingTasks.length) {
      parts.push(`Queued tasks: ${state.pendingTasks.length}`);
    }
    if (channel) {
      await channel.sendMessage(chatJid, parts.join('\n'));
    }
    return true;
  }

  if (/^\/sessions\b/i.test(lastMsg)) {
    // List available sessions for this group
    const transcriptDir = path.join(
      DATA_DIR,
      'sessions',
      group.folder,
      '.claude',
      'projects',
      '-workspace-group',
    );
    let lines: string[] = [];
    const currentId = sessions[group.folder];
    if (fs.existsSync(transcriptDir)) {
      const files = fs
        .readdirSync(transcriptDir)
        .filter((f) => f.endsWith('.jsonl'))
        .map((f) => {
          const stat = fs.statSync(path.join(transcriptDir, f));
          const id = f.replace('.jsonl', '');
          return { id, size: stat.size, modified: stat.mtime };
        })
        .sort((a, b) => b.modified.getTime() - a.modified.getTime());
      for (const s of files) {
        const kb = Math.round(s.size / 1024);
        const time = s.modified.toISOString().slice(0, 16).replace('T', ' ');
        const marker = s.id === currentId ? ' [active]' : '';
        lines.push(`${s.id.slice(0, 8)}  ${time}  ${kb}KB${marker}`);
      }
    }
    lastAgentTimestamp[cursorKey] =
      missedMessages[missedMessages.length - 1].timestamp;
    saveState();
    if (channel) {
      const msg =
        lines.length > 0
          ? `Sessions:\n${lines.join('\n')}\n\nUse /resume <id prefix> to switch.`
          : 'No sessions found.';
      await channel.sendMessage(chatJid, msg);
    }
    return true;
  }

  if (/^\/new\b/i.test(lastMsg)) {
    // Save current session so /resume can restore it
    if (sessions[group.folder]) {
      previousSessions[group.folder] = sessions[group.folder];
    }
    queue.markSessionCleared(chatJid);
    delete sessions[group.folder];
    deleteSession(group.folder);
    lastAgentTimestamp[cursorKey] =
      missedMessages[missedMessages.length - 1].timestamp;
    saveState();
    logger.info({ group: group.name }, 'Session reset via /new command');
    if (channel) {
      await channel.sendMessage(
        chatJid,
        'Session cleared. Next message starts a fresh conversation. Use /resume to restore the previous session.',
      );
    }
    return true;
  }

  if (/^\/resume\b/i.test(lastMsg)) {
    const arg = lastMsg.replace(/^\/resume\s*/i, '').trim();
    let targetId: string | undefined;

    if (arg) {
      // Find session by ID prefix
      const transcriptDir = path.join(
        DATA_DIR,
        'sessions',
        group.folder,
        '.claude',
        'projects',
        '-workspace-group',
      );
      if (fs.existsSync(transcriptDir)) {
        const match = fs
          .readdirSync(transcriptDir)
          .filter((f) => f.endsWith('.jsonl') && f.startsWith(arg))
          .map((f) => f.replace('.jsonl', ''));
        if (match.length === 1) {
          targetId = match[0];
        } else if (match.length > 1) {
          lastAgentTimestamp[cursorKey] =
            missedMessages[missedMessages.length - 1].timestamp;
          saveState();
          if (channel) {
            await channel.sendMessage(
              chatJid,
              `Ambiguous prefix "${arg}", matches: ${match.map((m) => m.slice(0, 8)).join(', ')}. Be more specific.`,
            );
          }
          return true;
        }
      }
    } else {
      // No argument — restore the session saved by /new
      targetId = previousSessions[group.folder];
    }

    if (targetId) {
      if (sessions[group.folder]) {
        previousSessions[group.folder] = sessions[group.folder];
      }
      sessions[group.folder] = targetId;
      setSession(group.folder, targetId);
      lastAgentTimestamp[cursorKey] =
        missedMessages[missedMessages.length - 1].timestamp;
      saveState();
      logger.info(
        { group: group.name, sessionId: targetId },
        'Session restored via /resume',
      );
      if (channel) {
        await channel.sendMessage(
          chatJid,
          `Session restored (${targetId.slice(0, 8)}...). Agent will continue with that session's history.`,
        );
      }
    } else {
      lastAgentTimestamp[cursorKey] =
        missedMessages[missedMessages.length - 1].timestamp;
      saveState();
      if (channel) {
        await channel.sendMessage(
          chatJid,
          'No session found. Use /sessions to list available sessions.',
        );
      }
    }
    return true;
  }

  if (/^\/compact\b/i.test(lastMsg)) {
    // Extract optional focus argument: /compact 保留飞书相关的记忆
    const focus = lastMsg.replace(/^\/compact\s*/i, '').trim();
    let compactPrompt: string;
    if (focus) {
      compactPrompt = `[SYSTEM] Compress and reorganize your current memory. Focus on: ${focus}. Discard everything not related to this focus. Summarize what you kept.`;
    } else {
      compactPrompt =
        '[SYSTEM] Summarize your current memory and context into a concise form. Forget unnecessary details, keep only the key facts, decisions, and ongoing tasks. After summarizing, confirm what you remember.';
    }
    missedMessages[missedMessages.length - 1] = {
      ...missedMessages[missedMessages.length - 1],
      content: compactPrompt,
    };
  }

  // For non-main groups, check if trigger is required and present
  if (!isMainGroup && group.requiresTrigger !== false) {
    const allowlistCfg = loadSenderAllowlist();
    const hasTrigger = missedMessages.some(
      (m) =>
        TRIGGER_PATTERN.test(m.content.trim()) &&
        (m.is_from_me || isTriggerAllowed(chatJid, m.sender, allowlistCfg)),
    );
    if (!hasTrigger) return true;
  }

  const prompt = formatMessages(missedMessages);

  // Advance cursor so the piping path in startMessageLoop won't re-fetch
  // these messages. Save the old cursor so we can roll back on error.
  const previousCursor = lastAgentTimestamp[cursorKey] || '';
  lastAgentTimestamp[cursorKey] =
    missedMessages[missedMessages.length - 1].timestamp;
  saveState();

  logger.info(
    { group: group.name, messageCount: missedMessages.length },
    'Processing messages',
  );

  // Track idle timer for closing stdin when agent is idle
  let idleTimer: ReturnType<typeof setTimeout> | null = null;

  const resetIdleTimer = () => {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      logger.debug(
        { group: group.name },
        'Idle timeout, closing container stdin',
      );
      queue.closeStdin(chatJid);
    }, IDLE_TIMEOUT);
  };

  await channel.setTyping?.(chatJid, true);
  let hadError = false;
  let outputSentToUser = false;

  const output = await runAgent(group, prompt, chatJid, async (result) => {
    // Streaming output callback — called for each agent result
    if (result.result) {
      const raw =
        typeof result.result === 'string'
          ? result.result
          : JSON.stringify(result.result);
      // Strip <internal>...</internal> blocks — agent uses these for internal reasoning
      const text = raw.replace(/<internal>[\s\S]*?<\/internal>/g, '').trim();
      logger.info({ group: group.name }, `Agent output: ${raw.slice(0, 200)}`);
      if (text) {
        await channel.sendMessage(chatJid, text);
        outputSentToUser = true;
      }
      // Only reset idle timer on actual results, not session-update markers (result: null)
      resetIdleTimer();
    }

    if (result.status === 'success') {
      queue.notifyIdle(chatJid);
    }

    if (result.status === 'error') {
      hadError = true;
    }
  });

  await channel.setTyping?.(chatJid, false);
  if (idleTimer) clearTimeout(idleTimer);

  if (output === 'error' || hadError) {
    // If we already sent output to the user, don't roll back the cursor —
    // the user got their response and re-processing would send duplicates.
    if (outputSentToUser) {
      logger.warn(
        { group: group.name },
        'Agent error after output was sent, skipping cursor rollback to prevent duplicates',
      );
      return true;
    }
    // Roll back cursor so retries can re-process these messages
    lastAgentTimestamp[cursorKey] = previousCursor;
    saveState();
    logger.warn(
      { group: group.name },
      'Agent error, rolled back message cursor for retry',
    );
    return false;
  }

  return true;
}

async function runAgent(
  group: RegisteredGroup,
  prompt: string,
  chatJid: string,
  onOutput?: (output: ContainerOutput) => Promise<void>,
): Promise<'success' | 'error'> {
  const isMain = group.isMain === true;
  const sessionId = sessions[group.folder];

  // Update tasks snapshot for container to read (filtered by group)
  const tasks = getAllTasks();
  writeTasksSnapshot(
    group.folder,
    isMain,
    tasks.map((t) => ({
      id: t.id,
      groupFolder: t.group_folder,
      prompt: t.prompt,
      schedule_type: t.schedule_type,
      schedule_value: t.schedule_value,
      status: t.status,
      next_run: t.next_run,
    })),
  );

  // Update available groups snapshot (main group only can see all groups)
  const availableGroups = getAvailableGroups();
  writeGroupsSnapshot(
    group.folder,
    isMain,
    availableGroups,
    new Set(Object.keys(registeredGroups)),
  );

  // Wrap onOutput to track session ID from streamed results.
  // Skip saving if /new cleared the session (dying container's output
  // must not re-save the old session ID).
  const wrappedOnOutput = onOutput
    ? async (output: ContainerOutput) => {
        if (output.newSessionId && !queue.isSessionCleared(chatJid)) {
          sessions[group.folder] = output.newSessionId;
          setSession(group.folder, output.newSessionId);
          if (group.folder === WEB_CHAT_FOLDER) {
            assignWebChatThreadSession(
              getActiveWebChatThreadId(),
              output.newSessionId,
            );
          }
        }
        await onOutput(output);
      }
    : undefined;

  try {
    const output = await runContainerAgent(
      group,
      {
        prompt,
        sessionId,
        groupFolder: group.folder,
        chatJid,
        isMain,
        assistantName: ASSISTANT_NAME,
      },
      (proc, containerName) =>
        queue.registerProcess(chatJid, proc, containerName, group.folder),
      wrappedOnOutput,
    );

    if (output.newSessionId && !queue.isSessionCleared(chatJid)) {
      sessions[group.folder] = output.newSessionId;
      setSession(group.folder, output.newSessionId);
      if (group.folder === WEB_CHAT_FOLDER) {
        assignWebChatThreadSession(
          getActiveWebChatThreadId(),
          output.newSessionId,
        );
      }
    }

    if (output.status === 'error') {
      logger.error(
        { group: group.name, error: output.error },
        'Container agent error',
      );
      return 'error';
    }

    return 'success';
  } catch (err) {
    logger.error({ group: group.name, err }, 'Agent error');
    return 'error';
  }
}

/**
 * Handle control commands (/stop, /new, /watch, /status, /help, /sessions)
 * that should work even while the agent container is running.
 * Returns true if the message was a control command and was handled.
 */
async function handleControlCommand(
  chatJid: string,
  group: RegisteredGroup,
  channel: Channel,
  messages: NewMessage[],
): Promise<boolean> {
  const threadId =
    group.folder === WEB_CHAT_FOLDER
      ? messages[messages.length - 1].thread_id || getActiveWebChatThreadId()
      : undefined;
  const cursorKey = getAgentCursorKey(chatJid, threadId);
  const lastMsg = messages[messages.length - 1].content.trim();

  if (/^\/stop\b/i.test(lastMsg)) {
    lastAgentTimestamp[cursorKey] = messages[messages.length - 1].timestamp;
    saveState();
    const state = queue.getState(chatJid);
    if (state?.active && state.process) {
      state.process.kill('SIGTERM');
      if (state.containerName) killContainer(state.containerName);
      logger.info({ group: group.name }, 'Agent stopped via /stop command');
      await channel.sendMessage(chatJid, 'Agent stopped.');
    } else {
      await channel.sendMessage(chatJid, 'No agent running.');
    }
    return true;
  }

  if (/^\/new\b/i.test(lastMsg)) {
    // Stop running container first
    const state = queue.getState(chatJid);
    if (state?.active && state.process) {
      // Mark session cleared BEFORE killing — prevents the dying container's
      // output callbacks from re-saving the old session ID, and blocks
      // sendMessage() from piping new messages to the old container.
      queue.markSessionCleared(chatJid);
      state.process.kill('SIGTERM');
      if (state.containerName) killContainer(state.containerName);
      logger.info({ group: group.name }, 'Agent stopped for /new command');
    }
    if (sessions[group.folder]) {
      previousSessions[group.folder] = sessions[group.folder];
    }
    delete sessions[group.folder];
    deleteSession(group.folder);
    lastAgentTimestamp[cursorKey] = messages[messages.length - 1].timestamp;
    saveState();
    logger.info({ group: group.name }, 'Session reset via /new command');
    await channel.sendMessage(
      chatJid,
      'Session cleared. Next message starts a fresh conversation. Use /resume to restore the previous session.',
    );
    return true;
  }

  if (/^\/watch\b/i.test(lastMsg)) {
    lastAgentTimestamp[cursorKey] = messages[messages.length - 1].timestamp;
    saveState();
    const liveLogPath = path.join(
      DATA_DIR,
      'groups',
      group.folder,
      'logs',
      'container-live.log',
    );
    if (!fs.existsSync(liveLogPath)) {
      await channel.sendMessage(chatJid, 'No agent activity yet.');
      return true;
    }
    const raw = fs.readFileSync(liveLogPath, 'utf-8');
    const allLines = raw.split('\n');
    const tail = allLines.slice(-200);
    const activities: string[] = [];
    for (const line of tail) {
      const m = line.match(/\[stderr\]\s*\[[\d:.]+\]\s*(.+)/);
      if (!m) continue;
      const content = m[1];
      if (content.startsWith('[ToolCall]')) {
        activities.push(content.replace('[ToolCall] ', ''));
      } else if (content.startsWith('[Assistant]')) {
        activities.push(
          'Assistant: ' + content.replace('[Assistant] ', '').slice(0, 100),
        );
      } else if (content.startsWith('[Thinking]')) {
        activities.push('Thinking...');
      } else if (content.startsWith('[Result')) {
        activities.push(content.slice(0, 120));
      }
    }
    const state = queue.getState(chatJid);
    const status = state?.active ? 'Running' : 'Idle';
    const recent = activities.slice(-10);
    const msg =
      recent.length > 0
        ? `Agent: ${status}\n\nRecent activity:\n${recent.join('\n')}`
        : `Agent: ${status}\n\nNo recent tool activity.`;
    await channel.sendMessage(chatJid, msg);
    return true;
  }

  if (/^\/status\b/i.test(lastMsg)) {
    lastAgentTimestamp[cursorKey] = messages[messages.length - 1].timestamp;
    saveState();
    const state = queue.getState(chatJid);
    const sid = sessions[group.folder];
    const parts: string[] = [];
    parts.push(`Group: ${group.name} (${group.folder})`);
    parts.push(
      `Agent: ${state?.active ? 'running' : 'idle'}${state?.containerName ? ` (${state.containerName})` : ''}`,
    );
    parts.push(`Session: ${sid ? sid.slice(0, 8) + '...' : 'none'}`);
    if (state?.pendingTasks.length) {
      parts.push(`Queued tasks: ${state.pendingTasks.length}`);
    }
    await channel.sendMessage(chatJid, parts.join('\n'));
    return true;
  }

  if (/^\/help\b/i.test(lastMsg)) {
    lastAgentTimestamp[cursorKey] = messages[messages.length - 1].timestamp;
    saveState();
    await channel.sendMessage(
      chatJid,
      [
        'Commands:',
        '/watch — see what agent is doing right now',
        '/status — agent status (running/idle, session, queue)',
        '/stop — force stop running agent',
        '/sessions — list all sessions',
        '/new — start fresh conversation',
        '/resume [id] — restore previous or specific session',
        '/compact [focus] — compress agent memory',
        '/help — this message',
      ].join('\n'),
    );
    return true;
  }

  if (/^\/sessions\b/i.test(lastMsg)) {
    const transcriptDir = path.join(
      DATA_DIR,
      'sessions',
      group.folder,
      '.claude',
      'projects',
      '-workspace-group',
    );
    const lines: string[] = [];
    const currentId = sessions[group.folder];
    if (fs.existsSync(transcriptDir)) {
      const files = fs
        .readdirSync(transcriptDir)
        .filter((f) => f.endsWith('.jsonl'))
        .map((f) => {
          const stat = fs.statSync(path.join(transcriptDir, f));
          const id = f.replace('.jsonl', '');
          return { id, size: stat.size, modified: stat.mtime };
        })
        .sort((a, b) => b.modified.getTime() - a.modified.getTime());
      for (const s of files) {
        const kb = Math.round(s.size / 1024);
        const time = s.modified.toISOString().slice(0, 16).replace('T', ' ');
        const marker = s.id === currentId ? ' [active]' : '';
        lines.push(`${s.id.slice(0, 8)}  ${time}  ${kb}KB${marker}`);
      }
    }
    lastAgentTimestamp[cursorKey] = messages[messages.length - 1].timestamp;
    saveState();
    const msg =
      lines.length > 0
        ? `Sessions:\n${lines.join('\n')}\n\nUse /resume <id prefix> to switch.`
        : 'No sessions found.';
    await channel.sendMessage(chatJid, msg);
    return true;
  }

  if (/^\/resume\b/i.test(lastMsg)) {
    // /resume requires stopping the running container first
    const state = queue.getState(chatJid);
    if (state?.active && state.process) {
      state.process.kill('SIGTERM');
      if (state.containerName) killContainer(state.containerName);
      logger.info({ group: group.name }, 'Agent stopped for /resume command');
    }

    const arg = lastMsg.replace(/^\/resume\s*/i, '').trim();
    let targetId: string | undefined;

    if (arg) {
      const transcriptDir = path.join(
        DATA_DIR,
        'sessions',
        group.folder,
        '.claude',
        'projects',
        '-workspace-group',
      );
      if (fs.existsSync(transcriptDir)) {
        const match = fs
          .readdirSync(transcriptDir)
          .filter((f) => f.endsWith('.jsonl'))
          .find((f) => f.startsWith(arg));
        if (match) targetId = match.replace('.jsonl', '');
      }
    } else {
      targetId = previousSessions[group.folder];
    }

    if (targetId) {
      sessions[group.folder] = targetId;
      setSession(group.folder, targetId);
      lastAgentTimestamp[cursorKey] = messages[messages.length - 1].timestamp;
      saveState();
      logger.info(
        { group: group.name, sessionId: targetId },
        'Session restored via /resume',
      );
      await channel.sendMessage(
        chatJid,
        `Session restored (${targetId.slice(0, 8)}...). Agent will continue with that session's history.`,
      );
    } else {
      lastAgentTimestamp[cursorKey] = messages[messages.length - 1].timestamp;
      saveState();
      await channel.sendMessage(
        chatJid,
        'No session found. Use /sessions to list available sessions.',
      );
    }
    return true;
  }

  return false;
}

// ── Startup Banner ──────────────────────────────────────────────────────

function printStartupBanner(): void {
  const ESC = '\x1b';
  const C = {
    reset: `${ESC}[0m`,
    bold: `${ESC}[1m`,
    dim: `${ESC}[2m`,
    underline: `${ESC}[4m`,
    green: `${ESC}[32m`,
    yellow: `${ESC}[33m`,
    cyan: `${ESC}[36m`,
    brightGreen: `${ESC}[92m`,
    brightYellow: `${ESC}[93m`,
    brightCyan: `${ESC}[96m`,
    brightWhite: `${ESC}[97m`,
    fg: (n: number) => `${ESC}[38;5;${n}m`,
  };
  const GRAD = [
    196, 197, 198, 199, 200, 164, 128, 92, 56, 57, 63, 69, 75, 81, 45, 39, 33,
    27,
  ];
  const W = Math.min(process.stdout.columns || 80, 76);

  const grad = (text: string) =>
    [...text]
      .map((ch, i) => {
        if (ch === ' ') return ch;
        const idx = Math.floor(
          (i / Math.max(text.length - 1, 1)) * (GRAD.length - 1),
        );
        return `${C.fg(GRAD[idx])}${C.bold}${ch}${C.reset}`;
      })
      .join('');

  const strip = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, '');
  const bTop = (t?: string) => {
    if (t) {
      const s = ` ${t} `;
      const r = Math.max(0, W - 4 - s.length);
      return `  ${C.dim}╭──${C.reset}${C.bold}${C.brightCyan}${s}${C.reset}${C.dim}${'─'.repeat(r)}╮${C.reset}`;
    }
    return `  ${C.dim}╭${'─'.repeat(W - 2)}╮${C.reset}`;
  };
  const bLine = (text: string) => {
    const vis = strip(text).length;
    const pad = Math.max(0, W - 4 - vis);
    return `  ${C.dim}│${C.reset} ${text}${' '.repeat(pad)} ${C.dim}│${C.reset}`;
  };
  const bDiv = () => `  ${C.dim}├${'─'.repeat(W - 2)}┤${C.reset}`;
  const bBot = () => `  ${C.dim}╰${'─'.repeat(W - 2)}╯${C.reset}`;
  const bEmpty = () => `  ${C.dim}│${' '.repeat(W - 2)}│${C.reset}`;

  const logo = [
    '  ███╗   ███╗ █████╗ ████████╗ ██████╗██╗      █████╗ ██╗    ██╗',
    '  ████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██║     ██╔══██╗██║    ██║',
    '  ██╔████╔██║███████║   ██║   ██║     ██║     ███████║██║ █╗ ██║',
    '  ██║╚██╔╝██║██╔══██║   ██║   ██║     ██║     ██╔══██║██║███╗██║',
    '  ██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╗███████╗██║  ██║╚███╔███╔╝',
    '  ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ ',
  ];

  console.log();
  for (const line of logo) console.log(grad(line));
  console.log();

  // Channel status
  console.log(bTop('Channels'));
  for (const ch of channels) {
    const name = ch.constructor.name.replace('Channel', '');
    console.log(
      bLine(
        `${C.brightGreen}✔${C.reset}  ${C.bold}${name}${C.reset} ${C.dim}connected${C.reset}`,
      ),
    );
  }
  // Show skipped channels
  for (const chName of getRegisteredChannelNames()) {
    if (
      !channels.find((ch) =>
        ch.constructor.name.toLowerCase().includes(chName.toLowerCase()),
      )
    ) {
      console.log(
        bLine(
          `${C.brightYellow}⚠${C.reset}  ${C.bold}${chName}${C.reset} ${C.dim}credentials missing${C.reset}`,
        ),
      );
    }
  }
  console.log(bDiv());
  const groupCount = Object.keys(registeredGroups).length;
  console.log(
    bLine(
      `${C.brightCyan}Groups:${C.reset} ${groupCount}  ${C.dim}│${C.reset}  ${C.brightCyan}Trigger:${C.reset} @${ASSISTANT_NAME}`,
    ),
  );
  console.log(bDiv());
  console.log(
    bLine(
      `${C.bold}/watch${C.reset}   ${C.dim}monitor${C.reset}  ${C.bold}/status${C.reset}  ${C.dim}info${C.reset}  ${C.bold}/stop${C.reset}  ${C.dim}halt${C.reset}  ${C.bold}/help${C.reset}  ${C.dim}all cmds${C.reset}`,
    ),
  );
  console.log(bBot());
  console.log();
}

async function startMessageLoop(): Promise<void> {
  if (messageLoopRunning) {
    logger.debug('Message loop already running, skipping duplicate start');
    return;
  }
  messageLoopRunning = true;

  logger.info(`MatClaw running (trigger: @${ASSISTANT_NAME})`);
  printStartupBanner();

  while (true) {
    try {
      const jids = Object.keys(registeredGroups);
      const { messages, newTimestamp } = getNewMessages(
        jids,
        lastTimestamp,
        ASSISTANT_NAME,
      );

      if (messages.length > 0) {
        logger.info({ count: messages.length }, 'New messages');

        // Advance the "seen" cursor for all messages immediately
        lastTimestamp = newTimestamp;
        saveState();

        // Deduplicate by group
        const messagesByGroup = new Map<string, NewMessage[]>();
        for (const msg of messages) {
          const existing = messagesByGroup.get(msg.chat_jid);
          if (existing) {
            existing.push(msg);
          } else {
            messagesByGroup.set(msg.chat_jid, [msg]);
          }
        }

        for (const [chatJid, groupMessages] of messagesByGroup) {
          const group = registeredGroups[chatJid];
          if (!group) continue;
          const activeThreadId =
            group.folder === WEB_CHAT_FOLDER
              ? getActiveWebChatThreadId()
              : undefined;
          const cursorKey = getAgentCursorKey(chatJid, activeThreadId);
          const scopedGroupMessages =
            group.folder === WEB_CHAT_FOLDER
              ? groupMessages.filter(
                  (msg) => (msg.thread_id || activeThreadId) === activeThreadId,
                )
              : groupMessages;
          if (scopedGroupMessages.length === 0) continue;

          const channel = findChannel(channels, chatJid);
          if (!channel) {
            logger.warn({ chatJid }, 'No channel owns JID, skipping messages');
            continue;
          }

          // Intercept control commands before piping to running container.
          // Commands like /stop, /new, /watch work even while agent is active.
          const handled = await handleControlCommand(
            chatJid,
            group,
            channel,
            scopedGroupMessages,
          );
          if (handled) continue;

          const isMainGroup = group.isMain === true;
          const needsTrigger = !isMainGroup && group.requiresTrigger !== false;

          // For non-main groups, only act on trigger messages.
          // Non-trigger messages accumulate in DB and get pulled as
          // context when a trigger eventually arrives.
          if (needsTrigger) {
            const allowlistCfg = loadSenderAllowlist();
            const hasTrigger = scopedGroupMessages.some(
              (m) =>
                TRIGGER_PATTERN.test(m.content.trim()) &&
                (m.is_from_me ||
                  isTriggerAllowed(chatJid, m.sender, allowlistCfg)),
            );
            if (!hasTrigger) continue;
          }

          // Pull all messages since lastAgentTimestamp so non-trigger
          // context that accumulated between triggers is included.
          const allPending = getMessagesSince(
            chatJid,
            lastAgentTimestamp[cursorKey] || '',
            ASSISTANT_NAME,
            activeThreadId,
          );
          const messagesToSend =
            allPending.length > 0 ? allPending : scopedGroupMessages;
          const formatted = formatMessages(messagesToSend);

          if (queue.sendMessage(chatJid, formatted)) {
            logger.debug(
              { chatJid, count: messagesToSend.length },
              'Piped messages to active container',
            );
            lastAgentTimestamp[cursorKey] =
              messagesToSend[messagesToSend.length - 1].timestamp;
            saveState();
            // Show typing indicator while the container processes the piped message
            channel
              .setTyping?.(chatJid, true)
              ?.catch((err) =>
                logger.warn({ chatJid, err }, 'Failed to set typing indicator'),
              );
          } else {
            // No active container — enqueue for a new one
            queue.enqueueMessageCheck(chatJid);
          }
        }
      }
    } catch (err) {
      logger.error({ err }, 'Error in message loop');
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));
  }
}

/**
 * Startup recovery: check for unprocessed messages in registered groups.
 * Handles crash between advancing lastTimestamp and processing messages.
 */
function recoverPendingMessages(): void {
  for (const [chatJid, group] of Object.entries(registeredGroups)) {
    const activeThreadId =
      group.folder === WEB_CHAT_FOLDER ? getActiveWebChatThreadId() : undefined;
    const cursorKey = getAgentCursorKey(chatJid, activeThreadId);
    const sinceTimestamp = lastAgentTimestamp[cursorKey] || '';
    const pending = getMessagesSince(
      chatJid,
      sinceTimestamp,
      ASSISTANT_NAME,
      activeThreadId,
    );
    if (pending.length > 0) {
      logger.info(
        { group: group.name, pendingCount: pending.length },
        'Recovery: found unprocessed messages',
      );
      queue.enqueueMessageCheck(chatJid);
    }
  }
}

function ensureContainerSystemRunning(): void {
  ensureContainerRuntimeRunning();
  ensureImageAvailable();
  cleanupOrphans();
}

async function main(): Promise<void> {
  ensureContainerSystemRunning();
  initDatabase();
  logger.info('Database initialized');
  loadState();
  syncWebChatRuntimeSession(getActiveWebChatThreadId());

  // Graceful shutdown handlers
  const shutdown = async (signal: string) => {
    logger.info({ signal }, 'Shutdown signal received');
    await queue.shutdown(10000);
    for (const ch of channels) await ch.disconnect();
    process.exit(0);
  };
  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));

  // Channel callbacks (shared by all channels)
  const channelOpts = {
    onMessage: (chatJid: string, msg: NewMessage) => {
      // Sender allowlist drop mode: discard messages from denied senders before storing
      if (!msg.is_from_me && !msg.is_bot_message && registeredGroups[chatJid]) {
        const cfg = loadSenderAllowlist();
        if (
          shouldDropMessage(chatJid, cfg) &&
          !isSenderAllowed(chatJid, msg.sender, cfg)
        ) {
          if (cfg.logDenied) {
            logger.debug(
              { chatJid, sender: msg.sender },
              'sender-allowlist: dropping message (drop mode)',
            );
          }
          return;
        }
      }
      storeMessage(msg);
    },
    onChatMetadata: (
      chatJid: string,
      timestamp: string,
      name?: string,
      channel?: string,
      isGroup?: boolean,
    ) => storeChatMetadata(chatJid, timestamp, name, channel, isGroup),
    registeredGroups: () => registeredGroups,
  };

  // Create and connect all registered channels.
  // Each channel self-registers via the barrel import above.
  // Factories return null when credentials are missing, so unconfigured channels are skipped.
  for (const channelName of getRegisteredChannelNames()) {
    const factory = getChannelFactory(channelName)!;
    const channel = factory(channelOpts);
    if (!channel) {
      logger.warn(
        { channel: channelName },
        'Channel installed but credentials missing — skipping. Check .env or re-run the channel skill.',
      );
      continue;
    }
    channels.push(channel);
    await channel.connect();
  }
  if (channels.length === 0) {
    logger.fatal('No channels connected');
    process.exit(1);
  }

  // Start subsystems (independently of connection handler)
  startSchedulerLoop({
    registeredGroups: () => registeredGroups,
    getSessions: () => sessions,
    queue,
    onProcess: (groupJid, proc, containerName, groupFolder) =>
      queue.registerProcess(groupJid, proc, containerName, groupFolder),
    sendMessage: async (jid, rawText) => {
      const channel = findChannel(channels, jid);
      if (!channel) {
        logger.warn({ jid }, 'No channel owns JID, cannot send message');
        return;
      }
      const text = formatOutbound(rawText);
      if (text) await channel.sendMessage(jid, text);
    },
  });
  startIpcWatcher({
    sendMessage: (jid, text) => {
      const channel = findChannel(channels, jid);
      if (!channel) throw new Error(`No channel for JID: ${jid}`);
      return channel.sendMessage(jid, text);
    },
    registeredGroups: () => registeredGroups,
    registerGroup,
    syncGroups: async (force: boolean) => {
      await Promise.all(
        channels
          .filter((ch) => ch.syncGroups)
          .map((ch) => ch.syncGroups!(force)),
      );
    },
    getAvailableGroups,
    writeGroupsSnapshot: (gf, im, ag, rj) =>
      writeGroupsSnapshot(gf, im, ag, rj),
  });
  queue.setProcessMessagesFn(processGroupMessages);
  recoverPendingMessages();
  startMessageLoop().catch((err) => {
    logger.fatal({ err }, 'Message loop crashed unexpectedly');
    process.exit(1);
  });

  // Start monitoring dashboard (with channelOpts for web chat)
  startDashboard(channelOpts, {
    listSessions: () => listWebChatThreads(),
    getActiveThreadId: () => getActiveWebChatThreadId(),
    createSession: () => {
      const thread = createWebChatThread();
      activateWebChatThread(thread.id);
      return getWebChatThread(thread.id)!;
    },
    switchSession: (threadId: string) => {
      activateWebChatThread(threadId);
      return getWebChatThread(threadId);
    },
    renameSession: (threadId: string, title: string) =>
      renameWebChatThread(threadId, title),
    deleteSession: (threadId: string) =>
      deleteAndReplaceWebChatThread(threadId),
  });
}

// Guard: only run when executed directly, not when imported by tests
const isDirectRun =
  process.argv[1] &&
  new URL(import.meta.url).pathname ===
    new URL(`file://${process.argv[1]}`).pathname;

if (isDirectRun) {
  main().catch((err) => {
    logger.error({ err }, 'Failed to start MatClaw');
    process.exit(1);
  });
}
