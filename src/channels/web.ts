/**
 * Web Chat Channel for MatClaw
 * Enables browser-based chat at http://localhost:3210 (Chat tab).
 * No credentials needed — always available.
 * Self-registers via registerChannel().
 */

import fs from 'fs';
import path from 'path';

import { ASSISTANT_NAME, GROUPS_DIR } from '../config.js';
import {
  getActiveWebChatThreadId,
  setRegisteredGroup,
  storeChatMetadata,
  storeMessageDirect,
  touchWebChatThread,
} from '../db.js';
import { logger } from '../logger.js';
import { Channel, NewMessage, RegisteredGroup } from '../types.js';
import { registerChannel, ChannelOpts } from './registry.js';

const WEB_JID = 'web:chat';
const WEB_FOLDER = 'web_chat';
const WEB_CHAT_DEFAULT_THREAD_ID = 'default';

// Broadcast function set by server.ts
let broadcastFn: ((msg: unknown) => void) | null = null;
let activeThreadId = WEB_CHAT_DEFAULT_THREAD_ID;

export function setWebBroadcast(fn: (msg: unknown) => void): void {
  broadcastFn = fn;
}

export function setWebChatActiveThread(threadId: string): void {
  activeThreadId = threadId || WEB_CHAT_DEFAULT_THREAD_ID;
}

export function getWebChatActiveThread(): string {
  return activeThreadId;
}

const DEFAULT_CLAUDE_MD = `# ${ASSISTANT_NAME}

You are ${ASSISTANT_NAME}, an AI materials scientist assistant. You help with computational materials science tasks including DFT calculations, molecular dynamics, Monte Carlo simulations, and materials property predictions.

## What You Can Do

- **Materials simulations** — Run DFT (Quantum ESPRESSO), MD (LAMMPS), MC (RASPA3) calculations
- **Machine learning potentials** — Use MACE and other MLIPs for accelerated simulations
- **Structure analysis** — Analyze crystal structures with pymatgen and ASE
- **Data analysis** — Process simulation results, plot data, compute properties
- Answer questions about materials science and computational methods
- Search the web and fetch content from URLs
- Read and write files in your workspace
- Run bash commands in your sandbox
- Schedule tasks to run later or on a recurring basis
- Send messages back to the chat

## Communication

Your output is sent to the user via the web chat interface.

You also have \`mcp__matclaw__send_message\` which sends a message immediately while you're still working.

### Internal thoughts

If part of your output is internal reasoning rather than something for the user, wrap it in \`<internal>\` tags:

\`\`\`
<internal>Compiled all three reports, ready to summarize.</internal>

Here are the key findings from the research...
\`\`\`

Text inside \`<internal>\` tags is logged but not sent to the user.

## Memory

The \`conversations/\` folder contains searchable history of past conversations. Use this to recall context from previous sessions.

When you learn something important:
- Create files for structured data (e.g., \`customers.md\`, \`preferences.md\`)
- Split files larger than 500 lines into folders

## Data Visualization

Always generate plots/figures to present results visually. Use matplotlib to create clear, publication-quality figures. Save plots as PNG files in your workspace and include them in your response using markdown image syntax:

\`\`\`
![K-point Convergence](k_convergence/convergence.png)
\`\`\`

Guidelines:
- Generate a plot for any numerical results (convergence tests, energy curves, DOS, band structures, RDFs, etc.)
- Use clear axis labels with units
- Add legends when multiple datasets are shown
- **CRITICAL: Always set a fixed figure size like \`figsize=(10, 6)\`. NEVER let figure height scale with data length. Max height: 2000 pixels.**
- Save as PNG with \`dpi=150\` for good quality
- If plotting long time series, downsample or use a rolling average
- Always include the plot in your final message to the user

## Sending Files

To send files (data, reports, etc.) to the user, include them in your response:

\`\`\`
[file:results/data.csv]
\`\`\`
`;

export class WebChannel implements Channel {
  name = 'web';

  private connected = false;
  private opts: ChannelOpts;

  constructor(opts: ChannelOpts) {
    this.opts = opts;
  }

  async connect(): Promise<void> {
    activeThreadId = getActiveWebChatThreadId();

    // Auto-register the web chat group if not already registered
    const groups = this.opts.registeredGroups();
    if (!groups[WEB_JID]) {
      const groupDir = path.join(GROUPS_DIR, WEB_FOLDER);
      fs.mkdirSync(path.join(groupDir, 'logs'), { recursive: true });

      const claudeMdPath = path.join(groupDir, 'CLAUDE.md');
      if (!fs.existsSync(claudeMdPath)) {
        fs.writeFileSync(claudeMdPath, DEFAULT_CLAUDE_MD);
      }

      const group: RegisteredGroup = {
        name: 'Web Chat',
        folder: WEB_FOLDER,
        trigger: `@${ASSISTANT_NAME}`,
        added_at: new Date().toISOString(),
        requiresTrigger: false,
        isMain: false,
      };
      setRegisteredGroup(WEB_JID, group);
      // Create chats row (FK target for messages table)
      storeChatMetadata(WEB_JID, new Date().toISOString(), 'Web Chat', 'web', false);
      // Also update in-memory map so message loop can find this group
      groups[WEB_JID] = group;
      logger.info('Web channel: registered web_chat group');
    }

    // Ensure chats row exists (FK target for messages table) — idempotent
    storeChatMetadata(WEB_JID, new Date().toISOString(), 'Web Chat', 'web', false);

    this.connected = true;
    logger.info('Web channel connected');
  }

  async sendMessage(_jid: string, text: string): Promise<void> {
    const now = new Date().toISOString();
    const msgId = `web_bot_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    // Store bot message in DB
    storeMessageDirect({
      id: msgId,
      chat_jid: WEB_JID,
      thread_id: activeThreadId,
      sender: 'assistant',
      sender_name: ASSISTANT_NAME,
      content: text,
      timestamp: now,
      is_from_me: true,
      is_bot_message: true,
    });
    touchWebChatThread(activeThreadId, text, now);

    // Broadcast to WS clients
    if (broadcastFn) {
      broadcastFn({
        type: 'chat:message',
        sender: 'assistant',
        text,
        timestamp: now,
        id: msgId,
        threadId: activeThreadId,
      });
    }
  }

  isConnected(): boolean {
    return this.connected;
  }

  ownsJid(jid: string): boolean {
    return jid === WEB_JID;
  }

  async disconnect(): Promise<void> {
    this.connected = false;
  }
}

/**
 * Handle an incoming web chat message from the browser.
 * Called by server.ts when a chat:send WS message arrives.
 */
export function handleWebChatMessage(
  text: string,
  opts: ChannelOpts,
  threadId = activeThreadId,
): void {
  const now = new Date().toISOString();
  const msgId = `web_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  activeThreadId = threadId || activeThreadId;
  touchWebChatThread(activeThreadId, text, now);

  const msg: NewMessage = {
    id: msgId,
    chat_jid: WEB_JID,
    thread_id: activeThreadId,
    sender: 'web_user',
    sender_name: 'User',
    content: text,
    timestamp: now,
    is_from_me: false,
    is_bot_message: false,
  };

  opts.onMessage(WEB_JID, msg);
}

// Self-register — always available, no credentials needed
registerChannel('web', (opts: ChannelOpts) => {
  return new WebChannel(opts);
});
