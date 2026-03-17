/**
 * MatClaw Agent Runner
 * Runs inside a container, receives config via stdin, outputs result to stdout
 *
 * Supports two agent engines (selected via AGENT_ENGINE env var):
 *   - claude (default): Uses @anthropic-ai/claude-agent-sdk
 *   - codex: Uses @openai/codex-sdk (any OpenAI-compatible API)
 *
 * Input protocol:
 *   Stdin: Full ContainerInput JSON (read until EOF)
 *   IPC:   Follow-up messages written as JSON files to /workspace/ipc/input/
 *          Files: {type:"message", text:"..."}.json — polled and consumed
 *          Sentinel: /workspace/ipc/input/_close — signals session end
 *
 * Stdout protocol:
 *   Each result is wrapped in OUTPUT_START_MARKER / OUTPUT_END_MARKER pairs.
 *   Multiple results may be emitted (one per agent turn).
 *   Final marker after loop ends signals completion.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import type { AgentEngine } from './engines/interface.js';

interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  assistantName?: string;
  secrets?: Record<string, string>;
}

interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

const IPC_INPUT_DIR = '/workspace/ipc/input';
const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
const IPC_SECRETS_PATH = '/workspace/ipc/_secrets.json';
const IPC_POLL_MS = 500;

async function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => { data += chunk; });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

const OUTPUT_START_MARKER = '---MATCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---MATCLAW_OUTPUT_END---';

function writeOutput(output: ContainerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);
}

function log(message: string): void {
  const ts = new Date().toISOString().slice(11, 23); // HH:mm:ss.SSS
  console.error(`[${ts}] ${message}`);
}

function shouldClose(): boolean {
  if (fs.existsSync(IPC_INPUT_CLOSE_SENTINEL)) {
    try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }
    return true;
  }
  return false;
}

function drainIpcInput(): string[] {
  try {
    fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
    const files = fs.readdirSync(IPC_INPUT_DIR)
      .filter(f => f.endsWith('.json'))
      .sort();

    const messages: string[] = [];
    for (const file of files) {
      const filePath = path.join(IPC_INPUT_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        fs.unlinkSync(filePath);
        if (data.type === 'message' && data.text) {
          messages.push(data.text);
        }
      } catch (err) {
        log(`Failed to process input file ${file}: ${err instanceof Error ? err.message : String(err)}`);
        try { fs.unlinkSync(filePath); } catch { /* ignore */ }
      }
    }
    return messages;
  } catch (err) {
    log(`IPC drain error: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

function waitForIpcMessage(): Promise<string | null> {
  return new Promise((resolve) => {
    const poll = () => {
      if (shouldClose()) {
        resolve(null);
        return;
      }
      const messages = drainIpcInput();
      if (messages.length > 0) {
        resolve(messages.join('\n'));
        return;
      }
      setTimeout(poll, IPC_POLL_MS);
    };
    poll();
  });
}

function refreshSdkEnv(sdkEnv: Record<string, string | undefined>): void {
  try {
    if (!fs.existsSync(IPC_SECRETS_PATH)) return;
    const secrets = JSON.parse(fs.readFileSync(IPC_SECRETS_PATH, 'utf-8'));
    let updated = false;
    for (const [key, value] of Object.entries(secrets)) {
      if (typeof value === 'string' && sdkEnv[key] !== value) {
        sdkEnv[key] = value;
        updated = true;
      }
    }
    if (updated) {
      log('Refreshed SDK secrets from IPC');
    }
  } catch {
    // Non-fatal: continue with existing secrets
  }
}

/**
 * Create the appropriate engine based on AGENT_ENGINE env var.
 * Uses dynamic import so only the selected engine's dependencies are loaded.
 */
async function createEngine(): Promise<AgentEngine> {
  const engineType = process.env['AGENT_ENGINE'] || 'claude';
  log(`Creating engine: ${engineType}`);

  switch (engineType) {
    case 'codex': {
      const { CodexEngine } = await import('./engines/codex.js');
      return new CodexEngine();
    }
    case 'gemini':
      throw new Error(
        'Gemini engine is not yet implemented. Use claude or codex for now.',
      );
    case 'claude':
    default: {
      const { ClaudeEngine } = await import('./engines/claude.js');
      return new ClaudeEngine();
    }
  }
}

async function main(): Promise<void> {
  let containerInput: ContainerInput;

  try {
    const stdinData = await readStdin();
    containerInput = JSON.parse(stdinData);
    // Delete the temp file the entrypoint wrote — it contains secrets
    try { fs.unlinkSync('/tmp/input.json'); } catch { /* may not exist */ }
    log(`Received input for group: ${containerInput.groupFolder}`);
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`
    });
    process.exit(1);
  }

  // Build SDK env: merge secrets into process.env for the SDK only.
  // Secrets never touch process.env itself, so Bash subprocesses can't see them.
  const sdkEnv: Record<string, string | undefined> = { ...process.env };
  for (const [key, value] of Object.entries(containerInput.secrets || {})) {
    sdkEnv[key] = value;
  }

  // Persist fresh secrets to IPC so refreshSdkEnv() doesn't overwrite with stale tokens
  if (containerInput.secrets && Object.keys(containerInput.secrets).length > 0) {
    try {
      fs.writeFileSync(IPC_SECRETS_PATH, JSON.stringify(containerInput.secrets));
    } catch {
      // Non-fatal: refreshSdkEnv will still work with stdin secrets in sdkEnv
    }
  }

  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const mcpServerPath = path.join(__dirname, 'ipc-mcp-stdio.js');

  let sessionId = containerInput.sessionId;
  fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });

  // Clean up stale _close sentinel from previous container runs
  try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }

  // Build initial prompt (drain any pending IPC messages too)
  let prompt = containerInput.prompt;
  if (containerInput.isScheduledTask) {
    prompt = `[SCHEDULED TASK - The following message was sent automatically and is not coming directly from the user or group.]\n\n${prompt}`;
  }
  const pending = drainIpcInput();
  if (pending.length > 0) {
    log(`Draining ${pending.length} pending IPC messages into initial prompt`);
    prompt += '\n' + pending.join('\n');
  }

  // Create the appropriate engine
  const engine = await createEngine();
  log(`Engine ready: ${engine.name}`);

  // Build engine context (shared callbacks for IPC, output, logging)
  const ctx = {
    mcpServerPath,
    chatJid: containerInput.chatJid,
    groupFolder: containerInput.groupFolder,
    isMain: containerInput.isMain,
    assistantName: containerInput.assistantName,
    sdkEnv,
    writeOutput,
    log,
    shouldClose,
    drainIpcInput,
    refreshSdkEnv,
  };

  // Query loop: run query → wait for IPC message → run new query → repeat
  let resumeAt: string | undefined;
  try {
    while (true) {
      // Refresh secrets before each query so we use the latest OAuth token
      refreshSdkEnv(sdkEnv);
      log(`Starting query (engine: ${engine.name}, session: ${sessionId || 'new'}, resumeAt: ${resumeAt || 'latest'})...`);

      const queryResult = await engine.runQuery(prompt, sessionId, ctx, resumeAt);
      if (queryResult.newSessionId) {
        sessionId = queryResult.newSessionId;
      }
      if (queryResult.lastAssistantUuid) {
        resumeAt = queryResult.lastAssistantUuid;
      }

      // If _close was consumed during the query, exit immediately.
      // Don't emit a session-update marker (it would reset the host's
      // idle timer and cause a 30-min delay before the next _close).
      if (queryResult.closedDuringQuery) {
        log('Close sentinel consumed during query, exiting');
        break;
      }

      // Emit session update so host can track it
      writeOutput({ status: 'success', result: null, newSessionId: sessionId });

      log('Query ended, waiting for next IPC message...');

      // Wait for the next message or _close sentinel
      const nextMessage = await waitForIpcMessage();
      if (nextMessage === null) {
        log('Close sentinel received, exiting');
        break;
      }

      log(`Got new message (${nextMessage.length} chars), starting new query`);
      prompt = nextMessage;
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    log(`Agent error: ${errorMessage}`);
    writeOutput({
      status: 'error',
      result: null,
      newSessionId: sessionId,
      error: errorMessage
    });
    process.exit(1);
  }
}

main();
