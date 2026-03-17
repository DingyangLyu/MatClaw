/**
 * Agent engine abstraction for MatClaw.
 * Allows swapping between Claude Agent SDK and Codex SDK (or any OpenAI-compatible provider).
 */

export interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

export interface QueryResult {
  newSessionId?: string;
  lastAssistantUuid?: string;
  closedDuringQuery: boolean;
}

export interface EngineContext {
  mcpServerPath: string;
  chatJid: string;
  groupFolder: string;
  isMain: boolean;
  assistantName?: string;
  sdkEnv: Record<string, string | undefined>;
  writeOutput: (output: ContainerOutput) => void;
  log: (message: string) => void;
  shouldClose: () => boolean;
  drainIpcInput: () => string[];
  refreshSdkEnv: (env: Record<string, string | undefined>) => void;
}

export interface AgentEngine {
  readonly name: string;
  runQuery(
    prompt: string,
    sessionId: string | undefined,
    ctx: EngineContext,
    resumeAt?: string,
  ): Promise<QueryResult>;
}
