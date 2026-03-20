import fs from 'fs';
import os from 'os';
import path from 'path';
import { logger } from './logger.js';

/**
 * Parse the .env file and return values for the requested keys.
 * Does NOT load anything into process.env — callers decide what to
 * do with the values. This keeps secrets out of the process environment
 * so they don't leak to child processes.
 */
export function readEnvFile(keys: string[]): Record<string, string> {
  const envFile = path.join(process.cwd(), '.env');
  let content: string;
  try {
    content = fs.readFileSync(envFile, 'utf-8');
  } catch (err) {
    logger.debug({ err }, '.env file not found, using defaults');
    return {};
  }

  const result: Record<string, string> = {};
  const wanted = new Set(keys);

  for (const line of content.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eqIdx = trimmed.indexOf('=');
    if (eqIdx === -1) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    if (!wanted.has(key)) continue;
    let value = trimmed.slice(eqIdx + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (value) result[key] = value;
  }

  // Fallback: if no API key or OAuth token configured, try reading
  // from Claude Code's credentials file (~/.claude/.credentials.json).
  // This lets developers use their Claude Max/Pro subscription without
  // manually copying tokens, while distributed users just set API keys.
  // Check the raw file content (not just the result) — callers may not
  // request ANTHROPIC_API_KEY but it could still be set in .env.
  const hasApiKey =
    result.ANTHROPIC_API_KEY || /^ANTHROPIC_API_KEY=.+/m.test(content);
  const hasOAuthToken =
    result.CLAUDE_CODE_OAUTH_TOKEN ||
    /^CLAUDE_CODE_OAUTH_TOKEN=.+/m.test(content);
  if (!hasApiKey && !hasOAuthToken) {
    const token = readClaudeOAuthToken();
    if (token) {
      result.CLAUDE_CODE_OAUTH_TOKEN = token;
    }
  }

  return result;
}

/**
 * Read OAuth access token from Claude Code's credentials file.
 * Searches multiple known locations across platforms:
 *   - ~/.claude/.credentials.json (Linux/macOS, Claude Code CLI)
 *   - ~/.claude/credentials.json  (alternate naming)
 *   - ~/Library/Application Support/Claude/credentials.json (macOS app)
 *   - ~/.config/claude/credentials.json (XDG Linux)
 * Returns undefined if not available.
 */
function readClaudeOAuthToken(): string | undefined {
  const home = os.homedir();
  const candidates = [
    path.join(home, '.claude', '.credentials.json'),
    path.join(home, '.claude', 'credentials.json'),
    ...(process.platform === 'darwin'
      ? [path.join(home, 'Library', 'Application Support', 'Claude', 'credentials.json')]
      : []),
    path.join(home, '.config', 'claude', 'credentials.json'),
  ];

  for (const credFile of candidates) {
    try {
      if (!fs.existsSync(credFile)) continue;
      const data = JSON.parse(fs.readFileSync(credFile, 'utf-8'));

      // Try multiple known token locations in the JSON structure
      const oauth = data.claudeAiOauth ?? data.oauth ?? data;
      const token = oauth?.accessToken ?? oauth?.access_token;
      if (!token) continue;

      // Check expiry if available
      const expiresAt = oauth.expiresAt ?? oauth.expires_at;
      if (expiresAt) {
        const expiresMs = expiresAt > 1e12 ? expiresAt : expiresAt * 1000;
        if (Date.now() > expiresMs) {
          logger.warn({ file: credFile }, 'Claude OAuth token expired, skipping');
          continue;
        }
      }

      logger.info({ file: credFile }, 'Using Claude OAuth token');
      return token;
    } catch {
      // Try next candidate
    }
  }

  return undefined;
}
