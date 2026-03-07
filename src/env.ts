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
  if (!result.ANTHROPIC_API_KEY && !result.CLAUDE_CODE_OAUTH_TOKEN) {
    const token = readClaudeOAuthToken();
    if (token) {
      result.CLAUDE_CODE_OAUTH_TOKEN = token;
    }
  }

  return result;
}

/**
 * Read OAuth access token from Claude Code's credentials file.
 * Returns undefined if not available.
 */
function readClaudeOAuthToken(): string | undefined {
  const credFile = path.join(os.homedir(), '.claude', '.credentials.json');
  try {
    const data = JSON.parse(fs.readFileSync(credFile, 'utf-8'));
    const oauth = data.claudeAiOauth;
    if (!oauth?.accessToken) return undefined;

    // Check expiry if available
    if (oauth.expiresAt) {
      const expiresMs =
        oauth.expiresAt > 1e12 ? oauth.expiresAt : oauth.expiresAt * 1000;
      if (Date.now() > expiresMs) {
        logger.warn('Claude OAuth token expired, skipping');
        return undefined;
      }
    }

    logger.info('Using Claude OAuth token from ~/.claude/.credentials.json');
    return oauth.accessToken;
  } catch {
    return undefined;
  }
}
