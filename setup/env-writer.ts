/**
 * Reusable .env file read/update/append utility.
 * Preserves comments, ordering, and blank lines.
 * Uses atomic write (write to .env.tmp, then rename).
 */
import fs from 'fs';
import path from 'path';

function envPath(projectRoot: string): string {
  return path.join(projectRoot, '.env');
}

/**
 * Update existing keys or append new key-value pairs to the .env file.
 */
export function writeEnvKeys(
  projectRoot: string,
  updates: Record<string, string>,
  sectionComment?: string,
): void {
  const file = envPath(projectRoot);
  let content = '';
  try {
    content = fs.readFileSync(file, 'utf-8');
  } catch {
    // File doesn't exist yet — start fresh
  }

  const toAppend: Record<string, string> = {};

  for (const [key, value] of Object.entries(updates)) {
    const regex = new RegExp(`^${key}=.*$`, 'm');
    if (regex.test(content)) {
      content = content.replace(regex, `${key}=${quoteValue(value)}`);
    } else {
      toAppend[key] = value;
    }
  }

  if (Object.keys(toAppend).length > 0) {
    // Ensure trailing newline before appending
    if (content && !content.endsWith('\n')) {
      content += '\n';
    }
    content += '\n';
    if (sectionComment) {
      content += `# ${sectionComment}\n`;
    }
    for (const [key, value] of Object.entries(toAppend)) {
      content += `${key}=${quoteValue(value)}\n`;
    }
  }

  // Atomic write
  const tmpFile = file + '.tmp';
  fs.writeFileSync(tmpFile, content);
  fs.renameSync(tmpFile, file);
}

/**
 * Remove specified keys from the .env file.
 */
export function removeEnvKeys(projectRoot: string, keys: string[]): void {
  const file = envPath(projectRoot);
  let content: string;
  try {
    content = fs.readFileSync(file, 'utf-8');
  } catch {
    return; // Nothing to remove
  }

  for (const key of keys) {
    content = content.replace(new RegExp(`^${key}=.*\n?`, 'gm'), '');
  }

  const tmpFile = file + '.tmp';
  fs.writeFileSync(tmpFile, content);
  fs.renameSync(tmpFile, file);
}

/**
 * Quote value if it contains spaces or special characters.
 */
function quoteValue(value: string): string {
  if (value.includes(' ') || value.includes('#') || value.includes('"')) {
    return `"${value.replace(/"/g, '\\"')}"`;
  }
  return value;
}
