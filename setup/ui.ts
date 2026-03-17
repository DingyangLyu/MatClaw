/**
 * Shared terminal UI helpers for MatClaw setup tools.
 * Box-drawing, ANSI colors, gradient text, animated output.
 */

// ── ANSI ────────────────────────────────────────────────────────────────────

const ESC = '\x1b';

export const c = {
  reset: `${ESC}[0m`,
  bold: `${ESC}[1m`,
  dim: `${ESC}[2m`,
  italic: `${ESC}[3m`,
  underline: `${ESC}[4m`,
  red: `${ESC}[31m`,
  green: `${ESC}[32m`,
  yellow: `${ESC}[33m`,
  blue: `${ESC}[34m`,
  magenta: `${ESC}[35m`,
  cyan: `${ESC}[36m`,
  white: `${ESC}[37m`,
  brightRed: `${ESC}[91m`,
  brightGreen: `${ESC}[92m`,
  brightYellow: `${ESC}[93m`,
  brightBlue: `${ESC}[94m`,
  brightMagenta: `${ESC}[95m`,
  brightCyan: `${ESC}[96m`,
  brightWhite: `${ESC}[97m`,
  fg: (n: number) => `${ESC}[38;5;${n}m`,
};

// 256-colour gradient: red → magenta → cyan → blue
export const GRADIENT = [196, 197, 198, 199, 200, 164, 128, 92, 56, 57, 63, 69, 75, 81, 45, 39, 33, 27];

export const BOX_W = Math.min(process.stdout.columns || 80, 76);

// ── Helpers ─────────────────────────────────────────────────────────────────

export function println(text = ''): void {
  console.log(text);
}

export function clearScreen(): void {
  process.stdout.write(`${ESC}[2J${ESC}[H`);
}

export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function stripAnsi(s: string): string {
  return s.replace(/\x1b\[[0-9;]*m/g, '');
}

/**
 * Calculate visual width of a string, accounting for CJK double-width chars.
 */
export function stringWidth(s: string): number {
  const clean = stripAnsi(s);
  let w = 0;
  for (const ch of clean) {
    const cp = ch.codePointAt(0) ?? 0;
    // CJK Unified Ideographs, CJK Compatibility, Fullwidth forms, etc.
    if (
      (cp >= 0x1100 && cp <= 0x115f) || // Hangul Jamo
      (cp >= 0x2e80 && cp <= 0x303e) || // CJK Radicals
      (cp >= 0x3040 && cp <= 0x33bf) || // Hiragana, Katakana, CJK Compat
      (cp >= 0x3400 && cp <= 0x4dbf) || // CJK Unified Extension A
      (cp >= 0x4e00 && cp <= 0xa4cf) || // CJK Unified
      (cp >= 0xac00 && cp <= 0xd7af) || // Hangul Syllables
      (cp >= 0xf900 && cp <= 0xfaff) || // CJK Compat Ideographs
      (cp >= 0xfe30 && cp <= 0xfe6f) || // CJK Compat Forms
      (cp >= 0xff01 && cp <= 0xff60) || // Fullwidth Forms
      (cp >= 0xffe0 && cp <= 0xffe6) || // Fullwidth Signs
      (cp >= 0x20000 && cp <= 0x2fffd) || // CJK Extension B+
      (cp >= 0x30000 && cp <= 0x3fffd)    // CJK Extension G+
    ) {
      w += 2;
    } else {
      w += 1;
    }
  }
  return w;
}

// ── Box Drawing ─────────────────────────────────────────────────────────────

export function boxTop(title?: string): string {
  if (title) {
    const t = ` ${title} `;
    const vis = stringWidth(t);
    const remaining = BOX_W - 2 - vis;
    const left = 2;
    const right = Math.max(0, remaining - left);
    return `  ${c.dim}╭${'─'.repeat(left)}${c.reset}${c.bold}${c.brightCyan}${t}${c.reset}${c.dim}${'─'.repeat(right)}╮${c.reset}`;
  }
  return `  ${c.dim}╭${'─'.repeat(BOX_W - 2)}╮${c.reset}`;
}

export function boxLine(text: string): string {
  const vis = stringWidth(text);
  const pad = Math.max(0, BOX_W - 4 - vis);
  return `  ${c.dim}│${c.reset} ${text}${' '.repeat(pad)} ${c.dim}│${c.reset}`;
}

export function boxEmpty(): string {
  return `  ${c.dim}│${' '.repeat(BOX_W - 2)}│${c.reset}`;
}

export function boxBottom(): string {
  return `  ${c.dim}╰${'─'.repeat(BOX_W - 2)}╯${c.reset}`;
}

export function boxDivider(): string {
  return `  ${c.dim}├${'─'.repeat(BOX_W - 2)}┤${c.reset}`;
}

// ── Gradient ────────────────────────────────────────────────────────────────

export function gradient(text: string): string {
  const chars = [...text];
  return chars.map((ch, i) => {
    if (ch === ' ') return ch;
    const idx = Math.floor((i / Math.max(chars.length - 1, 1)) * (GRADIENT.length - 1));
    return `${c.fg(GRADIENT[idx])}${c.bold}${ch}${c.reset}`;
  }).join('');
}

// ── Animated Typing ─────────────────────────────────────────────────────────

export async function typewrite(text: string, delay = 12): Promise<void> {
  for (const ch of text) {
    process.stdout.write(ch);
    await sleep(delay);
  }
  println();
}

// ── Status Icons ────────────────────────────────────────────────────────────

export function ok(msg: string): void {
  println(boxLine(`${c.brightGreen}✔${c.reset}  ${msg}`));
}

export function warn(msg: string): void {
  println(boxLine(`${c.brightYellow}⚠${c.reset}  ${msg}`));
}

export function fail(msg: string): void {
  println(boxLine(`${c.brightRed}✖${c.reset}  ${msg}`));
}

export function info(msg: string): void {
  println(boxLine(`${c.brightCyan}│${c.reset}  ${msg}`));
}

// ── Step Panel ──────────────────────────────────────────────────────────────

export function stepHeader(n: number, total: number, title: string): void {
  let bar = '';
  for (let i = 0; i < n; i++) {
    const idx = Math.floor((i / Math.max(total - 1, 1)) * (GRADIENT.length - 1));
    bar += `${c.fg(GRADIENT[idx])}━${c.reset}`;
  }
  bar += `${c.dim}${'╌'.repeat(total - n)}${c.reset}`;

  println(boxTop(`Step ${n}/${total}`));
  println(boxLine(`${bar}  ${c.bold}${c.brightWhite}${title}${c.reset}`));
  println(boxDivider());
}

export function stepFooter(): void {
  println(boxBottom());
  println();
}

// ── Phase Panel (lighter, for sub-sections) ─────────────────────────────────

export function phaseHeader(title: string): void {
  println();
  println(boxTop(title));
}

export function phaseFooter(): void {
  println(boxBottom());
  println();
}
