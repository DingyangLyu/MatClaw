#!/usr/bin/env npx tsx
/**
 * MatClaw Interactive Setup Wizard
 *
 * Claude Code–inspired terminal UI with box-drawn panels, animated
 * progress, gradient colors, and a polished feel.
 *
 * Usage: npm run setup
 */
import { execSync, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { select, confirm, checkbox } from '@inquirer/prompts';
import ora from 'ora';

// ── ANSI ────────────────────────────────────────────────────────────────────

const ESC = '\x1b';
const c = {
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
  // 256-colour for gradient
  fg: (n: number) => `${ESC}[38;5;${n}m`,
};

const cols = Math.min(process.stdout.columns || 80, 76);

// ── Utility ─────────────────────────────────────────────────────────────────

function println(text = ''): void {
  console.log(text);
}

function clearScreen(): void {
  process.stdout.write(`${ESC}[2J${ESC}[H`);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function commandExists(cmd: string): boolean {
  try { execSync(`command -v ${cmd}`, { stdio: 'ignore' }); return true; } catch { return false; }
}

function dockerRunning(): boolean {
  try { execSync('docker info', { stdio: 'ignore' }); return true; } catch { return false; }
}

function stripAnsi(s: string): string {
  return s.replace(/\x1b\[[0-9;]*m/g, '');
}

// ── Box Drawing ─────────────────────────────────────────────────────────────

const BOX_W = cols;

function boxTop(title?: string): string {
  if (title) {
    const t = ` ${title} `;
    const vis = stripAnsi(t).length;
    const remaining = BOX_W - 2 - vis;
    const left = 2;
    const right = Math.max(0, remaining - left);
    return `  ${c.dim}╭${'─'.repeat(left)}${c.reset}${c.bold}${c.brightCyan}${t}${c.reset}${c.dim}${'─'.repeat(right)}╮${c.reset}`;
  }
  return `  ${c.dim}╭${'─'.repeat(BOX_W - 2)}╮${c.reset}`;
}

function boxLine(text: string): string {
  const vis = stripAnsi(text).length;
  const pad = Math.max(0, BOX_W - 4 - vis);
  return `  ${c.dim}│${c.reset} ${text}${' '.repeat(pad)} ${c.dim}│${c.reset}`;
}

function boxEmpty(): string {
  return `  ${c.dim}│${' '.repeat(BOX_W - 2)}│${c.reset}`;
}

function boxBottom(): string {
  return `  ${c.dim}╰${'─'.repeat(BOX_W - 2)}╯${c.reset}`;
}

function boxDivider(): string {
  return `  ${c.dim}├${'─'.repeat(BOX_W - 2)}┤${c.reset}`;
}

// ── Gradient Text ───────────────────────────────────────────────────────────

// 256-colour gradient: red → magenta → cyan → blue
const GRADIENT = [196, 197, 198, 199, 200, 164, 128, 92, 56, 57, 63, 69, 75, 81, 45, 39, 33, 27];

function gradient(text: string): string {
  const chars = [...text];
  return chars.map((ch, i) => {
    if (ch === ' ') return ch;
    const idx = Math.floor((i / Math.max(chars.length - 1, 1)) * (GRADIENT.length - 1));
    return `${c.fg(GRADIENT[idx])}${c.bold}${ch}${c.reset}`;
  }).join('');
}

// ── Animated Typing ─────────────────────────────────────────────────────────

async function typewrite(text: string, delay = 12): Promise<void> {
  for (const ch of text) {
    process.stdout.write(ch);
    await sleep(delay);
  }
  println();
}

// ── Banner ──────────────────────────────────────────────────────────────────

async function printBanner(): Promise<void> {
  const logo = [
    '  ███╗   ███╗ █████╗ ████████╗ ██████╗██╗      █████╗ ██╗    ██╗',
    '  ████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██║     ██╔══██╗██║    ██║',
    '  ██╔████╔██║███████║   ██║   ██║     ██║     ███████║██║ █╗ ██║',
    '  ██║╚██╔╝██║██╔══██║   ██║   ██║     ██║     ██╔══██║██║███╗██║',
    '  ██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╗███████╗██║  ██║╚███╔███╔╝',
    '  ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ ',
  ];

  for (const line of logo) {
    println(gradient(line));
    await sleep(40);
  }
  println();
  await typewrite(`  ${c.dim}AI-Powered Autonomous Materials Science Agent${c.reset}`, 15);
  println();
}

// ── Step Header ─────────────────────────────────────────────────────────────

const TOTAL_STEPS = 7;

function stepHeader(n: number, title: string): void {
  const filled = n;
  const empty = TOTAL_STEPS - n;
  // Gradient progress bar
  let bar = '';
  for (let i = 0; i < filled; i++) {
    const idx = Math.floor((i / Math.max(TOTAL_STEPS - 1, 1)) * (GRADIENT.length - 1));
    bar += `${c.fg(GRADIENT[idx])}━${c.reset}`;
  }
  bar += `${c.dim}${'╌'.repeat(empty)}${c.reset}`;

  println(boxTop(`Step ${n}/${TOTAL_STEPS}`));
  println(boxLine(`${bar}  ${c.bold}${c.brightWhite}${title}${c.reset}`));
  println(boxDivider());
}

function stepFooter(): void {
  println(boxBottom());
  println();
}

// ── Status Icons ────────────────────────────────────────────────────────────

function ok(msg: string): void {
  println(boxLine(`${c.brightGreen}✔${c.reset}  ${msg}`));
}

function warn(msg: string): void {
  println(boxLine(`${c.brightYellow}⚠${c.reset}  ${msg}`));
}

function fail(msg: string): void {
  println(boxLine(`${c.brightRed}✖${c.reset}  ${msg}`));
}

function info(msg: string): void {
  println(boxLine(`${c.brightCyan}│${c.reset}  ${msg}`));
}

// ── Step 1: Environment ─────────────────────────────────────────────────────

async function step1(): Promise<{ docker: boolean }> {
  stepHeader(1, 'Environment');

  const checks: { name: string; cmd: string; getVer: () => string | null; required: boolean }[] = [
    { name: 'Node.js', cmd: 'node', required: true,
      getVer: () => { try { return execSync('node --version', { encoding: 'utf-8' }).trim(); } catch { return null; } } },
    { name: 'npm', cmd: 'npm', required: true,
      getVer: () => { try { return execSync('npm --version', { encoding: 'utf-8' }).trim(); } catch { return null; } } },
    { name: 'Docker', cmd: 'docker', required: false,
      getVer: () => { try { return execSync('docker --version', { encoding: 'utf-8' }).trim().replace('Docker version ', '').split(',')[0]; } catch { return null; } } },
    { name: 'Git', cmd: 'git', required: false,
      getVer: () => { try { return execSync('git --version', { encoding: 'utf-8' }).trim().replace('git version ', ''); } catch { return null; } } },
  ];

  let nodeOk = false;
  let dockerOk = false;

  for (const chk of checks) {
    const exists = commandExists(chk.cmd);
    const ver = exists ? chk.getVer() : null;
    await sleep(80);

    if (exists) {
      ok(`${c.bold}${chk.name}${c.reset} ${c.dim}${ver ?? ''}${c.reset}`);
      if (chk.cmd === 'node') nodeOk = true;
      if (chk.cmd === 'docker') dockerOk = true;
    } else if (chk.required) {
      fail(`${chk.name} ${c.red}— required${c.reset}`);
    } else {
      warn(`${chk.name} ${c.dim}— not found (optional)${c.reset}`);
    }
  }

  if (dockerOk) {
    await sleep(80);
    if (dockerRunning()) {
      ok(`Docker daemon ${c.dim}running${c.reset}`);
    } else {
      warn('Docker daemon not running');
      dockerOk = false;
    }
  }

  stepFooter();

  if (!nodeOk) {
    println(`  ${c.brightRed}Node.js is required. Install from https://nodejs.org/${c.reset}`);
    process.exit(1);
  }

  return { docker: dockerOk };
}

// ── Step 2: Dependencies ────────────────────────────────────────────────────

async function step2(): Promise<void> {
  stepHeader(2, 'Dependencies');

  const nmExists = fs.existsSync(path.join(process.cwd(), 'node_modules'));
  if (nmExists) {
    try {
      const lockTime = fs.statSync(path.join(process.cwd(), 'package-lock.json')).mtimeMs;
      const nmTime = fs.statSync(path.join(process.cwd(), 'node_modules')).mtimeMs;
      if (nmTime > lockTime) {
        ok('All dependencies up to date');
        stepFooter();
        return;
      }
    } catch { /* proceed to install */ }
  }

  println(boxLine(`${c.dim}Running npm install...${c.reset}`));
  stepFooter();

  const spinner = ora({ text: 'Installing dependencies...', indent: 4 }).start();
  try {
    execSync('npm install', { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] });
    spinner.succeed('Dependencies installed');
  } catch {
    spinner.fail('npm install failed — run manually to see errors');
    process.exit(1);
  }
  println();
}

// ── Step 3: Container ───────────────────────────────────────────────────────

async function step3(hasDocker: boolean): Promise<void> {
  stepHeader(3, 'Container');

  if (!hasDocker) {
    warn('Docker not available — skipping');
    info(`${c.dim}Run ./container/build.sh later to set up${c.reset}`);
    stepFooter();
    return;
  }

  let imageExists = false;
  try { execSync('docker image inspect matclaw-agent:latest', { stdio: 'ignore' }); imageExists = true; } catch {}

  if (imageExists) {
    ok('Container image found');
    stepFooter();

    const rebuild = await confirm({ message: '  Rebuild container image?', default: false });
    if (!rebuild) return;
    println();
    stepHeader(3, 'Container — Rebuild');
  }

  stepFooter();

  const method = await select({
    message: '  Container setup method',
    choices: [
      { value: 'pull',       name: `  ${c.brightGreen}⬇  Pull pre-built${c.reset}      ${c.dim}fastest, ~2 min${c.reset}` },
      { value: 'build',      name: `  ${c.yellow}🔨 Build from source${c.reset}   ${c.dim}~10 min, compiles QE${c.reset}` },
      { value: 'build-cuda', name: `  ${c.magenta}🎮 Build with CUDA${c.reset}     ${c.dim}~15 min, GPU support${c.reset}` },
      { value: 'skip',       name: `  ${c.dim}⏭  Skip${c.reset}` },
    ],
  });

  println();

  if (method === 'skip') return;

  if (method === 'pull') {
    const spinner = ora({ text: 'Pulling ghcr.io/dingyanglyu/matclaw-agent:latest ...', indent: 4 }).start();
    try {
      execSync('docker pull ghcr.io/dingyanglyu/matclaw-agent:latest', { stdio: ['ignore', 'pipe', 'pipe'] });
      execSync('docker tag ghcr.io/dingyanglyu/matclaw-agent:latest matclaw-agent:latest', { stdio: 'ignore' });
      spinner.succeed('Image pulled and tagged');
    } catch {
      spinner.fail('Pull failed — try building from source');
    }
  } else {
    const cuda = method === 'build-cuda' ? ' --cuda' : '';
    println(`  ${c.dim}Building container${cuda}...${c.reset}`);
    try {
      execSync(`bash ${path.join(process.cwd(), 'container', 'build.sh')}${cuda}`, { cwd: process.cwd(), stdio: 'inherit' });
    } catch {
      println(`  ${c.brightRed}Build failed${c.reset}`);
    }
  }

  println();
}

// ── Step 4: API ─────────────────────────────────────────────────────────────

async function step4(): Promise<void> {
  stepHeader(4, 'API Provider');

  const envFile = path.join(process.cwd(), '.env');
  let configured = false;
  if (fs.existsSync(envFile)) {
    const txt = fs.readFileSync(envFile, 'utf-8');
    configured = /^(ANTHROPIC_API_KEY|OPENAI_API_KEY|CLAUDE_CODE_OAUTH_TOKEN)=/m.test(txt);
  }

  if (configured) {
    ok('API credentials configured');
    stepFooter();
    const redo = await confirm({ message: '  Reconfigure API provider?', default: false });
    if (!redo) return;
    println();
  } else {
    info('No API credentials detected');
    stepFooter();
  }

  println(`  ${c.dim}Launching API wizard...${c.reset}\n`);

  try {
    const child = spawn('npx', ['tsx', 'setup/index.ts', '--step', 'configure-api'], { cwd: process.cwd(), stdio: 'inherit' });
    await new Promise<void>((resolve, reject) => {
      child.on('close', code => code === 0 ? resolve() : reject(new Error(`exit ${code}`)));
    });
  } catch {
    println(`  ${c.brightYellow}Skipped or failed — run npm run setup:api later${c.reset}`);
  }

  println();
}

// ── Step 5: Smoke Test ──────────────────────────────────────────────────────

async function step5(hasDocker: boolean): Promise<void> {
  stepHeader(5, 'Smoke Test');

  if (!hasDocker) { warn('Docker not available — skipping'); stepFooter(); return; }

  try { execSync('docker image inspect matclaw-agent:latest', { stdio: 'ignore' }); }
  catch { warn('No container image — skipping'); stepFooter(); return; }

  info('Verifies: QE, LAMMPS, MACE, pymatgen, ASE, PyTorch, Node.js');
  stepFooter();

  const run = await confirm({ message: '  Run smoke test?', default: true });
  if (!run) return;

  println();
  const spinner = ora({ text: 'Running smoke test inside container...', indent: 4 }).start();

  try {
    const smokeTest = path.join(process.cwd(), 'container', 'smoke-test.py');
    if (!fs.existsSync(smokeTest)) { spinner.info('smoke-test.py not found'); return; }

    const output = execSync(
      'docker run --rm -v ./container/smoke-test.py:/tmp/smoke-test.py matclaw-agent:latest bash -c "python3 /tmp/smoke-test.py"',
      { encoding: 'utf-8', cwd: process.cwd(), timeout: 180000 },
    );

    if (output.includes('FAILED')) {
      spinner.warn('Some checks failed');
      println();
      for (const line of output.trim().split('\n').slice(-12)) {
        if (line.includes('PASS')) println(`    ${c.brightGreen}✔${c.reset} ${line.trim()}`);
        else if (line.includes('FAIL')) println(`    ${c.brightRed}✖${c.reset} ${line.trim()}`);
        else println(`    ${c.dim}${line.trim()}${c.reset}`);
      }
    } else {
      spinner.succeed('All smoke tests passed');
    }
  } catch {
    spinner.fail('Smoke test failed or timed out');
  }

  println();
}

// ── Step 6: Channels ────────────────────────────────────────────────────────

async function step6(): Promise<void> {
  stepHeader(6, 'Messaging Channels');
  info(`${c.dim}The Web UI at localhost:3210 works without any channel.${c.reset}`);
  info(`${c.dim}Channels let you chat from Feishu, Telegram, etc.${c.reset}`);
  stepFooter();

  const add = await confirm({ message: '  Set up messaging channels now?', default: false });
  if (!add) {
    println(`  ${c.dim}Add anytime with /add-* commands in claude CLI${c.reset}\n`);
    return;
  }

  println();
  const channels = await checkbox({
    message: '  Select channels',
    choices: [
      { value: 'feishu',   name: `  ${c.brightBlue}飞书 Feishu${c.reset}     ${c.dim}WebSocket, no public URL${c.reset}` },
      { value: 'dingtalk', name: `  ${c.brightBlue}钉钉 DingTalk${c.reset}   ${c.dim}Stream Mode, no public URL${c.reset}` },
      { value: 'telegram', name: `  ${c.brightCyan}Telegram${c.reset}         ${c.dim}Bot API${c.reset}` },
      { value: 'discord',  name: `  ${c.brightMagenta}Discord${c.reset}          ${c.dim}Bot with slash commands${c.reset}` },
      { value: 'slack',    name: `  ${c.brightYellow}Slack${c.reset}            ${c.dim}Socket Mode${c.reset}` },
      { value: 'gmail',    name: `  ${c.brightRed}Gmail${c.reset}            ${c.dim}Email-based tasks${c.reset}` },
      { value: 'whatsapp', name: `  ${c.brightGreen}WhatsApp${c.reset}         ${c.dim}QR code auth${c.reset}` },
    ],
  });

  if (channels.length === 0) {
    println(`  ${c.dim}No channels selected${c.reset}\n`);
    return;
  }

  println();
  const guide: Record<string, string> = {
    feishu: 'docs/feishu-setup.md → then claude /add-feishu',
    dingtalk: 'docs/dingtalk-setup.md → then claude /add-dingtalk',
    telegram: 'claude /add-telegram',
    discord: 'claude /add-discord',
    slack: 'claude /add-slack',
    gmail: 'claude /add-gmail',
    whatsapp: 'claude /add-whatsapp',
  };
  for (const ch of channels) {
    println(`  ${c.brightCyan}→${c.reset} ${c.bold}${ch}${c.reset}: ${c.dim}${guide[ch]}${c.reset}`);
  }
  println();
}

// ── Step 7: Launch ──────────────────────────────────────────────────────────

async function step7(): Promise<void> {
  // ── Completion banner ──
  println();
  println(boxTop());
  println(boxEmpty());

  const ready = '✨  MatClaw is ready!';
  const readyGrad = gradient(ready);
  const readyVis = stripAnsi(ready).length;
  const readyPad = Math.max(0, BOX_W - 4 - readyVis);
  println(`  ${c.dim}│${c.reset} ${readyGrad}${' '.repeat(readyPad)} ${c.dim}│${c.reset}`);

  println(boxEmpty());
  println(boxDivider());
  println(boxLine(`${c.brightCyan}Quick commands${c.reset}`));
  println(boxLine(`  ${c.bold}npm run dev${c.reset}            Start in development mode`));
  println(boxLine(`  ${c.bold}npm run setup:api${c.reset}      Reconfigure API provider`));
  println(boxLine(`  ${c.bold}./container/build.sh${c.reset}   Rebuild container`));
  println(boxDivider());
  println(boxLine(`${c.brightCyan}Chat commands${c.reset}`));
  println(boxLine(`  ${c.bold}/watch${c.reset}    See what the agent is doing`));
  println(boxLine(`  ${c.bold}/status${c.reset}   Check agent status`));
  println(boxLine(`  ${c.bold}/stop${c.reset}     Stop running agent`));
  println(boxLine(`  ${c.bold}/help${c.reset}     All commands`));
  println(boxDivider());
  println(boxLine(`${c.dim}https://github.com/DingyangLyu/MatClaw${c.reset}`));
  println(boxBottom());
  println();

  stepHeader(7, 'Launch');
  stepFooter();

  const action = await select({
    message: '  How to start MatClaw?',
    choices: [
      { value: 'dev',     name: `  ${c.brightGreen}▶  Start now${c.reset}            ${c.dim}npm run dev (foreground)${c.reset}` },
      { value: 'service', name: `  ${c.brightBlue}⚙  Install as service${c.reset}   ${c.dim}auto-start on boot${c.reset}` },
      { value: 'skip',    name: `  ${c.dim}⏭  Don't start yet${c.reset}` },
    ],
  });

  if (action === 'skip') {
    println(`\n  ${c.dim}Start anytime with: npm run dev${c.reset}\n`);
    return;
  }

  if (action === 'service') {
    println();
    const spinner = ora({ text: 'Setting up system service...', indent: 4 }).start();
    try {
      execSync('npx tsx setup/index.ts --step service', { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] });
      spinner.succeed('Service installed and started');
      println(`  ${c.dim}Manage: systemctl --user {start|stop|restart} matclaw${c.reset}\n`);
    } catch {
      spinner.fail('Service setup failed — start manually with npm run dev');
    }
    return;
  }

  // dev mode
  println();
  println(`  ${c.brightCyan}🌐 Web UI:${c.reset}     ${c.underline}http://localhost:3210${c.reset}`);
  println(`  ${c.dim}   Ctrl+C to stop${c.reset}`);
  println();

  const child = spawn('npm', ['run', 'dev'], { cwd: process.cwd(), stdio: 'inherit' });
  child.on('close', code => process.exit(code ?? 0));
  process.on('SIGINT', () => child.kill('SIGINT'));
  process.on('SIGTERM', () => child.kill('SIGTERM'));
  await new Promise(() => {});
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  clearScreen();
  await printBanner();
  await sleep(200);

  const { docker } = await step1();
  await sleep(200);

  await step2();
  await sleep(150);

  await step3(docker);
  await sleep(150);

  await step4();
  await sleep(150);

  await step5(docker);
  await sleep(150);

  await step6();
  await sleep(150);

  await step7();
}

main().catch(err => {
  console.error(`\n${c.brightRed}Setup failed:${c.reset}`, err.message);
  process.exit(1);
});
