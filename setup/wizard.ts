#!/usr/bin/env npx tsx
/**
 * MatClaw Interactive Setup Wizard
 *
 * A beautiful, guided CLI that walks users through the complete
 * installation process — from zero to running agent.
 *
 * Usage: npm run setup:wizard
 */
import { execSync, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { select, confirm, checkbox } from '@inquirer/prompts';
import ora from 'ora';

// ── ANSI Colors & Styles ────────────────────────────────────────────────────

const c = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
  italic: '\x1b[3m',
  underline: '\x1b[4m',
  // Colors
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m',
  // Bright
  brightRed: '\x1b[91m',
  brightGreen: '\x1b[92m',
  brightYellow: '\x1b[93m',
  brightBlue: '\x1b[94m',
  brightMagenta: '\x1b[95m',
  brightCyan: '\x1b[96m',
  // BG
  bgBlue: '\x1b[44m',
  bgMagenta: '\x1b[45m',
  bgCyan: '\x1b[46m',
};

// ── UI Helpers ──────────────────────────────────────────────────────────────

function clearScreen(): void {
  process.stdout.write('\x1b[2J\x1b[H');
}

function println(text = ''): void {
  console.log(text);
}

function printBanner(): void {
  const claw = `${c.brightRed}
    ╔╦╗┌─┐┌┬┐${c.brightCyan}╔═╗┬  ┌─┐┬ ┬${c.reset}
    ${c.brightRed}║║║├─┤ │ ${c.brightCyan}║  │  ├─┤│││${c.reset}
    ${c.brightRed}╩ ╩┴ ┴ ┴ ${c.brightCyan}╚═╝┴─┘┴ ┴└┴┘${c.reset}`;

  println(claw);
  println();
  println(`  ${c.dim}AI-Powered Autonomous Materials Science Agent${c.reset}`);
  println(`  ${c.dim}────────────────────────────────────────────${c.reset}`);
  println();
}

function printStep(current: number, total: number, title: string): void {
  const progress = '█'.repeat(current) + '░'.repeat(total - current);
  println(`  ${c.brightCyan}[${progress}]${c.reset} ${c.bold}Step ${current}/${total}: ${title}${c.reset}`);
  println();
}

function printSuccess(msg: string): void {
  println(`  ${c.brightGreen}✓${c.reset} ${msg}`);
}

function printWarning(msg: string): void {
  println(`  ${c.brightYellow}⚠${c.reset} ${msg}`);
}

function printError(msg: string): void {
  println(`  ${c.brightRed}✗${c.reset} ${msg}`);
}

function printInfo(msg: string): void {
  println(`  ${c.brightBlue}ℹ${c.reset} ${msg}`);
}

function printDivider(): void {
  println(`  ${c.dim}${'─'.repeat(56)}${c.reset}`);
}

function commandExists(cmd: string): boolean {
  try {
    execSync(`command -v ${cmd}`, { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

function dockerRunning(): boolean {
  try {
    execSync('docker info', { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ── Step 1: Environment Check ───────────────────────────────────────────────

async function step1_environment(): Promise<{ docker: boolean; node: boolean }> {
  printStep(1, 7, 'Environment Check');

  const checks = [
    { name: 'Node.js', cmd: 'node', version: () => {
      try { return execSync('node --version', { encoding: 'utf-8' }).trim(); } catch { return null; }
    }},
    { name: 'npm', cmd: 'npm', version: () => {
      try { return execSync('npm --version', { encoding: 'utf-8' }).trim(); } catch { return null; }
    }},
    { name: 'Docker', cmd: 'docker', version: () => {
      try { return execSync('docker --version', { encoding: 'utf-8' }).trim().replace('Docker version ', ''); } catch { return null; }
    }},
    { name: 'Git', cmd: 'git', version: () => {
      try { return execSync('git --version', { encoding: 'utf-8' }).trim().replace('git version ', ''); } catch { return null; }
    }},
  ];

  let nodeOk = false;
  let dockerOk = false;

  for (const check of checks) {
    const exists = commandExists(check.cmd);
    const ver = exists ? check.version() : null;

    if (exists) {
      printSuccess(`${check.name} ${c.dim}${ver || ''}${c.reset}`);
      if (check.cmd === 'node') nodeOk = true;
      if (check.cmd === 'docker') dockerOk = true;
    } else {
      if (check.cmd === 'node' || check.cmd === 'npm') {
        printError(`${check.name} — ${c.red}required${c.reset}`);
      } else {
        printWarning(`${check.name} — not found (optional)`);
      }
    }
  }

  if (dockerOk) {
    if (dockerRunning()) {
      printSuccess(`Docker daemon ${c.dim}running${c.reset}`);
    } else {
      printWarning('Docker installed but not running — please start Docker');
      dockerOk = false;
    }
  }

  println();

  if (!nodeOk) {
    printError('Node.js is required. Install from https://nodejs.org/');
    process.exit(1);
  }

  return { docker: dockerOk, node: nodeOk };
}

// ── Step 2: Install Dependencies ────────────────────────────────────────────

async function step2_dependencies(): Promise<void> {
  printStep(2, 7, 'Install Dependencies');

  const nodeModulesExists = fs.existsSync(path.join(process.cwd(), 'node_modules'));

  if (nodeModulesExists) {
    const pkgLockTime = fs.statSync(path.join(process.cwd(), 'package-lock.json')).mtimeMs;
    const nodeModTime = fs.statSync(path.join(process.cwd(), 'node_modules')).mtimeMs;

    if (nodeModTime > pkgLockTime) {
      printSuccess('Dependencies already installed and up to date');
      println();
      return;
    }
  }

  const spinner = ora({ text: 'Installing npm dependencies...', indent: 2 }).start();

  try {
    execSync('npm install', {
      cwd: process.cwd(),
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    spinner.succeed('Dependencies installed');
  } catch (err) {
    spinner.fail('npm install failed');
    printError('Run `npm install` manually to see the error');
    process.exit(1);
  }

  println();
}

// ── Step 3: Container Setup ─────────────────────────────────────────────────

async function step3_container(hasDocker: boolean): Promise<void> {
  printStep(3, 7, 'Container Setup');

  if (!hasDocker) {
    printWarning('Docker not available — skipping container setup');
    printInfo('You can set up the container later with: ./container/build.sh');
    println();
    return;
  }

  // Check if image already exists
  let imageExists = false;
  try {
    execSync('docker image inspect matclaw-agent:latest', { stdio: 'ignore' });
    imageExists = true;
  } catch {
    // Image not found
  }

  if (imageExists) {
    const rebuild = await confirm({
      message: 'Container image already exists. Rebuild?',
      default: false,
    });
    if (!rebuild) {
      printSuccess('Using existing container image');
      println();
      return;
    }
  }

  const method = await select({
    message: 'How would you like to get the container?',
    choices: [
      {
        value: 'pull',
        name: `${c.brightGreen}⬇  Pull pre-built image${c.reset} ${c.dim}(fastest, ~2 min)${c.reset}`,
      },
      {
        value: 'build',
        name: `${c.yellow}🔨 Build from source${c.reset} ${c.dim}(~10 min, compiles QE 7.5)${c.reset}`,
      },
      {
        value: 'build-cuda',
        name: `${c.magenta}🎮 Build with GPU/CUDA${c.reset} ${c.dim}(~15 min, requires NVIDIA toolkit)${c.reset}`,
      },
      {
        value: 'skip',
        name: `${c.dim}⏭  Skip for now${c.reset}`,
      },
    ],
  });

  if (method === 'skip') {
    printWarning('Skipped — run `./container/build.sh` later');
    println();
    return;
  }

  if (method === 'pull') {
    const spinner = ora({ text: 'Pulling ghcr.io/dingyanglyu/matclaw-agent:latest...', indent: 2 }).start();
    try {
      execSync('docker pull ghcr.io/dingyanglyu/matclaw-agent:latest', {
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      execSync('docker tag ghcr.io/dingyanglyu/matclaw-agent:latest matclaw-agent:latest', {
        stdio: 'ignore',
      });
      spinner.succeed('Container image pulled and tagged');
    } catch {
      spinner.fail('Pull failed — try building from source instead');
    }
  } else if (method === 'build' || method === 'build-cuda') {
    const buildArgs = method === 'build-cuda' ? '--cuda' : '';
    printInfo(`Building container${buildArgs ? ' with CUDA' : ''}... this may take a while`);
    println();

    try {
      const buildScript = path.join(process.cwd(), 'container', 'build.sh');
      execSync(`bash ${buildScript} ${buildArgs}`, {
        cwd: process.cwd(),
        stdio: 'inherit',
      });
      printSuccess('Container built successfully');
    } catch {
      printError('Build failed — check the output above for errors');
    }
  }

  println();
}

// ── Step 4: API Configuration ───────────────────────────────────────────────

async function step4_api(): Promise<void> {
  printStep(4, 7, 'API Provider Configuration');

  // Check if already configured
  const envPath = path.join(process.cwd(), '.env');
  let alreadyConfigured = false;
  if (fs.existsSync(envPath)) {
    const content = fs.readFileSync(envPath, 'utf-8');
    if (/^(ANTHROPIC_API_KEY|OPENAI_API_KEY|CLAUDE_CODE_OAUTH_TOKEN)=/m.test(content)) {
      alreadyConfigured = true;
    }
  }

  if (alreadyConfigured) {
    printSuccess('API credentials already configured');
    const reconfigure = await confirm({
      message: 'Reconfigure API provider?',
      default: false,
    });
    if (!reconfigure) {
      println();
      return;
    }
  }

  printInfo('Launching API configuration wizard...');
  println();

  // Run the existing configure-api wizard
  try {
    const child = spawn('npx', ['tsx', 'setup/index.ts', '--step', 'configure-api'], {
      cwd: process.cwd(),
      stdio: 'inherit',
    });
    await new Promise<void>((resolve, reject) => {
      child.on('close', (code) => {
        if (code === 0) resolve();
        else reject(new Error(`configure-api exited with code ${code}`));
      });
    });
    printSuccess('API provider configured');
  } catch {
    printWarning('API configuration skipped or failed');
    printInfo('You can configure later with: npm run setup:api');
  }

  println();
}

// ── Step 5: Smoke Test ──────────────────────────────────────────────────────

async function step5_smokeTest(hasDocker: boolean): Promise<void> {
  printStep(5, 7, 'Smoke Test');

  if (!hasDocker) {
    printWarning('Docker not available — skipping smoke test');
    println();
    return;
  }

  // Check if image exists
  try {
    execSync('docker image inspect matclaw-agent:latest', { stdio: 'ignore' });
  } catch {
    printWarning('No container image found — skipping smoke test');
    println();
    return;
  }

  const runTest = await confirm({
    message: 'Run container smoke test? (verifies QE, LAMMPS, MACE, Python packages)',
    default: true,
  });

  if (!runTest) {
    printWarning('Skipped');
    println();
    return;
  }

  const spinner = ora({ text: 'Running smoke test inside container...', indent: 2 }).start();

  try {
    const smokeTestPath = path.join(process.cwd(), 'container', 'smoke-test.py');
    if (fs.existsSync(smokeTestPath)) {
      const output = execSync(
        'docker run --rm -v ./container/smoke-test.py:/tmp/smoke-test.py matclaw-agent:latest bash -c "python3 /tmp/smoke-test.py"',
        { encoding: 'utf-8', cwd: process.cwd(), timeout: 120000 }
      );
      const passed = output.includes('PASSED');
      const failed = output.includes('FAILED');

      if (failed) {
        spinner.warn('Some tests failed');
        println();
        // Print last few lines
        const lines = output.trim().split('\n').slice(-12);
        for (const line of lines) {
          if (line.includes('PASS')) printSuccess(line.trim());
          else if (line.includes('FAIL')) printError(line.trim());
          else printInfo(line.trim());
        }
      } else {
        spinner.succeed('All smoke tests passed');
      }
    } else {
      spinner.info('Smoke test script not found — skipping');
    }
  } catch {
    spinner.fail('Smoke test failed or timed out');
  }

  println();
}

// ── Step 6: Messaging Channels ──────────────────────────────────────────────

async function step6_channels(): Promise<void> {
  printStep(6, 7, 'Messaging Channels (Optional)');

  println(`  ${c.dim}MatClaw can connect to messaging platforms so you can${c.reset}`);
  println(`  ${c.dim}chat with your agent from anywhere. The Web UI at${c.reset}`);
  println(`  ${c.dim}localhost:3210 is always available without any setup.${c.reset}`);
  println();

  const addChannels = await confirm({
    message: 'Set up messaging channels now?',
    default: false,
  });

  if (!addChannels) {
    printInfo('Skipped — you can add channels anytime with /add-* commands in claude CLI');
    println();
    return;
  }

  const channels = await checkbox({
    message: 'Select channels to configure:',
    choices: [
      { value: 'feishu',    name: `${c.brightBlue}飞书 (Feishu)${c.reset}       ${c.dim}— WebSocket, no public URL needed. Recommended for China users${c.reset}` },
      { value: 'dingtalk',  name: `${c.brightBlue}钉钉 (DingTalk)${c.reset}     ${c.dim}— Stream Mode, no public URL needed${c.reset}` },
      { value: 'telegram',  name: `${c.brightCyan}Telegram${c.reset}             ${c.dim}— Bot API, works worldwide${c.reset}` },
      { value: 'discord',   name: `${c.brightMagenta}Discord${c.reset}              ${c.dim}— Bot with slash commands${c.reset}` },
      { value: 'slack',     name: `${c.brightYellow}Slack${c.reset}                ${c.dim}— Socket Mode, no public URL needed${c.reset}` },
      { value: 'gmail',     name: `${c.brightRed}Gmail${c.reset}                ${c.dim}— Send tasks via email, receive results${c.reset}` },
      { value: 'whatsapp',  name: `${c.brightGreen}WhatsApp${c.reset}             ${c.dim}— QR code authentication${c.reset}` },
    ],
  });

  if (channels.length === 0) {
    printInfo('No channels selected');
    println();
    return;
  }

  println();
  printInfo('Channel setup requires the claude CLI. For each channel:');
  println();

  for (const ch of channels) {
    const cmdMap: Record<string, string> = {
      feishu: 'See docs/feishu-setup.md for Feishu app creation, then run claude and type /add-feishu',
      dingtalk: 'See docs/dingtalk-setup.md for DingTalk app creation, then run claude and type /add-dingtalk',
      telegram: 'Run claude and type: /add-telegram',
      discord: 'Run claude and type: /add-discord',
      slack: 'Run claude and type: /add-slack',
      gmail: 'Run claude and type: /add-gmail',
      whatsapp: 'Run claude and type: /add-whatsapp',
    };
    printInfo(`${c.bold}${ch}${c.reset}: ${cmdMap[ch]}`);
  }

  println();
}

// ── Step 7: Launch ──────────────────────────────────────────────────────────

async function step7_launch(): Promise<void> {
  printStep(7, 7, 'Launch MatClaw');

  const action = await select({
    message: 'How would you like to start MatClaw?',
    choices: [
      {
        value: 'dev',
        name: `${c.brightGreen}▶  Start now${c.reset} ${c.dim}(npm run dev — foreground, with hot reload)${c.reset}`,
      },
      {
        value: 'service',
        name: `${c.brightBlue}⚙  Install as service${c.reset} ${c.dim}(auto-start on boot — systemd/launchd)${c.reset}`,
      },
      {
        value: 'skip',
        name: `${c.dim}⏭  Don't start yet${c.reset}`,
      },
    ],
  });

  if (action === 'skip') {
    printInfo('You can start anytime with: npm run dev');
    println();
    return;
  }

  if (action === 'service') {
    const spinner = ora({ text: 'Setting up system service...', indent: 2 }).start();
    try {
      execSync('npx tsx setup/index.ts --step service', {
        cwd: process.cwd(),
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      spinner.succeed('Service installed and started');
      printInfo('Manage with: systemctl --user {start|stop|restart} matclaw');
    } catch {
      spinner.fail('Service setup failed');
      printInfo('Start manually with: npm run dev');
    }
    println();
    return;
  }

  // action === 'dev'
  println();
  printDivider();
  println();
  printSuccess(`${c.bold}Setup complete!${c.reset} Starting MatClaw...`);
  println();
  println(`  ${c.brightCyan}🌐 Web UI:${c.reset}     http://localhost:3210`);
  println(`  ${c.brightCyan}📊 Dashboard:${c.reset}  http://localhost:3210`);
  println(`  ${c.dim}   Press Ctrl+C to stop${c.reset}`);
  println();
  printDivider();
  println();

  // Hand off to npm run dev
  const child = spawn('npm', ['run', 'dev'], {
    cwd: process.cwd(),
    stdio: 'inherit',
  });

  child.on('close', (code) => {
    process.exit(code ?? 0);
  });

  // Forward signals
  process.on('SIGINT', () => child.kill('SIGINT'));
  process.on('SIGTERM', () => child.kill('SIGTERM'));

  // Keep process alive
  await new Promise(() => {});
}

// ── Completion Banner ───────────────────────────────────────────────────────

function printCompletionBanner(): void {
  println();
  printDivider();
  println();
  println(`  ${c.brightGreen}${c.bold}✨ MatClaw is ready!${c.reset}`);
  println();
  println(`  ${c.brightCyan}Quick commands:${c.reset}`);
  println(`    ${c.bold}npm run dev${c.reset}           Start in development mode`);
  println(`    ${c.bold}npm run setup:api${c.reset}     Reconfigure API provider`);
  println(`    ${c.bold}./container/build.sh${c.reset}  Rebuild container`);
  println();
  println(`  ${c.brightCyan}In chat:${c.reset}`);
  println(`    ${c.bold}/watch${c.reset}   See what the agent is doing`);
  println(`    ${c.bold}/status${c.reset}  Check agent status`);
  println(`    ${c.bold}/stop${c.reset}    Stop running agent`);
  println(`    ${c.bold}/help${c.reset}    All commands`);
  println();
  println(`  ${c.dim}Documentation: https://github.com/DingyangLyu/MatClaw${c.reset}`);
  println();
  printDivider();
  println();
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  clearScreen();
  printBanner();

  const { docker } = await step1_environment();
  await sleep(300);

  await step2_dependencies();
  await sleep(200);

  await step3_container(docker);
  await sleep(200);

  await step4_api();
  await sleep(200);

  await step5_smokeTest(docker);
  await sleep(200);

  await step6_channels();
  await sleep(200);

  printCompletionBanner();

  await step7_launch();
}

main().catch((err) => {
  console.error(`\n${c.brightRed}Setup failed:${c.reset}`, err.message);
  process.exit(1);
});
