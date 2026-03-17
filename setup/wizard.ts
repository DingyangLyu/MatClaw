#!/usr/bin/env npx tsx
/**
 * MatClaw Interactive Setup Wizard
 *
 * Claude Code–inspired terminal UI with box-drawn panels, animated
 * progress, gradient colors, i18n (EN/ZH), and a polished feel.
 *
 * Usage: npm run setup
 */
import { execSync, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { select, confirm, checkbox } from '@inquirer/prompts';
import ora from 'ora';

import {
  c, GRADIENT, BOX_W,
  println, clearScreen, sleep, gradient, typewrite,
  boxTop, boxLine, boxEmpty, boxBottom, boxDivider,
  stepHeader, stepFooter, ok, warn, fail, info,
} from './ui.js';
import { type Locale, setLocale, getLocale, t } from './i18n.js';

// ── Utility ─────────────────────────────────────────────────────────────────

function commandExists(cmd: string): boolean {
  try { execSync(`command -v ${cmd}`, { stdio: 'ignore' }); return true; } catch { return false; }
}

function dockerRunning(): boolean {
  try { execSync('docker info', { stdio: 'ignore' }); return true; } catch { return false; }
}

// ── Language Selector ───────────────────────────────────────────────────────

async function selectLanguage(): Promise<Locale> {
  // Show a minimal gradient header before language choice
  println();
  println(gradient('  MatClaw Setup'));
  println();

  const locale = await select<Locale>({
    message: `  ${t('lang.prompt')}`,
    choices: [
      { value: 'en', name: `  ${c.brightCyan}English${c.reset}` },
      { value: 'zh', name: `  ${c.brightCyan}中文${c.reset}` },
    ],
  });

  setLocale(locale);
  return locale;
}

// ── Banner ──────────────────────────────────────────────────────────────────

const TOTAL_STEPS = 7;

async function printBanner(): Promise<void> {
  const logo = [
    '  ███╗   ███╗ █████╗ ████████╗ ██████╗██╗      █████╗ ██╗    ██╗',
    '  ████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██║     ██╔══██╗██║    ██║',
    '  ██╔████╔██║███████║   ██║   ██║     ██║     ███████║██║ █╗ ██║',
    '  ██║╚██╔╝██║██╔══██║   ██║   ██║     ██║     ██╔══██║██║███╗██║',
    '  ██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╗███████╗██║  ██║╚███╔███╔╝',
    '  ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ ',
  ];

  clearScreen();
  for (const line of logo) {
    println(gradient(line));
    await sleep(40);
  }
  println();
  await typewrite(`  ${c.dim}${t('banner.tagline')}${c.reset}`, 15);
  println();
}

// ── Step 1: Environment ─────────────────────────────────────────────────────

async function step1(): Promise<{ docker: boolean }> {
  stepHeader(1, TOTAL_STEPS, t('step.environment'));

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
      fail(`${chk.name} ${c.red}— ${t('env.required')}${c.reset}`);
    } else {
      warn(`${chk.name} ${c.dim}— ${t('env.notFound')}${c.reset}`);
    }
  }

  if (dockerOk) {
    await sleep(80);
    if (dockerRunning()) {
      ok(`Docker daemon ${c.dim}${t('env.running')}${c.reset}`);
    } else {
      warn(t('env.daemonNotRunning'));
      dockerOk = false;
    }
  }

  stepFooter();

  if (!nodeOk) {
    println(`  ${c.brightRed}${t('env.nodeRequired')}${c.reset}`);
    process.exit(1);
  }

  return { docker: dockerOk };
}

// ── Step 2: Dependencies ────────────────────────────────────────────────────

async function step2(): Promise<void> {
  stepHeader(2, TOTAL_STEPS, t('step.dependencies'));

  const nmExists = fs.existsSync(path.join(process.cwd(), 'node_modules'));
  if (nmExists) {
    try {
      const lockTime = fs.statSync(path.join(process.cwd(), 'package-lock.json')).mtimeMs;
      const nmTime = fs.statSync(path.join(process.cwd(), 'node_modules')).mtimeMs;
      if (nmTime > lockTime) {
        ok(t('deps.upToDate'));
        stepFooter();
        return;
      }
    } catch { /* proceed to install */ }
  }

  info(`${c.dim}${t('deps.installing')}${c.reset}`);
  stepFooter();

  const spinner = ora({ text: t('deps.spinnerInstalling'), indent: 4 }).start();
  try {
    execSync('npm install', { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] });
    spinner.succeed(t('deps.installed'));
  } catch {
    spinner.fail(t('deps.failed'));
    process.exit(1);
  }
  println();
}

// ── Step 3: Container ───────────────────────────────────────────────────────

async function step3(hasDocker: boolean): Promise<void> {
  stepHeader(3, TOTAL_STEPS, t('step.container'));

  if (!hasDocker) {
    warn(t('container.notAvailable'));
    info(`${c.dim}${t('container.setupLater')}${c.reset}`);
    stepFooter();
    return;
  }

  let imageExists = false;
  try { execSync('docker image inspect matclaw-agent:latest', { stdio: 'ignore' }); imageExists = true; } catch {}

  if (imageExists) {
    ok(t('container.imageFound'));
    stepFooter();

    const rebuild = await confirm({ message: `  ${t('container.rebuild')}`, default: false });
    if (!rebuild) return;
    println();
    stepHeader(3, TOTAL_STEPS, `${t('step.container')} — Rebuild`);
  }

  stepFooter();

  const method = await select({
    message: `  ${t('container.method')}`,
    choices: [
      { value: 'pull',       name: `  ${c.brightGreen}⬇  ${t('container.pull')}${c.reset}      ${c.dim}${t('container.pullDesc')}${c.reset}` },
      { value: 'build',      name: `  ${c.yellow}🔨 ${t('container.build')}${c.reset}   ${c.dim}${t('container.buildDesc')}${c.reset}` },
      { value: 'build-cuda', name: `  ${c.magenta}🎮 ${t('container.buildCuda')}${c.reset}     ${c.dim}${t('container.buildCudaDesc')}${c.reset}` },
      { value: 'skip',       name: `  ${c.dim}⏭  ${t('container.skip')}${c.reset}` },
    ],
  });

  println();

  if (method === 'skip') return;

  if (method === 'pull') {
    const spinner = ora({ text: t('container.pulling'), indent: 4 }).start();
    try {
      execSync('docker pull ghcr.io/dingyanglyu/matclaw-agent:latest', { stdio: ['ignore', 'pipe', 'pipe'] });
      execSync('docker tag ghcr.io/dingyanglyu/matclaw-agent:latest matclaw-agent:latest', { stdio: 'ignore' });
      spinner.succeed(t('container.pulled'));
    } catch {
      spinner.fail(t('container.pullFailed'));
    }
  } else {
    const cuda = method === 'build-cuda' ? ' --cuda' : '';
    println(`  ${c.dim}${cuda ? t('container.buildingCuda') : t('container.building')}${c.reset}`);
    try {
      execSync(`bash ${path.join(process.cwd(), 'container', 'build.sh')}${cuda}`, { cwd: process.cwd(), stdio: 'inherit' });
    } catch {
      println(`  ${c.brightRed}${t('container.buildFailed')}${c.reset}`);
    }
  }

  println();
}

// ── Step 4: API ─────────────────────────────────────────────────────────────

async function step4(): Promise<void> {
  stepHeader(4, TOTAL_STEPS, t('step.api'));

  const envFile = path.join(process.cwd(), '.env');
  let configured = false;
  if (fs.existsSync(envFile)) {
    const txt = fs.readFileSync(envFile, 'utf-8');
    configured = /^(ANTHROPIC_API_KEY|OPENAI_API_KEY|CLAUDE_CODE_OAUTH_TOKEN)=/m.test(txt);
  }

  if (configured) {
    ok(t('api.configured'));
    stepFooter();
    const redo = await confirm({ message: `  ${t('api.reconfigure')}`, default: false });
    if (!redo) return;
    println();
  } else {
    info(t('api.notDetected'));
    stepFooter();
  }

  println(`  ${c.dim}${t('api.launching')}${c.reset}\n`);

  try {
    const child = spawn('npx', ['tsx', 'setup/index.ts', '--step', 'configure-api', '--lang', getLocale()], {
      cwd: process.cwd(),
      stdio: 'inherit',
      env: { ...process.env, MATCLAW_LANG: getLocale() },
    });
    await new Promise<void>((resolve, reject) => {
      child.on('close', code => code === 0 ? resolve() : reject(new Error(`exit ${code}`)));
    });
    ok(t('api.done'));
  } catch {
    println(`  ${c.brightYellow}${t('api.skipped')}${c.reset}`);
  }

  println();
}

// ── Step 5: Smoke Test ──────────────────────────────────────────────────────

async function step5(hasDocker: boolean): Promise<void> {
  stepHeader(5, TOTAL_STEPS, t('step.smoke'));

  if (!hasDocker) { warn(t('smoke.notAvailable')); stepFooter(); return; }

  try { execSync('docker image inspect matclaw-agent:latest', { stdio: 'ignore' }); }
  catch { warn(t('smoke.noImage')); stepFooter(); return; }

  info(t('smoke.verifies'));
  stepFooter();

  const run = await confirm({ message: `  ${t('smoke.run')}`, default: true });
  if (!run) return;

  println();
  const spinner = ora({ text: t('smoke.running'), indent: 4 }).start();

  try {
    const smokeTest = path.join(process.cwd(), 'container', 'smoke-test.py');
    if (!fs.existsSync(smokeTest)) { spinner.info(t('smoke.notFound')); return; }

    const output = execSync(
      'docker run --rm -v ./container/smoke-test.py:/tmp/smoke-test.py matclaw-agent:latest bash -c "python3 /tmp/smoke-test.py"',
      { encoding: 'utf-8', cwd: process.cwd(), timeout: 180000 },
    );

    if (output.includes('FAILED')) {
      spinner.warn(t('smoke.someFailed'));
      println();
      for (const line of output.trim().split('\n').slice(-12)) {
        if (line.includes('PASS')) println(`    ${c.brightGreen}✔${c.reset} ${line.trim()}`);
        else if (line.includes('FAIL')) println(`    ${c.brightRed}✖${c.reset} ${line.trim()}`);
        else println(`    ${c.dim}${line.trim()}${c.reset}`);
      }
    } else {
      spinner.succeed(t('smoke.allPassed'));
    }
  } catch {
    spinner.fail(t('smoke.failed'));
  }

  println();
}

// ── Step 6: Channels ────────────────────────────────────────────────────────

async function step6(): Promise<void> {
  stepHeader(6, TOTAL_STEPS, t('step.channels'));
  info(`${c.dim}${t('channels.webNote')}${c.reset}`);
  info(`${c.dim}${t('channels.channelNote')}${c.reset}`);
  stepFooter();

  const add = await confirm({ message: `  ${t('channels.setup')}`, default: false });
  if (!add) {
    println(`  ${c.dim}${t('channels.addLater')}${c.reset}\n`);
    return;
  }

  println();
  const channelDefs = [
    { value: 'feishu',   name: t('channels.feishu'),   desc: t('channels.feishuDesc'),   color: c.brightBlue },
    { value: 'dingtalk', name: t('channels.dingtalk'), desc: t('channels.dingtalkDesc'), color: c.brightBlue },
    { value: 'telegram', name: t('channels.telegram'), desc: t('channels.telegramDesc'), color: c.brightCyan },
    { value: 'discord',  name: t('channels.discord'),  desc: t('channels.discordDesc'),  color: c.brightMagenta },
    { value: 'slack',    name: t('channels.slack'),    desc: t('channels.slackDesc'),    color: c.brightYellow },
    { value: 'gmail',    name: t('channels.gmail'),    desc: t('channels.gmailDesc'),    color: c.brightRed },
    { value: 'whatsapp', name: t('channels.whatsapp'), desc: t('channels.whatsappDesc'), color: c.brightGreen },
  ];

  const channels = await checkbox({
    message: `  ${t('channels.select')}`,
    choices: channelDefs.map(ch => ({
      value: ch.value,
      name: `  ${ch.color}${ch.name}${c.reset}     ${c.dim}${ch.desc}${c.reset}`,
    })),
  });

  if (channels.length === 0) {
    println(`  ${c.dim}${t('channels.noneSelected')}${c.reset}\n`);
    return;
  }

  println();
  for (const ch of channels) {
    const guideKey = `channels.guide.${ch}` as const;
    println(`  ${c.brightCyan}→${c.reset} ${c.bold}${ch}${c.reset}: ${c.dim}${t(guideKey)}${c.reset}`);
  }
  println();
}

// ── Step 7: Launch ──────────────────────────────────────────────────────────

async function step7(): Promise<void> {
  // ── Completion banner ──
  println();
  println(boxTop());
  println(boxEmpty());

  const ready = `✨  ${t('launch.ready')}`;
  const readyGrad = gradient(ready);
  println(boxLine(readyGrad));

  println(boxEmpty());
  println(boxDivider());
  println(boxLine(`${c.brightCyan}${t('launch.quickCmds')}${c.reset}`));
  println(boxLine(`  ${c.bold}npm run dev${c.reset}            ${getLocale() === 'zh' ? '开发模式启动' : 'Start in development mode'}`));
  println(boxLine(`  ${c.bold}npm run setup:api${c.reset}      ${getLocale() === 'zh' ? '重新配置 API' : 'Reconfigure API provider'}`));
  println(boxLine(`  ${c.bold}./container/build.sh${c.reset}   ${getLocale() === 'zh' ? '重建容器' : 'Rebuild container'}`));
  println(boxDivider());
  println(boxLine(`${c.brightCyan}${t('launch.chatCmds')}${c.reset}`));
  println(boxLine(`  ${c.bold}/watch${c.reset}    ${t('launch.watchDesc')}`));
  println(boxLine(`  ${c.bold}/status${c.reset}   ${t('launch.statusDesc')}`));
  println(boxLine(`  ${c.bold}/stop${c.reset}     ${t('launch.stopDesc')}`));
  println(boxLine(`  ${c.bold}/help${c.reset}     ${t('launch.helpDesc')}`));
  println(boxDivider());
  println(boxLine(`${c.dim}https://github.com/DingyangLyu/MatClaw${c.reset}`));
  println(boxBottom());
  println();

  stepHeader(7, TOTAL_STEPS, t('step.launch'));
  stepFooter();

  const action = await select({
    message: `  ${t('launch.how')}`,
    choices: [
      { value: 'dev',     name: `  ${c.brightGreen}▶  ${t('launch.startNow')}${c.reset}            ${c.dim}${t('launch.startNowDesc')}${c.reset}` },
      { value: 'service', name: `  ${c.brightBlue}⚙  ${t('launch.service')}${c.reset}   ${c.dim}${t('launch.serviceDesc')}${c.reset}` },
      { value: 'skip',    name: `  ${c.dim}⏭  ${t('launch.dontStart')}${c.reset}` },
    ],
  });

  if (action === 'skip') {
    println(`\n  ${c.dim}${t('launch.startLater')}${c.reset}\n`);
    return;
  }

  if (action === 'service') {
    println();
    const spinner = ora({ text: t('launch.serviceInstalling'), indent: 4 }).start();
    try {
      execSync('npx tsx setup/index.ts --step service', { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] });
      spinner.succeed(t('launch.serviceInstalled'));
      println(`  ${c.dim}${t('launch.serviceManage')}${c.reset}\n`);
    } catch {
      spinner.fail(t('launch.serviceFailed'));
    }
    return;
  }

  // dev mode
  println();
  println(`  ${c.brightCyan}🌐 ${t('launch.webUI')}${c.reset}     ${c.underline}http://localhost:3210${c.reset}`);
  println(`  ${c.dim}   ${t('launch.ctrlC')}${c.reset}`);
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

  // Language selection first
  await selectLanguage();

  // Then the animated banner
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
