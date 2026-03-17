/**
 * Feishu Bot Authentication Setup
 * Interactive guided setup with step-by-step instructions.
 * Run: npm run auth:feishu
 */

import fs from 'fs';
import path from 'path';
import * as Lark from '@larksuiteoapi/node-sdk';
import { input, password, confirm } from '@inquirer/prompts';
import ora from 'ora';

// Import shared UI (works when run from project root)
const ESC = '\x1b';
const c = {
  reset: `${ESC}[0m`,
  bold: `${ESC}[1m`,
  dim: `${ESC}[2m`,
  underline: `${ESC}[4m`,
  red: `${ESC}[31m`,
  green: `${ESC}[32m`,
  yellow: `${ESC}[33m`,
  cyan: `${ESC}[36m`,
  brightRed: `${ESC}[91m`,
  brightGreen: `${ESC}[92m`,
  brightYellow: `${ESC}[93m`,
  brightCyan: `${ESC}[96m`,
  brightWhite: `${ESC}[97m`,
  fg: (n: number) => `${ESC}[38;5;${n}m`,
};

const GRADIENT = [196, 197, 198, 199, 200, 164, 128, 92, 56, 57, 63, 69, 75, 81, 45, 39, 33, 27];

function gradient(text: string): string {
  return [...text].map((ch, i) => {
    if (ch === ' ') return ch;
    const idx = Math.floor((i / Math.max(text.length - 1, 1)) * (GRADIENT.length - 1));
    return `${c.fg(GRADIENT[idx])}${c.bold}${ch}${c.reset}`;
  }).join('');
}

function println(text = ''): void {
  console.log(text);
}

const BOX_W = Math.min(process.stdout.columns || 80, 76);

function boxTop(title?: string): string {
  if (title) {
    const t = ` ${title} `;
    const left = 2;
    const right = Math.max(0, BOX_W - 2 - t.length - left);
    return `  ${c.dim}╭${'─'.repeat(left)}${c.reset}${c.bold}${c.brightCyan}${t}${c.reset}${c.dim}${'─'.repeat(right)}╮${c.reset}`;
  }
  return `  ${c.dim}╭${'─'.repeat(BOX_W - 2)}╮${c.reset}`;
}

function boxLine(text: string): string {
  const vis = text.replace(/\x1b\[[0-9;]*m/g, '').length;
  const pad = Math.max(0, BOX_W - 4 - vis);
  return `  ${c.dim}│${c.reset} ${text}${' '.repeat(pad)} ${c.dim}│${c.reset}`;
}

function boxBottom(): string {
  return `  ${c.dim}╰${'─'.repeat(BOX_W - 2)}╯${c.reset}`;
}

function boxDivider(): string {
  return `  ${c.dim}├${'─'.repeat(BOX_W - 2)}┤${c.reset}`;
}

const STORE_DIR = path.join(process.cwd(), 'store');
const CREDS_PATH = path.join(STORE_DIR, 'feishu-credentials.json');

interface FeishuCredentials {
  appId: string;
  appSecret: string;
  encryptKey?: string;
  verificationToken?: string;
}

async function testConnection(
  creds: FeishuCredentials,
): Promise<{ success: boolean; botName?: string; error?: string }> {
  try {
    const client = new Lark.Client({
      appId: creds.appId,
      appSecret: creds.appSecret,
      appType: Lark.AppType.SelfBuild,
    });

    const response = await (client as any).request({
      method: 'GET',
      url: '/open-apis/bot/v3/info',
    });

    if (response.code === 0 && response.bot) {
      return { success: true, botName: response.bot.bot_name || response.bot.app_name };
    }
    return {
      success: false,
      error: response.msg || `Error code: ${response.code}`,
    };
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

async function main(): Promise<void> {
  println();
  println(gradient('  Feishu Bot Setup'));
  println();

  // ── Step 1: Create App ──
  println(boxTop('Step 1: Create App'));
  println(boxLine(`${c.brightCyan}1.${c.reset} Open ${c.underline}https://open.feishu.cn/app${c.reset}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} Click ${c.bold}Create Custom App${c.reset}`));
  println(boxLine(`${c.brightCyan}3.${c.reset} App name: e.g. "MatClaw", fill description`));
  println(boxBottom());
  println();

  // ── Step 2: Enable Bot ──
  println(boxTop('Step 2: Enable Bot'));
  println(boxLine(`${c.brightCyan}1.${c.reset} In app settings → ${c.bold}Add Capabilities${c.reset}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} Enable ${c.bold}Bot${c.reset} capability`));
  println(boxBottom());
  println();

  // ── Step 3: Event Subscriptions ──
  println(boxTop('Step 3: Event Subscriptions'));
  println(boxLine(`${c.brightCyan}1.${c.reset} Go to ${c.bold}Event Subscriptions${c.reset}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} Connection mode: ${c.bold}${c.brightGreen}Long Connection (WebSocket)${c.reset}`));
  println(boxLine(`   ${c.dim}No public URL needed — works behind NAT/firewall${c.reset}`));
  println(boxLine(`${c.brightCyan}3.${c.reset} Add event: ${c.bold}im.message.receive_v1${c.reset}`));
  println(boxBottom());
  println();

  // ── Step 4: Permissions ──
  println(boxTop('Step 4: Add Permissions'));
  println(boxLine(`Go to ${c.bold}Permissions & Scopes${c.reset} and add:`));
  println(boxDivider());
  println(boxLine(`${c.brightGreen}im:message${c.reset}                  ${c.dim}Receive messages${c.reset}`));
  println(boxLine(`${c.brightGreen}im:message:send_as_bot${c.reset}     ${c.dim}Send messages as bot${c.reset}`));
  println(boxLine(`${c.brightGreen}im:chat:readonly${c.reset}           ${c.dim}Read chat/group info${c.reset}`));
  println(boxLine(`${c.brightGreen}im:resource${c.reset}                ${c.dim}Upload images (for plots)${c.reset}`));
  println(boxLine(`${c.brightGreen}contact:contact.base:readonly${c.reset} ${c.dim}Read user names${c.reset}`));
  println(boxBottom());
  println();

  // ── Step 5: Publish ──
  println(boxTop('Step 5: Publish App'));
  println(boxLine(`${c.brightCyan}1.${c.reset} Go to ${c.bold}Version Management${c.reset}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} Create and ${c.bold}publish${c.reset} a new version`));
  println(boxLine(`${c.brightCyan}3.${c.reset} Wait for admin approval ${c.dim}(enterprise accounts)${c.reset}`));
  println(boxBottom());
  println();

  const ready = await confirm({
    message: '  Completed the steps above?',
    default: true,
  });

  if (!ready) {
    println(`\n  ${c.dim}Come back after completing the steps. Run: npm run auth:feishu${c.reset}\n`);
    process.exit(0);
  }

  // ── Step 6: Enter Credentials ──
  println();
  println(boxTop('Step 6: Enter Credentials'));
  println(boxLine(`${c.dim}Find these on your app's Credentials & Basic Info page${c.reset}`));
  println(boxBottom());
  println();

  // Check existing
  if (fs.existsSync(CREDS_PATH)) {
    try {
      JSON.parse(fs.readFileSync(CREDS_PATH, 'utf-8'));
      const overwrite = await confirm({
        message: '  Existing credentials found. Overwrite?',
        default: false,
      });
      if (!overwrite) {
        println(`  ${c.dim}Keeping existing credentials.${c.reset}`);
        const spinner = ora({ text: 'Testing connection...', indent: 4 }).start();
        const existing: FeishuCredentials = JSON.parse(fs.readFileSync(CREDS_PATH, 'utf-8'));
        const result = await testConnection(existing);
        if (result.success) {
          spinner.succeed(`Connected! Bot: ${c.bold}${result.botName || 'Unknown'}${c.reset}`);
        } else {
          spinner.fail(`Connection failed: ${result.error}`);
        }
        process.exit(result.success ? 0 : 1);
      }
    } catch {
      // Invalid file — continue
    }
  }

  const appId = await input({
    message: '  App ID:',
    validate: (val) => val.trim() ? true : 'App ID is required',
  });

  const appSecret = await password({
    message: '  App Secret:',
    mask: '*',
    validate: (val) => val.trim() ? true : 'App Secret is required',
  });

  println(`\n  ${c.dim}Optional security settings (Enter to skip):${c.reset}`);
  const encryptKey = await input({ message: '  Encrypt Key (optional):' });
  const verificationToken = await input({ message: '  Verification Token (optional):' });

  const creds: FeishuCredentials = {
    appId: appId.trim(),
    appSecret: appSecret.trim(),
    ...(encryptKey.trim() && { encryptKey: encryptKey.trim() }),
    ...(verificationToken.trim() && { verificationToken: verificationToken.trim() }),
  };

  // ── Test Connection ──
  println();
  const spinner = ora({ text: 'Testing connection to Feishu API...', indent: 4 }).start();
  const testResult = await testConnection(creds);

  if (!testResult.success) {
    spinner.fail(`Connection failed: ${testResult.error}`);
    println();
    println(boxTop('Troubleshooting'));
    println(boxLine(`${c.brightYellow}1.${c.reset} Check App ID and App Secret are correct`));
    println(boxLine(`${c.brightYellow}2.${c.reset} Ensure the app version is published`));
    println(boxLine(`${c.brightYellow}3.${c.reset} Check network/VPN — feishu.cn must be reachable`));
    println(boxBottom());
    process.exit(1);
  }

  spinner.succeed(`Connected! Bot: ${c.bold}${testResult.botName}${c.reset}`);

  // ── Save ──
  fs.mkdirSync(STORE_DIR, { recursive: true });
  fs.writeFileSync(CREDS_PATH, JSON.stringify(creds, null, 2));
  fs.chmodSync(CREDS_PATH, 0o600);

  // ── Next Steps ──
  println();
  println(boxTop('Done!'));
  println(boxLine(`${c.brightGreen}✔${c.reset}  Credentials saved to ${c.dim}store/feishu-credentials.json${c.reset}`));
  println(boxDivider());
  println(boxLine(`${c.bold}Next:${c.reset}`));
  println(boxLine(`${c.brightCyan}1.${c.reset} Add the bot to your Feishu group or DM it directly`));
  println(boxLine(`${c.brightCyan}2.${c.reset} Start MatClaw: ${c.bold}npm run dev${c.reset}`));
  println(boxLine(`${c.brightCyan}3.${c.reset} Send a message — the chat auto-registers as a group`));
  println(boxDivider());
  println(boxLine(`${c.dim}Group chat: @mention the bot to trigger${c.reset}`));
  println(boxLine(`${c.dim}Direct message: all messages are processed${c.reset}`));
  println(boxBottom());
  println();

  process.exit(0);
}

main().catch((err) => {
  console.error(`\n${c.brightRed}Error:${c.reset}`, err.message);
  process.exit(1);
});
