/**
 * Feishu Bot Authentication Setup
 * Interactive guided setup with step-by-step instructions and i18n.
 * Run: npm run auth:feishu
 */

import fs from 'fs';
import path from 'path';
import * as Lark from '@larksuiteoapi/node-sdk';
import { input, password, confirm } from '@inquirer/prompts';
import ora from 'ora';
import {
  c, gradient, println,
  boxTop, boxLine, boxBottom, boxDivider,
} from '../setup/ui.js';
import { setLocale, detectLocale, t, type Locale } from '../setup/i18n.js';

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
    return { success: false, error: response.msg || `Error code: ${response.code}` };
  } catch (err) {
    return { success: false, error: err instanceof Error ? err.message : String(err) };
  }
}

async function main(): Promise<void> {
  // Detect language
  const langArg = process.argv.find((_, i, a) => a[i - 1] === '--lang') as Locale | undefined;
  setLocale(langArg || detectLocale());

  println();
  println(gradient(`  ${t('feishu.title')}`));
  println();

  // ── Step 1: Create App ──
  println(boxTop(t('feishu.step1')));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('feishu.step1.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('feishu.step1.2')}`));
  println(boxLine(`${c.brightCyan}3.${c.reset} ${t('feishu.step1.3')}`));
  println(boxBottom());
  println();

  // ── Step 2: Enable Bot ──
  println(boxTop(t('feishu.step2')));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('feishu.step2.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('feishu.step2.2')}`));
  println(boxBottom());
  println();

  // ── Step 3: Event Subscriptions ──
  println(boxTop(t('feishu.step3')));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('feishu.step3.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('feishu.step3.2')}`));
  println(boxLine(`   ${c.dim}${t('feishu.step3.note')}${c.reset}`));
  println(boxLine(`${c.brightCyan}3.${c.reset} ${t('feishu.step3.3')}`));
  println(boxBottom());
  println();

  // ── Step 4: Permissions ──
  println(boxTop(t('feishu.step4')));
  println(boxLine(`${t('feishu.step4.intro')}`));
  println(boxDivider());
  println(boxLine(`${c.brightGreen}im:message${c.reset}                  ${c.dim}${t('feishu.perm.message')}${c.reset}`));
  println(boxLine(`${c.brightGreen}im:message:send_as_bot${c.reset}     ${c.dim}${t('feishu.perm.send')}${c.reset}`));
  println(boxLine(`${c.brightGreen}im:chat:readonly${c.reset}           ${c.dim}${t('feishu.perm.chat')}${c.reset}`));
  println(boxLine(`${c.brightGreen}im:resource${c.reset}                ${c.dim}${t('feishu.perm.resource')}${c.reset}`));
  println(boxLine(`${c.brightGreen}contact:contact.base:readonly${c.reset} ${c.dim}${t('feishu.perm.contact')}${c.reset}`));
  println(boxBottom());
  println();

  // ── Step 5: Publish ──
  println(boxTop(t('feishu.step5')));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('feishu.step5.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('feishu.step5.2')}`));
  println(boxLine(`${c.brightCyan}3.${c.reset} ${t('feishu.step5.3')}`));
  println(boxBottom());
  println();

  const ready = await confirm({
    message: `  ${t('feishu.ready')}`,
    default: true,
  });

  if (!ready) {
    println(`\n  ${c.dim}${t('feishu.comeback')}${c.reset}\n`);
    process.exit(0);
  }

  // ── Step 6: Enter Credentials ──
  println();
  println(boxTop(t('feishu.step6')));
  println(boxLine(`${c.dim}${t('feishu.step6.hint')}${c.reset}`));
  println(boxBottom());
  println();

  // Check existing
  if (fs.existsSync(CREDS_PATH)) {
    try {
      JSON.parse(fs.readFileSync(CREDS_PATH, 'utf-8'));
      const overwrite = await confirm({
        message: `  ${t('feishu.existingFound')}`,
        default: false,
      });
      if (!overwrite) {
        println(`  ${c.dim}${t('feishu.keeping')}${c.reset}`);
        const spinner = ora({ text: t('feishu.testing'), indent: 4 }).start();
        const existing: FeishuCredentials = JSON.parse(fs.readFileSync(CREDS_PATH, 'utf-8'));
        const result = await testConnection(existing);
        if (result.success) {
          spinner.succeed(t('feishu.connected', { name: result.botName || 'Unknown' }));
        } else {
          spinner.fail(t('feishu.connFailed', { error: result.error || '' }));
        }
        process.exit(result.success ? 0 : 1);
      }
    } catch {
      // Invalid file — continue
    }
  }

  const appId = await input({
    message: `  ${t('feishu.appId')}`,
    validate: (val) => val.trim() ? true : t('feishu.appIdRequired'),
  });

  const appSecret = await password({
    message: `  ${t('feishu.appSecret')}`,
    mask: '*',
    validate: (val) => val.trim() ? true : t('feishu.appSecretRequired'),
  });

  println(`\n  ${c.dim}${t('feishu.optional')}${c.reset}`);
  const encryptKey = await input({ message: `  ${t('feishu.encryptKey')}` });
  const verificationToken = await input({ message: `  ${t('feishu.verifyToken')}` });

  const creds: FeishuCredentials = {
    appId: appId.trim(),
    appSecret: appSecret.trim(),
    ...(encryptKey.trim() && { encryptKey: encryptKey.trim() }),
    ...(verificationToken.trim() && { verificationToken: verificationToken.trim() }),
  };

  // ── Test Connection ──
  println();
  const spinner = ora({ text: t('feishu.testingApi'), indent: 4 }).start();
  const testResult = await testConnection(creds);

  if (!testResult.success) {
    spinner.fail(t('feishu.connFailed', { error: testResult.error || '' }));
    println();
    println(boxTop(t('feishu.troubleshoot')));
    println(boxLine(`${c.brightYellow}1.${c.reset} ${t('feishu.troubleshoot.1')}`));
    println(boxLine(`${c.brightYellow}2.${c.reset} ${t('feishu.troubleshoot.2')}`));
    println(boxLine(`${c.brightYellow}3.${c.reset} ${t('feishu.troubleshoot.3')}`));
    println(boxBottom());
    process.exit(1);
  }

  spinner.succeed(t('feishu.connected', { name: testResult.botName || '' }));

  // ── Save ──
  fs.mkdirSync(STORE_DIR, { recursive: true });
  fs.writeFileSync(CREDS_PATH, JSON.stringify(creds, null, 2));
  fs.chmodSync(CREDS_PATH, 0o600);

  // ── Next Steps ──
  println();
  println(boxTop(t('feishu.done')));
  println(boxLine(`${c.brightGreen}✔${c.reset}  ${t('feishu.saved')}`));
  println(boxDivider());
  println(boxLine(`${c.bold}${t('feishu.next')}${c.reset}`));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('feishu.next.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('feishu.next.2')}`));
  println(boxLine(`${c.brightCyan}3.${c.reset} ${t('feishu.next.3')}`));
  println(boxDivider());
  println(boxLine(`${c.dim}${t('feishu.next.group')}${c.reset}`));
  println(boxLine(`${c.dim}${t('feishu.next.dm')}${c.reset}`));
  println(boxBottom());
  println();

  process.exit(0);
}

main().catch((err) => {
  console.error(`\n${c.brightRed}Error:${c.reset}`, err.message);
  process.exit(1);
});
