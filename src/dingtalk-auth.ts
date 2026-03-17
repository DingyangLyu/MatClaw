/**
 * DingTalk Bot Authentication Setup
 * Interactive guided setup with step-by-step instructions and i18n.
 * Run: npm run auth:dingtalk
 */

import fs from 'fs';
import path from 'path';
import { input, password, confirm } from '@inquirer/prompts';
import ora from 'ora';
import {
  c,
  gradient,
  println,
  boxTop,
  boxLine,
  boxBottom,
  boxDivider,
} from '../setup/ui.js';
import { setLocale, detectLocale, t, type Locale } from '../setup/i18n.js';

const STORE_DIR = path.join(process.cwd(), 'store');
const CREDS_PATH = path.join(STORE_DIR, 'dingtalk-credentials.json');

interface DingTalkCredentials {
  clientId: string;
  clientSecret: string;
}

async function testConnection(
  creds: DingTalkCredentials,
): Promise<{ success: boolean; error?: string }> {
  try {
    const response = await fetch(
      'https://api.dingtalk.com/v1.0/oauth2/accessToken',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          appKey: creds.clientId,
          appSecret: creds.clientSecret,
        }),
      },
    );

    if (!response.ok) {
      return {
        success: false,
        error: `HTTP ${response.status}: ${await response.text()}`,
      };
    }

    const data: any = await response.json();
    if (data.accessToken) {
      return { success: true };
    }
    return {
      success: false,
      error: data.message || `Unexpected response: ${JSON.stringify(data)}`,
    };
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

async function main(): Promise<void> {
  // Detect language
  const langArg = process.argv.find((_, i, a) => a[i - 1] === '--lang') as
    | Locale
    | undefined;
  setLocale(langArg || detectLocale());

  println();
  println(gradient(`  ${t('dingtalk.title')}`));
  println();

  // -- Step 1: Create App --
  println(boxTop(t('dingtalk.step1')));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('dingtalk.step1.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('dingtalk.step1.2')}`));
  println(boxLine(`${c.brightCyan}3.${c.reset} ${t('dingtalk.step1.3')}`));
  println(boxBottom());
  println();

  // -- Step 2: Enable Robot --
  println(boxTop(t('dingtalk.step2')));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('dingtalk.step2.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('dingtalk.step2.2')}`));
  println(boxLine(`   ${c.dim}${t('dingtalk.step2.note')}${c.reset}`));
  println(boxBottom());
  println();

  // -- Step 3: Add Permissions --
  println(boxTop(t('dingtalk.step3')));
  println(boxLine(`${t('dingtalk.step3.intro')}`));
  println(boxDivider());
  println(
    boxLine(
      `${c.brightGreen}qyapi_robot_sendmsg${c.reset}   ${c.dim}${t('dingtalk.perm.sendmsg')}${c.reset}`,
    ),
  );
  println(
    boxLine(
      `${c.brightGreen}qyapi_chat_manage${c.reset}     ${c.dim}${t('dingtalk.perm.chatmanage')}${c.reset}`,
    ),
  );
  println(boxBottom());
  println();

  // -- Step 4: Publish App --
  println(boxTop(t('dingtalk.step4')));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('dingtalk.step4.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('dingtalk.step4.2')}`));
  println(boxBottom());
  println();

  const ready = await confirm({
    message: `  ${t('dingtalk.ready')}`,
    default: true,
  });

  if (!ready) {
    println(`\n  ${c.dim}${t('dingtalk.comeback')}${c.reset}\n`);
    process.exit(0);
  }

  // -- Step 5: Enter Credentials --
  println();
  println(boxTop(t('dingtalk.step5')));
  println(boxLine(`${c.dim}${t('dingtalk.step5.hint')}${c.reset}`));
  println(boxBottom());
  println();

  // Check existing
  if (fs.existsSync(CREDS_PATH)) {
    try {
      JSON.parse(fs.readFileSync(CREDS_PATH, 'utf-8'));
      const overwrite = await confirm({
        message: `  ${t('dingtalk.existingFound')}`,
        default: false,
      });
      if (!overwrite) {
        println(`  ${c.dim}${t('dingtalk.keeping')}${c.reset}`);
        const spinner = ora({ text: t('dingtalk.testing'), indent: 4 }).start();
        const existing: DingTalkCredentials = JSON.parse(
          fs.readFileSync(CREDS_PATH, 'utf-8'),
        );
        const result = await testConnection(existing);
        if (result.success) {
          spinner.succeed(t('dingtalk.connected'));
        } else {
          spinner.fail(t('dingtalk.connFailed', { error: result.error || '' }));
        }
        process.exit(result.success ? 0 : 1);
      }
    } catch {
      // Invalid file -- continue
    }
  }

  const clientId = await input({
    message: `  ${t('dingtalk.clientId')}`,
    validate: (val) => (val.trim() ? true : t('dingtalk.clientIdRequired')),
  });

  const clientSecret = await password({
    message: `  ${t('dingtalk.clientSecret')}`,
    mask: '*',
    validate: (val) => (val.trim() ? true : t('dingtalk.clientSecretRequired')),
  });

  const creds: DingTalkCredentials = {
    clientId: clientId.trim(),
    clientSecret: clientSecret.trim(),
  };

  // -- Test Connection --
  println();
  const spinner = ora({ text: t('dingtalk.testingApi'), indent: 4 }).start();
  const testResult = await testConnection(creds);

  if (!testResult.success) {
    spinner.fail(t('dingtalk.connFailed', { error: testResult.error || '' }));
    println();
    println(boxTop(t('dingtalk.troubleshoot')));
    println(
      boxLine(`${c.brightYellow}1.${c.reset} ${t('dingtalk.troubleshoot.1')}`),
    );
    println(
      boxLine(`${c.brightYellow}2.${c.reset} ${t('dingtalk.troubleshoot.2')}`),
    );
    println(
      boxLine(`${c.brightYellow}3.${c.reset} ${t('dingtalk.troubleshoot.3')}`),
    );
    println(boxBottom());
    process.exit(1);
  }

  spinner.succeed(t('dingtalk.connected'));

  // -- Save --
  fs.mkdirSync(STORE_DIR, { recursive: true });
  fs.writeFileSync(CREDS_PATH, JSON.stringify(creds, null, 2));
  fs.chmodSync(CREDS_PATH, 0o600);

  // -- Next Steps --
  println();
  println(boxTop(t('dingtalk.done')));
  println(boxLine(`${c.brightGreen}✔${c.reset}  ${t('dingtalk.saved')}`));
  println(boxDivider());
  println(boxLine(`${c.bold}${t('dingtalk.next')}${c.reset}`));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('dingtalk.next.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('dingtalk.next.2')}`));
  println(boxLine(`${c.brightCyan}3.${c.reset} ${t('dingtalk.next.3')}`));
  println(boxDivider());
  println(boxLine(`${c.dim}${t('dingtalk.next.group')}${c.reset}`));
  println(boxBottom());
  println();

  process.exit(0);
}

main().catch((err) => {
  console.error(`\n${c.brightRed}Error:${c.reset}`, err.message);
  process.exit(1);
});
