/**
 * DingTalk Bot Authentication Setup
 * Interactive script to configure DingTalk bot credentials.
 * Run: npm run auth:dingtalk
 */

import fs from 'fs';
import path from 'path';
import readline from 'readline';

const STORE_DIR = path.join(process.cwd(), 'store');
const CREDS_PATH = path.join(STORE_DIR, 'dingtalk-credentials.json');

interface DingTalkCredentials {
  clientId: string;
  clientSecret: string;
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function ask(question: string): Promise<string> {
  return new Promise((resolve) => {
    rl.question(question, (answer) => resolve(answer.trim()));
  });
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
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║           DingTalk Bot Authentication Setup                ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  console.log('');
  console.log('Setup Instructions:');
  console.log('1. Go to https://open-dev.dingtalk.com/fe/app#/corp/app');
  console.log('2. Click "Create App" (H5 Micro App)');
  console.log('3. In "Robot" section, enable "Robot configuration"');
  console.log('4. Set message receiving mode to "Stream Mode"');
  console.log('5. Copy the Client ID (AppKey) and Client Secret (AppSecret)');
  console.log(
    '6. In "Permissions", add: qyapi_robot_sendmsg, qyapi_chat_manage',
  );
  console.log('7. Publish the app and add the bot to your DingTalk group');
  console.log('');

  // Check for existing credentials
  if (fs.existsSync(CREDS_PATH)) {
    try {
      const existing: DingTalkCredentials = JSON.parse(
        fs.readFileSync(CREDS_PATH, 'utf-8'),
      );
      console.log('Existing credentials found.');
      const overwrite = await ask('Overwrite? (y/N): ');
      if (overwrite.toLowerCase() !== 'y') {
        console.log('Keeping existing credentials.');
        const testResult = await testConnection(existing);
        if (!testResult.success) {
          console.error('Connection failed:', testResult.error);
          rl.close();
          process.exit(1);
        }
        console.log('Connection successful!');
        rl.close();
        process.exit(0);
      }
    } catch {
      // Invalid existing file — continue to collect new credentials
    }
  }

  const clientId = await ask('Client ID (AppKey): ');
  if (!clientId) {
    console.error('Client ID is required');
    rl.close();
    process.exit(1);
  }

  const clientSecret = await ask('Client Secret (AppSecret): ');
  if (!clientSecret) {
    console.error('Client Secret is required');
    rl.close();
    process.exit(1);
  }

  const creds: DingTalkCredentials = { clientId, clientSecret };

  console.log('');
  console.log('Testing connection to DingTalk API...');

  const testResult = await testConnection(creds);
  if (!testResult.success) {
    console.error('Connection failed:', testResult.error);
    console.log('Please check your Client ID and Client Secret and try again.');
    rl.close();
    process.exit(1);
  }

  console.log('Connection successful!');

  fs.mkdirSync(STORE_DIR, { recursive: true });
  fs.writeFileSync(CREDS_PATH, JSON.stringify(creds, null, 2));
  fs.chmodSync(CREDS_PATH, 0o600);

  console.log('');
  console.log(`Credentials saved to: ${CREDS_PATH}`);
  console.log('');
  console.log('Next steps:');
  console.log('1. Add the bot to your DingTalk group');
  console.log('2. Send a message mentioning the bot (@bot) in the group');
  console.log('3. The group will be auto-registered on first message');
  console.log('4. Restart MatClaw: npm run dev');

  rl.close();
  process.exit(0);
}

main().catch((err) => {
  console.error('Error:', err);
  rl.close();
  process.exit(1);
});
