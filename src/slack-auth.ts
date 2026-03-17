/**
 * Slack Bot Authentication Setup
 * Interactive guided setup with step-by-step instructions and i18n.
 * Run: npm run auth:slack
 */

import { password, confirm } from '@inquirer/prompts';
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
import { writeEnvKeys } from '../setup/env-writer.js';

async function main(): Promise<void> {
  // Detect language
  const langArg = process.argv.find((_, i, a) => a[i - 1] === '--lang') as
    | Locale
    | undefined;
  setLocale(langArg || detectLocale());

  println();
  println(gradient(`  ${t('slack.title')}`));
  println();

  // -- Step 1: Create App --
  println(boxTop(t('slack.step1')));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('slack.step1.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('slack.step1.2')}`));
  println(boxLine(`${c.brightCyan}3.${c.reset} ${t('slack.step1.3')}`));
  println(boxLine(`${c.brightCyan}4.${c.reset} ${t('slack.step1.4')}`));
  println(boxBottom());
  println();

  // -- Step 2: Add OAuth Scopes --
  println(boxTop(t('slack.step2')));
  println(boxLine(`${t('slack.step2.intro')}`));
  println(boxDivider());
  println(
    boxLine(
      `${c.brightGreen}chat:write${c.reset}              ${c.dim}${t('slack.scope.chatWrite')}${c.reset}`,
    ),
  );
  println(
    boxLine(
      `${c.brightGreen}channels:history${c.reset}         ${c.dim}${t('slack.scope.channelsHistory')}${c.reset}`,
    ),
  );
  println(
    boxLine(
      `${c.brightGreen}channels:read${c.reset}            ${c.dim}${t('slack.scope.channelsRead')}${c.reset}`,
    ),
  );
  println(
    boxLine(
      `${c.brightGreen}groups:history${c.reset}           ${c.dim}${t('slack.scope.groupsHistory')}${c.reset}`,
    ),
  );
  println(
    boxLine(
      `${c.brightGreen}groups:read${c.reset}              ${c.dim}${t('slack.scope.groupsRead')}${c.reset}`,
    ),
  );
  println(
    boxLine(
      `${c.brightGreen}im:history${c.reset}               ${c.dim}${t('slack.scope.imHistory')}${c.reset}`,
    ),
  );
  println(
    boxLine(
      `${c.brightGreen}im:read${c.reset}                  ${c.dim}${t('slack.scope.imRead')}${c.reset}`,
    ),
  );
  println(
    boxLine(
      `${c.brightGreen}files:write${c.reset}              ${c.dim}${t('slack.scope.filesWrite')}${c.reset}`,
    ),
  );
  println(
    boxLine(
      `${c.brightGreen}users:read${c.reset}               ${c.dim}${t('slack.scope.usersRead')}${c.reset}`,
    ),
  );
  println(boxBottom());
  println();

  const ready = await confirm({
    message: `  ${t('slack.ready')}`,
    default: true,
  });

  if (!ready) {
    println(`\n  ${c.dim}${t('slack.comeback')}${c.reset}\n`);
    process.exit(0);
  }

  // -- Step 3: Enter Tokens --
  println();
  println(boxTop(t('slack.step3')));
  println(boxLine(`${c.dim}${t('slack.step3.hint')}${c.reset}`));
  println(boxBottom());
  println();

  const botToken = await password({
    message: `  ${t('slack.botToken')}`,
    mask: '*',
    validate: (val) => {
      if (!val.trim()) return t('slack.botTokenRequired');
      if (!val.trim().startsWith('xoxb-')) return t('slack.botTokenPrefix');
      return true;
    },
  });

  const appToken = await password({
    message: `  ${t('slack.appToken')}`,
    mask: '*',
    validate: (val) => {
      if (!val.trim()) return t('slack.appTokenRequired');
      if (!val.trim().startsWith('xapp-')) return t('slack.appTokenPrefix');
      return true;
    },
  });

  // -- Save --
  writeEnvKeys(
    process.cwd(),
    {
      SLACK_BOT_TOKEN: botToken.trim(),
      SLACK_APP_TOKEN: appToken.trim(),
    },
    'Slack',
  );

  // -- Done --
  println();
  println(boxTop(t('slack.done')));
  println(boxLine(`${c.brightGreen}✔${c.reset}  ${t('slack.saved')}`));
  println(boxDivider());
  println(boxLine(`${c.bold}${t('slack.next')}${c.reset}`));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('slack.next.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('slack.next.2')}`));
  println(boxLine(`${c.brightCyan}3.${c.reset} ${t('slack.next.3')}`));
  println(boxDivider());
  println(boxLine(`${c.dim}${t('slack.next.channel')}${c.reset}`));
  println(boxLine(`${c.dim}${t('slack.next.dm')}${c.reset}`));
  println(boxBottom());
  println();

  process.exit(0);
}

main().catch((err) => {
  console.error(`\n${c.brightRed}Error:${c.reset}`, err.message);
  process.exit(1);
});
