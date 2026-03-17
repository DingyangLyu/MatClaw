/**
 * Discord Bot Authentication Setup
 * Interactive guided setup with step-by-step instructions and i18n.
 * Run: npm run auth:discord
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
  println(gradient(`  ${t('discord.title')}`));
  println();

  // -- Step 1: Create App --
  println(boxTop(t('discord.step1')));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('discord.step1.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('discord.step1.2')}`));
  println(boxLine(`${c.brightCyan}3.${c.reset} ${t('discord.step1.3')}`));
  println(boxLine(`${c.brightCyan}4.${c.reset} ${t('discord.step1.4')}`));
  println(boxBottom());
  println();

  // -- Step 2: Enable Intents --
  println(boxTop(t('discord.step2')));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('discord.step2.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('discord.step2.2')}`));
  println(boxDivider());
  println(boxLine(`${t('discord.step2.intents')}`));
  println(boxDivider());
  println(
    boxLine(
      `${c.brightGreen}Message Content Intent${c.reset}   ${c.dim}${t('discord.intent.message')}${c.reset}`,
    ),
  );
  println(
    boxLine(
      `${c.brightGreen}Server Members Intent${c.reset}    ${c.dim}${t('discord.intent.members')}${c.reset}`,
    ),
  );
  println(boxBottom());
  println();

  const ready = await confirm({
    message: `  ${t('discord.ready')}`,
    default: true,
  });

  if (!ready) {
    println(`\n  ${c.dim}${t('discord.comeback')}${c.reset}\n`);
    process.exit(0);
  }

  // -- Step 3: Enter Token --
  println();
  println(boxTop(t('discord.step3')));
  println(boxLine(`${c.dim}${t('discord.step3.hint')}${c.reset}`));
  println(boxBottom());
  println();

  const token = await password({
    message: `  ${t('discord.token')}`,
    mask: '*',
    validate: (val) => (val.trim() ? true : t('discord.tokenRequired')),
  });

  // -- Save --
  writeEnvKeys(process.cwd(), { DISCORD_BOT_TOKEN: token.trim() }, 'Discord');

  // -- Done --
  println();
  println(boxTop(t('discord.done')));
  println(boxLine(`${c.brightGreen}✔${c.reset}  ${t('discord.saved')}`));
  println(boxDivider());
  println(boxLine(`${c.bold}${t('discord.next')}${c.reset}`));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('discord.next.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('discord.next.2')}`));
  println(boxLine(`${c.brightCyan}3.${c.reset} ${t('discord.next.3')}`));
  println(boxDivider());
  println(boxLine(`${c.dim}${t('discord.next.invite')}${c.reset}`));
  println(boxBottom());
  println();

  process.exit(0);
}

main().catch((err) => {
  console.error(`\n${c.brightRed}Error:${c.reset}`, err.message);
  process.exit(1);
});
