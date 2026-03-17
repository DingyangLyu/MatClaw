/**
 * Telegram Bot Authentication Setup
 * Interactive guided setup with step-by-step instructions and i18n.
 * Run: npm run auth:telegram
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
  println(gradient(`  ${t('telegram.title')}`));
  println();

  // -- Step 1: Create Bot --
  println(boxTop(t('telegram.step1')));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('telegram.step1.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('telegram.step1.2')}`));
  println(boxLine(`${c.brightCyan}3.${c.reset} ${t('telegram.step1.3')}`));
  println(boxLine(`${c.brightCyan}4.${c.reset} ${t('telegram.step1.4')}`));
  println(boxBottom());
  println();

  const ready = await confirm({
    message: `  ${t('telegram.ready')}`,
    default: true,
  });

  if (!ready) {
    println(`\n  ${c.dim}${t('telegram.comeback')}${c.reset}\n`);
    process.exit(0);
  }

  // -- Step 2: Enter Token --
  println();
  println(boxTop(t('telegram.step2')));
  println(boxLine(`${c.dim}${t('telegram.step2.hint')}${c.reset}`));
  println(boxBottom());
  println();

  const token = await password({
    message: `  ${t('telegram.token')}`,
    mask: '*',
    validate: (val) => (val.trim() ? true : t('telegram.tokenRequired')),
  });

  // -- Save --
  writeEnvKeys(process.cwd(), { TELEGRAM_BOT_TOKEN: token.trim() }, 'Telegram');

  // -- Done --
  println();
  println(boxTop(t('telegram.done')));
  println(boxLine(`${c.brightGreen}✔${c.reset}  ${t('telegram.saved')}`));
  println(boxDivider());
  println(boxLine(`${c.bold}${t('telegram.next')}${c.reset}`));
  println(boxLine(`${c.brightCyan}1.${c.reset} ${t('telegram.next.1')}`));
  println(boxLine(`${c.brightCyan}2.${c.reset} ${t('telegram.next.2')}`));
  println(boxLine(`${c.brightCyan}3.${c.reset} ${t('telegram.next.3')}`));
  println(boxDivider());
  println(boxLine(`${c.dim}${t('telegram.next.group')}${c.reset}`));
  println(boxLine(`${c.dim}${t('telegram.next.dm')}${c.reset}`));
  println(boxBottom());
  println();

  process.exit(0);
}

main().catch((err) => {
  console.error(`\n${c.brightRed}Error:${c.reset}`, err.message);
  process.exit(1);
});
