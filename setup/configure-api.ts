/**
 * Interactive API configuration wizard.
 * Usage: npx tsx setup/index.ts --step configure-api
 *
 * Walks users through provider selection, API key input, validation,
 * and .env file updates — no manual editing required.
 */
import fs from 'fs';
import os from 'os';
import path from 'path';
import ora from 'ora';
import { select, input, password, confirm, Separator } from '@inquirer/prompts';
import { readEnvFile } from '../src/env.js';
import { writeEnvKeys, removeEnvKeys } from './env-writer.js';
import { emitStatus } from './status.js';
import { c, println, boxTop, boxLine, boxBottom, boxDivider, ok, warn, fail, info, phaseHeader, phaseFooter } from './ui.js';
import { setLocale, getLocale, detectLocale, t, type Locale } from './i18n.js';

// ── Provider definitions ──

interface ProviderConfig {
  id: string;
  label: string;       // Machine-readable label (used in emitStatus, detectExistingConfig)
  labelEn: string;     // English display name
  labelZh: string;     // Chinese display name
  descEn?: string;     // English description
  descZh?: string;     // Chinese description
  engine: string;
  apiKeyEnvName: string;
  apiKeyPrefix?: string;
  defaultBaseUrl?: string;
  defaultModel?: string;
  needsBaseUrl?: boolean;
  needsModel?: boolean;
  keyOptional?: boolean; // e.g. Ollama doesn't need an API key
  authMethods: ('api_key' | 'oauth_auto' | 'env_ref')[];
}

function providerLabel(p: ProviderConfig): string {
  return getLocale() === 'zh' ? p.labelZh : p.labelEn;
}

function providerDesc(p: ProviderConfig): string | undefined {
  return getLocale() === 'zh' ? p.descZh : p.descEn;
}

const PROVIDERS: ProviderConfig[] = [
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  //  推荐
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    id: 'anthropic',
    label: 'Anthropic Claude（推荐）',
    labelEn: 'Anthropic Claude (Recommended)',
    labelZh: 'Anthropic Claude（推荐）',
    descEn: 'Claude Opus/Sonnet/Haiku | API Key or Claude Pro/Max OAuth',
    descZh: 'Claude Opus/Sonnet/Haiku | API Key 或 Claude Pro/Max OAuth',
    engine: 'claude',
    apiKeyEnvName: 'ANTHROPIC_API_KEY',
    apiKeyPrefix: 'sk-ant-',
    authMethods: ['api_key', 'oauth_auto', 'env_ref'],
  },

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  //  主流国际供应商
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    id: 'gemini',
    label: 'Google Gemini [engine 开发中]',
    labelEn: 'Google Gemini [engine in dev]',
    labelZh: 'Google Gemini [engine 开发中]',
    descEn: 'Gemini 2.5 Pro/Flash | Free tier available',
    descZh: 'Gemini 2.5 Pro/Flash | 免费额度可用',
    engine: 'gemini',
    apiKeyEnvName: 'GOOGLE_API_KEY',
    apiKeyPrefix: 'AIza',
    defaultModel: 'gemini-2.5-flash',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'openai',
    label: 'OpenAI',
    labelEn: 'OpenAI',
    labelZh: 'OpenAI',
    descEn: 'GPT-4.1 / o4-mini | Pay per use',
    descZh: 'GPT-4.1 / o4-mini | 按量付费',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    apiKeyPrefix: 'sk-',
    defaultModel: 'o4-mini',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'deepseek',
    label: 'DeepSeek（深度求索）',
    labelEn: 'DeepSeek',
    labelZh: 'DeepSeek（深度求索）',
    descEn: 'DeepSeek-V3 / R1 | Low cost, high performance',
    descZh: 'DeepSeek-V3 / R1 | 低成本高性能',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    apiKeyPrefix: 'sk-',
    defaultBaseUrl: 'https://api.deepseek.com/v1',
    defaultModel: 'deepseek-chat',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'mistral',
    label: 'Mistral AI',
    labelEn: 'Mistral AI',
    labelZh: 'Mistral AI',
    descEn: 'Mistral Large / Codestral | 262K context',
    descZh: 'Mistral Large / Codestral | 262K 上下文',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://api.mistral.ai/v1',
    defaultModel: 'mistral-large-latest',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'xai',
    label: 'xAI（Grok）',
    labelEn: 'xAI (Grok)',
    labelZh: 'xAI（Grok）',
    descEn: 'Grok 4 / Grok 3 | 131K context | Free tier',
    descZh: 'Grok 4 / Grok 3 | 131K 上下文 | 免费额度',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    apiKeyPrefix: 'xai-',
    defaultBaseUrl: 'https://api.x.ai/v1',
    defaultModel: 'grok-4',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'perplexity',
    label: 'Perplexity AI',
    labelEn: 'Perplexity AI',
    labelZh: 'Perplexity AI',
    descEn: 'Sonar Pro | Built-in search',
    descZh: 'Sonar Pro | 内置搜索增强',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://api.perplexity.ai',
    defaultModel: 'sonar-pro',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  //  国内供应商
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    id: 'qwen',
    label: '阿里通义千问（Qwen）',
    labelEn: 'Alibaba Qwen',
    labelZh: '阿里通义千问（Qwen）',
    descEn: 'Qwen3.5 / Qwen-Coder | 1M context | Coding Plan available',
    descZh: 'Qwen3.5 / Qwen-Coder | 1M 上下文 | 提供 Coding Plan 订阅',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    apiKeyPrefix: 'sk-',
    defaultBaseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    defaultModel: 'qwen-plus',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'qwen-coding',
    label: '阿里通义 Coding Plan（国内 / OpenAI 兼容）',
    labelEn: 'Qwen Coding Plan (China / OpenAI)',
    labelZh: '通义 Coding Plan（国内 / OpenAI 兼容）',
    descEn: 'Qwen3.5 | Subscription | 1M context | Includes GLM/Kimi/MiniMax',
    descZh: 'Qwen3.5 | 订阅制 | 1M 上下文 | 包含 GLM/Kimi/MiniMax',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    apiKeyPrefix: 'sk-',
    defaultBaseUrl: 'https://coding.dashscope.aliyuncs.com/v1',
    defaultModel: 'qwen3.5-plus',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'qwen-coding-anthropic',
    label: '阿里通义 Coding Plan（国内 / Anthropic 兼容）',
    labelEn: 'Qwen Coding Plan (China / Anthropic)',
    labelZh: '通义 Coding Plan（国内 / Anthropic 兼容）',
    descEn: 'Qwen3.5 | Subscription | Claude Agent SDK',
    descZh: 'Qwen3.5 | 订阅制 | Claude Agent SDK 直接调用',
    engine: 'claude',
    apiKeyEnvName: 'ANTHROPIC_API_KEY',
    apiKeyPrefix: 'sk-',
    defaultBaseUrl: 'https://coding.dashscope.aliyuncs.com/apps/anthropic',
    defaultModel: 'qwen3.5-plus',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'qwen-coding-intl',
    label: '阿里通义 Coding Plan（国际 / OpenAI 兼容）',
    labelEn: 'Qwen Coding Plan (Intl / OpenAI)',
    labelZh: '通义 Coding Plan（国际 / OpenAI 兼容）',
    descEn: 'Qwen3.5 | Subscription | Global endpoint',
    descZh: 'Qwen3.5 | 订阅制 | 海外加速端点',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    apiKeyPrefix: 'sk-',
    defaultBaseUrl: 'https://coding-intl.dashscope.aliyuncs.com/v1',
    defaultModel: 'qwen3.5-plus',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'qwen-coding-intl-anthropic',
    label: '阿里通义 Coding Plan（国际 / Anthropic 兼容）',
    labelEn: 'Qwen Coding Plan (Intl / Anthropic)',
    labelZh: '通义 Coding Plan（国际 / Anthropic 兼容）',
    descEn: 'Qwen3.5 | Subscription | Global | Claude Agent SDK',
    descZh: 'Qwen3.5 | 订阅制 | 海外 | Claude Agent SDK',
    engine: 'claude',
    apiKeyEnvName: 'ANTHROPIC_API_KEY',
    apiKeyPrefix: 'sk-',
    defaultBaseUrl: 'https://coding-intl.dashscope.aliyuncs.com/apps/anthropic',
    defaultModel: 'qwen3.5-plus',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'zhipu',
    label: '智谱 AI（GLM）',
    labelEn: 'Zhipu AI (GLM)',
    labelZh: '智谱 AI（GLM）',
    descEn: 'GLM-5 / GLM-4.7 | 200K context | Free tier',
    descZh: 'GLM-5 / GLM-4.7 | 200K 上下文 | 免费额度',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://open.bigmodel.cn/api/paas/v4',
    defaultModel: 'glm-5',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'zhipu-anthropic',
    label: '智谱 GLM Coding Plan（Anthropic 兼容）',
    labelEn: 'Zhipu GLM Coding Plan (Anthropic)',
    labelZh: '智谱 GLM Coding Plan（Anthropic 兼容）',
    descEn: 'GLM-5 | Subscription | Claude Agent SDK',
    descZh: 'GLM-5 | 订阅制 | Claude Agent SDK 直接调用',
    engine: 'claude',
    apiKeyEnvName: 'ANTHROPIC_API_KEY',
    defaultBaseUrl: 'https://open.bigmodel.cn/api/anthropic',
    defaultModel: 'glm-5',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'zhipu-coding',
    label: '智谱 GLM Coding Plan（国内 OpenAI 兼容）',
    labelEn: 'Zhipu GLM Coding Plan (China OpenAI)',
    labelZh: '智谱 GLM Coding Plan（国内 OpenAI 兼容）',
    descEn: 'GLM-5 | Subscription | Coding endpoint',
    descZh: 'GLM-5 | 订阅制 | Coding 端点',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://open.bigmodel.cn/api/coding/paas/v4',
    defaultModel: 'glm-5',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'zhipu-coding-intl',
    label: '智谱 GLM Coding Plan（国际 OpenAI 兼容）',
    labelEn: 'Zhipu GLM Coding Plan (Intl OpenAI)',
    labelZh: '智谱 GLM Coding Plan（国际 OpenAI 兼容）',
    descEn: 'GLM-5 | Subscription | Z.AI global',
    descZh: 'GLM-5 | 订阅制 | Z.AI 海外',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://api.z.ai/api/coding/paas/v4',
    defaultModel: 'glm-5',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'moonshot',
    label: '月之暗面 Kimi（.cn 国内）',
    labelEn: 'Moonshot Kimi (.cn)',
    labelZh: '月之暗面 Kimi（.cn 国内）',
    descEn: 'Kimi K2.5 | 256K context | Free tier',
    descZh: 'Kimi K2.5 | 256K 上下文 | 免费额度',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://api.moonshot.cn/v1',
    defaultModel: 'kimi-k2.5',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'moonshot-intl',
    label: '月之暗面 Kimi（.ai 国际）',
    labelEn: 'Moonshot Kimi (.ai Intl)',
    labelZh: '月之暗面 Kimi（.ai 国际）',
    descEn: 'Kimi K2.5 | 256K context | Global endpoint',
    descZh: 'Kimi K2.5 | 256K 上下文 | 海外加速端点',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://api.moonshot.ai/v1',
    defaultModel: 'kimi-k2.5',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'kimi-coding',
    label: 'Kimi Coding Plan',
    labelEn: 'Kimi Coding Plan',
    labelZh: 'Kimi Coding Plan',
    descEn: 'Kimi-Code | Subscription | Anthropic API | 262K context',
    descZh: 'Kimi-Code | 订阅制 | Anthropic 兼容 | 262K 上下文',
    engine: 'claude',
    apiKeyEnvName: 'ANTHROPIC_API_KEY',
    defaultBaseUrl: 'https://api.kimi.com/coding/',
    defaultModel: 'kimi-code',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'qianfan',
    label: '百度千帆（文心一言）',
    labelEn: 'Baidu Qianfan (ERNIE)',
    labelZh: '百度千帆（文心一言）',
    descEn: 'ERNIE-5.0 / DeepSeek-V3.2 | Free tier',
    descZh: 'ERNIE-5.0 / DeepSeek-V3.2 | 免费额度',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://qianfan.baidubce.com/v2',
    defaultModel: 'deepseek-v3.2',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'volcengine',
    label: '火山引擎（豆包 / OpenAI 兼容）',
    labelEn: 'Volcengine (Doubao / OpenAI)',
    labelZh: '火山引擎（豆包 / OpenAI 兼容）',
    descEn: 'Doubao-Seed-Code | 256K context | ByteDance',
    descZh: 'Doubao-Seed-Code | 256K 上下文 | 字节跳动旗下',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://ark.cn-beijing.volces.com/api/v3',
    defaultModel: 'doubao-seed-code',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'volcengine-anthropic',
    label: '火山引擎 Coding Plan（Anthropic 兼容）',
    labelEn: 'Volcengine Coding Plan (Anthropic)',
    labelZh: '火山引擎 Coding Plan（Anthropic 兼容）',
    descEn: 'Doubao-Seed-Code | Subscription | Claude Agent SDK',
    descZh: 'Doubao-Seed-Code | 订阅制 | Claude Agent SDK 直接调用',
    engine: 'claude',
    apiKeyEnvName: 'ANTHROPIC_API_KEY',
    defaultBaseUrl: 'https://ark.cn-beijing.volces.com/api/coding',
    defaultModel: 'doubao-seed-code',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'byteplus',
    label: 'BytePlus（火山海外版）',
    labelEn: 'BytePlus (Volcengine Intl)',
    labelZh: 'BytePlus（火山海外版）',
    descEn: 'Seed 1.8 / Kimi / GLM | 256K context | Global',
    descZh: 'Seed 1.8 / Kimi / GLM | 256K 上下文 | 海外可用',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://ark.ap-southeast.bytepluses.com/api/v3',
    defaultModel: 'seed-1-8-251228',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'minimax',
    label: 'MiniMax（海螺 AI / OpenAI 兼容）',
    labelEn: 'MiniMax (OpenAI)',
    labelZh: 'MiniMax（海螺 AI / OpenAI 兼容）',
    descEn: 'MiniMax-M2.5 | 200K context',
    descZh: 'MiniMax-M2.5 | 200K 上下文',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://api.minimax.chat/v1',
    defaultModel: 'MiniMax-M2.5',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'minimax-anthropic',
    label: 'MiniMax（海螺 AI / Anthropic 兼容）',
    labelEn: 'MiniMax (Anthropic)',
    labelZh: 'MiniMax（海螺 AI / Anthropic 兼容）',
    descEn: 'MiniMax-M2.5 | Claude Agent SDK',
    descZh: 'MiniMax-M2.5 | Claude Agent SDK 直接调用',
    engine: 'claude',
    apiKeyEnvName: 'ANTHROPIC_API_KEY',
    defaultBaseUrl: 'https://api.minimax.io/anthropic',
    defaultModel: 'MiniMax-M2.5',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'minimax-anthropic-cn',
    label: 'MiniMax（海螺 AI / Anthropic 兼容 / 国内）',
    labelEn: 'MiniMax (Anthropic / China)',
    labelZh: 'MiniMax（海螺 AI / Anthropic / 国内）',
    descEn: 'MiniMax-M2.5 | Claude Agent SDK | China endpoint',
    descZh: 'MiniMax-M2.5 | Claude Agent SDK | 国内端点',
    engine: 'claude',
    apiKeyEnvName: 'ANTHROPIC_API_KEY',
    defaultBaseUrl: 'https://api.minimaxi.com/anthropic',
    defaultModel: 'MiniMax-M2.5',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'baichuan',
    label: '百川智能',
    labelEn: 'Baichuan AI',
    labelZh: '百川智能',
    descEn: 'Baichuan4 | General chat',
    descZh: 'Baichuan4 | 通用对话模型',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://api.baichuan-ai.com/v1',
    defaultModel: 'Baichuan4',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'stepfun',
    label: '阶跃星辰（Step）',
    labelEn: 'StepFun (Step)',
    labelZh: '阶跃星辰（Step）',
    descEn: 'Step-2 | 256K context',
    descZh: 'Step-2 | 256K 上下文',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://api.stepfun.com/v1',
    defaultModel: 'step-2-16k',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'xiaomi',
    label: '小米 MiMo',
    labelEn: 'Xiaomi MiMo',
    labelZh: '小米 MiMo',
    descEn: 'MiMo-V2-Flash | 262K context | Anthropic API',
    descZh: 'MiMo-V2-Flash | 262K 上下文 | Anthropic 兼容 API',
    engine: 'claude',
    apiKeyEnvName: 'ANTHROPIC_API_KEY',
    defaultBaseUrl: 'https://api.xiaomimimo.com/anthropic',
    defaultModel: 'mimo-v2-flash',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  //  API 聚合平台
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    id: 'openrouter',
    label: 'OpenRouter（OpenAI 兼容）',
    labelEn: 'OpenRouter (OpenAI)',
    labelZh: 'OpenRouter（OpenAI 兼容）',
    descEn: 'API aggregator | 320+ models | Pay per use',
    descZh: 'API 聚合 | 320+ 模型 | 按量付费',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    apiKeyPrefix: 'sk-or-',
    defaultBaseUrl: 'https://openrouter.ai/api/v1',
    defaultModel: 'anthropic/claude-sonnet-4',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'openrouter-anthropic',
    label: 'OpenRouter（Anthropic 兼容）',
    labelEn: 'OpenRouter (Anthropic)',
    labelZh: 'OpenRouter（Anthropic 兼容）',
    descEn: 'API aggregator | Claude Agent SDK | Multi-provider failover',
    descZh: 'API 聚合 | Claude Agent SDK | 多供应商自动 failover',
    engine: 'claude',
    apiKeyEnvName: 'ANTHROPIC_API_KEY',
    apiKeyPrefix: 'sk-or-',
    defaultBaseUrl: 'https://openrouter.ai/api',
    defaultModel: 'anthropic/claude-sonnet-4',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'siliconflow',
    label: '硅基流动（SiliconFlow）',
    labelEn: 'SiliconFlow',
    labelZh: '硅基流动（SiliconFlow）',
    descEn: 'API aggregator | DeepSeek/Qwen/Llama | China accelerated',
    descZh: 'API 聚合 | DeepSeek/Qwen/Llama | 国内加速',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://api.siliconflow.cn/v1',
    defaultModel: 'deepseek-ai/DeepSeek-V3',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'together',
    label: 'Together AI',
    labelEn: 'Together AI',
    labelZh: 'Together AI',
    descEn: 'API aggregator | Llama/DeepSeek/GLM/Kimi',
    descZh: 'API 聚合 | Llama/DeepSeek/GLM/Kimi',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://api.together.xyz/v1',
    defaultModel: 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'nvidia',
    label: 'NVIDIA NIM',
    labelEn: 'NVIDIA NIM',
    labelZh: 'NVIDIA NIM',
    descEn: 'Llama / Nemotron | Free tier | GPU optimized',
    descZh: 'Llama / Nemotron | 免费额度 | GPU 优化推理',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://integrate.api.nvidia.com/v1',
    defaultModel: 'meta/llama-3.3-70b-instruct',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'venice',
    label: 'Venice AI',
    labelEn: 'Venice AI',
    labelZh: 'Venice AI',
    descEn: 'Privacy-first | No logs | Open source models',
    descZh: '隐私优先 | 无日志 | 开源模型',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://api.venice.ai/api/v1',
    defaultModel: 'kimi-k2-5',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'huggingface',
    label: 'Hugging Face Inference',
    labelEn: 'Hugging Face Inference',
    labelZh: 'Hugging Face Inference',
    descEn: 'Open source models | Free inference API',
    descZh: '海量开源模型 | 免费推理 API',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    apiKeyPrefix: 'hf_',
    defaultBaseUrl: 'https://api-inference.huggingface.co/v1',
    defaultModel: 'meta-llama/Llama-3.3-70B-Instruct',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  //  云平台
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    id: 'bedrock',
    label: 'Amazon Bedrock',
    labelEn: 'Amazon Bedrock',
    labelZh: 'Amazon Bedrock',
    descEn: 'Claude / Llama / Mistral via AWS',
    descZh: 'Claude / Llama / Mistral via AWS | 需要 AWS 凭据',
    engine: 'claude',
    apiKeyEnvName: 'ANTHROPIC_API_KEY',
    needsBaseUrl: true,
    needsModel: true,
    defaultModel: 'anthropic.claude-sonnet-4-20250514-v1:0',
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'github-copilot',
    label: 'GitHub Copilot',
    labelEn: 'GitHub Copilot',
    labelZh: 'GitHub Copilot',
    descEn: 'GPT-4.1 / Claude | Copilot subscription required',
    descZh: 'GPT-4.1 / Claude | 需要 Copilot 订阅',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://api.githubcopilot.com',
    defaultModel: 'gpt-4.1',
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  //  本地 / 自托管
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    id: 'ollama',
    label: 'Ollama',
    labelEn: 'Ollama',
    labelZh: 'Ollama',
    descEn: 'Local models | No API key needed | Qwen/Llama/DeepSeek',
    descZh: '本地模型 | 无需 API Key | 支持 Qwen/Llama/DeepSeek',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'http://host.docker.internal:11434/v1',
    defaultModel: 'qwen2.5:72b',
    needsModel: true,
    keyOptional: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'vllm',
    label: 'vLLM',
    labelEn: 'vLLM',
    labelZh: 'vLLM',
    descEn: 'Self-hosted | High-performance inference | OpenAI compatible',
    descZh: '自托管 | 高性能推理引擎 | OpenAI 兼容',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    needsBaseUrl: true,
    needsModel: true,
    keyOptional: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'sglang',
    label: 'SGLang',
    labelEn: 'SGLang',
    labelZh: 'SGLang',
    descEn: 'Self-hosted | Ultra-fast inference | OpenAI compatible',
    descZh: '自托管 | 超快推理 | OpenAI 兼容',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    needsBaseUrl: true,
    needsModel: true,
    keyOptional: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'lmstudio',
    label: 'LM Studio',
    labelEn: 'LM Studio',
    labelZh: 'LM Studio',
    descEn: 'Desktop local inference | OpenAI compatible | Free',
    descZh: '桌面本地推理 | OpenAI 兼容 | 免费',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    defaultBaseUrl: 'http://host.docker.internal:1234/v1',
    needsModel: true,
    keyOptional: true,
    authMethods: ['api_key', 'env_ref'],
  },

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  //  自定义
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    id: 'custom-openai',
    label: '自定义 OpenAI 兼容 API',
    labelEn: 'Custom OpenAI-compatible API',
    labelZh: '自定义 OpenAI 兼容 API',
    descEn: 'Any OpenAI Chat Completions endpoint',
    descZh: '任何兼容 OpenAI Chat Completions 的端点',
    engine: 'codex',
    apiKeyEnvName: 'OPENAI_API_KEY',
    needsBaseUrl: true,
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
  {
    id: 'custom-anthropic',
    label: '自定义 Anthropic 兼容 API',
    labelEn: 'Custom Anthropic-compatible API',
    labelZh: '自定义 Anthropic 兼容 API',
    descEn: 'Any Anthropic Messages endpoint (proxy, gateway, etc.)',
    descZh: '任何兼容 Anthropic Messages 的端点（代理、网关等）',
    engine: 'claude',
    apiKeyEnvName: 'ANTHROPIC_API_KEY',
    needsBaseUrl: true,
    needsModel: true,
    authMethods: ['api_key', 'env_ref'],
  },
];

// Keys belonging to each engine (for cleanup when switching)
const ENGINE_KEYS: Record<string, string[]> = {
  claude: ['ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL', 'CLAUDE_CODE_OAUTH_TOKEN'],
  codex: ['CODEX_API_KEY', 'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'CODEX_MODEL'],
  gemini: ['GOOGLE_API_KEY'],
};

// ── Helpers ──

function sanitizeApiKey(raw: string): string {
  let key = raw.trim();
  // Strip shell assignment: export KEY=value or KEY=value
  // Only match ALL_CAPS_SNAKE_CASE= patterns to avoid stripping valid key content
  key = key.replace(/^export\s+[A-Z_][A-Z0-9_]*=/, '');
  key = key.replace(/^[A-Z_][A-Z0-9_]*=/, '');
  // Strip surrounding quotes
  if (
    (key.startsWith('"') && key.endsWith('"')) ||
    (key.startsWith("'") && key.endsWith("'")) ||
    (key.startsWith('`') && key.endsWith('`'))
  ) {
    key = key.slice(1, -1);
  }
  // Strip trailing semicolons
  key = key.replace(/;+$/, '');
  return key.trim();
}

function maskKey(key: string): string {
  if (key.length <= 8) return '****';
  return key.slice(0, 4) + '****' + key.slice(-4);
}

function detectOAuthToken(): string | undefined {
  const home = os.homedir();
  const candidates = [
    path.join(home, '.claude', '.credentials.json'),
    path.join(home, '.claude', 'credentials.json'),
    ...(process.platform === 'darwin'
      ? [path.join(home, 'Library', 'Application Support', 'Claude', 'credentials.json')]
      : []),
    path.join(home, '.config', 'claude', 'credentials.json'),
  ];

  for (const credFile of candidates) {
    try {
      if (!fs.existsSync(credFile)) continue;
      const data = JSON.parse(fs.readFileSync(credFile, 'utf-8'));
      const oauth = data.claudeAiOauth ?? data.oauth ?? data;
      const token = oauth?.accessToken ?? oauth?.access_token;
      if (!token) continue;
      const expiresAt = oauth.expiresAt ?? oauth.expires_at;
      if (expiresAt) {
        const expiresMs = expiresAt > 1e12 ? expiresAt : expiresAt * 1000;
        if (Date.now() > expiresMs) continue;
      }
      return token;
    } catch { /* try next */ }
  }
  return undefined;
}

interface ExistingConfig {
  engine: string;
  provider: string;
  maskedKey: string;
}

function detectExistingConfig(projectRoot: string): ExistingConfig | null {
  const env = readEnvFile([
    'AGENT_ENGINE',
    'ANTHROPIC_API_KEY',
    'ANTHROPIC_BASE_URL',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'OPENAI_API_KEY',
    'OPENAI_BASE_URL',
    'GOOGLE_API_KEY',
  ]);

  const engine = env.AGENT_ENGINE || 'claude';

  if (env.ANTHROPIC_API_KEY) {
    // Identify provider from base URL
    const baseUrl = env.ANTHROPIC_BASE_URL || '';
    const provider = baseUrl
      ? PROVIDERS.find((p) => {
          try {
            return p.engine === 'claude' && p.defaultBaseUrl && baseUrl.includes(new URL(p.defaultBaseUrl).hostname);
          } catch { return false; }
        })
      : undefined;
    const label = provider ? provider.label : 'Anthropic Claude';
    return { engine, provider: label, maskedKey: maskKey(env.ANTHROPIC_API_KEY) };
  }
  if (env.CLAUDE_CODE_OAUTH_TOKEN) {
    return { engine, provider: 'Claude OAuth', maskedKey: 'OAuth token (auto-detected)' };
  }
  if (env.GOOGLE_API_KEY) {
    return { engine, provider: 'Google Gemini', maskedKey: maskKey(env.GOOGLE_API_KEY) };
  }
  if (env.OPENAI_API_KEY) {
    // Identify provider from base URL
    const baseUrl = env.OPENAI_BASE_URL || '';
    const provider = baseUrl
      ? PROVIDERS.find((p) => {
          try {
            return p.engine === 'codex' && p.defaultBaseUrl && baseUrl.includes(new URL(p.defaultBaseUrl).hostname);
          } catch { return false; }
        })
      : undefined;
    const label = provider ? provider.label : (engine === 'codex' ? 'OpenAI' : 'OpenAI-compatible');
    return { engine, provider: label, maskedKey: maskKey(env.OPENAI_API_KEY) };
  }

  // Check OAuth fallback
  const oauthToken = detectOAuthToken();
  if (oauthToken) {
    return { engine: 'claude', provider: 'Claude OAuth', maskedKey: 'OAuth token (auto-detected)' };
  }

  return null;
}

// ── API Validation ──

interface ValidationResult {
  ok: boolean;
  error?: string;
}

async function validateAnthropicKey(
  apiKey: string,
  baseUrl?: string,
  model?: string,
): Promise<ValidationResult> {
  const base = (baseUrl || 'https://api.anthropic.com').replace(/\/+$/, '');
  const url = `${base}/v1/messages`;
  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: model || 'claude-sonnet-4-20250514',
        max_tokens: 1,
        messages: [{ role: 'user', content: 'hi' }],
      }),
      signal: AbortSignal.timeout(15000),
    });
    if (resp.ok || resp.status === 200) return { ok: true };
    // 400 with "max_tokens" is still a valid key, just bad request
    if (resp.status === 400) return { ok: true };
    const body = await resp.text().catch(() => '');
    if (resp.status === 401) return { ok: false, error: t('api.invalid401') };
    if (resp.status === 403) return { ok: false, error: t('api.forbidden403') };
    if (resp.status === 429) return { ok: true }; // Rate limited but key is valid
    return { ok: false, error: `HTTP ${resp.status}: ${body.slice(0, 200)}` };
  } catch (err) {
    return { ok: false, error: t('api.networkError', { error: err instanceof Error ? err.message : String(err) }) };
  }
}

async function validateOpenAIKey(
  apiKey: string,
  baseUrl?: string,
  model?: string,
): Promise<ValidationResult> {
  // Base URLs already include version path (/v1, /v2, /v3, /v4), just append /chat/completions
  const base = (baseUrl || 'https://api.openai.com/v1').replace(/\/+$/, '');
  const url = `${base}/chat/completions`;
  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: model || 'gpt-4o-mini',
        max_tokens: 1,
        messages: [{ role: 'user', content: 'hi' }],
      }),
      signal: AbortSignal.timeout(15000),
    });
    if (resp.ok) return { ok: true };
    if (resp.status === 400) return { ok: true };
    if (resp.status === 429) return { ok: true };
    if (resp.status === 401) return { ok: false, error: t('api.invalid401') };
    const body = await resp.text().catch(() => '');
    return { ok: false, error: `HTTP ${resp.status}: ${body.slice(0, 200)}` };
  } catch (err) {
    return { ok: false, error: t('api.networkError', { error: err instanceof Error ? err.message : String(err) }) };
  }
}

async function validateGeminiKey(
  apiKey: string,
  model?: string,
): Promise<ValidationResult> {
  const m = model || 'gemini-2.5-flash';
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${m}:generateContent?key=${apiKey}`;
  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: 'hi' }] }],
      }),
      signal: AbortSignal.timeout(15000),
    });
    if (resp.ok) return { ok: true };
    if (resp.status === 400) return { ok: true };
    if (resp.status === 429) return { ok: true };
    if (resp.status === 401 || resp.status === 403) {
      return { ok: false, error: t('api.invalid401') };
    }
    const body = await resp.text().catch(() => '');
    return { ok: false, error: `HTTP ${resp.status}: ${body.slice(0, 200)}` };
  } catch (err) {
    return { ok: false, error: t('api.networkError', { error: err instanceof Error ? err.message : String(err) }) };
  }
}

// ── Main wizard ──

export async function run(_args: string[]): Promise<void> {
  const projectRoot = process.cwd();

  // Non-interactive mode
  if (!process.stdin.isTTY) {
    const existing = detectExistingConfig(projectRoot);
    if (existing) {
      emitStatus('CONFIGURE_API', {
        STATUS: 'ok',
        PROVIDER: existing.provider,
        ENGINE: existing.engine,
        KEY: existing.maskedKey,
        MODE: 'non-interactive',
      });
    } else {
      emitStatus('CONFIGURE_API', {
        STATUS: 'failed',
        ERROR: 'No API key configured. Run interactively: npx tsx setup/index.ts --step configure-api',
      });
      process.exit(1);
    }
    return;
  }

  // Detect language from env or args
  const langArg = _args.find((_, i, a) => a[i - 1] === '--lang') as Locale | undefined;
  setLocale(langArg || detectLocale());

  // Suppress pino logger output during interactive wizard
  process.env.LOG_LEVEL = 'silent';

  println();
  println(boxTop(t('api.title')));
  println(boxLine(`${c.dim}${t('api.detecting')}${c.reset}`));

  // Step 1: Auto-detection
  const existing = detectExistingConfig(projectRoot);

  if (existing) {
    info(t('api.current', { provider: existing.provider, key: existing.maskedKey }));
    println(boxBottom());
    const keepExisting = await confirm({
      message: t('api.keepCurrent'),
      default: true,
    });
    if (keepExisting) {
      ok(t('api.keeping'));
      emitStatus('CONFIGURE_API', {
        STATUS: 'ok',
        PROVIDER: existing.provider,
        ENGINE: existing.engine,
        KEY: existing.maskedKey,
        MODE: 'kept-existing',
      });
      return;
    }
  } else {
    info(t('api.noExisting'));
    println(boxBottom());
  }

  // Step 2: Provider selection (categorized)
  const categories: { label: string; ids: string[] }[] = [
    { label: t('api.cat.recommended'),
      ids: ['anthropic'] },
    { label: t('api.cat.international'),
      ids: ['gemini', 'openai', 'deepseek', 'mistral', 'xai', 'perplexity'] },
    { label: t('api.cat.domestic'),
      ids: ['qwen', 'zhipu', 'moonshot', 'qianfan', 'volcengine', 'baichuan', 'stepfun', 'xiaomi'] },
    { label: t('api.cat.codingPlan'),
      ids: ['qwen-coding', 'qwen-coding-anthropic', 'qwen-coding-intl', 'qwen-coding-intl-anthropic',
            'zhipu-anthropic', 'zhipu-coding', 'zhipu-coding-intl', 'moonshot-intl', 'kimi-coding',
            'volcengine-anthropic', 'minimax', 'minimax-anthropic', 'minimax-anthropic-cn', 'byteplus'] },
    { label: t('api.cat.aggregator'),
      ids: ['openrouter', 'openrouter-anthropic', 'siliconflow', 'together', 'nvidia', 'venice', 'huggingface'] },
    { label: t('api.cat.cloud'),
      ids: ['bedrock', 'github-copilot'] },
    { label: t('api.cat.selfHosted'),
      ids: ['ollama', 'vllm', 'sglang', 'lmstudio'] },
    { label: t('api.cat.custom'),
      ids: ['custom-openai', 'custom-anthropic'] },
  ];

  const categorizedIds = new Set(categories.flatMap((cat) => cat.ids));
  const providerChoices: (Separator | { name: string; value: string })[] = [];

  for (const cat of categories) {
    providerChoices.push(new Separator(`\n  ${c.brightCyan}━━ ${cat.label} ━━${c.reset}`));
    for (const id of cat.ids) {
      const p = PROVIDERS.find((pr) => pr.id === id);
      if (p) {
        providerChoices.push({
          name: `  ${providerLabel(p)}  ${c.dim}${providerDesc(p) || ''}${c.reset}`,
          value: p.id,
        });
      }
    }
  }

  // Append any providers not in a category
  const uncategorized = PROVIDERS.filter((p) => !categorizedIds.has(p.id));
  if (uncategorized.length > 0) {
    providerChoices.push(new Separator(`\n  ${c.brightCyan}━━ Other ━━${c.reset}`));
    for (const p of uncategorized) {
      providerChoices.push({
        name: `  ${providerLabel(p)}  ${c.dim}${providerDesc(p) || ''}${c.reset}`,
        value: p.id,
      });
    }
  }

  const provider = await select({
    message: t('api.selectProvider'),
    choices: providerChoices,
    pageSize: 20,
  });

  const providerConfig = PROVIDERS.find((p) => p.id === provider)!;

  // Step 3: Auth method
  let apiKey = '';
  let useOAuth = false;
  let useEnvRef = false;
  let envRefName = '';

  if (providerConfig.authMethods.length > 1) {
    const choices: { name: string; value: string }[] = [];
    if (providerConfig.authMethods.includes('api_key')) {
      choices.push({ name: t('api.authPasteKey'), value: 'api_key' });
    }
    if (providerConfig.authMethods.includes('oauth_auto')) {
      const oauthToken = detectOAuthToken();
      const label = oauthToken
        ? t('api.authOAuthDetected')
        : t('api.authOAuthNotDetected');
      choices.push({ name: label, value: 'oauth_auto' });
    }
    if (providerConfig.authMethods.includes('env_ref')) {
      choices.push({ name: t('api.authEnvImport'), value: 'env_ref' });
    }

    const authMethod = await select({
      message: t('api.authMethod'),
      choices,
    });

    if (authMethod === 'oauth_auto') {
      const token = detectOAuthToken();
      if (!token) {
        ora().fail(t('api.authOAuthFail'));
        process.exit(1);
      }
      useOAuth = true;
      ora().succeed(t('api.authOAuthSuccess'));
    } else if (authMethod === 'env_ref') {
      useEnvRef = true;
      envRefName = await input({
        message: t('api.envVarName'),
        default: providerConfig.apiKeyEnvName,
        validate: (val) => {
          if (!/^[A-Z_][A-Z0-9_]*$/.test(val)) {
            return t('api.envVarInvalid');
          }
          if (!process.env[val]) {
            return t('api.envVarNotSet', { name: val });
          }
          return true;
        },
      });
      apiKey = process.env[envRefName]!;
    } else {
      // api_key — fall through to key input below
    }
  }

  // Step 4: API Key input
  if (!useOAuth && !useEnvRef && !apiKey) {
    if (providerConfig.keyOptional) {
      const wantKey = await confirm({
        message: t('api.keyOptionalPrompt', { provider: providerLabel(providerConfig) }),
        default: false,
      });
      if (wantKey) {
        const raw = await input({
          message: t('api.keyOptional'),
        });
        apiKey = sanitizeApiKey(raw);
        if (apiKey) console.log(`   Key: ${maskKey(apiKey)}`);
      }
      if (!apiKey) {
        apiKey = 'ollama'; // Placeholder — Ollama accepts any non-empty key
      }
    } else {
      const raw = await password({
        message: t('api.enterKey', { provider: providerLabel(providerConfig) }),
        mask: '*',
        validate: (val) => {
          const cleaned = sanitizeApiKey(val);
          if (!cleaned) return t('api.keyEmpty');
          if (
            providerConfig.apiKeyPrefix &&
            !cleaned.startsWith(providerConfig.apiKeyPrefix)
          ) {
            return t('api.keyBadPrefix', { prefix: providerConfig.apiKeyPrefix });
          }
          return true;
        },
      });
      apiKey = sanitizeApiKey(raw);
      console.log(`   Key: ${maskKey(apiKey)}`);
    }
  }

  // Step 5: Base URL
  let baseUrl = providerConfig.defaultBaseUrl || '';
  if (providerConfig.needsBaseUrl) {
    baseUrl = await input({
      message: t('api.baseUrl'),
      default: providerConfig.defaultBaseUrl || '',
      validate: (val) => {
        if (!val) return t('api.baseUrlEmpty');
        try {
          new URL(val);
          return true;
        } catch {
          return t('api.baseUrlInvalid');
        }
      },
    });
  }

  // Step 6: Model
  let model = providerConfig.defaultModel || '';
  if (providerConfig.needsModel) {
    model = await input({
      message: t('api.modelName'),
      default: providerConfig.defaultModel || '',
      validate: (val) => (val ? true : t('api.modelEmpty')),
    });
  }

  // Step 7: API Validation (skip for local providers with placeholder keys)
  if (!useOAuth && apiKey && !providerConfig.keyOptional) {
    const spinner = ora({ text: t('api.validating'), spinner: 'dots' }).start();
    let result: ValidationResult;

    if (providerConfig.engine === 'claude') {
      result = await validateAnthropicKey(apiKey, baseUrl || undefined, model || undefined);
    } else if (providerConfig.engine === 'gemini') {
      result = await validateGeminiKey(apiKey, model || undefined);
    } else {
      result = await validateOpenAIKey(apiKey, baseUrl || undefined, model || undefined);
    }

    if (result.ok) {
      spinner.succeed(t('api.validated'));
    } else {
      spinner.fail(t('api.validationFailed', { error: result.error || '' }));
      const proceed = await confirm({
        message: t('api.saveAnyway'),
        default: false,
      });
      if (!proceed) {
        ora().fail(t('api.cancelled'));
        process.exit(1);
      }
    }
  }

  // Warn if engine is not yet implemented
  if (providerConfig.engine === 'gemini') {
    ora().warn(t('api.geminiWarning'));
    const proceed = await confirm({
      message: t('api.saveConfig'),
      default: true,
    });
    if (!proceed) {
      ora().fail(t('api.cancelled'));
      process.exit(1);
    }
  }

  // Step 8: Write to .env
  // Clean up keys from other engines
  const currentEnv = readEnvFile(['AGENT_ENGINE', 'ANTHROPIC_API_KEY', 'CLAUDE_CODE_OAUTH_TOKEN']);
  const currentEngine = currentEnv.AGENT_ENGINE || 'claude';
  if (currentEngine !== providerConfig.engine) {
    const otherKeys = Object.entries(ENGINE_KEYS)
      .filter(([eng]) => eng !== providerConfig.engine)
      .flatMap(([, keys]) => keys);
    removeEnvKeys(projectRoot, otherKeys);
  }
  // Always clean stale base URL / model from same engine (e.g. switching DeepSeek → OpenAI)
  // They share the same env var names, so writeEnvKeys will overwrite the key,
  // but if the new provider has no base URL we must remove the old one.
  if (!baseUrl && providerConfig.engine === 'codex') {
    removeEnvKeys(projectRoot, ['OPENAI_BASE_URL']);
  }
  if (!baseUrl && providerConfig.engine === 'claude') {
    removeEnvKeys(projectRoot, ['ANTHROPIC_BASE_URL']);
  }

  // Always clean stale model settings before writing new ones.
  // Prevents e.g. AGENT_MODEL=glm-5 lingering when switching to Anthropic (no model).
  removeEnvKeys(projectRoot, ['AGENT_MODEL', 'CODEX_MODEL']);

  // Build updates
  const updates: Record<string, string> = {
    AGENT_ENGINE: providerConfig.engine,
  };

  if (useOAuth) {
    // OAuth auto-detected — remove stale API key so OAuth takes effect
    removeEnvKeys(projectRoot, ['ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL']);
  } else if (apiKey) {
    updates[providerConfig.apiKeyEnvName] = apiKey;
    // Using explicit API key — remove stale OAuth token from .env (if any)
    if (providerConfig.engine === 'claude') {
      removeEnvKeys(projectRoot, ['CLAUDE_CODE_OAUTH_TOKEN']);
    }
  }

  if (baseUrl) {
    if (providerConfig.engine === 'codex') {
      updates.OPENAI_BASE_URL = baseUrl;
    } else if (providerConfig.engine === 'claude') {
      updates.ANTHROPIC_BASE_URL = baseUrl;
    }
  }

  if (model) {
    if (providerConfig.engine === 'codex') {
      updates.CODEX_MODEL = model;
    } else {
      updates.AGENT_MODEL = model;
    }
  }

  const saveSpinner = ora({ text: t('api.saving'), spinner: 'dots' }).start();
  writeEnvKeys(projectRoot, updates);
  saveSpinner.succeed(t('api.saved', { provider: providerLabel(providerConfig) }));

  emitStatus('CONFIGURE_API', {
    STATUS: 'ok',
    PROVIDER: providerConfig.label,
    ENGINE: providerConfig.engine,
    KEY: useOAuth ? 'OAuth' : maskKey(apiKey),
  });
}
