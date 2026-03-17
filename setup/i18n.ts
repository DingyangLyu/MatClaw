/**
 * Lightweight i18n for MatClaw setup tools.
 * No dependencies — plain TypeScript objects.
 */

export type Locale = 'en' | 'zh';

let currentLocale: Locale = 'en';

export function setLocale(l: Locale): void {
  currentLocale = l;
}

export function getLocale(): Locale {
  return currentLocale;
}

/**
 * Auto-detect locale from environment.
 * Used when configure-api runs standalone (not spawned from wizard).
 */
export function detectLocale(): Locale {
  // Explicit MatClaw setting takes priority
  if (process.env.MATCLAW_LANG === 'zh' || process.env.MATCLAW_LANG === 'en') {
    return process.env.MATCLAW_LANG;
  }
  // Check system locale
  const lang = process.env.LANG || process.env.LC_ALL || process.env.LANGUAGE || '';
  if (lang.startsWith('zh')) return 'zh';
  return 'en';
}

type TranslationMap = Record<string, string>;

/**
 * Translate a key, with optional variable interpolation.
 * Variables use {name} syntax: t('hello', { name: 'world' }) → "Hello world"
 * Falls back to English, then to the raw key.
 */
export function t(key: string, vars?: Record<string, string | number>): string {
  let text = translations[currentLocale]?.[key] ?? translations['en']?.[key] ?? key;
  if (vars) {
    for (const [k, v] of Object.entries(vars)) {
      text = text.replaceAll(`{${k}}`, String(v));
    }
  }
  return text;
}

// ── Translations ────────────────────────────────────────────────────────────

const en: TranslationMap = {
  // ── Language selector ──
  'lang.prompt': 'Language / 语言',

  // ── Banner ──
  'banner.tagline': 'AI-Powered Autonomous Materials Science Agent',

  // ── Step titles ──
  'step.environment': 'Environment',
  'step.dependencies': 'Dependencies',
  'step.container': 'Container',
  'step.api': 'API Provider',
  'step.smoke': 'Smoke Test',
  'step.channels': 'Messaging Channels',
  'step.launch': 'Launch',

  // ── Step 1: Environment ──
  'env.required': 'required',
  'env.notFound': 'not found (optional)',
  'env.running': 'running',
  'env.daemonNotRunning': 'Docker daemon not running',
  'env.nodeRequired': 'Node.js is required. Install from https://nodejs.org/',

  // ── Step 2: Dependencies ──
  'deps.upToDate': 'All dependencies up to date',
  'deps.installing': 'Running npm install...',
  'deps.spinnerInstalling': 'Installing dependencies...',
  'deps.installed': 'Dependencies installed',
  'deps.failed': 'npm install failed — run manually to see errors',

  // ── Step 3: Container ──
  'container.notAvailable': 'Docker not available — skipping',
  'container.setupLater': 'Run ./container/build.sh later to set up',
  'container.imageFound': 'Container image found',
  'container.rebuild': 'Rebuild container image?',
  'container.method': 'Container setup method',
  'container.pull': 'Pull pre-built',
  'container.pullDesc': 'fastest, ~2 min',
  'container.build': 'Build from source',
  'container.buildDesc': '~10 min, compiles QE',
  'container.buildCuda': 'Build with CUDA',
  'container.buildCudaDesc': '~15 min, GPU support',
  'container.skip': 'Skip',
  'container.pulling': 'Pulling ghcr.io/dingyanglyu/matclaw-agent:latest ...',
  'container.pulled': 'Image pulled and tagged',
  'container.pullFailed': 'Pull failed — try building from source',
  'container.building': 'Building container...',
  'container.buildingCuda': 'Building container with CUDA...',
  'container.buildFailed': 'Build failed',

  // ── Step 4: API ──
  'api.configured': 'API credentials configured',
  'api.reconfigure': 'Reconfigure API provider?',
  'api.notDetected': 'No API credentials detected',
  'api.launching': 'Launching API wizard...',
  'api.done': 'API provider configured',
  'api.skipped': 'Skipped or failed — run npm run setup:api later',

  // ── Step 5: Smoke Test ──
  'smoke.notAvailable': 'Docker not available — skipping',
  'smoke.noImage': 'No container image — skipping',
  'smoke.verifies': 'Verifies: QE, LAMMPS, MACE, pymatgen, ASE, PyTorch, Node.js',
  'smoke.run': 'Run smoke test?',
  'smoke.running': 'Running smoke test inside container...',
  'smoke.notFound': 'smoke-test.py not found',
  'smoke.someFailed': 'Some checks failed',
  'smoke.allPassed': 'All smoke tests passed',
  'smoke.failed': 'Smoke test failed or timed out',

  // ── Step 6: Channels ──
  'channels.webNote': 'The Web UI at localhost:3210 works without any channel.',
  'channels.channelNote': 'Channels let you chat from Feishu, Telegram, etc.',
  'channels.setup': 'Set up messaging channels now?',
  'channels.addLater': 'Add anytime with /add-* commands in claude CLI',
  'channels.select': 'Select channels (Space to toggle, Enter to confirm)',
  'channels.noneSelected': 'No channels selected',
  'channels.feishu': 'Feishu',
  'channels.feishuDesc': 'WebSocket, no public URL',
  'channels.dingtalk': 'DingTalk',
  'channels.dingtalkDesc': 'Stream Mode, no public URL',
  'channels.telegram': 'Telegram',
  'channels.telegramDesc': 'Bot API',
  'channels.discord': 'Discord',
  'channels.discordDesc': 'Bot with slash commands',
  'channels.slack': 'Slack',
  'channels.slackDesc': 'Socket Mode',
  'channels.gmail': 'Gmail',
  'channels.gmailDesc': 'Email-based tasks',
  'channels.whatsapp': 'WhatsApp',
  'channels.whatsappDesc': 'QR code auth',
  'channels.guide.feishu': 'docs/feishu-setup.md → then claude /add-feishu',
  'channels.guide.dingtalk': 'docs/dingtalk-setup.md → then claude /add-dingtalk',
  'channels.guide.telegram': 'claude /add-telegram',
  'channels.guide.discord': 'claude /add-discord',
  'channels.guide.slack': 'claude /add-slack',
  'channels.guide.gmail': 'claude /add-gmail',
  'channels.guide.whatsapp': 'claude /add-whatsapp',

  // ── Step 7: Launch ──
  'launch.ready': 'MatClaw is ready!',
  'launch.quickCmds': 'Quick commands',
  'launch.chatCmds': 'Chat commands',
  'launch.watchDesc': 'See what the agent is doing',
  'launch.statusDesc': 'Check agent status',
  'launch.stopDesc': 'Stop running agent',
  'launch.helpDesc': 'All commands',
  'launch.how': 'How to start MatClaw?',
  'launch.startNow': 'Start now',
  'launch.startNowDesc': 'npm run dev (foreground)',
  'launch.service': 'Install as service',
  'launch.serviceDesc': 'auto-start on boot',
  'launch.dontStart': "Don't start yet",
  'launch.serviceInstalling': 'Setting up system service...',
  'launch.serviceInstalled': 'Service installed and started',
  'launch.serviceManage': 'Manage: systemctl --user {start|stop|restart} matclaw',
  'launch.serviceFailed': 'Service setup failed — start manually with npm run dev',
  'launch.startLater': 'Start anytime with: npm run dev',
  'launch.webUI': 'Web UI:',
  'launch.ctrlC': 'Ctrl+C to stop',

  // ── configure-api ──
  'api.title': 'API Configuration',
  'api.detecting': 'Detecting existing configuration...',
  'api.current': 'Current: {provider} ({key})',
  'api.keepCurrent': 'Keep current configuration?',
  'api.keeping': 'Keeping current configuration.',
  'api.noExisting': 'No existing configuration found.',
  'api.selectProvider': 'Select AI provider:',

  // Provider categories
  'api.cat.recommended': 'Recommended',
  'api.cat.international': 'International',
  'api.cat.domestic': 'China Domestic',
  'api.cat.codingPlan': 'Coding Plans',
  'api.cat.aggregator': 'API Aggregators',
  'api.cat.cloud': 'Cloud Platforms',
  'api.cat.selfHosted': 'Self-hosted',
  'api.cat.custom': 'Custom',

  // Auth methods
  'api.authMethod': 'Authentication method:',
  'api.authPasteKey': 'Paste API Key',
  'api.authOAuthDetected': 'Claude Code OAuth (valid token detected)',
  'api.authOAuthNotDetected': 'Claude Code OAuth (not detected)',
  'api.authEnvImport': 'Import from environment variable (one-time copy to .env)',
  'api.authOAuthSuccess': 'Using Claude Code OAuth token (auto-detected).',
  'api.authOAuthFail': 'No valid Claude Code OAuth token found. Run `claude` to log in.',

  // Key input
  'api.envVarName': 'Environment variable name:',
  'api.envVarInvalid': 'Use uppercase letters, numbers, underscore (e.g. OPENAI_API_KEY)',
  'api.envVarNotSet': 'Environment variable {name} is not set or empty',
  'api.keyOptionalPrompt': '{provider} usually needs no API Key. Set one?',
  'api.keyOptional': 'API Key (optional):',
  'api.enterKey': 'Enter {provider} API Key:',
  'api.keyEmpty': 'API Key cannot be empty',
  'api.keyBadPrefix': 'Invalid format, should start with {prefix}',
  'api.baseUrl': 'API Base URL:',
  'api.baseUrlEmpty': 'Base URL cannot be empty',
  'api.baseUrlInvalid': 'Enter a valid URL (e.g. https://api.example.com/v1)',
  'api.modelName': 'Model name:',
  'api.modelEmpty': 'Model name cannot be empty',

  // Validation
  'api.validating': 'Validating API key...',
  'api.validated': 'API key validated.',
  'api.validationFailed': 'Validation failed: {error}',
  'api.saveAnyway': 'Still save this key? (might be a network issue)',
  'api.cancelled': 'Cancelled.',
  'api.invalid401': 'Invalid API Key (401 Unauthorized)',
  'api.forbidden403': 'Access denied (403 Forbidden)',
  'api.networkError': 'Network error: {error}',

  // Engine warning
  'api.geminiWarning': 'Gemini engine is not yet implemented. Key saved for future use.',
  'api.saveConfig': 'Still save configuration?',

  // Save
  'api.saving': 'Writing configuration...',
  'api.saved': 'Configuration saved to .env ({provider})',
};

const zh: TranslationMap = {
  // ── Language selector ──
  'lang.prompt': 'Language / 语言',

  // ── Banner ──
  'banner.tagline': 'AI 驱动的自主材料科学智能体',

  // ── Step titles ──
  'step.environment': '环境检查',
  'step.dependencies': '安装依赖',
  'step.container': '容器设置',
  'step.api': 'API 配置',
  'step.smoke': '冒烟测试',
  'step.channels': '消息通道',
  'step.launch': '启动',

  // ── Step 1 ──
  'env.required': '必需',
  'env.notFound': '未找到（可选）',
  'env.running': '运行中',
  'env.daemonNotRunning': 'Docker 守护进程未运行',
  'env.nodeRequired': '需要 Node.js，请从 https://nodejs.org/ 安装',

  // ── Step 2 ──
  'deps.upToDate': '所有依赖已是最新',
  'deps.installing': '正在运行 npm install...',
  'deps.spinnerInstalling': '正在安装依赖...',
  'deps.installed': '依赖安装完成',
  'deps.failed': 'npm install 失败 — 请手动运行查看错误',

  // ── Step 3 ──
  'container.notAvailable': 'Docker 不可用 — 跳过',
  'container.setupLater': '稍后运行 ./container/build.sh 设置',
  'container.imageFound': '已找到容器镜像',
  'container.rebuild': '重新构建容器镜像？',
  'container.method': '容器设置方式',
  'container.pull': '拉取预构建镜像',
  'container.pullDesc': '最快，约 2 分钟',
  'container.build': '从源码构建',
  'container.buildDesc': '约 10 分钟，编译 QE',
  'container.buildCuda': '构建 CUDA 版',
  'container.buildCudaDesc': '约 15 分钟，GPU 加速',
  'container.skip': '跳过',
  'container.pulling': '正在拉取 ghcr.io/dingyanglyu/matclaw-agent:latest ...',
  'container.pulled': '镜像拉取并标记完成',
  'container.pullFailed': '拉取失败 — 请尝试从源码构建',
  'container.building': '正在构建容器...',
  'container.buildingCuda': '正在构建 CUDA 容器...',
  'container.buildFailed': '构建失败',

  // ── Step 4 ──
  'api.configured': 'API 凭据已配置',
  'api.reconfigure': '重新配置 API 供应商？',
  'api.notDetected': '未检测到 API 凭据',
  'api.launching': '正在启动 API 配置向导...',
  'api.done': 'API 供应商配置完成',
  'api.skipped': '已跳过 — 稍后运行 npm run setup:api 配置',

  // ── Step 5 ──
  'smoke.notAvailable': 'Docker 不可用 — 跳过',
  'smoke.noImage': '未找到容器镜像 — 跳过',
  'smoke.verifies': '检查项：QE、LAMMPS、MACE、pymatgen、ASE、PyTorch、Node.js',
  'smoke.run': '运行冒烟测试？',
  'smoke.running': '正在容器中运行冒烟测试...',
  'smoke.notFound': '未找到 smoke-test.py',
  'smoke.someFailed': '部分检查失败',
  'smoke.allPassed': '所有冒烟测试通过',
  'smoke.failed': '冒烟测试失败或超时',

  // ── Step 6 ──
  'channels.webNote': 'Web 界面 (localhost:3210) 无需任何通道即可使用。',
  'channels.channelNote': '消息通道让你可以从飞书、Telegram 等平台与 Agent 对话。',
  'channels.setup': '现在设置消息通道？',
  'channels.addLater': '随时通过 claude CLI 中的 /add-* 命令添加',
  'channels.select': '选择要配置的通道（空格选中，回车确认）',
  'channels.noneSelected': '未选择任何通道',
  'channels.feishu': '飞书',
  'channels.feishuDesc': 'WebSocket，无需公网 URL',
  'channels.dingtalk': '钉钉',
  'channels.dingtalkDesc': 'Stream 模式，无需公网 URL',
  'channels.telegram': 'Telegram',
  'channels.telegramDesc': 'Bot API',
  'channels.discord': 'Discord',
  'channels.discordDesc': 'Bot + 斜杠命令',
  'channels.slack': 'Slack',
  'channels.slackDesc': 'Socket 模式',
  'channels.gmail': 'Gmail',
  'channels.gmailDesc': '邮件任务',
  'channels.whatsapp': 'WhatsApp',
  'channels.whatsappDesc': '扫码认证',
  'channels.guide.feishu': 'docs/feishu-setup.md → 然后 claude /add-feishu',
  'channels.guide.dingtalk': 'docs/dingtalk-setup.md → 然后 claude /add-dingtalk',
  'channels.guide.telegram': 'claude /add-telegram',
  'channels.guide.discord': 'claude /add-discord',
  'channels.guide.slack': 'claude /add-slack',
  'channels.guide.gmail': 'claude /add-gmail',
  'channels.guide.whatsapp': 'claude /add-whatsapp',

  // ── Step 7 ──
  'launch.ready': 'MatClaw 已就绪！',
  'launch.quickCmds': '常用命令',
  'launch.chatCmds': '聊天命令',
  'launch.watchDesc': '查看 Agent 正在做什么',
  'launch.statusDesc': '检查 Agent 状态',
  'launch.stopDesc': '停止运行中的 Agent',
  'launch.helpDesc': '所有命令',
  'launch.how': '如何启动 MatClaw？',
  'launch.startNow': '立即启动',
  'launch.startNowDesc': 'npm run dev（前台运行）',
  'launch.service': '安装为系统服务',
  'launch.serviceDesc': '开机自启',
  'launch.dontStart': '暂不启动',
  'launch.serviceInstalling': '正在设置系统服务...',
  'launch.serviceInstalled': '服务已安装并启动',
  'launch.serviceManage': '管理：systemctl --user {start|stop|restart} matclaw',
  'launch.serviceFailed': '服务设置失败 — 请手动运行 npm run dev',
  'launch.startLater': '随时通过 npm run dev 启动',
  'launch.webUI': 'Web 界面：',
  'launch.ctrlC': 'Ctrl+C 停止',

  // ── configure-api ──
  'api.title': 'API 配置',
  'api.detecting': '正在检测已有配置...',
  'api.current': '当前：{provider}（{key}）',
  'api.keepCurrent': '保留当前配置？',
  'api.keeping': '保留当前配置。',
  'api.noExisting': '未检测到已有配置。',
  'api.selectProvider': '选择 AI 供应商：',

  // Provider categories
  'api.cat.recommended': '推荐',
  'api.cat.international': '国际供应商',
  'api.cat.domestic': '国内供应商',
  'api.cat.codingPlan': 'Coding 计划',
  'api.cat.aggregator': 'API 聚合平台',
  'api.cat.cloud': '云平台',
  'api.cat.selfHosted': '本地部署',
  'api.cat.custom': '自定义',

  // Auth methods
  'api.authMethod': '认证方式：',
  'api.authPasteKey': '粘贴 API Key',
  'api.authOAuthDetected': 'Claude Code OAuth（已检测到有效 token）',
  'api.authOAuthNotDetected': 'Claude Code OAuth（未检测到）',
  'api.authEnvImport': '从环境变量导入（一次性复制到 .env）',
  'api.authOAuthSuccess': '已使用 Claude Code OAuth token（自动检测）。',
  'api.authOAuthFail': '未找到有效的 Claude Code OAuth token。请运行 `claude` 登录。',

  // Key input
  'api.envVarName': '环境变量名：',
  'api.envVarInvalid': '请使用大写字母、数字和下划线（如 OPENAI_API_KEY）',
  'api.envVarNotSet': '环境变量 {name} 未设置或为空',
  'api.keyOptionalPrompt': '{provider} 通常不需要 API Key，是否设置？',
  'api.keyOptional': 'API Key（可选）：',
  'api.enterKey': '输入 {provider} API Key：',
  'api.keyEmpty': 'API Key 不能为空',
  'api.keyBadPrefix': '格式不正确，应以 {prefix} 开头',
  'api.baseUrl': 'API Base URL：',
  'api.baseUrlEmpty': 'Base URL 不能为空',
  'api.baseUrlInvalid': '请输入有效的 URL（如 https://api.example.com/v1）',
  'api.modelName': '模型名称：',
  'api.modelEmpty': '模型名称不能为空',

  // Validation
  'api.validating': '正在验证 API Key...',
  'api.validated': 'API Key 验证通过。',
  'api.validationFailed': '验证失败：{error}',
  'api.saveAnyway': '仍然保存此 Key？（可能是网络问题）',
  'api.cancelled': '已取消。',
  'api.invalid401': 'API Key 无效 (401 Unauthorized)',
  'api.forbidden403': '访问被拒 (403 Forbidden)',
  'api.networkError': '网络错误：{error}',

  // Engine warning
  'api.geminiWarning': 'Gemini engine 尚未实现。Key 已保存，待 engine 完成后可用。',
  'api.saveConfig': '仍然保存配置？',

  // Save
  'api.saving': '正在写入配置...',
  'api.saved': '配置已保存到 .env（{provider}）',
};

const translations: Record<Locale, TranslationMap> = { en, zh };
