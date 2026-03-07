# MatClaw

基于 Claude Agent SDK 的材料科学 AI 助手。构建于 [NanoClaw](https://github.com/qwibitai/nanoclaw) 架构之上，集成了计算材料科学工具。

通过 **飞书** 或 **Gmail** 从手机提交任务 —— MatClaw 运行 DFT、MD、MC 模拟并将结果返回到您的聊天。

## 功能

- **计算材料科学** — Quantum ESPRESSO (DFT)、LAMMPS (MD)、RASPA3 (MC)、MACE (MLIPs)、pymatgen、ASE
- **多渠道消息** — 通过飞书或 Gmail 与 MatClaw 对话。频道在收到第一条消息时自动注册。
- **容器隔离** — 每个任务在独立的 Docker 容器中运行，拥有自己的文件系统
- **定时任务** — 设置定期计算或监控作业
- **网络访问** — 搜索论文、从材料数据库获取数据

## 快速开始

```bash
git clone <your-repo-url>
cd matclaw
npm install
```

### 1. 配置 AI 模型

在项目根目录创建 `.env`：

```bash
ANTHROPIC_API_KEY=your-api-key-here
# 可选：使用兼容的 API 端点（如 DeepSeek）
# ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
```

### 2. 构建智能体容器

```bash
./container/build.sh
```

构建包含所有计算工具的 Docker 镜像（QE、LAMMPS、MACE、pymatgen、ASE 等）。

### 3. 添加消息渠道

至少设置一个渠道与 MatClaw 通信：

#### 飞书（推荐）

1. 前往[飞书开放平台](https://open.feishu.cn/app)创建自建应用
2. 启用**机器人**能力
3. 在**事件与回调**中，订阅方式选择**长连接**，添加事件 `im.message.receive_v1`
4. 添加权限：`im:message`、`im:message:send_as_bot`、`im:chat:readonly`、`contact:contact.base:readonly`
5. 发布应用版本
6. 运行认证：

```bash
npm run auth:feishu
```

输入 App ID 和 App Secret 即可。启动 MatClaw 后给机器人发消息，聊天会**自动注册**为群组，无需手动操作数据库。

#### Gmail

1. 在 GCP 项目中启用 Gmail API
2. 创建 OAuth 2.0 凭据（桌面应用类型）
3. 将 `client_secret_*.json` 放在项目根目录
4. 运行 Gmail 技能添加渠道：

```bash
npx tsx scripts/apply-skill.ts .claude/skills/add-gmail
npm run build
```

5. 认证：

```bash
npm run auth
```

### 4. 启动 MatClaw

```bash
npm run dev
```

MatClaw 连接所有已配置的渠道并开始监听消息。

## 使用方法

在飞书聊天或 Gmail 中给 MatClaw 发消息：

```
帮我设置一个硅的能带结构 QE 计算

用 MACE-MP-0 计算 FCC 铝的状态方程

运行一个铜在 300K 下使用 EAM 势的 LAMMPS MD 模拟

用 pymatgen 从 Materials Project 查找 Li-Fe-O 体系的所有稳定化合物
```

在群聊中需要加触发词（默认 `@MatClaw`）：

```
@MatClaw MgO 的形成能是多少？
```

在私聊（直接消息）中不需要触发词，所有消息都会被自动处理。

## 架构

```
飞书/Gmail --> SQLite --> 轮询循环 --> Docker 容器 (Claude Agent SDK) --> 回复
```

- **单 Node.js 进程** — 编排渠道、消息路由和容器生命周期
- **渠道自注册** — 启动时自动检测凭据
- **飞书自动注册** — 新聊天在收到第一条消息时自动注册为群组
- **容器隔离** — 每个任务在独立 Docker 容器中运行
- **群组独立记忆** — 每个群组有隔离的 `CLAUDE.md`、对话历史和文件系统

### 关键文件

| 文件 | 用途 |
|------|------|
| `src/index.ts` | 编排器：状态管理、消息循环、智能体调用 |
| `src/channels/feishu.ts` | 飞书渠道（WebSocket，自动注册） |
| `src/channels/gmail.ts` | Gmail 渠道（轮询） |
| `src/channels/registry.ts` | 渠道注册表（自注册） |
| `src/container-runner.ts` | 生成智能体容器 |
| `src/db.ts` | SQLite 操作 |
| `groups/*/CLAUDE.md` | 群组智能体记忆和人格 |
| `container/` | 包含计算工具的 Docker 镜像 |

## 配置

MatClaw 使用代码而非配置文件。要自定义行为，编辑 `groups/{name}/CLAUDE.md` 调整智能体人格，或直接修改源文件。

### 环境变量（`.env`）

| 变量 | 必需 | 说明 |
|------|------|------|
| `ANTHROPIC_API_KEY` | 是 | 模型 API 密钥 |
| `ANTHROPIC_BASE_URL` | 否 | 自定义 API 端点 |
| `ASSISTANT_NAME` | 否 | 触发词（默认：`MatClaw`） |

### 飞书凭据（`store/feishu-credentials.json`）

由 `npm run auth:feishu` 自动创建。包含 App ID 和 App Secret。

### Gmail 凭据（`store/gmail-credentials.json`）

由 `npm run auth` 自动创建。包含 OAuth 令牌。

## 开发

```bash
npm run dev          # 热重载运行
npm run build        # 编译 TypeScript
npm run test         # 运行测试
./container/build.sh # 重建智能体容器
```

## 系统要求

- Linux 或 macOS
- Node.js 20+
- Docker
- 飞书机器人应用 和/或 Gmail API 凭据

## 常见问题

**飞书消息没有被处理？**
- 检查应用版本是否已在飞书开发者后台发布
- 确认事件订阅方式设置为"长连接"
- 确保已添加 `im.message.receive_v1` 事件
- 检查权限是否已授予：`im:message`、`im:message:send_as_bot`

**Gmail 连接错误？**
- 网络问题（VPN/代理干扰）可能导致间歇性 ECONNRESET
- 如果使用 TUN 模式代理，尝试关闭

**容器无法启动？**
- 确保用户在 docker 组中：`sudo usermod -aG docker $USER`（需重新登录）
- 验证镜像已构建：`docker images | grep matclaw`

## 许可证

MIT
