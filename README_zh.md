<p align="center">
  <img src="assets/matclaw-logo.svg" alt="MatClaw" width="720">
</p>

<p align="center">
  <strong>告诉它要算什么，它自动写脚本、跑 DFT/MD/MC/MLIP，然后返回结果。</strong>
</p>

<p align="center">
  <a href="README.md">English</a>&nbsp; · &nbsp;
  <img src="https://img.shields.io/badge/QE-7.5-4F46E5" alt="QE 7.5">&nbsp;
  <img src="https://img.shields.io/badge/LAMMPS-2021-7C3AED" alt="LAMMPS">&nbsp;
  <img src="https://img.shields.io/badge/RASPA3-3.0.16-0D9488" alt="RASPA3">&nbsp;
  <img src="https://img.shields.io/badge/MACE--MP--0-latest-D97706" alt="MACE">&nbsp;
  <img src="https://img.shields.io/badge/测试-通过-brightgreen" alt="Tests">
</p>

---

## MatClaw 是什么？

MatClaw 是一个**能自主执行材料科学计算的 AI Agent**。你用自然语言描述任务，它自动编写 Python/Shell 脚本，在隔离的 Docker 容器中运行（容器内预装了完整的计算工具链），然后返回结果。

```
你：  "用 MACE-MP-0 medium 模型计算 2 原子硅金刚石晶胞的能量。"

MatClaw: 正在编写 si_energy.py...
         正在运行计算...
         ✅ 总能量: -10.8248 eV (-5.4124 eV/atom)
         力: [0, 0, 0] eV/Å（平衡态）
         结构已保存到 si_diamond_primitive.xyz
```

不需要手写脚本，不需要调试输入文件，直接拿结果。

## 核心特性

- **自主计算** — 理解任务、编写代码、执行计算、分析输出、遇错自动重试
- **开箱即用** — QE 7.5、LAMMPS、RASPA3、MACE、pymatgen、ASE、PyTorch 全部预装在一个容器内
- **安全隔离** — 每次计算都在一次性 Docker 容器中运行，文件系统隔离
- **灵活的 LLM 后端** — 支持 Anthropic Claude、DeepSeek 或任何 Anthropic 兼容 API
- **多通道接入** — 通过 WhatsApp、Telegram、Discord、Slack 对话（基于 [NanoClaw](https://github.com/qwibitai/nanoclaw) 技能系统）
- **可扩展** — 容器内有 conda/pip，Agent 可以按需安装额外的包

## 计算引擎

| 引擎 | 版本 | 方法 | 应用场景 |
|------|------|------|---------|
| [Quantum ESPRESSO](https://www.quantum-espresso.org/) | 7.5 | DFT | 电子结构、带隙、态密度、声子、弹性常数 |
| [LAMMPS](https://www.lammps.org/) | 2021 | MD | 热学性质、扩散系数、力学性质、相变 |
| [RASPA3](https://github.com/iRASPA/RASPA3) | 3.0.16 | MC | MOF/沸石中的气体吸附、吸附等温线、Henry 常数 |
| [MACE-MP-0](https://github.com/ACEsuit/mace) | latest | MLIP | 通用机器学习势，快速能量/力/应力预测 |

### Python 材料科学工具栈

全部预装在 conda base 环境中：

| 包 | 用途 |
|---|------|
| [pymatgen](https://pymatgen.org/) | 晶体结构操作、相图、电子结构分析 |
| [ASE](https://wiki.fysik.dtu.dk/ase/) | 原子对象、计算器、结构优化、分子动力学 |
| [MACE-torch](https://github.com/ACEsuit/mace) | 通用机器学习原子间势 |
| [mp-api](https://materialsproject.org/) | Materials Project 数据库访问 |
| [spglib](https://spglib.github.io/spglib/) | 空间群/对称性分析 |
| [PyTorch](https://pytorch.org/) | 机器学习框架（CPU 版） |
| numpy, scipy, matplotlib, pandas, seaborn | 科学计算与可视化 |

## 快速开始

### 前置要求

- Linux（推荐 Ubuntu 20.04+）或 macOS
- [Docker](https://docs.docker.com/get-docker/)
- Anthropic 兼容的 API 密钥（Claude、DeepSeek 等）

### 1. 构建容器

```bash
git clone https://gitee.com/baiyuan1/mat-claw.git
cd mat-claw
./container/build.sh
```

首次构建约需 10 分钟（从源码编译 QE 7.5），后续构建使用 Docker 缓存。

### 2. 运行计算

```bash
echo '{
  "prompt": "用 MACE-MP-0 计算体硅的能量",
  "groupFolder": "test",
  "chatJid": "test@g.us",
  "isMain": false,
  "secrets": {
    "ANTHROPIC_API_KEY": "your-api-key",
    "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
  }
}' | docker run -i -v ./workspace:/workspace/group matclaw-agent:latest
```

### 3. 完整 Agent 部署 + 消息通道

```bash
npm install
```

配置至少一个消息通道：

- **[飞书配置指南](docs/feishu-setup.md)** — 推荐中国用户使用，WebSocket 长连接，无需公网 URL
- **[Gmail 配置指南](docs/gmail-setup.md)** — 通过邮件发送任务

然后启动 MatClaw：

```bash
npm run dev
```

## 示例工作流

### DFT：Quantum ESPRESSO 硅 SCF 计算

```
"用 QE 对硅进行 SCF 计算，使用 PAW 赝势，4×4×4 k 点网格，ecutwfc=30 Ry。"
```

Agent 会自动：
1. 从 QE 仓库下载 Si 赝势文件
2. 生成包含 `&CONTROL`、`&SYSTEM`、`&ELECTRONS` 的 `si_scf.in`
3. 运行 `mpirun -np 2 pw.x < si_scf.in`
4. 解析输出：总能量、收敛信息、受力
5. 汇报结果

### MD：LAMMPS 铜的分子动力学

```
"用 LAMMPS 模拟 500 个 FCC 铜原子在 300K 下运行 10ps，使用 LJ 势。
 报告最终温度和总能量。"
```

### MLIP：MACE 快速能量筛选

```
"用 MACE-MP-0 计算 Li、Na、K、Rb、Cs 的 BCC 结构能量，
 与实验内聚能进行比较。"
```

### MC：RASPA3 甲烷吸附

```
"用 RASPA3 在 300K、1 atm 下对甲烷进行巨正则蒙特卡洛模拟。"
```

## 架构

```
┌──────────────────────────────────────────────────────┐
│  宿主机 (Node.js)                                     │
│  ┌────────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  消息通道    │→│  SQLite   │→│  容器运行器       │ │
│  │ (WhatsApp,  │  │ (消息,    │  │ (启动 Docker     │ │
│  │  Telegram,  │  │  任务,    │  │  容器)           │ │
│  │  Discord…)  │  │  状态)    │  └────────┬─────────┘ │
│  └────────────┘  └──────────┘           │           │
└──────────────────────────────────────────┼───────────┘
                                           │ stdin/stdout JSON
┌──────────────────────────────────────────┼───────────┐
│  容器 (Ubuntu 24.04)                      │           │
│  ┌───────────────────────────────────────┘         │ │
│  │  Agent Runner (Claude Agent SDK)                │ │
│  │  ┌─────────────────────────────────────────┐    │ │
│  │  │  LLM ←→ 工具调用 (bash, browser, MCP)   │    │ │
│  │  └─────────────────────────────────────────┘    │ │
│  │                                                  │ │
│  │  计算工具:                                        │ │
│  │  ┌─────────┐ ┌────────┐ ┌───────┐ ┌──────┐     │ │
│  │  │ QE 7.5  │ │ LAMMPS │ │RASPA3 │ │ MACE │     │ │
│  │  └─────────┘ └────────┘ └───────┘ └──────┘     │ │
│  │  ┌──────────────────────────────────────────┐   │ │
│  │  │ Python: pymatgen, ASE, torch, numpy, …   │   │ │
│  │  └──────────────────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**工作原理：**
1. 用户发送自然语言 prompt（通过 stdin JSON 或消息通道）
2. 宿主机编排器将其路由到新的 Docker 容器
3. 容器内 Claude Agent SDK 接收 prompt，迭代执行：
   - 编写计算脚本（Python、Shell、QE 输入文件、LAMMPS 脚本…）
   - 通过 bash 工具执行
   - 读取并分析输出
   - 出错时自动调试和重试
4. 最终结果通过 stdout 标记返回给用户

## 配置

### API 密钥

MatClaw 支持任何 Anthropic 兼容 API。通过 stdin JSON 传入凭据：

```json
{
  "secrets": {
    "ANTHROPIC_API_KEY": "your-key",
    "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
  }
}
```

**已测试的提供商：**

| 提供商 | Base URL | 备注 |
|--------|----------|------|
| [Anthropic](https://www.anthropic.com/) | `https://api.anthropic.com` | Claude 系列模型，推荐 |
| [DeepSeek](https://www.deepseek.com/) | `https://api.deepseek.com/anthropic` | 性价比高，支持 tool_use |

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CONTAINER_RUNTIME` | `docker` | 容器运行时（`docker`、`podman`、`nerdctl`） |
| `MAX_CONCURRENT_CONTAINERS` | `5` | 最大并行 Agent 容器数 |
| `AGENT_TIMEOUT` | `300` | Agent 执行超时（秒） |

## 测试

**最新测试结果（全部通过）：**

| 测试 | 结果 | 详情 |
|------|------|------|
| MACE 能量 | **-10.68 eV** | 2 原子 Si 金刚石晶胞 |
| QE SCF | **-93.44 Ry** | Si，PAW 赝势，4×4×4 k 点网格 |
| LAMMPS MD | **50 步** | FCC Cu，LJ 势，NVE 系综 |
| RASPA3 MC | **98.66 kg/m³** | 甲烷盒子，300K |
| Python 包 | **8/8 完整** | pymatgen, ASE, MACE, torch, numpy, scipy, matplotlib, spglib |
| Agent (E2E) | **-10.82 eV** | 自主完成 Si 能量计算 |

## 基于 NanoClaw

MatClaw 基于 [NanoClaw](https://github.com/qwibitai/nanoclaw) 构建，NanoClaw 是 [qwibitai](https://github.com/qwibitai) 开发的轻量级个人 AI 助手框架。NanoClaw 提供了编排器架构、通道系统、容器隔离和技能框架。MatClaw 在此基础上扩展了完整的材料科学计算环境。

## 项目结构

```
matclaw/
├── src/                        # 宿主机编排器
│   ├── index.ts                # 主循环：消息、Agent、调度
│   ├── container-runner.ts     # 启动隔离的 Docker 容器
│   ├── db.ts                   # SQLite（消息、任务、状态）
│   ├── channels/               # 消息通道注册表
│   └── ...
├── container/
│   ├── Dockerfile              # 多阶段构建（QE 编译 + 运行时）
│   ├── agent-runner/           # Claude Agent SDK 运行器（容器内）
│   └── skills/
│       ├── materials-compute/  # 计算引擎文档
│       └── agent-browser/      # 浏览器自动化
└── groups/                     # 按组隔离的记忆
```

## 路线图

- [ ] GPU 支持（CUDA 容器，加速 PyTorch/MACE）
- [ ] 更多 MLIP 模型（CHGNet、SevenNet、ALIGNN）
- [ ] 工作流自动化（多步计算流水线）
- [ ] Materials Project 集成（查询 + 计算工作流）
- [ ] 自动生成 Jupyter Notebook 以确保可复现性

## 引用

如果你在研究中使用了 MatClaw，请引用：

```bibtex
@software{matclaw2026,
  title  = {MatClaw: AI-Powered Autonomous Materials Science Agent},
  author = {Yuan Bai},
  year   = {2026},
  url    = {https://gitee.com/baiyuan1/mat-claw}
}
```

## 许可证

[MIT](LICENSE)
