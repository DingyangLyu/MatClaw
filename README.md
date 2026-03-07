# MatClaw

Materials science AI assistant powered by Claude Agent SDK. Built on [NanoClaw](https://github.com/qwibitai/nanoclaw) architecture with integrated computational materials science tools.

Submit tasks from your phone via **Feishu** or **Gmail** — MatClaw runs DFT, MD, MC simulations and returns results to your chat.

## Features

- **Computational materials science** — Quantum ESPRESSO (DFT), LAMMPS (MD), RASPA3 (MC), MACE (MLIPs), pymatgen, ASE
- **Multi-channel messaging** — Talk to MatClaw from Feishu or Gmail. Channels auto-register on first message.
- **Container isolation** — Each task runs in a sandboxed Docker container with its own filesystem
- **Scheduled tasks** — Set up recurring calculations or monitoring jobs
- **Web access** — Search papers, fetch data from materials databases

## Quick Start

```bash
git clone <your-repo-url>
cd matclaw
npm install
```

### 1. Configure the AI model

Create `.env` in the project root:

```bash
ANTHROPIC_API_KEY=your-api-key-here
# Optional: use a compatible API endpoint (e.g., DeepSeek)
# ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
```

### 2. Build the agent container

```bash
./container/build.sh
```

This builds the Docker image with all computational tools pre-installed (QE, LAMMPS, MACE, pymatgen, ASE, etc.).

### 3. Add a messaging channel

Set up at least one channel to communicate with MatClaw:

#### Feishu (recommended for Chinese users)

1. Go to [Feishu Open Platform](https://open.feishu.cn/app) and create a Custom App
2. Enable **Bot** capability
3. In **Event Subscriptions**, set mode to **Long Connection (WebSocket)** and add `im.message.receive_v1`
4. Add permissions: `im:message`, `im:message:send_as_bot`, `im:chat:readonly`, `contact:contact.base:readonly`
5. Publish a version of the app
6. Run authentication:

```bash
npm run auth:feishu
```

Enter your App ID and App Secret when prompted. That's it — when you start MatClaw and message the bot, the chat is **auto-registered** as a group. No manual database setup needed.

#### Gmail

1. Set up a GCP project with Gmail API enabled
2. Create OAuth 2.0 credentials (Desktop app type)
3. Place the `client_secret_*.json` file in the project root
4. Run the Gmail skill to add the channel:

```bash
npx tsx scripts/apply-skill.ts .claude/skills/add-gmail
npm run build
```

5. Authenticate:

```bash
npm run auth
```

### 4. Start MatClaw

```bash
npm run dev
```

MatClaw connects to all configured channels and starts listening for messages.

## Usage

Send messages to MatClaw in your Feishu chat or via Gmail:

```
Help me set up a QE calculation for silicon band structure

Calculate the equation of state for FCC aluminum using MACE-MP-0

Run a LAMMPS MD simulation of copper at 300K with EAM potential

Use pymatgen to find all stable compounds in the Li-Fe-O system from Materials Project
```

For group chats, prefix with the trigger word (default `@MatClaw`):

```
@MatClaw what's the formation energy of MgO?
```

In p2p (direct message) chats, no trigger word is needed — all messages are processed automatically.

## Architecture

```
Feishu/Gmail --> SQLite --> Polling loop --> Docker container (Claude Agent SDK) --> Reply
```

- **Single Node.js process** — orchestrates channels, message routing, and container lifecycle
- **Channel self-registration** — channels auto-detect credentials at startup
- **Feishu auto-registration** — new chats are automatically registered as groups on first message
- **Container isolation** — each task runs in its own Docker container with mounted workspace
- **Per-group memory** — each group has isolated `CLAUDE.md`, conversation history, and filesystem

### Key Files

| File | Purpose |
|------|---------|
| `src/index.ts` | Orchestrator: state, message loop, agent invocation |
| `src/channels/feishu.ts` | Feishu channel (WebSocket, auto-registration) |
| `src/channels/gmail.ts` | Gmail channel (polling) |
| `src/channels/registry.ts` | Channel registry (self-registration) |
| `src/container-runner.ts` | Spawns agent containers |
| `src/db.ts` | SQLite operations |
| `groups/*/CLAUDE.md` | Per-group agent memory and personality |
| `container/` | Docker image with computational tools |

## Configuration

MatClaw uses code over configuration. To customize behavior, edit `groups/{name}/CLAUDE.md` for the agent personality, or modify source files directly.

### Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | API key for the model |
| `ANTHROPIC_BASE_URL` | No | Custom API endpoint |
| `ASSISTANT_NAME` | No | Trigger word (default: `MatClaw`) |

### Feishu Credentials (`store/feishu-credentials.json`)

Created automatically by `npm run auth:feishu`. Contains App ID and App Secret.

### Gmail Credentials (`store/gmail-credentials.json`)

Created automatically by `npm run auth`. Contains OAuth tokens.

## Development

```bash
npm run dev          # Run with hot reload
npm run build        # Compile TypeScript
npm run test         # Run tests
./container/build.sh # Rebuild agent container
```

## Requirements

- Linux or macOS
- Node.js 20+
- Docker
- Feishu bot app and/or Gmail API credentials

## Troubleshooting

**Feishu messages not being processed?**
- Check that the app version is published in Feishu Developer Console
- Verify event subscription mode is set to "Long Connection"
- Ensure `im.message.receive_v1` event is added
- Check permissions are granted: `im:message`, `im:message:send_as_bot`

**Gmail connection errors?**
- Network issues (VPN/proxy interference) can cause intermittent ECONNRESET
- Try disabling TUN-mode proxies if present

**Container not starting?**
- Ensure your user is in the `docker` group: `sudo usermod -aG docker $USER` (log out and back in)
- Verify the image is built: `docker images | grep matclaw`

## License

MIT
