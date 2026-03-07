# Feishu (飞书) Setup Guide

## Prerequisites

- A Feishu account (personal or enterprise)
- Access to [Feishu Open Platform](https://open.feishu.cn/app)

## Step 1: Create a Custom App

1. Go to [Feishu Open Platform](https://open.feishu.cn/app)
2. Click **Create Custom App**
3. Fill in app name (e.g., "MatClaw") and description

## Step 2: Enable Bot Capability

1. In the app settings, go to **Add Capabilities**
2. Enable **Bot**

## Step 3: Configure Event Subscriptions

1. Go to **Event Subscriptions**
2. Set the mode to **Long Connection (WebSocket)** — no public URL needed
3. Add event: `im.message.receive_v1`

## Step 4: Add Permissions

Go to **Permissions & Scopes** and add:

| Permission | Purpose |
|-----------|---------|
| `im:message` | Receive messages |
| `im:message:send_as_bot` | Send messages as bot |
| `im:chat:readonly` | Read chat info |
| `contact:contact.base:readonly` | Read user info |
| `im:resource` | Upload images (optional, for sending plots) |

## Step 5: Publish the App

1. Go to **Version Management**
2. Create and publish a new version
3. Wait for admin approval (if enterprise account)

## Step 6: Authenticate MatClaw

```bash
npm run auth:feishu
```

Enter your **App ID** and **App Secret** when prompted. These are found on the app's **Credentials & Basic Info** page.

## Usage

Start MatClaw and send a message to the bot — the chat is **auto-registered** as a group. No manual database setup needed.

- **Direct message (P2P)**: All messages are processed automatically
- **Group chat**: Add the bot to a group, then prefix messages with `@MatClaw`

## Troubleshooting

**Bot not receiving messages?**
- Check that the app version is published
- Verify event subscription mode is "Long Connection" (not "Request URL")
- Ensure `im.message.receive_v1` event is added

**Images not sending?**
- Add `im:resource` permission in the developer console
- Publish a new app version after adding the permission

**Connection timeout?**
- If using a VPN/proxy with TUN mode, add `open.feishu.cn` and `*.larksuite.com` to direct connection rules
