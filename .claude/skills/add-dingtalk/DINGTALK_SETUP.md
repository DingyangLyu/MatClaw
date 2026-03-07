# DingTalk Bot Setup Guide

## Step 1: Create DingTalk App

1. Open [DingTalk Open Platform](https://open-dev.dingtalk.com/fe/app#/corp/app)
2. Log in with your DingTalk admin account
3. Click **"Create App"** (创建应用)
4. Choose **"Enterprise Internal App"** (企业内部开发)
5. Fill in the app name (e.g., "MatClaw") and description

## Step 2: Enable Robot

1. In the app settings, go to **"Add Capability"** (添加能力)
2. Select **"Robot"** (机器人)
3. In Robot configuration:
   - Set message receiving mode to **"Stream Mode"** (Stream 模式)
   - This uses WebSocket — no public URL or webhook endpoint needed

## Step 3: Get Credentials

1. Go to **"Credentials & Basic Info"** (凭证与基本信息)
2. Copy the **Client ID** (AppKey / 应用标识)
3. Copy the **Client Secret** (AppSecret / 应用密钥)

## Step 4: Configure Permissions

Go to **"Permissions"** (权限管理) and grant:

| Permission | Description |
|-----------|-------------|
| `qyapi_robot_sendmsg` | Send robot messages |
| `qyapi_chat_manage` | Manage group chats |

## Step 5: Publish the App

1. Go to **"Release"** (版本管理与发布)
2. Click **"Create Release"** (创建版本)
3. Set visibility scope (which departments/users can use the bot)
4. Submit for release

## Step 6: Add Bot to Group

1. Open a DingTalk group chat
2. Click the group settings icon (top right)
3. Go to **"Robot"** (智能群助手 / 机器人)
4. Click **"Add Robot"** (添加机器人)
5. Find your app's robot and add it

## Step 7: Configure MatClaw

Run the auth script:

```bash
npm run auth:dingtalk
```

Enter the Client ID and Client Secret when prompted.

## Step 8: Test

1. Restart MatClaw: `npm run dev`
2. In the DingTalk group, send: `@MatClaw hello`
3. The bot should respond automatically

## Token Reference

| Value | Format | Where to Find |
|-------|--------|---------------|
| Client ID (AppKey) | `ding...` or alphanumeric | App credentials page |
| Client Secret (AppSecret) | Long alphanumeric string | App credentials page |

## Troubleshooting

### "Stream connection failed"

- Verify Stream Mode is enabled (not HTTP mode)
- Check that the app has been published
- Ensure the Client ID and Client Secret are correct

### Bot added but not responding

- The bot only responds when @mentioned in group chats
- In 1-on-1 chats, any message triggers a response
- Check MatClaw logs: `tail -f logs/matclaw.log | grep -i dingtalk`

### Permission denied errors

- Ensure `qyapi_robot_sendmsg` permission is granted
- After adding permissions, you may need to re-publish the app
- The visibility scope must include the users/groups trying to use the bot
