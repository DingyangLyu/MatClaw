# Gmail Setup Guide

## Prerequisites

- A Google account
- Access to [Google Cloud Console](https://console.cloud.google.com/)

## Step 1: Create a GCP Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (e.g., "MatClaw")

## Step 2: Enable Gmail API

1. Go to **APIs & Services** → **Library**
2. Search for "Gmail API" and enable it

## Step 3: Create OAuth 2.0 Credentials

1. Go to **APIs & Services** → **Credentials**
2. Click **Create Credentials** → **OAuth client ID**
3. Application type: **Desktop app**
4. Download the `client_secret_*.json` file
5. Place it in the MatClaw project root

## Step 4: Install Gmail Channel

```bash
npx tsx scripts/apply-skill.ts .claude/skills/add-gmail
npm run build
```

## Step 5: Authenticate

```bash
npm run auth
```

This opens a browser for Google OAuth consent. After authorization, tokens are saved to `~/.gmail-mcp/`.

## Usage

Send an email to the authenticated Gmail address with `@MatClaw` in the subject or body. MatClaw will process the request and reply via email.

## Troubleshooting

**Connection errors (ECONNRESET)?**
- Network issues or VPN/proxy interference
- Add `gmail.googleapis.com` to proxy direct connection rules

**OAuth token expired?**
- Tokens auto-refresh. If persistent, delete `~/.gmail-mcp/` and re-run `npm run auth`
