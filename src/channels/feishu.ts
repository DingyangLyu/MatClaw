/**
 * Feishu (Lark) Channel Implementation for MatClaw
 * Handles Feishu bot communication using WebSocket (long connection mode)
 * Self-registers via registerChannel() — no core file modifications required.
 *
 * Auto-registration: When a message arrives from an unregistered Feishu chat,
 * the channel automatically registers the group, creates the folder & CLAUDE.md,
 * and delivers the message — no manual SQL or setup needed.
 */

import * as Lark from '@larksuiteoapi/node-sdk';
import { execFileSync } from 'child_process';
import fs from 'fs';
import path from 'path';

import { STORE_DIR, ASSISTANT_NAME, GROUPS_DIR } from '../config.js';
import { storeChatMetadata, setRegisteredGroup } from '../db.js';
import { logger } from '../logger.js';
import {
  Channel,
  OnInboundMessage,
  OnChatMetadata,
  RegisteredGroup,
} from '../types.js';
import { registerChannel, ChannelOpts } from './registry.js';

// --- Types ---

interface FeishuCredentials {
  appId: string;
  appSecret: string;
}

interface FeishuMessageEvent {
  event_id?: string;
  sender: {
    sender_id: {
      open_id?: string;
      user_id?: string;
      union_id?: string;
    };
    sender_type?: string;
    tenant_key?: string;
  };
  message: {
    message_id: string;
    root_id?: string;
    parent_id?: string;
    chat_id: string;
    chat_type: 'p2p' | 'group';
    message_type: string;
    content: string;
    create_time?: string;
    mentions?: Array<{
      key: string;
      id: { open_id?: string; user_id?: string; union_id?: string };
      name: string;
    }>;
  };
}

// Default CLAUDE.md content for auto-registered Feishu groups
const DEFAULT_CLAUDE_MD = `# ${ASSISTANT_NAME}

You are ${ASSISTANT_NAME}, an AI materials scientist assistant. You help with computational materials science tasks including DFT calculations, molecular dynamics, Monte Carlo simulations, and materials property predictions.

## What You Can Do

- **Materials simulations** — Run DFT (Quantum ESPRESSO), MD (LAMMPS), MC (RASPA3) calculations
- **Machine learning potentials** — Use MACE and other MLIPs for accelerated simulations
- **Structure analysis** — Analyze crystal structures with pymatgen and ASE
- **Data analysis** — Process simulation results, plot data, compute properties
- Answer questions about materials science and computational methods
- Search the web and fetch content from URLs
- Read and write files in your workspace
- Run bash commands in your sandbox
- Schedule tasks to run later or on a recurring basis
- Send messages back to the chat

## Communication

Your output is sent to the user or group via Feishu.

You also have \`mcp__matclaw__send_message\` which sends a message immediately while you're still working.

### Internal thoughts

If part of your output is internal reasoning rather than something for the user, wrap it in \`<internal>\` tags:

\`\`\`
<internal>Compiled all three reports, ready to summarize.</internal>

Here are the key findings from the research...
\`\`\`

Text inside \`<internal>\` tags is logged but not sent to the user.

## Memory

The \`conversations/\` folder contains searchable history of past conversations. Use this to recall context from previous sessions.

When you learn something important:
- Create files for structured data (e.g., \`customers.md\`, \`preferences.md\`)
- Split files larger than 500 lines into folders

## Data Visualization

Always generate plots/figures to present results visually. Use matplotlib to create clear, publication-quality figures. Save plots as PNG files in your workspace and include them in your response using markdown image syntax:

\`\`\`
![K-point Convergence](k_convergence/convergence.png)
\`\`\`

Guidelines:
- Generate a plot for any numerical results (convergence tests, energy curves, DOS, band structures, RDFs, etc.)
- Use clear axis labels with units
- Add legends when multiple datasets are shown
- **CRITICAL: Always set a fixed figure size like \`figsize=(10, 6)\`. NEVER let figure height scale with data length. Max height: 2000 pixels.**
- Save as PNG with \`dpi=150\` for good quality
- If plotting long time series, downsample or use a rolling average
- Always include the plot in your final message to the user

## Sending Files

To send files (data, reports, etc.) to the user, include them in your response:

\`\`\`
[file:results/data.csv]
\`\`\`

## Feishu Formatting

Use markdown formatting for Feishu messages:
- **Bold** (double asterisks)
- *Italic* (single asterisks)
- Bullet points
- \`\`\`Code blocks\`\`\` (triple backticks)

Keep messages clear and well-structured.
`;

// --- FeishuChannel class ---

export class FeishuChannel implements Channel {
  name = 'feishu';

  private client: Lark.Client | null = null;
  private wsClient: Lark.WSClient | null = null;
  private eventDispatcher: Lark.EventDispatcher | null = null;
  private credentials: FeishuCredentials | null = null;
  private botOpenId: string | null = null;
  private connected = false;

  private opts: ChannelOpts;

  constructor(opts: ChannelOpts) {
    this.opts = opts;
  }

  // --- Public Channel interface ---

  async connect(): Promise<void> {
    const credsPath = path.join(STORE_DIR, 'feishu-credentials.json');

    if (!fs.existsSync(credsPath)) {
      logger.warn(
        'Feishu credentials not found. Run `npm run auth:feishu` first.',
      );
      return;
    }

    try {
      this.credentials = JSON.parse(fs.readFileSync(credsPath, 'utf-8'));
    } catch (err) {
      logger.error({ err }, 'Failed to parse Feishu credentials');
      return;
    }

    if (!this.credentials?.appId || !this.credentials?.appSecret) {
      logger.error('Feishu credentials missing appId or appSecret');
      return;
    }

    // REST client for API calls (send messages, lookup user/chat info)
    this.client = new Lark.Client({
      appId: this.credentials.appId,
      appSecret: this.credentials.appSecret,
      appType: Lark.AppType.SelfBuild,
    });

    // Verify credentials & get bot open_id for self-message filtering
    try {
      const response = await (this.client as any).request({
        method: 'GET',
        url: '/open-apis/bot/v3/info',
      });
      if (response.code === 0 && response.bot) {
        this.botOpenId = response.bot.open_id || null;
        logger.info(
          { botName: response.bot.bot_name, botOpenId: this.botOpenId },
          'Connected to Feishu',
        );
      }
    } catch (err) {
      logger.error({ err }, 'Failed to verify Feishu connection');
      return;
    }

    await this.setupWebSocket();
    this.connected = true;
  }

  async sendMessage(jid: string, text: string): Promise<void> {
    if (!this.client) {
      logger.warn({ jid }, 'Feishu: cannot send — client not connected');
      return;
    }

    if (!jid.startsWith('oc_') && !jid.startsWith('ou_')) {
      logger.warn({ jid }, 'Feishu: invalid chat_id format, skipping send');
      return;
    }

    const receiveIdType: 'chat_id' | 'open_id' = jid.startsWith('oc_')
      ? 'chat_id'
      : 'open_id';
    const groupFolder = this.getGroupFolder(jid);

    // Extract file attachments [file:path/to/file.pdf] from text
    const fileRegex = /\[file:([^\]]+)\]/g;
    const filePaths: string[] = [];
    let cleanText = text;
    let fileMatch: RegExpExecArray | null;
    while ((fileMatch = fileRegex.exec(text)) !== null) {
      filePaths.push(fileMatch[1]);
    }
    if (filePaths.length > 0) {
      cleanText = text.replace(fileRegex, '').trim();
    }

    // Send text + images as a rich post
    try {
      const { paragraphs, imageCount } = await this.buildPostContent(
        jid,
        cleanText,
      );

      const content = JSON.stringify({
        zh_cn: { content: paragraphs },
      });

      const response = await this.client.im.message.create({
        params: { receive_id_type: receiveIdType },
        data: {
          receive_id: jid,
          content,
          msg_type: 'post',
        },
      });

      if (response.code !== 0) {
        throw new Error(
          `Feishu send failed: ${response.msg || `code ${response.code}`}`,
        );
      }

      logger.info(
        { jid, messageId: response.data?.message_id, imageCount },
        'Feishu: message sent',
      );
    } catch (err) {
      logger.error({ jid, err }, 'Feishu: failed to send message');
    }

    // Send each file as a separate file message
    for (const filePath of filePaths) {
      await this.sendFileMessage(jid, receiveIdType, filePath, groupFolder);
    }
  }

  /**
   * Upload and send a file as a Feishu file message.
   */
  private async sendFileMessage(
    jid: string,
    receiveIdType: 'chat_id' | 'open_id',
    filePath: string,
    groupFolder: string | null,
  ): Promise<void> {
    if (!this.client) return;

    // Resolve path
    let resolvedPath = filePath;
    if (filePath.startsWith('/workspace/group/')) {
      if (!groupFolder) return;
      resolvedPath = path.join(
        GROUPS_DIR,
        groupFolder,
        filePath.slice('/workspace/group/'.length),
      );
    } else if (!path.isAbsolute(filePath) && groupFolder) {
      resolvedPath = path.join(GROUPS_DIR, groupFolder, filePath);
    }

    if (!fs.existsSync(resolvedPath)) {
      logger.warn(
        { filePath, resolvedPath },
        'Feishu: file not found, skipping',
      );
      return;
    }

    // Map extension to Feishu file_type
    const ext = path.extname(resolvedPath).toLowerCase();
    const fileTypeMap: Record<
      string,
      'pdf' | 'doc' | 'xls' | 'ppt' | 'mp4' | 'opus' | 'stream'
    > = {
      '.pdf': 'pdf',
      '.doc': 'doc',
      '.docx': 'doc',
      '.xls': 'xls',
      '.xlsx': 'xls',
      '.ppt': 'ppt',
      '.pptx': 'ppt',
      '.mp4': 'mp4',
      '.opus': 'opus',
      '.ogg': 'opus',
    };
    const fileType = fileTypeMap[ext] || 'stream';
    const fileName = path.basename(resolvedPath);

    try {
      const fileStream = fs.createReadStream(resolvedPath);
      const uploadRes = await this.client.im.file.create({
        data: {
          file_type: fileType,
          file_name: fileName,
          file: fileStream,
        },
      });

      if (!uploadRes?.file_key) {
        logger.warn({ resolvedPath }, 'Feishu: file upload returned no key');
        return;
      }

      logger.info(
        { resolvedPath, fileKey: uploadRes.file_key },
        'Feishu: file uploaded',
      );

      const response = await this.client.im.message.create({
        params: { receive_id_type: receiveIdType },
        data: {
          receive_id: jid,
          content: JSON.stringify({ file_key: uploadRes.file_key }),
          msg_type: 'file',
        },
      });

      if (response.code !== 0) {
        throw new Error(
          `Feishu file send failed: ${response.msg || `code ${response.code}`}`,
        );
      }

      logger.info(
        { jid, fileName, messageId: response.data?.message_id },
        'Feishu: file sent',
      );
    } catch (err) {
      logger.error({ resolvedPath, err }, 'Feishu: failed to send file');
    }
  }

  isConnected(): boolean {
    return this.connected;
  }

  ownsJid(jid: string): boolean {
    return jid.startsWith('oc_') || jid.startsWith('ou_');
  }

  async disconnect(): Promise<void> {
    this.connected = false;
  }

  async setTyping(_jid: string, _isTyping: boolean): Promise<void> {
    // Feishu bot API does not expose a typing indicator.
  }

  async syncGroups(_force: boolean): Promise<void> {
    // Feishu group metadata is fetched on-demand during message handling.
  }

  // --- Private helpers ---

  private async setupWebSocket(): Promise<void> {
    if (!this.credentials) return;

    this.wsClient = new Lark.WSClient({
      appId: this.credentials.appId,
      appSecret: this.credentials.appSecret,
      loggerLevel: Lark.LoggerLevel.warn,
    });

    // WebSocket mode delivers events pre-decrypted — do NOT pass
    // encryptKey/verificationToken here or the SDK will try to AES-decrypt
    // already-plaintext data, resulting in undefined payloads.
    this.eventDispatcher = new Lark.EventDispatcher({});

    this.eventDispatcher.register({
      'im.message.receive_v1': async (data) => {
        try {
          await this.handleMessageEvent(data as unknown as FeishuMessageEvent);
        } catch (err) {
          logger.error({ err }, 'Feishu: error handling message event');
        }
      },
      'im.message.message_read_v1': async (_data) => {
        // Ignore read receipts
      },
      'im.chat.member.bot.added_v1': async (data) => {
        const event = data as unknown as { chat_id: string };
        logger.info({ chatId: event.chat_id }, 'Feishu: bot added to chat');
      },
      'im.chat.member.bot.deleted_v1': async (data) => {
        const event = data as unknown as { chat_id: string };
        logger.info({ chatId: event.chat_id }, 'Feishu: bot removed from chat');
      },
    });

    this.wsClient.start({ eventDispatcher: this.eventDispatcher });
    logger.info('Feishu: WebSocket client started');
  }

  private async handleMessageEvent(event: FeishuMessageEvent): Promise<void> {
    const chatId = event.message.chat_id;
    const senderOpenId = event.sender.sender_id.open_id;
    const senderUserId = event.sender.sender_id.user_id;

    // Skip bot's own messages
    if (
      (senderOpenId && senderOpenId === this.botOpenId) ||
      (senderUserId && senderUserId === this.botOpenId)
    ) {
      return;
    }

    if (!chatId.startsWith('oc_') && !chatId.startsWith('ou_')) {
      logger.warn({ chatId }, 'Feishu: invalid chat_id, skipping');
      return;
    }

    const messageType = event.message.message_type;
    const timestamp = event.message.create_time
      ? new Date(parseInt(event.message.create_time, 10)).toISOString()
      : new Date().toISOString();

    const { text: content, attachment } = this.parseContent(
      event.message.content,
      messageType,
    );

    const senderName = await this.resolveSenderName(senderOpenId || '');
    const chatName = await this.getChatName(chatId);
    const isGroup = event.message.chat_type === 'group';

    // Notify host of chat metadata (for group discovery)
    this.opts.onChatMetadata(chatId, timestamp, chatName, 'feishu', isGroup);

    // Auto-register unregistered Feishu chats
    let groups = this.opts.registeredGroups();
    if (!groups[chatId]) {
      this.autoRegisterGroup(chatId, chatName, isGroup);
      // Re-read groups after registration
      groups = this.opts.registeredGroups();
    }

    if (groups[chatId]) {
      // Download attachment (image/file) if present
      let finalContent = content;
      if (attachment && attachment.key) {
        const groupFolder = groups[chatId].folder;
        const savedPath = await this.downloadAttachment(
          event.message.message_id,
          attachment.key,
          attachment.type,
          attachment.name,
          groupFolder,
        );
        if (savedPath) {
          const label = attachment.type === 'image' ? 'image' : 'file';
          finalContent = `[Attached ${label}: ${savedPath}]`;
        }
      }

      this.opts.onMessage(chatId, {
        id: event.message.message_id,
        chat_jid: chatId,
        sender: senderOpenId || senderUserId || 'unknown',
        sender_name: senderName,
        content: finalContent,
        timestamp,
        is_from_me: false,
        is_bot_message: false,
      });
    }
  }

  /**
   * Auto-register a Feishu chat as a group.
   * Creates the group folder, CLAUDE.md, and database entry automatically.
   */
  private autoRegisterGroup(
    chatId: string,
    chatName: string,
    isGroup: boolean,
  ): void {
    // Generate a unique folder name: feishu_{short-id}
    const shortId = chatId.replace(/^oc_/, '').slice(0, 12);
    const folder = `feishu_${shortId}`;
    const displayName =
      chatName && chatName !== chatId
        ? `Feishu: ${chatName}`
        : `Feishu ${isGroup ? 'Group' : 'Chat'}`;

    try {
      // Create group folder and CLAUDE.md
      const groupDir = path.join(GROUPS_DIR, folder);
      fs.mkdirSync(groupDir, { recursive: true });

      const claudeMdPath = path.join(groupDir, 'CLAUDE.md');
      if (!fs.existsSync(claudeMdPath)) {
        fs.writeFileSync(claudeMdPath, DEFAULT_CLAUDE_MD);
      }

      // Register in database
      const group: RegisteredGroup = {
        name: displayName,
        folder,
        trigger: `@${ASSISTANT_NAME}`,
        added_at: new Date().toISOString(),
        requiresTrigger: isGroup, // p2p chats don't need trigger
      };
      setRegisteredGroup(chatId, group);

      logger.info(
        { chatId, folder, displayName, isGroup },
        'Feishu: auto-registered new group',
      );
    } catch (err) {
      logger.error({ chatId, err }, 'Feishu: failed to auto-register group');
    }
  }

  /**
   * Build Feishu post content from text, replacing markdown images with
   * uploaded Feishu images. Supports ![alt](path) syntax.
   * Container paths (/workspace/group/...) are resolved to host paths.
   */
  private async buildPostContent(
    jid: string,
    text: string,
  ): Promise<{
    paragraphs: Array<Array<Record<string, string>>>;
    imageCount: number;
  }> {
    // Regex for markdown images: ![alt text](image/path.png)
    const imageRegex = /!\[([^\]]*)\]\(([^)]+)\)/g;
    const groupFolder = this.getGroupFolder(jid);

    // Split text into segments: text parts and image references
    const paragraphs: Array<Array<Record<string, string>>> = [];
    let lastIndex = 0;
    let imageCount = 0;
    let match: RegExpExecArray | null;

    while ((match = imageRegex.exec(text)) !== null) {
      // Add text before the image
      const textBefore = text.slice(lastIndex, match.index).trim();
      if (textBefore) {
        paragraphs.push([
          {
            tag: 'md',
            text:
              lastIndex === 0 ? `${ASSISTANT_NAME}: ${textBefore}` : textBefore,
          },
        ]);
      } else if (lastIndex === 0) {
        // Ensure the first paragraph has the assistant name
        paragraphs.push([{ tag: 'md', text: `${ASSISTANT_NAME}:` }]);
      }

      // Try to upload the image
      const imagePath = match[2];
      const imageKey = await this.uploadImageFromPath(imagePath, groupFolder);
      if (imageKey) {
        paragraphs.push([{ tag: 'img', image_key: imageKey }]);
        imageCount++;
      } else {
        // Fallback: show as text if upload fails
        paragraphs.push([
          { tag: 'md', text: `[Image: ${match[1] || imagePath}]` },
        ]);
      }

      lastIndex = match.index + match[0].length;
    }

    // Add remaining text after last image
    const remaining = text.slice(lastIndex).trim();
    if (remaining) {
      paragraphs.push([
        {
          tag: 'md',
          text: lastIndex === 0 ? `${ASSISTANT_NAME}: ${remaining}` : remaining,
        },
      ]);
    }

    // If no content was added (empty text), add a minimal message
    if (paragraphs.length === 0) {
      paragraphs.push([{ tag: 'md', text: `${ASSISTANT_NAME}: (empty)` }]);
    }

    return { paragraphs, imageCount };
  }

  /**
   * Upload an image file to Feishu and return the image_key.
   * Resolves container paths (/workspace/group/...) to host paths.
   */
  private async uploadImageFromPath(
    imagePath: string,
    groupFolder: string | null,
  ): Promise<string | null> {
    if (!this.client) return null;

    // Resolve the image path
    let resolvedPath = imagePath;

    // Container path → host path
    if (imagePath.startsWith('/workspace/group/')) {
      if (!groupFolder) return null;
      resolvedPath = path.join(
        GROUPS_DIR,
        groupFolder,
        imagePath.slice('/workspace/group/'.length),
      );
    } else if (!path.isAbsolute(imagePath) && groupFolder) {
      // Relative paths are relative to the group folder
      resolvedPath = path.join(GROUPS_DIR, groupFolder, imagePath);
    }

    // Check file exists and is an image
    if (!fs.existsSync(resolvedPath)) {
      logger.debug({ imagePath, resolvedPath }, 'Feishu: image file not found');
      return null;
    }

    const ext = path.extname(resolvedPath).toLowerCase();
    if (
      ![
        '.png',
        '.jpg',
        '.jpeg',
        '.gif',
        '.webp',
        '.bmp',
        '.tiff',
        '.ico',
      ].includes(ext)
    ) {
      logger.debug({ resolvedPath, ext }, 'Feishu: unsupported image format');
      return null;
    }

    try {
      // Resize oversized images to prevent sending giant blank images
      const uploadPath = this.validateAndResizeImage(resolvedPath);
      if (!uploadPath) return null; // Skip broken images
      const imageStream = fs.createReadStream(uploadPath);
      const res = await this.client.im.image.create({
        data: {
          image_type: 'message',
          image: imageStream,
        },
      });

      // Clean up temp resized file
      if (uploadPath !== resolvedPath) {
        try {
          fs.unlinkSync(uploadPath);
        } catch {
          /* ignore */
        }
      }

      if (res?.image_key) {
        logger.info(
          { resolvedPath, imageKey: res.image_key },
          'Feishu: image uploaded',
        );
        return res.image_key;
      }
      logger.warn(
        { resolvedPath, res },
        'Feishu: image upload returned no key',
      );
      return null;
    } catch (err) {
      logger.error({ resolvedPath, err }, 'Feishu: failed to upload image');
      return null;
    }
  }

  /**
   * Resize an image if it exceeds max dimensions (4096x4096).
   * Returns the path to use for upload (original or temp resized file).
   */
  /**
   * Validate and resize image before upload.
   * - Reject images with extreme aspect ratio (broken plots)
   * - Resize large but reasonable images to fit within bounds
   * Returns null to skip upload, or the path to use.
   */
  private validateAndResizeImage(imagePath: string): string | null {
    const MAX_W = 2048;
    const MAX_H = 2048;
    const script = [
      'import sys',
      'from PIL import Image',
      'Image.MAX_IMAGE_PIXELS = None',
      'src, dst = sys.argv[1], sys.argv[2]',
      'mw, mh = int(sys.argv[3]), int(sys.argv[4])',
      'img = Image.open(src)',
      'w, h = img.size',
      '# Reject extreme aspect ratios (broken plots)',
      'if h > w * 5 or w > h * 5:',
      '    print("reject " + str(w) + "x" + str(h))',
      '    sys.exit(0)',
      'if w <= mw and h <= mh:',
      '    print("ok")',
      '    sys.exit(0)',
      'img.thumbnail((mw, mh), Image.LANCZOS)',
      'img.save(dst, dpi=(150, 150))',
      'print("resized")',
    ].join('\n');
    const tmpPath = imagePath.replace(/(\.\w+)$/, '_resized$1');
    try {
      const result = execFileSync(
        'python3',
        ['-c', script, imagePath, tmpPath, String(MAX_W), String(MAX_H)],
        { timeout: 60000, encoding: 'utf-8' },
      ).trim();
      if (result.startsWith('reject')) {
        logger.warn(
          { imagePath, dimensions: result },
          'Feishu: image has extreme aspect ratio, skipping upload',
        );
        return null;
      }
      if (result === 'resized') {
        logger.info({ imagePath, tmpPath }, 'Feishu: resized oversized image');
        return tmpPath;
      }
      return imagePath;
    } catch (err) {
      logger.warn(
        { imagePath, err },
        'Feishu: failed to validate image, skipping',
      );
      return null;
    }
  }

  /**
   * Get the group folder name for a given JID.
   */
  private getGroupFolder(jid: string): string | null {
    const groups = this.opts.registeredGroups();
    return groups[jid]?.folder || null;
  }

  /**
   * Download a message attachment (image/file) and save to group uploads folder.
   * Returns the relative path (e.g., "uploads/123_photo.png") or null on failure.
   */
  private async downloadAttachment(
    messageId: string,
    fileKey: string,
    type: 'image' | 'file',
    filename: string,
    groupFolder: string,
  ): Promise<string | null> {
    if (!this.client || !groupFolder) return null;

    const uploadDir = path.join(GROUPS_DIR, groupFolder, 'uploads');
    fs.mkdirSync(uploadDir, { recursive: true });

    const safeName = `${Date.now()}_${filename.replace(/[^a-zA-Z0-9._\-\u4e00-\u9fff]/g, '_')}`;
    const filePath = path.join(uploadDir, safeName);

    try {
      const resp = await this.client.im.messageResource.get({
        path: { message_id: messageId, file_key: fileKey },
        params: { type },
      });

      await (resp as any).writeFile(filePath);
      logger.info(
        { messageId, fileKey, filePath },
        'Feishu: downloaded attachment',
      );
      return `uploads/${safeName}`;
    } catch (err) {
      logger.error(
        { messageId, fileKey, err },
        'Feishu: failed to download attachment',
      );
      return null;
    }
  }

  private parseContent(
    rawContent: string,
    messageType: string,
  ): {
    text: string;
    attachment?: { type: 'image' | 'file'; key: string; name: string };
  } {
    try {
      const parsed = JSON.parse(rawContent);
      switch (messageType) {
        case 'text':
          return { text: parsed.text || '' };
        case 'post': {
          const title =
            parsed.zh_cn?.title || parsed.en_us?.title || parsed.title || '';
          const blocks =
            parsed.zh_cn?.content ||
            parsed.en_us?.content ||
            parsed.content ||
            [];
          let text = title ? `${title}\n\n` : '';
          for (const paragraph of blocks) {
            if (Array.isArray(paragraph)) {
              for (const el of paragraph) {
                if (el.tag === 'text') text += el.text || '';
                else if (el.tag === 'a') text += el.text || el.href || '';
                else if (el.tag === 'at')
                  text += `@${el.user_name || el.user_id || ''}`;
                else if (el.tag === 'img') text += '<media:image>';
              }
              text += '\n';
            }
          }
          return { text: text.trim() || '[Rich Text]' };
        }
        case 'image':
          return {
            text: '<media:image>',
            attachment: {
              type: 'image',
              key: parsed.image_key || '',
              name: 'image.png',
            },
          };
        case 'file':
          return {
            text: `<media:file:${parsed.file_name || 'unknown'}>`,
            attachment: {
              type: 'file',
              key: parsed.file_key || '',
              name: parsed.file_name || 'file',
            },
          };
        case 'audio':
          return {
            text: '<media:audio>',
            attachment: {
              type: 'file',
              key: parsed.file_key || '',
              name: parsed.file_name || 'audio.ogg',
            },
          };
        case 'video':
          return {
            text: '<media:video>',
            attachment: {
              type: 'file',
              key: parsed.file_key || '',
              name: parsed.file_name || 'video.mp4',
            },
          };
        case 'sticker':
          return { text: '<media:sticker>' };
        default:
          return { text: `[${messageType}]` };
      }
    } catch {
      return { text: rawContent };
    }
  }

  private async resolveSenderName(openId: string): Promise<string> {
    if (!this.client || !openId) return 'Unknown';
    try {
      const res = await this.client.contact.user.get({
        path: { user_id: openId },
        params: { user_id_type: 'open_id' },
      });
      const user = res.data?.user;
      if (user) return user.name || user.en_name || openId;
    } catch {
      // Best effort
    }
    return openId;
  }

  private async getChatName(chatId: string): Promise<string> {
    if (!this.client) return chatId;
    try {
      const res = await this.client.im.chat.get({
        path: { chat_id: chatId },
      });
      return res.data?.name || chatId;
    } catch {
      return chatId;
    }
  }
}

// Self-register the channel — triggers when this module is imported.
// Returns null if FEISHU credentials are absent (graceful no-op).
registerChannel('feishu', (opts: ChannelOpts) => {
  const credsPath = path.join(STORE_DIR, 'feishu-credentials.json');
  if (!fs.existsSync(credsPath)) {
    logger.debug('Feishu credentials not found — channel not registered');
    return null;
  }
  return new FeishuChannel(opts);
});
