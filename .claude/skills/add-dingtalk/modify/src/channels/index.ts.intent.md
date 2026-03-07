# Intent: Add DingTalk channel import

Add `import './dingtalk.js';` to the channel barrel file so the DingTalk
module self-registers with the channel registry on startup.

This is an append-only change — existing import lines for other channels
must be preserved.
