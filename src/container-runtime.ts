/**
 * Container runtime abstraction for MatClaw.
 * All runtime-specific logic lives here so swapping runtimes means changing one file.
 */
import { execSync } from 'child_process';

import { CONTAINER_IMAGE, CONTAINER_IMAGE_REMOTE } from './config.js';
import { logger } from './logger.js';

/** The container runtime binary name. */
export const CONTAINER_RUNTIME_BIN = 'docker';

/** Returns CLI args for a readonly bind mount. */
export function readonlyMountArgs(
  hostPath: string,
  containerPath: string,
): string[] {
  return ['-v', `${hostPath}:${containerPath}:ro`];
}

/** Returns the shell command to stop a container by name. */
export function stopContainer(name: string): string {
  return `${CONTAINER_RUNTIME_BIN} stop ${name}`;
}

/** Kill a running container by name (fire-and-forget, ignores errors). */
export function killContainer(name: string): void {
  try {
    execSync(`${CONTAINER_RUNTIME_BIN} kill ${name}`, {
      stdio: 'pipe',
      timeout: 10000,
    });
  } catch {
    // Container may already be stopped — ignore
  }
}

/** Ensure the container runtime is running, starting it if needed. */
export function ensureContainerRuntimeRunning(): void {
  try {
    execSync(`${CONTAINER_RUNTIME_BIN} info`, {
      stdio: 'pipe',
      timeout: 10000,
    });
    logger.debug('Container runtime already running');
  } catch (err) {
    logger.error({ err }, 'Failed to reach container runtime');
    console.error(
      '\n╔════════════════════════════════════════════════════════════════╗',
    );
    console.error(
      '║  FATAL: Container runtime failed to start                      ║',
    );
    console.error(
      '║                                                                ║',
    );
    console.error(
      '║  Agents cannot run without a container runtime. To fix:        ║',
    );
    console.error(
      '║  1. Ensure Docker is installed and running                     ║',
    );
    console.error(
      '║  2. Run: docker info                                           ║',
    );
    console.error(
      '║  3. Restart MatClaw                                           ║',
    );
    console.error(
      '╚════════════════════════════════════════════════════════════════╝\n',
    );
    throw new Error('Container runtime is required but failed to start');
  }
}

/** Ensure the container image is available locally; pull from GHCR if missing. */
export function ensureImageAvailable(): void {
  try {
    execSync(`${CONTAINER_RUNTIME_BIN} image inspect ${CONTAINER_IMAGE}`, {
      stdio: 'pipe',
      timeout: 10000,
    });
    logger.debug({ image: CONTAINER_IMAGE }, 'Container image found locally');
    return;
  } catch {
    // Image not found locally
  }

  logger.info(
    { image: CONTAINER_IMAGE, remote: CONTAINER_IMAGE_REMOTE },
    'Local image not found, pulling from GHCR...',
  );
  console.log(`\n  Container image "${CONTAINER_IMAGE}" not found locally.`);
  console.log(`  Pulling from ${CONTAINER_IMAGE_REMOTE} ...\n`);

  try {
    execSync(`${CONTAINER_RUNTIME_BIN} pull ${CONTAINER_IMAGE_REMOTE}`, {
      stdio: 'inherit',
      timeout: 600000,
    });
    execSync(
      `${CONTAINER_RUNTIME_BIN} tag ${CONTAINER_IMAGE_REMOTE} ${CONTAINER_IMAGE}`,
      { stdio: 'pipe', timeout: 10000 },
    );
    logger.info('Container image pulled and tagged successfully');
  } catch (err) {
    logger.warn(
      { err },
      'Failed to pull image from GHCR. Build locally with: ./container/build.sh',
    );
    console.error(
      `\n  Failed to pull image. Build locally instead:\n  ./container/build.sh\n`,
    );
  }
}

/** Kill orphaned MatClaw containers from previous runs. */
export function cleanupOrphans(): void {
  try {
    const output = execSync(
      `${CONTAINER_RUNTIME_BIN} ps --filter name=matclaw- --format '{{.Names}}'`,
      { stdio: ['pipe', 'pipe', 'pipe'], encoding: 'utf-8' },
    );
    const orphans = output.trim().split('\n').filter(Boolean);
    for (const name of orphans) {
      try {
        execSync(stopContainer(name), { stdio: 'pipe' });
      } catch {
        /* already stopped */
      }
    }
    if (orphans.length > 0) {
      logger.info(
        { count: orphans.length, names: orphans },
        'Stopped orphaned containers',
      );
    }
  } catch (err) {
    logger.warn({ err }, 'Failed to clean up orphaned containers');
  }
}
