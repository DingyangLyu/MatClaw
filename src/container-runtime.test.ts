import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock config
vi.mock('./config.js', () => ({
  CONTAINER_IMAGE: 'matclaw-agent:latest',
  CONTAINER_IMAGE_REMOTE: 'ghcr.io/dingyangLyu/matclaw-agent:latest',
}));

// Mock logger
vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// Mock child_process — store the mock fn so tests can configure it
const mockExecSync = vi.fn();
vi.mock('child_process', () => ({
  execSync: (...args: unknown[]) => mockExecSync(...args),
}));

import {
  CONTAINER_RUNTIME_BIN,
  readonlyMountArgs,
  stopContainer,
  ensureContainerRuntimeRunning,
  ensureImageAvailable,
  cleanupOrphans,
} from './container-runtime.js';
import { logger } from './logger.js';

beforeEach(() => {
  vi.clearAllMocks();
});

// --- Pure functions ---

describe('readonlyMountArgs', () => {
  it('returns -v flag with :ro suffix', () => {
    const args = readonlyMountArgs('/host/path', '/container/path');
    expect(args).toEqual(['-v', '/host/path:/container/path:ro']);
  });
});

describe('stopContainer', () => {
  it('returns stop command using CONTAINER_RUNTIME_BIN', () => {
    expect(stopContainer('matclaw-test-123')).toBe(
      `${CONTAINER_RUNTIME_BIN} stop matclaw-test-123`,
    );
  });
});

// --- ensureContainerRuntimeRunning ---

describe('ensureContainerRuntimeRunning', () => {
  it('does nothing when runtime is already running', () => {
    mockExecSync.mockReturnValueOnce('');

    ensureContainerRuntimeRunning();

    expect(mockExecSync).toHaveBeenCalledTimes(1);
    expect(mockExecSync).toHaveBeenCalledWith(`${CONTAINER_RUNTIME_BIN} info`, {
      stdio: 'pipe',
      timeout: 10000,
    });
    expect(logger.debug).toHaveBeenCalledWith(
      'Container runtime already running',
    );
  });

  it('throws when docker info fails', () => {
    mockExecSync.mockImplementationOnce(() => {
      throw new Error('Cannot connect to the Docker daemon');
    });

    expect(() => ensureContainerRuntimeRunning()).toThrow(
      'Container runtime is required but failed to start',
    );
    expect(logger.error).toHaveBeenCalled();
  });
});

// --- ensureImageAvailable ---

describe('ensureImageAvailable', () => {
  it('does nothing when image exists locally', () => {
    mockExecSync.mockReturnValueOnce(''); // docker image inspect succeeds

    ensureImageAvailable();

    expect(mockExecSync).toHaveBeenCalledTimes(1);
    expect(logger.debug).toHaveBeenCalledWith(
      { image: 'matclaw-agent:latest' },
      'Container image found locally',
    );
  });

  it('pulls from GHCR when local image is missing', () => {
    // docker image inspect fails
    mockExecSync.mockImplementationOnce(() => {
      throw new Error('No such image');
    });
    // docker pull succeeds
    mockExecSync.mockReturnValueOnce('');
    // docker tag succeeds
    mockExecSync.mockReturnValueOnce('');

    ensureImageAvailable();

    expect(mockExecSync).toHaveBeenCalledTimes(3);
    expect(mockExecSync).toHaveBeenNthCalledWith(
      2,
      `${CONTAINER_RUNTIME_BIN} pull ghcr.io/dingyangLyu/matclaw-agent:latest`,
      expect.objectContaining({ timeout: 600000 }),
    );
    expect(mockExecSync).toHaveBeenNthCalledWith(
      3,
      `${CONTAINER_RUNTIME_BIN} tag ghcr.io/dingyangLyu/matclaw-agent:latest matclaw-agent:latest`,
      expect.objectContaining({ timeout: 10000 }),
    );
    expect(logger.info).toHaveBeenCalledWith(
      'Container image pulled and tagged successfully',
    );
  });

  it('warns when pull fails', () => {
    // docker image inspect fails
    mockExecSync.mockImplementationOnce(() => {
      throw new Error('No such image');
    });
    // docker pull fails
    mockExecSync.mockImplementationOnce(() => {
      throw new Error('network error');
    });

    ensureImageAvailable(); // should not throw

    expect(logger.warn).toHaveBeenCalledWith(
      expect.objectContaining({ err: expect.any(Error) }),
      'Failed to pull image from GHCR. Build locally with: ./container/build.sh',
    );
  });
});

// --- cleanupOrphans ---

describe('cleanupOrphans', () => {
  it('stops orphaned matclaw containers', () => {
    // docker ps returns container names, one per line
    mockExecSync.mockReturnValueOnce(
      'matclaw-group1-111\nmatclaw-group2-222\n',
    );
    // stop calls succeed
    mockExecSync.mockReturnValue('');

    cleanupOrphans();

    // ps + 2 stop calls
    expect(mockExecSync).toHaveBeenCalledTimes(3);
    expect(mockExecSync).toHaveBeenNthCalledWith(
      2,
      `${CONTAINER_RUNTIME_BIN} stop matclaw-group1-111`,
      { stdio: 'pipe' },
    );
    expect(mockExecSync).toHaveBeenNthCalledWith(
      3,
      `${CONTAINER_RUNTIME_BIN} stop matclaw-group2-222`,
      { stdio: 'pipe' },
    );
    expect(logger.info).toHaveBeenCalledWith(
      { count: 2, names: ['matclaw-group1-111', 'matclaw-group2-222'] },
      'Stopped orphaned containers',
    );
  });

  it('does nothing when no orphans exist', () => {
    mockExecSync.mockReturnValueOnce('');

    cleanupOrphans();

    expect(mockExecSync).toHaveBeenCalledTimes(1);
    expect(logger.info).not.toHaveBeenCalled();
  });

  it('warns and continues when ps fails', () => {
    mockExecSync.mockImplementationOnce(() => {
      throw new Error('docker not available');
    });

    cleanupOrphans(); // should not throw

    expect(logger.warn).toHaveBeenCalledWith(
      expect.objectContaining({ err: expect.any(Error) }),
      'Failed to clean up orphaned containers',
    );
  });

  it('continues stopping remaining containers when one stop fails', () => {
    mockExecSync.mockReturnValueOnce('matclaw-a-1\nmatclaw-b-2\n');
    // First stop fails
    mockExecSync.mockImplementationOnce(() => {
      throw new Error('already stopped');
    });
    // Second stop succeeds
    mockExecSync.mockReturnValueOnce('');

    cleanupOrphans(); // should not throw

    expect(mockExecSync).toHaveBeenCalledTimes(3);
    expect(logger.info).toHaveBeenCalledWith(
      { count: 2, names: ['matclaw-a-1', 'matclaw-b-2'] },
      'Stopped orphaned containers',
    );
  });
});
