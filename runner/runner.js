import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';

const MAX_SOURCE_CHARS = 24000;
const MAX_OUTPUT_CHARS = 48000;
const RUN_TIMEOUT_MS = 8000;
const workspaceRoot = '/tmp/axon-workspace';

function emit(result) {
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

function isSafePath(filePath) {
  return typeof filePath === 'string' &&
    filePath.length > 0 &&
    !filePath.startsWith('/') &&
    !filePath.includes('\\') &&
    !/^[a-zA-Z]:/.test(filePath) &&
    filePath.split('/').every((part) => part !== '' && part !== '.' && part !== '..');
}

function run(command, args, cwd, stdin = '') {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env: {
        HOME: '/tmp',
        PATH: process.env.PATH ?? '',
        GOCACHE: '/tmp/go-cache',
        GOPATH: '/tmp/go-path',
        TMPDIR: '/tmp',
        JAVA_TOOL_OPTIONS: '-XX:ActiveProcessorCount=1 -Xmx96m',
      },
      detached: true,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    let timedOut = false;
    let outputExceeded = false;
    const killProcessGroup = () => {
      if (!child.pid) {
        child.kill('SIGKILL');
        return;
      }
      try {
        process.kill(-child.pid, 'SIGKILL');
      } catch (error) {
        if (error instanceof Error && error.code === 'ESRCH') return;
        throw error;
      }
    };
    const timeout = setTimeout(() => {
      timedOut = true;
      killProcessGroup();
    }, RUN_TIMEOUT_MS);

    const collect = (stream, chunk) => {
      const next = stream === 'stdout' ? stdout + chunk : stderr + chunk;
      const remaining = Math.max(0, MAX_OUTPUT_CHARS - (stream === 'stdout' ? stderr.length : stdout.length));
      if (stream === 'stdout') stdout = next.slice(0, remaining);
      else stderr = next.slice(0, remaining);
      if (next.length > remaining) {
        outputExceeded = true;
        killProcessGroup();
      }
    };

    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => collect('stdout', chunk));
    child.stderr.on('data', (chunk) => collect('stderr', chunk));
    child.stdin.end(stdin);
    child.once('error', reject);
    child.once('close', (code, signal) => {
      clearTimeout(timeout);
      resolve({ code, signal, stdout, stderr, timedOut, outputExceeded });
    });
  });
}

try {
  let input = '';
  for await (const chunk of process.stdin) input += chunk;
  const request = JSON.parse(input);
  if (!request || !Array.isArray(request.files) || typeof request.entrypoint !== 'string') {
    throw new Error('Provide workspace files and an entrypoint.');
  }
  const stdin = request.stdin ?? '';
  if (typeof stdin !== 'string' || stdin.length > 4000) {
    throw new Error('Program input must be text with at most 4,000 characters.');
  }

  let totalChars = 0;
  const files = new Map();
  for (const file of request.files) {
    if (!file || !isSafePath(file.path) || typeof file.content !== 'string') {
      throw new Error('A workspace file has an invalid path or content.');
    }
    totalChars += file.content.length;
    if (totalChars > MAX_SOURCE_CHARS) {
      throw new Error('Workspace source exceeds the 24,000 character runner limit.');
    }
    files.set(file.path, file.content);
  }
  if (!files.has(request.entrypoint)) throw new Error('The selected entrypoint is not in the workspace.');

  const extension = path.extname(request.entrypoint).toLowerCase();
  const target = path.join(workspaceRoot, request.entrypoint);
  const targetBase = path.basename(target, extension);
  await mkdir(workspaceRoot, { recursive: true });
  for (const [filePath, content] of files) {
    const destination = path.join(workspaceRoot, filePath);
    await mkdir(path.dirname(destination), { recursive: true });
    await writeFile(destination, content, { encoding: 'utf8', mode: 0o600 });
  }

  const compilers = {
    '.c': ['gcc', ['-O0', '-std=c11', target, '-o', '/tmp/axon-program'], ['/tmp/axon-program']],
    '.cpp': ['g++', ['-O0', '-std=c++17', target, '-o', '/tmp/axon-program'], ['/tmp/axon-program']],
    '.go': ['go', ['run', target], null],
    '.java': ['javac', ['-d', '/tmp', target], ['java', '-cp', '/tmp', targetBase]],
    '.rs': ['rustc', [target, '-o', '/tmp/axon-program'], ['/tmp/axon-program']],
  };
  const interpreted = {
    '.js': ['node', [target]],
    '.mjs': ['node', [target]],
    '.py': ['python3', [target]],
  };
  const compilation = compilers[extension];
  const execution = interpreted[extension];
  if (!compilation && !execution) {
    throw new Error(`No isolated runtime is configured for ${extension || 'files without an extension'}.`);
  }

  let result;
  if (compilation) {
    result = await run(compilation[0], compilation[1], workspaceRoot);
    if (result.code === 0 && compilation[2]) {
      const program = compilation[2];
      result = await run(program[0] ?? compilation[0], program.slice(1), workspaceRoot, stdin);
    }
  } else if (execution) {
    result = await run(execution[0], execution[1], workspaceRoot, stdin);
  }

  emit({
    ok: result.code === 0 && !result.timedOut && !result.outputExceeded,
    exitCode: result.code,
    signal: result.signal,
    stdout: result.stdout,
    stderr: result.stderr,
    timedOut: result.timedOut,
    outputExceeded: result.outputExceeded,
  });
} catch (error) {
  emit({ ok: false, error: error instanceof Error ? error.message : 'Runner could not execute this workspace.' });
}
