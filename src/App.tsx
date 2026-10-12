import { ChangeEvent, KeyboardEvent as ReactKeyboardEvent, useEffect, useMemo, useRef, useState } from 'react';
import { strToU8, zipSync } from 'fflate';

type ChatMessage = {
  id: number;
  role: 'user' | 'assistant';
  text: string;
};

type ModeKey = 'build' | 'fix' | 'refactor' | 'explain' | 'design';

type FileItem = {
  id: string;
  name: string;
  language: string;
  value: string;
};

type GeneratedFilePayload = {
  path: string;
  content: string;
};

type AgentPlanStep = {
  id: string;
  label: string;
  status: 'planned' | 'active' | 'done';
};

type ModelStatus = 'checking' | 'not-loaded' | 'loading' | 'loaded' | 'unavailable' | 'offline';

type ModelHealth = {
  status: Exclude<ModelStatus, 'checking' | 'offline'>;
  progress: number | null;
};

type RunnerResult = {
  ok: boolean;
  exitCode?: number | null;
  signal?: string | null;
  stdout?: string;
  stderr?: string;
  timedOut?: boolean;
  outputExceeded?: boolean;
  error?: string;
};

async function fetchModelHealth(): Promise<ModelHealth> {
  const response = await fetch('/api/health');
  if (!response.ok) throw new Error('Model health request failed');
  return response.json();
}

function isGeneratedFilePayload(value: unknown): value is GeneratedFilePayload {
  if (typeof value !== 'object' || value === null) return false;
  const candidate = value as { path?: unknown; content?: unknown };
  return typeof candidate.path === 'string' &&
    isSafeWorkspacePath(candidate.path) &&
    typeof candidate.content === 'string';
}

function isSafeWorkspacePath(path: string): boolean {
  return !path.startsWith('/') &&
    !path.includes('\\') &&
    !/^[a-zA-Z]:/.test(path) &&
    path.split('/').every((part) => part !== '' && part !== '.' && part !== '..');
}

const starterFiles: FileItem[] = [
  {
    id: 'app',
    name: 'App.tsx',
    language: 'tsx',
    value: `export default function App() {
  return (
    <main className="ax-shell">
      <section className="ax-card">
        <span className="ax-badge">AXON</span>
        <h1>Private AI product studio</h1>
        <p>Built for browser-native reasoning and rapid product generation.</p>
      </section>
    </main>
  );
}`,
  },
  {
    id: 'styles',
    name: 'styles.css',
    language: 'css',
    value: `.ax-shell {
  min-height: 100vh;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, #020817, #111827, #0f172a);
  color: white;
  padding: 2rem;
}

.ax-card {
  max-width: 720px;
  background: rgba(15, 23, 42, 0.8);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 24px;
  padding: 2rem;
}`,
  },
  {
    id: 'readme',
    name: 'README.md',
    language: 'md',
    value: '# AXON\n\nLocal-first AI workspace.',
  },
];

const newFileTypes = [
  ['tsx', 'TypeScript React'], ['ts', 'TypeScript'], ['js', 'JavaScript'], ['html', 'HTML'],
  ['css', 'CSS'], ['json', 'JSON'], ['py', 'Python'], ['c', 'C'], ['cpp', 'C++'],
  ['rs', 'Rust'], ['go', 'Go'], ['java', 'Java'], ['cs', 'C#'], ['rb', 'Ruby'],
  ['php', 'PHP'], ['swift', 'Swift'], ['kt', 'Kotlin'], ['sql', 'SQL'], ['sh', 'Shell'],
  ['yaml', 'YAML'], ['toml', 'TOML'], ['md', 'Markdown'],
] as const;

const quickPrompts = [
  'Build a SaaS analytics dashboard with charts and customer metrics',
  'Create a modern AI coding workspace UI with a dark editor and agent panel',
  'Design a subscription landing page for an AI startup',
  'Fix a React app with state bugs and broken async flows',
];

const newId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

function getLanguageFromPath(path: string): string {
  const extension = path.split('.').pop()?.toLowerCase() ?? '';
  const languages: Record<string, string> = {
    tsx: 'tsx', ts: 'typescript', js: 'javascript', mjs: 'javascript', jsx: 'jsx',
    html: 'html', css: 'css', json: 'json', py: 'python', c: 'c',
    h: 'c', cpp: 'cpp', cc: 'cpp', cxx: 'cpp', hpp: 'cpp', hxx: 'cpp',
    rs: 'rust', go: 'go', java: 'java',
    cs: 'csharp', rb: 'ruby', php: 'php', swift: 'swift', kt: 'kotlin',
    sql: 'sql', sh: 'shell', yaml: 'yaml', yml: 'yaml', toml: 'toml', md: 'markdown',
  };
  return languages[extension] ?? 'text';
}

function isRunnableWorkspaceFile(path: string): boolean {
  return /\.(c|cc|cpp|cxx|go|java|mjs|js|py|rs)$/i.test(path);
}

function buildProjectBlueprint(prompt: string) {
  const text = prompt.toLowerCase();

  if (text.includes('dashboard')) return 'analytics dashboard';
  if (text.includes('landing') || text.includes('saas') || text.includes('startup')) return 'startup landing page';
  if (text.includes('auth') || text.includes('login')) return 'auth system';
  if (text.includes('crm') || text.includes('sales')) return 'CRM workspace';
  if (text.includes('portfolio') || text.includes('personal')) return 'portfolio site';
  if (text.includes('ecommerce') || text.includes('shop')) return 'ecommerce storefront';
  return 'AI product workspace';
}

function createAxonResponse(mode: ModeKey, prompt: string) {
  const basePrompt = prompt.trim() || 'an elegant product experience';
  const blueprint = buildProjectBlueprint(basePrompt);

  const appCode = `import { useMemo, useState } from 'react';

const metrics = [
  { label: 'Performance', value: '96.4%' },
  { label: 'Mrr', value: '$184k' },
  { label: 'Velocity', value: '2.4x' },
];

const tasks = [
  'Map product requirements',
  'Design the system architecture',
  'Ship the working interface',
];

export default function App() {
  const [activeIndex, setActiveIndex] = useState(0);

  const activeTask = useMemo(() => tasks[activeIndex], [activeIndex]);

  return (
    <main className="axon-shell">
      <section className="axon-panel">
        <header className="axon-header">
          <div>
            <span className="axon-badge">AXON</span>
            <h1>${blueprint}</h1>
          </div>
          <button className="axon-button">Launch</button>
        </header>

        <div className="axon-metrics">
          {metrics.map((metric) => (
            <article key={metric.label} className="axon-metric">
              <small>{metric.label}</small>
              <strong>{metric.value}</strong>
            </article>
          ))}
        </div>

        <div className="axon-content">
          <div className="axon-card">
            <p className="axon-label">Current focus</p>
            <h3>{activeTask}</h3>
            <p>
              AXON interprets the request, structures the plan, and generates a coherent product flow for ${basePrompt}.
            </p>
          </div>

          <div className="axon-sidebar">
            {tasks.map((task, index) => (
              <button
                key={task}
                className={'axon-task ' + (index === activeIndex ? 'selected' : '')}
                onClick={() => setActiveIndex(index)}
              >
                {task}
              </button>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}`;

  const stylesCode = `
.axon-shell {
  min-height: 100vh;
  display: grid;
  place-items: center;
  background:
    radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 30%),
    linear-gradient(135deg, #020817, #0b1220 40%, #111827 100%);
  color: #edf6ff;
  padding: 32px;
}

.axon-panel {
  width: min(980px, 100%);
  background: rgba(15, 23, 42, 0.82);
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 24px;
  box-shadow: 0 30px 80px rgba(15, 23, 42, 0.45);
  padding: 28px;
}

.axon-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}

.axon-badge {
  display: inline-block;
  padding: 0.42rem 0.7rem;
  border-radius: 999px;
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  background: rgba(96, 165, 250, 0.16);
  color: #dbeafe;
}

h1 {
  margin: 0.8rem 0 0;
  font-size: clamp(2rem, 3vw, 3.2rem);
}

.axon-metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin-top: 24px;
}

.axon-metric,
.axon-card {
  background: rgba(15, 23, 42, 0.68);
  border: 1px solid rgba(148, 163, 184, 0.12);
  border-radius: 18px;
  padding: 18px;
}

.axon-metric small,
.axon-label {
  color: #93c5fd;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 11px;
}

.axon-metric strong {
  display: block;
  margin-top: 10px;
  font-size: 1.8rem;
}

.axon-content {
  display: grid;
  grid-template-columns: 1.3fr 0.7fr;
  gap: 16px;
  margin-top: 24px;
}

.axon-card {
  min-height: 220px;
}

.axon-sidebar {
  display: grid;
  gap: 10px;
}

.axon-task {
  padding: 0.8rem 1rem;
  border-radius: 12px;
  border: 1px solid rgba(148, 163, 184, 0.16);
  background: rgba(15, 23, 42, 0.7);
  color: #e2e8f0;
  text-align: left;
}

.axon-task.selected {
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.22), rgba(168, 85, 247, 0.18));
  border-color: rgba(96, 165, 250, 0.52);
}

.axon-button {
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  border: none;
  border-radius: 12px;
  color: white;
  padding: 0.8rem 1.1rem;
  cursor: pointer;
}
`;

  const readmeCode = `# AXON generated project

## Goal
${basePrompt}

## Stack
- React
- TypeScript
- Browser-first workflow

## Run
npm install
npm run dev -- --host 0.0.0.0
`;

  const nextPlan: AgentPlanStep[] = [
    { id: 'understand', label: 'Understand user goal', status: 'done' },
    { id: 'architect', label: 'Map architecture', status: 'done' },
    { id: 'implement', label: 'Generate core UI', status: 'active' },
    { id: 'verify', label: 'Validate flow', status: 'planned' },
  ];

  const actionList = {
    build: ['Architect the system', 'Generate the core UI', 'Validate the interaction flow'],
    fix: ['Trace the root cause', 'Patch state and logic', 'Check edge cases'],
    refactor: ['Reduce duplication', 'Split responsibilities', 'Re-check behavior'],
    explain: ['Map the system flow', 'Identify constraints', 'Recommend the cleanest path'],
    design: ['Shape the visual language', 'Refine the interaction model', 'Balance clarity and conversion'],
  } as const;

  return {
    summary: `AXON reviewed the request for ${basePrompt} and mapped a working execution plan for a ${blueprint}.`,
    plan: nextPlan,
    actions: actionList[mode],
    files: [
      { id: 'app', name: 'App.tsx', language: 'tsx', value: appCode },
      { id: 'styles', name: 'styles.css', language: 'css', value: stylesCode },
      { id: 'readme', name: 'README.md', language: 'md', value: readmeCode },
    ],
  };
}

function App() {
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 1,
      role: 'assistant',
      text: 'AXON workspace ready. Describe a coding task to generate project files with the configured local model adapter.',
    },
  ]);
  const [files, setFiles] = useState<FileItem[]>(starterFiles);
  const [selectedFile, setSelectedFile] = useState<string>('app');
  const [newFileExtension, setNewFileExtension] = useState<(typeof newFileTypes)[number][0]>('tsx');
  const [isLoading, setIsLoading] = useState(false);
  const [requestError, setRequestError] = useState('');
  const [runnerBusy, setRunnerBusy] = useState(false);
  const [runnerStdin, setRunnerStdin] = useState('');
  const [runnerError, setRunnerError] = useState('');
  const [runnerResult, setRunnerResult] = useState<RunnerResult | null>(null);
  const [runnerFileName, setRunnerFileName] = useState('');
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const [deepReasoning, setDeepReasoning] = useState(true);
  const [modelStatus, setModelStatus] = useState<ModelStatus>('checking');
  const [modelProgress, setModelProgress] = useState<number | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, isLoading]);

  const startNewChat = () => {
    setMessages([{
      id: 1,
      role: 'assistant',
      text: 'AXON workspace ready. Describe a coding task to generate project files with the configured local model adapter.',
    }]);
    setPrompt('');
    setRequestError('');
    setRunnerError('');
    setRunnerResult(null);
  };

  useEffect(() => {
    let active = true;
    fetchModelHealth()
      .then((health) => {
        if (!active) return;
        setModelStatus(health.status);
        setModelProgress(health.progress);
      })
      .catch(() => {
        if (active) setModelStatus('offline');
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (modelStatus !== 'loading') return;
    let active = true;
    const interval = window.setInterval(() => {
      fetchModelHealth()
        .then((health) => {
          if (!active) return;
          setModelStatus(health.status);
          setModelProgress(health.progress);
        })
        .catch(() => {
          if (active) setModelStatus('offline');
        });
    }, 1500);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [modelStatus]);

  const currentFile = useMemo(
    () => files.find((file) => file.id === selectedFile) ?? files[0],
    [files, selectedFile],
  );

  const handleGenerate = async () => {
    const request = prompt.trim();
    if (!request || isLoading) return;
    const requestText = request;
    const conversation = [
      ...messages.filter((message) => message.role !== 'assistant' || message.id !== 1),
      { id: Date.now(), role: 'user' as const, text: requestText },
    ].slice(-20);

    setIsLoading(true);
    setRequestError('');
    setMessages(conversation);

    try {
      const response = await fetch('/api/agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: conversation.map(({ role, text }) => ({ role, content: text })),
          files: files.map(({ name, value }) => ({ path: name, content: value })),
          deepReasoning,
        }),
      });

      let payload: {
        message?: unknown;
        error?: unknown;
        files?: unknown;
      };
      try {
        payload = await response.json();
      } catch {
        throw new Error('AXOM received an invalid response from its local backend.');
      }
      if (!response.ok) throw new Error(typeof payload.error === 'string' ? payload.error : 'AXOM could not complete the request.');
      if (typeof payload.message !== 'string' || !payload.message.trim()) {
        throw new Error('The local model returned an empty response. Please try again.');
      }
      const assistantText = payload.message.trim();

      const generatedFiles = payload.files === undefined
        ? []
        : Array.isArray(payload.files) && payload.files.every(isGeneratedFilePayload)
          ? payload.files.map((file) => ({
            id: `generated-${newId()}`,
            name: file.path,
            language: file.path.split('.').pop()?.toLowerCase() || 'text',
            value: file.content,
          }))
          : (() => { throw new Error('AXOM returned an invalid project file list.'); })();

      setMessages((previous) => [
        ...conversation,
        { id: Date.now() + 1, role: 'assistant', text: assistantText },
      ]);
      if (generatedFiles.length > 0) {
        setFiles((previous) => {
          const byPath = new Map(previous.map((file) => [file.name.toLowerCase(), file]));
          generatedFiles.forEach((file) => {
            const existing = byPath.get(file.name.toLowerCase());
            byPath.set(file.name.toLowerCase(), { ...file, id: existing?.id ?? file.id });
          });
          return [...byPath.values()];
        });
        const existingSelectedFile = files.find(
          (file) => file.name.toLowerCase() === generatedFiles[0].name.toLowerCase(),
        );
        setSelectedFile(existingSelectedFile?.id ?? generatedFiles[0].id);
      }
      setPrompt('');
      setModelStatus('loaded');
      setModelProgress(100);
    } catch (error) {
      const message = error instanceof TypeError
        ? 'Could not connect to AXOM’s local model backend. Start the backend with `npm run dev:server`, then try again.'
        : error instanceof Error
          ? error.message
          : 'AXOM could not run the local model.';
      setRequestError(message);
      setModelStatus(error instanceof TypeError ? 'offline' : 'unavailable');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePromptKeyDown = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void handleGenerate();
    }
  };

  const handleEditorChange = (value: string) => {
    setRunnerResult(null);
    setFiles((prev) =>
      prev.map((file) =>
        file.id === selectedFile ? { ...file, value } : file,
      ),
    );
  };

  const handleRunCurrentFile = async () => {
    if (!isRunnableWorkspaceFile(currentFile.name) || runnerBusy) return;
    setRunnerBusy(true);
    setRunnerError('');
    setRunnerResult(null);
    setRunnerFileName(currentFile.name);

    try {
      const response = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          entrypoint: currentFile.name,
          files: files.map(({ name, value }) => ({ path: name, content: value })),
          stdin: runnerStdin,
        }),
      });
      let payload: RunnerResult;
      try {
        payload = await response.json();
      } catch {
        throw new Error('AXON received an invalid response from the isolated runner.');
      }
      if (!response.ok) {
        throw new Error(payload.error || 'AXON could not start the isolated runner.');
      }
      if (typeof payload.ok !== 'boolean') {
        throw new Error('The isolated runner returned an invalid execution result.');
      }
      setRunnerResult(payload);
      setRunnerFileName(currentFile.name);
    } catch (error) {
      setRunnerError(error instanceof TypeError
        ? 'Could not connect to AXON. Start the local app and Docker, then try again.'
        : error instanceof Error
          ? error.message
          : 'AXON could not run this file.');
    } finally {
      setRunnerBusy(false);
    }
  };

  const handleSendRunnerOutput = () => {
    if (!runnerResult) return;
    const details = [
      `Please diagnose and fix this execution result for ${runnerFileName}.`,
      `Exit code: ${runnerResult.exitCode ?? 'unknown'}`,
      runnerResult.stdout ? `stdout:\n${runnerResult.stdout}` : '',
      runnerResult.stderr ? `stderr:\n${runnerResult.stderr}` : '',
      runnerResult.error ? `Runner error:\n${runnerResult.error}` : '',
      runnerResult.timedOut ? 'The runner reported that execution timed out.' : '',
      runnerResult.outputExceeded ? 'The runner output exceeded its limit.' : '',
    ].filter(Boolean).join('\n\n');
    setPrompt((current) => current.trim() ? `${current.trim()}\n\n${details}` : details);
  };

  const handleNewFile = () => {
    const [extension] = newFileTypes.find(([candidate]) => candidate === newFileExtension) ?? newFileTypes[0];
    const id = `file-${Date.now()}`;
    const name = `module-${files.length + 1}.${extension}`;
    const file: FileItem = {
      id,
      name,
      language: getLanguageFromPath(name),
      value: extension === 'tsx'
        ? `export default function Module() {\n  return <section>New AXON module</section>;\n}`
        : '',
    };

    setFiles((prev) => [...prev, file]);
    setSelectedFile(id);
  };

  const handleImportFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(event.target.files ?? []);
    event.target.value = '';
    if (selected.length === 0) return;

    try {
      const imported = await Promise.all(selected.map(async (file) => {
        if (file.size > 48000) {
          throw new Error(`${file.name} is too large to import as a text workspace file.`);
        }
        const value = await file.text();
        if (value.length > 12000 || value.includes('\0')) {
          throw new Error(`${file.name} is too large or is not a text source file.`);
        }
        const name = file.name.replace(/\\/g, '/').split('/').pop() || 'imported.txt';
        return {
          id: `imported-${newId()}`,
          name,
          language: getLanguageFromPath(name),
          value,
        };
      }));
      const merged = new Map(files.map((file) => [file.name.toLowerCase(), file]));
      imported.forEach((file) => merged.set(file.name.toLowerCase(), file));
      const nextFiles = [...merged.values()];
      if (nextFiles.reduce((total, file) => total + file.value.length, 0) > 24000) {
        throw new Error('The workspace exceeds the 24,000 character model-context limit.');
      }
      setFiles(nextFiles);
      setSelectedFile(imported[imported.length - 1].id);
      setRequestError('');
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : 'AXON could not import those files.');
    }
  };

  const downloadFile = (file: FileItem) => {
    const blob = new Blob([file.value], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.href = url;
    link.download = file.name;
    link.click();
    URL.revokeObjectURL(url);
  };

  const downloadProject = () => {
    const archive = zipSync(Object.fromEntries(
      files.map((file) => [file.name, strToU8(file.value)]),
    ));
    const archiveBuffer = new ArrayBuffer(archive.byteLength);
    new Uint8Array(archiveBuffer).set(archive);
    const blob = new Blob([archiveBuffer], { type: 'application/zip' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.href = url;
    link.download = 'axon-project.zip';
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark">A</div>
          <div>
            <p className="muted">Workspace</p>
            <h2>AXON</h2>
          </div>
        </div>

        <button className="new-chat-button" onClick={startNewChat}>
          <span aria-hidden="true">＋</span>
          <span>New chat</span>
        </button>

        <div className="sidebar-panel conversation-sidebar">
          <div className="panel-label">Chats</div>
          <button className="nav-item active current-chat" onClick={startNewChat}>
            <span aria-hidden="true">◌</span>
            <span>Current conversation</span>
          </button>
        </div>

        <div className="panel-block">
          <div className="panel-header">
            <span>Workspace files</span>
            <span className="badge">{files.length}</span>
          </div>
          <div className="file-list">
            {files.map((file) => (
              <button
                key={file.id}
                className={`file-item ${selectedFile === file.id ? 'selected' : ''}`}
                onClick={() => {
                  setSelectedFile(file.id);
                  setWorkspaceOpen(true);
                }}
              >
                <span className="ext">{file.language}</span>
                {file.name}
              </button>
            ))}
          </div>
        </div>
        <div className="sidebar-footer">
          <span className={`sidebar-status-dot ${modelStatus === 'loaded' ? 'ready' : ''}`} />
          <span>{modelStatus === 'offline' ? 'Local model offline' : 'Local workspace'}</span>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="chat-title">
            <button className="mobile-new-chat" onClick={startNewChat} aria-label="Start a new chat">＋</button>
            <strong>AXON</strong>
            <span className="model-title">Local coding assistant</span>
          </div>

          <div className="status-row">
            <span className={`status-pill ${modelStatus === 'loaded' ? 'online' : ''}`}>
              {modelStatus === 'loading' && modelProgress !== null
                ? `Loading model ${modelProgress}%`
                : modelStatus === 'not-loaded'
                  ? 'Model not loaded'
                  : modelStatus === 'unavailable'
                    ? 'Model unavailable'
                    : modelStatus === 'offline'
                      ? 'Backend offline'
                      : modelStatus === 'checking'
                        ? 'Checking model'
                        : 'Model loaded'}
            </span>
            <button
              className={`action-button workspace-toggle ${workspaceOpen ? 'selected' : ''}`}
              onClick={() => setWorkspaceOpen((open) => !open)}
              aria-pressed={workspaceOpen}
            >
              {workspaceOpen ? 'Close workspace' : 'Workspace'}
            </button>
          </div>
        </header>

        {requestError && (
          <div className="request-error" role="alert">
            <span>{requestError}</span>
            <button onClick={() => setRequestError('')} aria-label="Dismiss error">×</button>
          </div>
        )}

        <section className={`workspace-panel ${workspaceOpen ? 'workspace-visible' : 'workspace-hidden'}`}>
          {workspaceOpen && (
            <div className="editor-column workspace-drawer">
              <div className="workspace-drawer-header">
                <div className="editor-header">
                  <span>{currentFile.name}</span>
                  <span className="tiny">{currentFile.language.toUpperCase()}</span>
                </div>
                <button className="workspace-close" onClick={() => setWorkspaceOpen(false)} aria-label="Close workspace">
                  ×
                </button>
              </div>
              <div className="workspace-toolbar">
                <label className="new-file-language">
                  <span className="visually-hidden">New file language</span>
                  <select
                    value={newFileExtension}
                    onChange={(event) => {
                      const match = newFileTypes.find(([extension]) => extension === event.target.value);
                      if (match) setNewFileExtension(match[0]);
                    }}
                  >
                    {newFileTypes.map(([extension, label]) => (
                      <option value={extension} key={extension}>{label}</option>
                    ))}
                  </select>
                </label>
                <button className="action-button" onClick={handleNewFile}>New file</button>
                <label className="action-button import-files">
                  Import
                  <input type="file" multiple onChange={handleImportFiles} />
                </label>
                <button className="action-button secondary" onClick={() => downloadFile(currentFile)}>
                  Download
                </button>
                <button className="action-button secondary" onClick={downloadProject}>
                  ZIP
                </button>
              </div>
              <button
                className="action-button run-code-button workspace-run-button"
                onClick={handleRunCurrentFile}
                disabled={!isRunnableWorkspaceFile(currentFile.name) || runnerBusy}
                title={isRunnableWorkspaceFile(currentFile.name)
                  ? 'Run this file in the isolated local container'
                  : 'Run supports JavaScript, Python, C, C++, Rust, Go, and Java files'}
              >
                {runnerBusy ? 'Running…' : '▶ Run code'}
              </button>
              <textarea
                className="workspace-editor"
                value={currentFile.value}
                onChange={(event) => handleEditorChange(event.target.value)}
                spellCheck={false}
                aria-label={`Code editor for ${currentFile.name}`}
              />
              <label className="runner-stdin-field">
                <span>Program input (stdin, optional)</span>
                <textarea
                  value={runnerStdin}
                  onChange={(event) => setRunnerStdin(event.target.value)}
                  maxLength={4000}
                  placeholder="Enter test input here"
                />
              </label>
              {(runnerError || runnerResult) && (
                <section className="runner-output" aria-live="polite">
                  <div className="runner-output-header">
                    <strong>
                      {runnerError ? 'Runner unavailable' : runnerResult?.ok ? 'Run completed' : 'Run failed'}
                      {runnerFileName ? ` · ${runnerFileName}` : ''}
                    </strong>
                    {runnerResult && (
                      <span>
                        {runnerResult.timedOut
                          ? 'Timed out'
                          : runnerResult.exitCode === null || runnerResult.exitCode === undefined
                            ? 'No exit code'
                            : `Exit code ${runnerResult.exitCode}`}
                      </span>
                    )}
                  </div>
                  {runnerError
                    ? <p className="runner-error">{runnerError}</p>
                    : (
                      <>
                        {runnerResult?.error && <p className="runner-error">{runnerResult.error}</p>}
                        {runnerResult?.stdout && <pre><strong>stdout</strong>{runnerResult.stdout}</pre>}
                        {runnerResult?.stderr && <pre><strong>stderr</strong>{runnerResult.stderr}</pre>}
                        {!runnerResult?.stdout && !runnerResult?.stderr && !runnerResult?.error && (
                          <p className="runner-empty">No output.</p>
                        )}
                        <button className="action-button secondary" onClick={handleSendRunnerOutput}>
                          Ask AXON to debug this result
                        </button>
                      </>
                    )}
                </section>
              )}
            </div>
          )}

          <div className="assistant-column">
            <div className={`chat-window ${messages.some((message) => message.role === 'user') ? 'has-messages' : 'welcome-chat'}`}>
              {messages.some((message) => message.role === 'user')
                ? messages.filter((message) => message.id !== 1).map((msg) => (
                  <div key={msg.id} className={`message ${msg.role}`}>
                    {msg.role === 'user' && <strong>You</strong>}
                    <p>{msg.text}</p>
                  </div>
                ))
                : (
                  <div className="chat-welcome">
                    <div className="welcome-mark">A</div>
                    <h1>What can I help you build?</h1>
                    <p>Ask a question, describe a bug, or start a project.</p>
                    <div className="welcome-prompts">
                      {quickPrompts.map((example) => (
                        <button key={example} className="prompt-chip" onClick={() => setPrompt(example)}>
                          {example}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              {isLoading && (
                <div className="message assistant">
                  <p className="chat-thinking">AXON is working…</p>
                </div>
              )}
              <div ref={transcriptEndRef} />
            </div>

            <div className="prompt-box">
              {requestError && (
                <div className="composer-error" role="alert">
                  <span>{requestError}</span>
                  <button onClick={() => setRequestError('')} aria-label="Dismiss error">×</button>
                </div>
              )}
              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                onKeyDown={handlePromptKeyDown}
                placeholder="Message AXON"
                aria-label="Message AXON"
              />
              <div className="prompt-actions">
                <button className="workspace-shortcut" onClick={() => setWorkspaceOpen((open) => !open)}>
                  {workspaceOpen ? 'Hide workspace' : `Workspace · ${files.length} files`}
                </button>
                <span className="composer-hint">Enter to send · Shift+Enter for a new line</span>
                <button
                  className="send-button"
                  onClick={handleGenerate}
                  disabled={isLoading || !prompt.trim()}
                  aria-label={isLoading ? 'AXON is thinking' : 'Send message'}
                >
                  {isLoading ? '…' : '↑'}
                </button>
              </div>
            </div>
          </div>
        </section>

      </main>
    </div>
  );
}

export default App;
