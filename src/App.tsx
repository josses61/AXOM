import { ChangeEvent, KeyboardEvent as ReactKeyboardEvent, useEffect, useMemo, useRef, useState } from 'react';
import { strToU8, Unzip, UnzipInflate, UnzipPassThrough, zipSync } from 'fflate';

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

type WorkspaceAsset = {
  id: string;
  name: string;
  data: Uint8Array;
};

type ArchiveEntry = {
  path: string;
  data: Uint8Array;
};

type GeneratedFilePayload = {
  path: string;
  content: string;
};

type ManagedAgent = {
  id: string;
  name: string;
  role: string;
  enabled: boolean;
  team: string;
  managerId: string;
};

type AgentChatMessage = {
  id: string;
  sender: string;
  role: 'user' | 'assistant';
  text: string;
};

type ScrapedPage = {
  title: string;
  url: string;
  content: string;
  description: string;
  author: string;
  publishedAt: string;
  canonicalUrl: string;
  headings: string[];
};

type ManagedTask = {
  id: string;
  title: string;
  assigneeId: string;
  status: 'queued' | 'in-progress' | 'done';
};

type GitHubRepository = {
  nameWithOwner: string;
  description: string | null;
  isPrivate: boolean;
};

type ModelProfile = 'AXOM1.0' | 'AXOM1.1';

type AgentPlanStep = {
  id: string;
  label: string;
  status: 'planned' | 'active' | 'done';
};

type ModelStatus = 'checking' | 'not-loaded' | 'switching' | 'loading' | 'loaded' | 'unavailable' | 'offline';

type ModelHealth = {
  status: Exclude<ModelStatus, 'checking' | 'offline'>;
  progress: number | null;
  profile?: ModelProfile;
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

const MAX_WORKSPACE_FILES = 64;
const MAX_ARCHIVE_INPUT_BYTES = 20 * 1024 * 1024;
const MAX_ARCHIVE_OUTPUT_BYTES = 32 * 1024 * 1024;
const MAX_ASSET_BYTES = 16 * 1024 * 1024;
const MAX_SOURCE_FILE_CHARS = 12000;
const MAX_SOURCE_CONTEXT_CHARS = 24000;

function extractZipEntries(data: Uint8Array, maxOutputBytes = MAX_ARCHIVE_OUTPUT_BYTES): Promise<ArchiveEntry[]> {
  return new Promise((resolve, reject) => {
    const entries: ArchiveEntry[] = [];
    let totalBytes = 0;
    let fileCount = 0;
    let pendingFiles = 0;
    let streamEnded = false;
    let failure: Error | null = null;
    const fail = (message: string) => {
      if (failure) return;
      failure = new Error(message);
      reject(failure);
    };
    const finishIfReady = () => {
      if (streamEnded && pendingFiles === 0 && !failure) {
        if (entries.length === 0) fail('The ZIP archive contains no files.');
        else resolve(entries);
      }
    };
    const archive = new Unzip((file) => {
      if (failure || file.name.endsWith('/')) return;
      if (!isSafeWorkspacePath(file.name)) {
        fail(`The archive contains an unsafe file path: ${file.name}`);
        return;
      }
      fileCount += 1;
      if (fileCount > MAX_WORKSPACE_FILES) {
        fail(`ZIP archives can contain up to ${MAX_WORKSPACE_FILES} files.`);
        return;
      }
      if ((file.originalSize || 0) > maxOutputBytes - totalBytes) {
        fail('The expanded ZIP archive exceeds the 32 MB import limit.');
        return;
      }

      pendingFiles += 1;
      let fileBytes = 0;
      const chunks: Uint8Array[] = [];
      file.ondata = (error, chunk, final) => {
        if (failure) return;
        if (error) {
          fail(`Could not read ${file.name} from the ZIP archive.`);
          return;
        }
        fileBytes += chunk.byteLength;
        totalBytes += chunk.byteLength;
        if (totalBytes > maxOutputBytes) {
          fail('The expanded ZIP archive exceeds the 32 MB import limit.');
          return;
        }
        chunks.push(chunk);
        if (!final) return;
        const content = new Uint8Array(fileBytes);
        let offset = 0;
        chunks.forEach((part) => {
          content.set(part, offset);
          offset += part.byteLength;
        });
        entries.push({ path: file.name, data: content });
        pendingFiles -= 1;
        finishIfReady();
      };
      try {
        file.start();
      } catch {
        fail(`The ZIP archive uses an unsupported compression method for ${file.name}.`);
      }
    });
    archive.register(UnzipPassThrough);
    archive.register(UnzipInflate);
    try {
      archive.push(data, true);
      streamEnded = true;
      finishIfReady();
    } catch {
      fail('The selected ZIP archive is invalid or damaged.');
    }
  });
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

const agentRoleTemplates = [
  { name: 'Researcher', role: 'Find relevant documentation, APIs, constraints, and current best practices. Cite sources and separate facts from assumptions.' },
  { name: 'Builder', role: 'Turn the request into a coherent implementation. Identify files to change, make focused code proposals, and call out validation steps.' },
  { name: 'Reviewer', role: 'Review proposed work for bugs, security risks, missing requirements, and useful tests. Prioritize actionable findings.' },
  { name: 'Designer', role: 'Shape accessible, clear interface structure and visual details. Consider responsive behavior and real user workflows.' },
  { name: 'Security', role: 'Threat-model the request, find trust boundaries and data risks, and suggest practical mitigations.' },
  { name: 'Writer', role: 'Create concise, accurate documentation and user-facing copy that matches the product and implementation.' },
];

const newId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

function loadManagedTasks(): ManagedTask[] {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem('axom-tasks') || '[]');
    if (!Array.isArray(stored)) return [];
    return stored.flatMap((task): ManagedTask[] =>
      task && typeof task.id === 'string' && typeof task.title === 'string' &&
      typeof task.assigneeId === 'string' &&
      (task.status === 'queued' || task.status === 'in-progress' || task.status === 'done')
        ? [{ id: task.id, title: task.title.slice(0, 180), assigneeId: task.assigneeId, status: task.status }]
        : [],
    ).slice(0, 100);
  } catch {
    return [];
  }
}

const defaultAgents: ManagedAgent[] = [
  { id: 'research', name: 'Scout', role: 'Research relevant APIs, documentation, constraints, and current best practices.', enabled: true, team: 'Studio', managerId: '' },
  { id: 'builder', name: 'Builder', role: 'Turn the request into a coherent implementation and identify the files or steps required.', enabled: true, team: 'Studio', managerId: 'research' },
  { id: 'reviewer', name: 'Reviewer', role: 'Review the proposed solution for bugs, security risks, missing requirements, and useful tests.', enabled: true, team: 'Studio', managerId: 'research' },
];

function loadAgentChats(): Record<string, AgentChatMessage[]> {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem('axom-agent-chats') || '{}');
    if (!stored || typeof stored !== 'object' || Array.isArray(stored)) return {};
    return Object.fromEntries(Object.entries(stored).flatMap(([team, messages]) => {
      if (!Array.isArray(messages)) return [];
      const valid = messages.flatMap((message): AgentChatMessage[] =>
        message && typeof message.id === 'string' && typeof message.sender === 'string' &&
        (message.role === 'user' || message.role === 'assistant') && typeof message.text === 'string'
          ? [{ id: message.id, sender: message.sender.slice(0, 48), role: message.role, text: message.text.slice(0, 6000) }]
          : [],
      ).slice(-40);
      return valid.length > 0 ? [[team.slice(0, 48), valid]] : [];
    }));
  } catch {
    return {};
  }
}

function loadManagedAgents(): ManagedAgent[] {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem('axom-agents') || 'null');
    if (!Array.isArray(stored)) return defaultAgents;
    const agents = stored.flatMap((agent): ManagedAgent[] =>
      agent &&
      typeof agent.id === 'string' &&
      typeof agent.name === 'string' &&
      typeof agent.role === 'string'
        ? [{
          id: agent.id,
          name: agent.name.slice(0, 48),
          role: agent.role.slice(0, 400),
          enabled: Boolean(agent.enabled),
          team: typeof agent.team === 'string' ? agent.team.slice(0, 48) : 'Studio',
          managerId: typeof agent.managerId === 'string' ? agent.managerId.slice(0, 80) : '',
        }]
        : [],
    );
    return agents.slice(0, 8);
  } catch {
    return defaultAgents;
  }
}

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
  const [assets, setAssets] = useState<WorkspaceAsset[]>([]);
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
  const [modelProfile, setModelProfile] = useState<ModelProfile>('AXOM1.1');
  const [agents, setAgents] = useState<ManagedAgent[]>(loadManagedAgents);
  const [tasks, setTasks] = useState<ManagedTask[]>(loadManagedTasks);
  const [queueBusy, setQueueBusy] = useState(false);
  const [agentDraftName, setAgentDraftName] = useState('');
  const [agentDraftTeam, setAgentDraftTeam] = useState('Studio');
  const [agentDraftRole, setAgentDraftRole] = useState(agentRoleTemplates[1].role);
  const [agentDraftTemplate, setAgentDraftTemplate] = useState('Builder');
  const [agentDraftManagerId, setAgentDraftManagerId] = useState('');
  const [agentChats, setAgentChats] = useState<Record<string, AgentChatMessage[]>>(loadAgentChats);
  const [selectedAgentId, setSelectedAgentId] = useState('research');
  const [agentChatDraft, setAgentChatDraft] = useState('');
  const [agentChatBusy, setAgentChatBusy] = useState(false);
  const [agentChatError, setAgentChatError] = useState('');
  const [taskTitle, setTaskTitle] = useState('');
  const [agentsOpen, setAgentsOpen] = useState(false);
  const [emailAddress, setEmailAddress] = useState(() => localStorage.getItem('axom-email') || '');
  const [webUrl, setWebUrl] = useState('');
  const [webLoading, setWebLoading] = useState(false);
  const [webError, setWebError] = useState('');
  const [scrapedPage, setScrapedPage] = useState<ScrapedPage | null>(null);
  const [githubAccount, setGithubAccount] = useState<{ login: string; url: string } | null>(null);
  const [githubRepositories, setGithubRepositories] = useState<GitHubRepository[]>([]);
  const [githubRepository, setGithubRepository] = useState('');
  const [githubTrustedAuthors, setGithubTrustedAuthors] = useState<string[]>(() => {
    try {
      const stored: unknown = JSON.parse(localStorage.getItem('axom-github-trusted-authors') || '[]');
      return Array.isArray(stored) ? stored.filter((author): author is string => typeof author === 'string').slice(0, 50) : [];
    } catch {
      return [];
    }
  });
  const [githubAuthorInput, setGithubAuthorInput] = useState('');
  const [newRepositoryName, setNewRepositoryName] = useState('');
  const [newRepositoryPrivate, setNewRepositoryPrivate] = useState(true);
  const [githubLoading, setGithubLoading] = useState(false);
  const [githubError, setGithubError] = useState('');
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef(messages);
  const filesRef = useRef(files);
  const activeAgents = agents.filter((agent) => agent.enabled);
  const agentTeams = [...new Set(agents.map((agent) => agent.team.trim()).filter(Boolean))];
  const selectedAgent = agents.find((agent) => agent.id === selectedAgentId) || activeAgents[0] || agents[0];
  const selectedAgentMessages = selectedAgent ? agentChats[selectedAgent.id] || [] : [];
  const selectedGitHubOwner = githubRepository.split('/')[0] || '';
  const selectedGitHubOwnerTrusted = githubTrustedAuthors.some((author) => author.toLowerCase() === selectedGitHubOwner.toLowerCase());

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [selectedAgentMessages, agentChatBusy]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    filesRef.current = files;
  }, [files]);

  useEffect(() => {
    localStorage.setItem('axom-agents', JSON.stringify(agents));
  }, [agents]);

  useEffect(() => {
    localStorage.setItem('axom-tasks', JSON.stringify(tasks));
  }, [tasks]);

  useEffect(() => {
    localStorage.setItem('axom-agent-chats', JSON.stringify(agentChats));
  }, [agentChats]);

  useEffect(() => {
    localStorage.setItem('axom-github-trusted-authors', JSON.stringify(githubTrustedAuthors));
  }, [githubTrustedAuthors]);

  useEffect(() => {
    localStorage.setItem('axom-email', emailAddress);
  }, [emailAddress]);

  const startNewChat = () => {
    if (selectedAgent) {
      setAgentChats((current) => ({ ...current, [selectedAgent.id]: [] }));
    }
    const nextMessages: ChatMessage[] = [{
      id: 1,
      role: 'assistant',
      text: 'AXON workspace ready. Describe a coding task to generate project files with the configured local model adapter.',
    }];
    messagesRef.current = nextMessages;
    setMessages(nextMessages);
    setPrompt('');
    setRequestError('');
    setAgentChatError('');
    setAgentChatDraft('');
    setRunnerError('');
    setRunnerResult(null);
  };

  useEffect(() => {
    let active = true;
    fetch('/api/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
      .then((response) => {
        if (!response.ok) throw new Error('The local model could not be started.');
        return fetchModelHealth();
      })
      .then((health) => {
        if (!active) return;
        setModelStatus(health.status);
        setModelProgress(health.progress);
        if (health.profile) setModelProfile(health.profile);
      })
      .catch(() => {
        if (active) setModelStatus('unavailable');
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (modelStatus !== 'loading' && modelStatus !== 'switching') return;
    let active = true;
    const interval = window.setInterval(() => {
      fetchModelHealth()
        .then((health) => {
          if (!active) return;
          setModelStatus(health.status);
          setModelProgress(health.progress);
          if (health.profile) setModelProfile(health.profile);
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

  const refreshGitHub = async () => {
    setGithubLoading(true);
    setGithubError('');
    try {
      const statusResponse = await fetch('/api/github/status');
      const status = await statusResponse.json() as {
        connected?: boolean;
        account?: { login?: string; url?: string };
        error?: string;
      };
      if (!status.connected || !status.account?.login) {
        setGithubAccount(null);
        setGithubRepositories([]);
        setGithubError(status.error || 'Sign in with `gh auth login` in the local terminal to connect GitHub.');
        return;
      }
      setGithubAccount({ login: status.account.login, url: status.account.url || 'https://github.com' });
      const repositoriesResponse = await fetch('/api/github/repos');
      const repositoriesPayload = await repositoriesResponse.json() as {
        repositories?: GitHubRepository[];
        error?: string;
      };
      if (!repositoriesResponse.ok || !Array.isArray(repositoriesPayload.repositories)) {
        throw new Error(repositoriesPayload.error || 'Could not load GitHub repositories.');
      }
      setGithubRepositories(repositoriesPayload.repositories);
      setGithubRepository((current) => current || repositoriesPayload.repositories?.[0]?.nameWithOwner || '');
    } catch (error) {
      setGithubError(error instanceof Error ? error.message : 'Could not connect to GitHub.');
    } finally {
      setGithubLoading(false);
    }
  };

  const handleModelProfileChange = async (profile: ModelProfile) => {
    setModelProfile(profile);
    setModelStatus('switching');
    setRequestError('');
    try {
      const response = await fetch('/api/model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile }),
      });
      const payload = await response.json() as { error?: string };
      if (!response.ok) throw new Error(payload.error || 'The model profile could not be changed.');
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : 'The model profile could not be changed.');
      setModelStatus('unavailable');
    }
  };

  const handleImportGitHubRepository = async () => {
    if (!githubRepository || githubLoading) return;
    setGithubLoading(true);
    setGithubError('');
    try {
      const response = await fetch('/api/github/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repository: githubRepository }),
      });
      const payload = await response.json() as {
        repository?: { fullName?: string; defaultBranch?: string };
        files?: Array<{ path: string; content: string }>;
        error?: string;
      };
      if (!response.ok || !Array.isArray(payload.files)) {
        throw new Error(payload.error || 'Could not import this repository.');
      }
      const merged = new Map(files.map((file) => [file.name.toLowerCase(), file]));
      const imported = payload.files.map((file) => ({
        id: `github-${newId()}`,
        name: file.path,
        language: getLanguageFromPath(file.path),
        value: file.content,
      }));
      imported.forEach((file) => merged.set(file.name.toLowerCase(), file));
      const nextFiles = [...merged.values()];
      if (nextFiles.reduce((total, file) => total + file.value.length, 0) > 24000) {
        throw new Error('The imported repository would exceed the 24,000 character workspace context limit.');
      }
      setFiles(nextFiles);
      if (imported.length > 0) setSelectedFile(imported[0].id);
      setWorkspaceOpen(true);
      setGithubError(`Imported ${imported.length} text files from ${payload.repository?.fullName || githubRepository}.`);
    } catch (error) {
      setGithubError(error instanceof Error ? error.message : 'Could not import this repository.');
    } finally {
      setGithubLoading(false);
    }
  };

  const handlePublishGitHubRepository = async () => {
    if (!githubRepository || githubLoading) return;
    setGithubLoading(true);
    setGithubError('');
    try {
      const response = await fetch('/api/github/publish', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repository: githubRepository,
          files: files.map(({ name, value }) => ({ path: name, content: value })),
        }),
      });
      const payload = await response.json() as { publishedCount?: number; error?: string };
      if (!response.ok) throw new Error(payload.error || 'Could not publish this workspace.');
      setGithubError(`Published ${payload.publishedCount || 0} files to ${githubRepository}.`);
    } catch (error) {
      setGithubError(error instanceof Error ? error.message : 'Could not publish this workspace.');
    } finally {
      setGithubLoading(false);
    }
  };

  const handleScrapeWebPage = async () => {
    if (!webUrl.trim() || webLoading) return;
    setWebLoading(true);
    setWebError('');
    setScrapedPage(null);
    try {
      const response = await fetch('/api/web/scrape', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: webUrl.trim() }),
      });
      const payload = await response.json() as {
        title?: string;
        url?: string;
        content?: string;
        description?: string;
        author?: string;
        publishedAt?: string;
        canonicalUrl?: string;
        headings?: string[];
        error?: string;
      };
      if (!response.ok || !payload.url || !payload.content) {
        throw new Error(payload.error || 'Could not read this web page.');
      }
      setScrapedPage({
        title: payload.title || payload.url,
        url: payload.url,
        content: payload.content,
        description: payload.description || '',
        author: payload.author || '',
        publishedAt: payload.publishedAt || '',
        canonicalUrl: payload.canonicalUrl || '',
        headings: Array.isArray(payload.headings) ? payload.headings.filter((heading) => typeof heading === 'string').slice(0, 12) : [],
      });
    } catch (error) {
      setWebError(error instanceof Error ? error.message : 'Could not read this web page.');
    } finally {
      setWebLoading(false);
    }
  };

  const addTask = () => {
    const title = taskTitle.trim();
    if (!title) return;
    const task: ManagedTask = {
      id: newId(),
      title: title.slice(0, 180),
      assigneeId: selectedAgent?.id || activeAgents[0]?.id || '',
      status: 'queued',
    };
    setTasks((current) => [task, ...current].slice(0, 100));
    setTaskTitle('');
  };

  const handleAddAgent = (template?: (typeof agentRoleTemplates)[number]) => {
    const requestedName = template?.name || agentDraftName.trim() ||
      (agentDraftTemplate === 'Custom' ? 'Agent' : agentDraftTemplate);
    const existingNames = new Set(agents.map((agent) => agent.name.toLowerCase()));
    let name = requestedName;
    let suffix = 2;
    while (existingNames.has(name.toLowerCase())) {
      name = `${requestedName} ${suffix}`;
      suffix += 1;
    }
    const team = agentDraftTeam.trim() || 'Studio';
    const role = template?.role || agentDraftRole.trim();
    if (!role || agents.length >= 8) return;
    const managerId = agents.some((agent) =>
      agent.id === agentDraftManagerId && agent.team === team && !agent.managerId,
    ) ? agentDraftManagerId : '';
    const id = newId();
    setAgents((current) => [...current, {
      id,
      name: name.slice(0, 48),
      role: role.slice(0, 400),
      enabled: true,
      team: team.slice(0, 48),
      managerId,
    }]);
    setSelectedAgentId(id);
    setAgentDraftName('');
    setAgentDraftManagerId('');
  };

  const handleRunQueuedTasks = async () => {
    if (queueBusy || isLoading) return;
    const queuedTasks = tasks.filter((task) => task.status === 'queued');
    if (queuedTasks.length === 0 || activeAgents.length === 0) return;

    setQueueBusy(true);
    for (const task of queuedTasks) {
      const assignee = activeAgents.find((agent) => agent.id === task.assigneeId) || activeAgents[0];
      setTasks((current) => current.map((item) => item.id === task.id
        ? { ...item, assigneeId: assignee.id, status: 'in-progress' }
        : item));
      const completed = await handleGenerate(
        `Work item for ${assignee.name} (${assignee.role}): ${task.title}`,
        assignee.id,
        true,
      );
      setTasks((current) => current.map((item) => item.id === task.id
        ? { ...item, status: completed ? 'done' : 'queued' }
        : item));
      if (!completed) break;
    }
    setQueueBusy(false);
  };

  const handleAgentChatSend = async () => {
    const text = agentChatDraft.trim();
    const chatAgent = selectedAgent;
    if (!text || !chatAgent || !chatAgent.enabled || agentChatBusy || activeAgents.length === 0) return;
    const userMessage: AgentChatMessage = { id: newId(), sender: 'You', role: 'user', text };
    const conversation = [...(agentChats[chatAgent.id] || []), userMessage].slice(-20);
    setAgentChats((current) => ({
      ...current,
      [chatAgent.id]: [...(current[chatAgent.id] || []), userMessage].slice(-40),
    }));
    setAgentChatDraft('');
    setAgentChatBusy(true);
    setAgentChatError('');
    try {
      const response = await fetch('/api/agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: conversation.map(({ role, text: content }) => ({ role, content })),
          files: filesRef.current.map(({ name, value }) => ({ path: name, content: value })),
          deepReasoning,
          focusAgentId: chatAgent.id,
          agents: activeAgents.map(({ id, name, role, managerId, team }) => ({ id, name, role, managerId, team })),
        }),
      });
      const payload = await response.json() as {
        message?: string;
        error?: string;
        files?: unknown;
        teamMessages?: Array<{ agentId?: string; name?: string; text?: string }>;
      };
      if (!response.ok || typeof payload.message !== 'string' || !payload.message.trim()) {
        throw new Error(payload.error || 'The agents could not complete this chat turn.');
      }
      if (payload.files !== undefined) {
        if (!Array.isArray(payload.files) || !payload.files.every(isGeneratedFilePayload)) {
          throw new Error('The team returned an invalid project file list.');
        }
        const generatedFiles: FileItem[] = payload.files.map((file) => ({
          id: `generated-${newId()}`,
          name: file.path,
          language: getLanguageFromPath(file.path),
          value: file.content,
        }));
        const byPath = new Map(filesRef.current.map((file) => [file.name.toLowerCase(), file]));
        generatedFiles.forEach((file) => {
          const existing = byPath.get(file.name.toLowerCase());
          byPath.set(file.name.toLowerCase(), { ...file, id: existing?.id ?? file.id });
        });
        const nextFiles = [...byPath.values()];
        if (nextFiles.reduce((total, file) => total + file.value.length, 0) > MAX_SOURCE_CONTEXT_CHARS) {
          throw new Error('The generated project exceeds the 24,000 character source limit.');
        }
        filesRef.current = nextFiles;
        setFiles(nextFiles);
        setAssets((current) => current.filter((asset) =>
          !generatedFiles.some((file) => file.name.toLowerCase() === asset.name.toLowerCase()),
        ));
        if (generatedFiles.length > 0) {
          const firstFile = nextFiles.find((file) => file.name.toLowerCase() === generatedFiles[0].name.toLowerCase());
          if (firstFile) setSelectedFile(firstFile.id);
          setWorkspaceOpen(true);
        }
      }
      const agentReplies = (payload.teamMessages || []).flatMap((agentMessage): AgentChatMessage[] =>
        typeof agentMessage.name === 'string' && typeof agentMessage.text === 'string' && agentMessage.text.trim()
          ? [{ id: newId(), sender: agentMessage.name, role: 'assistant', text: agentMessage.text.trim() }]
          : [],
      );
      const replies: AgentChatMessage[] = [
        ...agentReplies,
        { id: newId(), sender: chatAgent.name, role: 'assistant', text: payload.message.trim() },
      ];
      setAgentChats((current) => ({
        ...current,
        [chatAgent.id]: [...(current[chatAgent.id] || []), ...replies].slice(-40),
      }));
    } catch (error) {
      const message = error instanceof Error ? error.message : 'The agent chat request failed.';
      setAgentChatError(message);
      const reply: AgentChatMessage = {
        id: newId(),
        sender: 'AXOM',
        role: 'assistant',
        text: `This request could not be completed: ${message}`,
      };
      setAgentChats((current) => ({
        ...current,
        [chatAgent.id]: [...(current[chatAgent.id] || []), reply].slice(-40),
      }));
    } finally {
      setAgentChatBusy(false);
    }
  };

  const handleTrustGitHubAuthor = (input = githubAuthorInput) => {
    const author = input.trim().replace(/^@/, '');
    if (!/^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37})$/.test(author)) {
      setGithubError('Enter a valid GitHub username to add to your local trusted-author list.');
      return;
    }
    setGithubTrustedAuthors((current) => current.some((item) => item.toLowerCase() === author.toLowerCase())
      ? current
      : [...current, author].slice(-50));
    setGithubAuthorInput('');
    setGithubError('');
  };

  const handleCreateGitHubRepository = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const name = newRepositoryName.trim();
    if (!/^[A-Za-z0-9._-]{1,100}$/.test(name) || name === '.' || name === '..' || githubLoading) return;
    setGithubLoading(true);
    setGithubError('');
    try {
      const response = await fetch('/api/github/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, private: newRepositoryPrivate }),
      });
      const payload = await response.json() as { repository?: string; error?: string };
      if (!response.ok || !payload.repository) throw new Error(payload.error || 'Could not create this repository.');
      setNewRepositoryName('');
      await refreshGitHub();
      setGithubRepository(payload.repository);
      setGithubError(`Created ${payload.repository}${newRepositoryPrivate ? ' as a private repository' : ''}.`);
    } catch (error) {
      setGithubError(error instanceof Error ? error.message : 'Could not create this repository.');
    } finally {
      setGithubLoading(false);
    }
  };

  useEffect(() => {
    void refreshGitHub();
  }, []);

  const handleGenerate = async (requestOverride?: string, selectedAgentId?: string, fromQueue = false): Promise<boolean> => {
    const request = (requestOverride ?? prompt).trim();
    if (!request || isLoading || (queueBusy && !fromQueue)) return false;
    const requestAgents = selectedAgentId
      ? agents.filter((agent) => agent.id === selectedAgentId)
      : activeAgents;
    const requestText = request;
    const conversation = [
      ...messagesRef.current.filter((message) => message.role !== 'assistant' || message.id !== 1),
      { id: Date.now(), role: 'user' as const, text: requestText },
    ].slice(-20);

    setIsLoading(true);
    setRequestError('');
    messagesRef.current = conversation;
    setMessages(conversation);

    try {
      const response = await fetch('/api/agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: conversation.map(({ role, text }) => ({ role, content: text })),
          files: filesRef.current.map(({ name, value }) => ({ path: name, content: value })),
          deepReasoning,
          agents: requestAgents.map(({ id, name, role, managerId, team }) => ({ id, name, role, managerId, team })),
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

      const nextMessages: ChatMessage[] = [
        ...conversation,
        { id: Date.now() + 1, role: 'assistant', text: assistantText },
      ];
      messagesRef.current = nextMessages;
      setMessages(nextMessages);
      if (generatedFiles.length > 0) {
        const byPath = new Map(filesRef.current.map((file) => [file.name.toLowerCase(), file]));
        generatedFiles.forEach((file) => {
          const existing = byPath.get(file.name.toLowerCase());
          byPath.set(file.name.toLowerCase(), { ...file, id: existing?.id ?? file.id });
        });
        const nextFiles = [...byPath.values()];
        filesRef.current = nextFiles;
        setFiles(nextFiles);
        setAssets((current) => current.filter((asset) =>
          !generatedFiles.some((file) => file.name.toLowerCase() === asset.name.toLowerCase()),
        ));
        const existingSelectedFile = filesRef.current.find(
          (file) => file.name.toLowerCase() === generatedFiles[0].name.toLowerCase(),
        );
        setSelectedFile(existingSelectedFile?.id ?? generatedFiles[0].id);
      }
      setPrompt('');
      setModelStatus('loaded');
      setModelProgress(100);
      return true;
    } catch (error) {
      const message = error instanceof TypeError
        ? 'Could not connect to AXOM’s local model backend. Start the backend with `npm run dev:server`, then try again.'
        : error instanceof Error
          ? error.message
          : 'AXOM could not run the local model.';
      setRequestError(message);
      setModelStatus(error instanceof TypeError ? 'offline' : 'unavailable');
      return false;
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
      const selectedBytes = selected.reduce((total, file) => total + file.size, 0);
      if (selectedBytes > MAX_ARCHIVE_INPUT_BYTES) {
        throw new Error('Selected uploads must total 20 MB or less.');
      }
      const importedEntries: ArchiveEntry[] = [];
      let expandedBytes = 0;
      for (const file of selected) {
        const data = new Uint8Array(await file.arrayBuffer());
        const isZip = file.name.toLowerCase().endsWith('.zip') ||
          file.type === 'application/zip' ||
          (data[0] === 0x50 && data[1] === 0x4b);
        if (isZip) {
          const archiveEntries = await extractZipEntries(data, MAX_ARCHIVE_OUTPUT_BYTES - expandedBytes);
          expandedBytes += archiveEntries.reduce((total, entry) => total + entry.data.byteLength, 0);
          importedEntries.push(...archiveEntries);
        } else {
          if (data.byteLength > MAX_ARCHIVE_OUTPUT_BYTES - expandedBytes) {
            throw new Error('Expanded uploads exceed the 32 MB import limit.');
          }
          expandedBytes += data.byteLength;
          const path = file.name.replace(/\\/g, '/').split('/').pop() || 'imported-file';
          importedEntries.push({ path, data });
        }
        if (importedEntries.length > MAX_WORKSPACE_FILES) {
          throw new Error(`The project can contain up to ${MAX_WORKSPACE_FILES} files.`);
        }
      }

      const nextFiles = new Map(files.map((file) => [file.name.toLowerCase(), file]));
      const nextAssets = new Map(assets.map((asset) => [asset.name.toLowerCase(), asset]));
      let selectedSourceId = '';
      for (const entry of importedEntries) {
        if (!isSafeWorkspacePath(entry.path)) throw new Error(`Unsafe project path: ${entry.path}`);
        let value: string | null = null;
        if (entry.data.byteLength <= MAX_SOURCE_FILE_CHARS) {
          try {
            const decoded = new TextDecoder('utf-8', { fatal: true }).decode(entry.data);
            if (!decoded.includes('\0')) value = decoded;
          } catch {
            value = null;
          }
        }
        const key = entry.path.toLowerCase();
        if (value !== null) {
          const importedFile: FileItem = {
            id: `imported-${newId()}`,
            name: entry.path,
            language: getLanguageFromPath(entry.path),
            value,
          };
          nextFiles.set(key, importedFile);
          nextAssets.delete(key);
          selectedSourceId = importedFile.id;
        } else {
          nextFiles.delete(key);
          nextAssets.set(key, { id: `asset-${newId()}`, name: entry.path, data: entry.data });
        }
      }

      const nextFileList = [...nextFiles.values()];
      const nextAssetList = [...nextAssets.values()];
      if (nextFileList.length + nextAssetList.length > MAX_WORKSPACE_FILES) {
        throw new Error(`The project can contain up to ${MAX_WORKSPACE_FILES} files.`);
      }
      if (nextFileList.reduce((total, file) => total + file.value.length, 0) > MAX_SOURCE_CONTEXT_CHARS) {
        throw new Error('Editable source files exceed the 24,000 character model-context limit.');
      }
      if (nextAssetList.reduce((total, asset) => total + asset.data.byteLength, 0) > MAX_ASSET_BYTES) {
        throw new Error('Project attachments exceed the 16 MB asset limit.');
      }
      setFiles(nextFileList);
      setAssets(nextAssetList);
      if (selectedSourceId) setSelectedFile(selectedSourceId);
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
    const archiveEntries: Record<string, Uint8Array> = {};
    files.forEach((file) => { archiveEntries[file.name] = strToU8(file.value); });
    assets.forEach((asset) => { archiveEntries[asset.name] = asset.data; });
    const archive = zipSync(archiveEntries, { level: 1 });
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

  const downloadAsset = (asset: WorkspaceAsset) => {
    const buffer = new ArrayBuffer(asset.data.byteLength);
    new Uint8Array(buffer).set(asset.data);
    const url = URL.createObjectURL(new Blob([buffer]));
    const link = document.createElement('a');
    link.href = url;
    link.download = asset.name.split('/').pop() || 'project-asset';
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark">A</div>
          <div>
            <p className="muted">Team operations</p>
            <h2>AXON</h2>
          </div>
        </div>

        <button className="new-chat-button" onClick={startNewChat}>
          <span aria-hidden="true">＋</span>
          <span>New chat</span>
        </button>

        <div className="sidebar-panel conversation-sidebar">
          <div className="panel-label">Direct agent chats</div>
        </div>

        <section className="channel-navigation" aria-label="Individual agents">
          <div className="channel-navigation-heading">
            <span>Agents</span>
            <button onClick={() => setAgentsOpen(true)} title="Manage agents" aria-label="Manage agents">＋</button>
          </div>
          <div className="channel-list">
            {agents.map((agent) => (
              <button
                key={agent.id}
                className={`channel-link ${selectedAgent?.id === agent.id ? 'selected' : ''}`}
                onClick={() => {
                  setSelectedAgentId(agent.id);
                  setAgentChatError('');
                  setAgentsOpen(false);
                }}
              >
                <span className="channel-avatar agent">{agent.name.slice(0, 1).toUpperCase()}</span>
                <span className="channel-name">{agent.name}</span>
                <span className={`agent-presence ${agent.enabled ? 'enabled' : ''}`}>
                  {agent.enabled ? 'Ready' : 'Off'}
                </span>
              </button>
            ))}
          </div>
        </section>

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
            <span className="model-title">{selectedAgent ? `Chat with ${selectedAgent.name}` : 'No agents yet'}</span>
          </div>

          <label className="mobile-channel-control">
            <span className="visually-hidden">Choose an agent</span>
            <select
              value={selectedAgent?.id || ''}
              onChange={(event) => setSelectedAgentId(event.target.value)}
            >
              {agents.map((agent) => <option key={agent.id} value={agent.id}>{agent.name}</option>)}
            </select>
          </label>

          <div className="status-row">
            <label className="model-profile-control">
              <span className="visually-hidden">Local model profile</span>
              <select
                value={modelProfile}
                onChange={(event) => void handleModelProfileChange(event.target.value as ModelProfile)}
                disabled={modelStatus === 'switching' || modelStatus === 'loading'}
              >
                <option value="AXOM1.0">AXOM 1.0 · 0.5B</option>
                <option value="AXOM1.1">AXOM 1.1 · 1.5B</option>
              </select>
            </label>
            <span className={`status-pill ${modelStatus === 'loaded' ? 'online' : ''}`}>
              {(modelStatus === 'loading' || modelStatus === 'switching') && modelProgress !== null
                ? `Loading model ${modelProgress}%`
                : modelStatus === 'loading' || modelStatus === 'switching'
                  ? 'Loading model'
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
              {workspaceOpen ? 'Close build' : 'Build'}
            </button>
            <button
              className={`action-button workspace-toggle ${agentsOpen ? 'selected' : ''}`}
              onClick={() => setAgentsOpen((open) => !open)}
              aria-expanded={agentsOpen}
              aria-controls="agent-manager"
            >
              Agents · {activeAgents.length}
            </button>
          </div>
        </header>

        {agentsOpen && (
          <aside className="agent-drawer" id="agent-manager" aria-label="Agents and GitHub">
            <div className="agent-drawer-header">
              <div>
                <h2>Agents &amp; work</h2>
                <p>{activeAgents.length} active specialists · {tasks.filter((task) => task.status !== 'done').length} open tasks</p>
              </div>
              <button className="workspace-close" onClick={() => setAgentsOpen(false)} aria-label="Close agent manager">×</button>
            </div>
            <section className="agent-create-panel">
              <div className="agent-create-heading">
                <div><h3>Create an agent</h3><p>Give each agent a direct chat; names are assigned automatically.</p></div>
                <span>{agents.length}/8</span>
              </div>
              <div className="agent-quick-add" aria-label="Quick add specialist">
                <span>Quick add</span>
                {agentRoleTemplates.map((template) => (
                  <button
                    type="button"
                    key={template.name}
                    title={template.role}
                    disabled={agents.length >= 8}
                    onClick={() => handleAddAgent(template)}
                  >
                    + {template.name}
                  </button>
                ))}
              </div>
              <form className="agent-create-form" onSubmit={(event) => { event.preventDefault(); handleAddAgent(); }}>
                <label className="agent-field">
                  <span>Name</span>
                  <input
                    value={agentDraftName}
                    onChange={(event) => setAgentDraftName(event.target.value)}
                    placeholder="Optional — assigned automatically"
                    maxLength={48}
                  />
                </label>
                <label className="agent-field">
                  <span>Team</span>
                  <input
                    value={agentDraftTeam}
                    onChange={(event) => {
                      const team = event.target.value;
                      setAgentDraftTeam(team);
                      setAgentDraftManagerId(agents.find((agent) => agent.team === team && !agent.managerId)?.id || '');
                    }}
                    placeholder="Create or choose a team"
                    maxLength={48}
                    list="agent-team-options"
                  />
                  <datalist id="agent-team-options">{agentTeams.map((team) => <option value={team} key={team} />)}</datalist>
                </label>
                <label className="agent-field">
                  <span>Role template</span>
                  <select
                    value={agentDraftTemplate}
                    onChange={(event) => {
                      const template = agentRoleTemplates.find((item) => item.name === event.target.value);
                      setAgentDraftTemplate(template?.name || 'Custom');
                      if (template) setAgentDraftRole(template.role);
                    }}
                  >
                    {agentRoleTemplates.map((template) => <option key={template.name} value={template.name}>{template.name}</option>)}
                    <option value="Custom">Custom role</option>
                  </select>
                </label>
                <label className="agent-field">
                  <span>Role instructions</span>
                  <textarea
                    value={agentDraftRole}
                    onChange={(event) => {
                      setAgentDraftRole(event.target.value);
                      setAgentDraftTemplate('Custom');
                    }}
                    maxLength={400}
                    rows={3}
                    required
                  />
                </label>
                <label className="agent-field">
                  <span>Reports to</span>
                  <select value={agentDraftManagerId} onChange={(event) => setAgentDraftManagerId(event.target.value)}>
                    <option value="">Group lead</option>
                    {agents.filter((agent) => agent.team === agentDraftTeam.trim() && !agent.managerId)
                      .map((agent) => <option key={agent.id} value={agent.id}>{agent.name || 'Unnamed agent'}</option>)}
                  </select>
                </label>
                <button className="action-button agent-add" type="submit" disabled={agents.length >= 8 || !agentDraftRole.trim()}>
                  {agents.length >= 8 ? 'Agent limit reached' : 'Create agent'}
                </button>
              </form>
            </section>
            <div className="agent-list">
              {agents.map((agent) => (
                <details className="agent-item" key={agent.id}>
                  <summary className="agent-summary">
                    <span className="agent-summary-name">{agent.name || 'Unnamed agent'}</span>
                    <span className="agent-summary-meta">{agent.team || 'No team'} · {agent.managerId ? 'Specialist' : 'Group lead'}</span>
                  </summary>
                  <div className="agent-editor-fields">
                  <div className="agent-item-topline">
                    <label className="agent-enabled">
                      <input
                        type="checkbox"
                        checked={agent.enabled}
                        onChange={(event) => setAgents((current) => current.map((item) =>
                          item.id === agent.id ? { ...item, enabled: event.target.checked } : item))}
                      />
                      <span>Assigned</span>
                    </label>
                    <span className="agent-rank">
                      {agent.managerId ? 'Specialist' : `Group lead · ${agents.filter((item) => item.managerId === agent.id).length}`}
                    </span>
                    <button
                      className="agent-remove"
                      onClick={() => setAgents((current) => current.filter((item) => item.id !== agent.id))}
                      aria-label={`Remove ${agent.name}`}
                    >
                      Remove
                    </button>
                  </div>
                  <label className="agent-field">
                    <span>Name</span>
                    <input
                      value={agent.name}
                      maxLength={48}
                      onChange={(event) => setAgents((current) => current.map((item) =>
                        item.id === agent.id ? { ...item, name: event.target.value } : item))}
                    />
                  </label>
                  <label className="agent-field">
                    <span>Team</span>
                    <input
                      value={agent.team}
                      maxLength={48}
                      onChange={(event) => setAgents((current) => current.map((item) =>
                        item.id === agent.id ? { ...item, team: event.target.value, managerId: '' } : item))}
                    />
                  </label>
                  <label className="agent-field">
                    <span>Reports to</span>
                    <select
                      value={agent.managerId}
                      onChange={(event) => setAgents((current) => current.map((item) =>
                        item.id === agent.id ? { ...item, managerId: event.target.value } : item))}
                    >
                      <option value="">Group lead</option>
                      {agents.filter((candidate) => candidate.id !== agent.id && candidate.team === agent.team && !candidate.managerId)
                        .map((candidate) => <option key={candidate.id} value={candidate.id}>{candidate.name}</option>)}
                    </select>
                  </label>
                  <label className="agent-field">
                    <span>Role</span>
                    <textarea
                      value={agent.role}
                      maxLength={400}
                      rows={3}
                      placeholder="Describe this agent's specialty and how it should approach work."
                      onChange={(event) => setAgents((current) => current.map((item) =>
                        item.id === agent.id ? { ...item, role: event.target.value } : item))}
                    />
                  </label>
                  </div>
                </details>
              ))}
            </div>

            <section className="tasks-section">
              <div className="github-heading">
                <h3>Work queue</h3>
                <div className="task-queue-actions">
                  <span className="task-count">{tasks.filter((task) => task.status === 'queued').length} queued</span>
                  <button
                    className="task-run-all"
                    type="button"
                    onClick={() => void handleRunQueuedTasks()}
                    disabled={queueBusy || isLoading || !tasks.some((task) => task.status === 'queued') || activeAgents.length === 0}
                  >
                    {queueBusy ? 'Working…' : 'Run queued'}
                  </button>
                </div>
              </div>
              <form className="task-create" onSubmit={(event) => { event.preventDefault(); addTask(); }}>
                <input
                  aria-label="New task"
                  value={taskTitle}
                  onChange={(event) => setTaskTitle(event.target.value)}
                  placeholder="Add a task for the team"
                  maxLength={180}
                />
                <button type="submit" disabled={!taskTitle.trim()} aria-label="Add task">+</button>
              </form>
              {tasks.length === 0 ? (
                <p className="empty-tasks">No work assigned yet.</p>
              ) : (
                <div className="task-list">
                  {tasks.map((task) => {
                    const assignedAgent = agents.find((agent) => agent.id === task.assigneeId);
                    return (
                      <article className={`task-item ${task.status}`} key={task.id}>
                        <div className="task-item-heading">
                          <span className="task-status-dot" />
                          <p>{task.title}</p>
                          <button
                            className="agent-remove"
                            onClick={() => setTasks((current) => current.filter((item) => item.id !== task.id))}
                            aria-label={`Remove task ${task.title}`}
                          >×</button>
                        </div>
                        <div className="task-controls">
                          <label className="visually-hidden" htmlFor={`assignee-${task.id}`}>Assigned teammate</label>
                          <select
                            id={`assignee-${task.id}`}
                            value={task.assigneeId}
                            disabled={queueBusy}
                            onChange={(event) => setTasks((current) => current.map((item) =>
                              item.id === task.id ? { ...item, assigneeId: event.target.value } : item))}
                          >
                            <option value="">Unassigned</option>
                            {agents.map((agent) => <option key={agent.id} value={agent.id}>{agent.name}</option>)}
                          </select>
                          <select
                            aria-label={`Status for ${task.title}`}
                            value={task.status}
                            disabled={queueBusy}
                            onChange={(event) => setTasks((current) => current.map((item) =>
                              item.id === task.id ? { ...item, status: event.target.value as ManagedTask['status'] } : item))}
                          >
                            <option value="queued">Queued</option>
                            <option value="in-progress">In progress</option>
                            <option value="done">Done</option>
                          </select>
                          {task.status !== 'done' && task.assigneeId && (
                            <button
                              className="task-run"
                              disabled={isLoading || queueBusy}
                              onClick={async () => {
                                setTasks((current) => current.map((item) =>
                                  item.id === task.id ? { ...item, status: 'in-progress' } : item));
                                const agent = agents.find((item) => item.id === task.assigneeId);
                                const completed = await handleGenerate(
                                  `Work item for ${agent?.name || 'the team'} (${agent?.role || 'general support'}): ${task.title}`,
                                  task.assigneeId,
                                );
                                setTasks((current) => current.map((item) => item.id === task.id
                                  ? { ...item, status: completed ? 'done' : 'queued' }
                                  : item));
                              }}
                            >
                              Run
                            </button>
                          )}
                        </div>
                      </article>
                    );
                  })}
                </div>
              )}
            </section>

            <section className="web-section">
              <div className="github-heading"><h3>Web research</h3></div>
              <form className="web-scrape-form" onSubmit={(event) => { event.preventDefault(); void handleScrapeWebPage(); }}>
                <input
                  type="url"
                  aria-label="Public web page URL"
                  value={webUrl}
                  onChange={(event) => setWebUrl(event.target.value)}
                  placeholder="https://example.com/article"
                  required
                />
                <button type="submit" disabled={webLoading}>{webLoading ? 'Reading…' : 'Read page'}</button>
              </form>
              {webError && <p className="github-message" role="alert">{webError}</p>}
              {scrapedPage && (
                <article className="scraped-result">
                  <a href={scrapedPage.url} target="_blank" rel="noreferrer">{scrapedPage.title}</a>
                  {scrapedPage.description && <p className="scraped-description">{scrapedPage.description}</p>}
                  {(scrapedPage.author || scrapedPage.publishedAt) && (
                    <small>{[scrapedPage.author, scrapedPage.publishedAt].filter(Boolean).join(' · ')}</small>
                  )}
                  {scrapedPage.headings.length > 0 && (
                    <ul>{scrapedPage.headings.slice(0, 5).map((heading, index) => <li key={`${heading}-${index}`}>{heading}</li>)}</ul>
                  )}
                  <p>{scrapedPage.content.slice(0, 420)}{scrapedPage.content.length > 420 ? '…' : ''}</p>
                  <button
                    className="agent-refresh"
                    onClick={() => setPrompt(`Use this public source as reference for my team:\n${scrapedPage.title} (${scrapedPage.canonicalUrl || scrapedPage.url})${scrapedPage.author ? `\nAuthor: ${scrapedPage.author}` : ''}${scrapedPage.publishedAt ? `\nPublished: ${scrapedPage.publishedAt}` : ''}${scrapedPage.description ? `\nSummary: ${scrapedPage.description}` : ''}\n\n${scrapedPage.content.slice(0, 3000)}`)}
                  >
                    Add research to chat
                  </button>
                </article>
              )}
            </section>

            <section className="github-section">
              <div className="github-heading">
                <h3>GitHub</h3>
                <button className="agent-refresh" onClick={() => void refreshGitHub()} disabled={githubLoading}>
                  {githubLoading ? 'Working…' : 'Refresh'}
                </button>
              </div>
              {githubAccount ? (
                <a className="github-account" href={githubAccount.url} target="_blank" rel="noreferrer">
                  <span className="github-dot connected" />
                  @{githubAccount.login}
                </a>
              ) : (
                <p className="github-account"><span className="github-dot" />Not connected</p>
              )}
              {githubRepositories.length > 0 && (
                <>
                  <label className="agent-field">
                    <span>Repository</span>
                    <select value={githubRepository} onChange={(event) => setGithubRepository(event.target.value)}>
                      {githubRepositories.map((repository) => (
                        <option key={repository.nameWithOwner} value={repository.nameWithOwner}>
                          {repository.nameWithOwner}{repository.isPrivate ? ' · private' : ''}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button className="action-button github-import" onClick={() => void handleImportGitHubRepository()} disabled={githubLoading || !githubRepository}>
                    Import repository files
                  </button>
                  <button className="action-button github-publish" onClick={() => void handlePublishGitHubRepository()} disabled={githubLoading || !githubRepository || files.length === 0}>
                    Publish workspace to repository
                  </button>
                </>
              )}
              {githubAccount && (
                <form className="github-create-form" onSubmit={(event) => void handleCreateGitHubRepository(event)}>
                  <div className="github-heading"><h3>New repository</h3><span className="email-state">Private by default</span></div>
                  <label className="agent-field">
                    <span>Repository name</span>
                    <input
                      value={newRepositoryName}
                      onChange={(event) => setNewRepositoryName(event.target.value)}
                      placeholder="my-new-project"
                      maxLength={100}
                      required
                    />
                  </label>
                  <label className="github-visibility">
                    <input type="checkbox" checked={newRepositoryPrivate} onChange={(event) => setNewRepositoryPrivate(event.target.checked)} />
                    <span>Private repository</span>
                  </label>
                  <button className="action-button github-create" type="submit" disabled={githubLoading || !newRepositoryName.trim()}>
                    {githubLoading ? 'Working…' : 'Create repository'}
                  </button>
                </form>
              )}
              <section className="github-trust-section">
                <div className="github-heading">
                  <h3>Trusted authors</h3>
                  {selectedGitHubOwner && (
                    <button className="agent-refresh" type="button" disabled={selectedGitHubOwnerTrusted}
                      onClick={() => handleTrustGitHubAuthor(selectedGitHubOwner)}>
                      {selectedGitHubOwnerTrusted ? 'Trusted' : `Trust @${selectedGitHubOwner}`}
                    </button>
                  )}
                </div>
                <form className="github-trust-form" onSubmit={(event) => { event.preventDefault(); handleTrustGitHubAuthor(); }}>
                  <input
                    value={githubAuthorInput}
                    onChange={(event) => setGithubAuthorInput(event.target.value)}
                    placeholder="GitHub username"
                    aria-label="GitHub author to trust"
                    maxLength={39}
                  />
                  <button type="submit" disabled={!githubAuthorInput.trim()}>Add</button>
                </form>
                {githubTrustedAuthors.length > 0 ? (
                  <ul className="trusted-author-list">
                    {githubTrustedAuthors.map((author) => (
                      <li key={author}>
                        <a href={`https://github.com/${encodeURIComponent(author)}`} target="_blank" rel="noreferrer">@{author}</a>
                        <button type="button" className="agent-remove" onClick={() => setGithubTrustedAuthors((current) => current.filter((item) => item !== author))} aria-label={`Remove ${author} from trusted authors`}>Remove</button>
                      </li>
                    ))}
                  </ul>
                ) : <p className="empty-tasks">No authors marked trusted.</p>}
                <p className="github-note">Trust labels are stored in this browser only. Review imported code before running or publishing it.</p>
              </section>
              {githubError && <p className="github-message" role="status">{githubError}</p>}
              <p className="github-note">Sign in through the local GitHub CLI with <code>gh auth login</code>. AXOM does not receive your GitHub password or token.</p>
            </section>

            <section className="email-section">
              <div className="github-heading"><h3>Email</h3><span className="email-state">Local</span></div>
              <label className="agent-field">
                <span>Team contact address</span>
                <input
                  type="email"
                  value={emailAddress}
                  onChange={(event) => setEmailAddress(event.target.value)}
                  placeholder="team@company.com"
                />
              </label>
              {emailAddress && (
                <a className="action-button email-compose" href={`mailto:${emailAddress}?subject=${encodeURIComponent('AXON team update')}`}>
                  Compose email to team
                </a>
              )}
              <p className="github-note">Opens your email app with a draft. No mailbox access or email is sent by AXON.</p>
            </section>
          </aside>
        )}

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
                  <label className="workspace-file-control">
                    <span className="visually-hidden">Choose a source file</span>
                    <select
                      value={currentFile.id}
                      onChange={(event) => setSelectedFile(event.target.value)}
                    >
                      {files.map((file) => <option value={file.id} key={file.id}>{file.name}</option>)}
                    </select>
                  </label>
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
                  Add files or ZIP
                  <input type="file" multiple accept="*/*,.zip" onChange={handleImportFiles} />
                </label>
                <button className="action-button secondary" onClick={() => downloadFile(currentFile)}>
                  Download
                </button>
                <button className="action-button secondary" onClick={downloadProject}>
                  Export ZIP
                </button>
              </div>
              {assets.length > 0 && (
                <section className="workspace-assets" aria-label="Project attachments">
                  <div className="workspace-assets-heading">Attachments · included in ZIP</div>
                  {assets.map((asset) => (
                    <div className="workspace-asset" key={asset.id}>
                      <span title={asset.name}>{asset.name}</span>
                      <button type="button" onClick={() => downloadAsset(asset)} aria-label={`Download ${asset.name}`}>Download</button>
                      <button
                        type="button"
                        onClick={() => setAssets((current) => current.filter((item) => item.id !== asset.id))}
                        aria-label={`Remove ${asset.name}`}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </section>
              )}
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
            <section className="team-channel-view">
              <header className="team-channel-header">
                <div>
                  <h1>{selectedAgent?.name || 'Create an agent to begin'}</h1>
                  <p>
                    {selectedAgent
                      ? `${selectedAgent.role} · Other enabled agents will collaborate in this chat`
                      : 'Add an agent to start a direct chat and delegate work.'}
                  </p>
                </div>
                {selectedAgent && (
                  <button className="channel-details-button" onClick={() => setAgentsOpen(true)} aria-label="Open agent settings">
                    <span className="channel-avatar agent">{selectedAgent.name.slice(0, 1).toUpperCase()}</span>
                    <span>Agent settings</span>
                  </button>
                )}
              </header>
              <div className="team-channel-transcript" aria-live="polite">
                {selectedAgentMessages.length === 0 ? (
                  <div className="channel-empty-state">
                    <span className="channel-avatar agent">{selectedAgent?.name.slice(0, 1).toUpperCase() || 'A'}</span>
                    <h2>{selectedAgent ? `Start a chat with ${selectedAgent.name}` : 'Create your first agent'}</h2>
                    <p>
                      {selectedAgent
                        ? 'Describe a task or ask a question. This agent will lead while the other enabled agents discuss their findings here. Paste a public page URL to have them read it.'
                        : 'Use the Agents button to create a specialist. Each agent gets its own chat.'}
                    </p>
                  </div>
                ) : selectedAgentMessages.map((message) => (
                  <article className={`channel-message ${message.role}`} key={message.id}>
                    <div className={`channel-avatar ${message.role === 'user' ? 'human' : 'agent'}`}>
                      {message.sender.slice(0, 1).toUpperCase()}
                    </div>
                    <div className="channel-message-body">
                      <div className="channel-message-meta">
                        <strong>{message.sender}</strong>
                        <time>{message.role === 'assistant' ? 'agent response' : 'you'}</time>
                      </div>
                      <p>{message.text}</p>
                    </div>
                  </article>
                ))}
                {agentChatBusy && (
                  <article className="channel-message assistant">
                    <div className="channel-avatar agent">{selectedAgent?.name.slice(0, 1).toUpperCase() || 'A'}</div>
                    <div className="channel-message-body">
                      <strong>{selectedAgent?.name || 'Agents'} and collaborators</strong>
                      <p className="chat-thinking">Working together…</p>
                    </div>
                  </article>
                )}
                {agentChatError && <p className="channel-error" role="alert">{agentChatError}</p>}
                <div ref={transcriptEndRef} />
              </div>
              <form className="team-channel-composer" onSubmit={(event) => { event.preventDefault(); void handleAgentChatSend(); }}>
                <input
                  value={agentChatDraft}
                  onChange={(event) => setAgentChatDraft(event.target.value)}
                  placeholder={selectedAgent ? `Message ${selectedAgent.name} or start a task…` : 'Create an agent to begin'}
                  maxLength={16000}
                  aria-label={selectedAgent ? `Message ${selectedAgent.name}` : 'Message an agent'}
                  disabled={!selectedAgent?.enabled || activeAgents.length === 0}
                />
                <button
                  type="submit"
                  disabled={!agentChatDraft.trim() || agentChatBusy || !selectedAgent?.enabled || activeAgents.length === 0}
                  aria-label="Send message to agent"
                >↑</button>
              </form>
            </section>
          </div>
        </section>

      </main>
    </div>
  );
}

export default App;
