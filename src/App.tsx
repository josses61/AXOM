import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from 'react';
import { strToU8, zipSync } from 'fflate';

type Role = 'user' | 'assistant';
type ModeKey = 'build' | 'project' | 'improve' | 'fix' | 'refactor' | 'explain' | 'design';
type ModelProfile = 'AXOM1.0' | 'AXOM1.1';
type ModelStatus = 'checking' | 'not-loaded' | 'loading' | 'switching' | 'loaded' | 'unavailable' | 'offline';

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  sources?: LearningSource[];
  learningWarning?: string;
};

type LearningSource = {
  name: string;
  url: string;
  language: string;
};

type FileItem = {
  id: string;
  name: string;
  language: string;
  value: string;
};

type GeneratedProjectFile = {
  path: string;
  content: string;
};

type ModelHealth = {
  status: Exclude<ModelStatus, 'checking' | 'offline'>;
  profile: ModelProfile;
  activeProfile: ModelProfile;
  model: string;
  progress: number | null;
  error: string | null;
};

const modes: { id: ModeKey; label: string }[] = [
  { id: 'build', label: 'Build' },
  { id: 'project', label: 'Create files' },
  { id: 'improve', label: 'Improve files' },
  { id: 'fix', label: 'Fix' },
  { id: 'refactor', label: 'Refactor' },
  { id: 'explain', label: 'Explain' },
  { id: 'design', label: 'Design' },
];

const suggestions = [
  'Help me plan and build a polished SaaS dashboard',
  'Review this code for bugs and suggest a focused fix',
  'Explain how to structure a full-stack React application',
  'Design an accessible sign-in flow with validation',
];

const newId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

function isModelProfile(value: string): value is ModelProfile {
  return value === 'AXOM1.0' || value === 'AXOM1.1';
}

function isGeneratedProjectFile(file: unknown): file is GeneratedProjectFile {
  return typeof file === 'object' &&
    file !== null &&
    typeof (file as { path?: unknown }).path === 'string' &&
    typeof (file as { content?: unknown }).content === 'string';
}

function isLearningSource(source: unknown): source is LearningSource {
  if (typeof source !== 'object' || source === null) return false;
  const candidate = source as { name?: unknown; url?: unknown; language?: unknown };
  return typeof candidate.name === 'string' &&
    typeof candidate.url === 'string' &&
    candidate.url.startsWith('https://github.com/') &&
    typeof candidate.language === 'string';
}

function CopyCodeButton({ code }: { code: string }) {
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle');

  const copyCode = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopyState('copied');
    } catch (error) {
      console.error('Could not copy code to the clipboard:', error);
      setCopyState('failed');
    }
    window.setTimeout(() => setCopyState('idle'), 1800);
  };

  return (
    <button className="code-copy-button" onClick={copyCode} type="button">
      {copyState === 'copied' ? 'Copied' : copyState === 'failed' ? 'Copy failed' : 'Copy code'}
    </button>
  );
}

function AssistantContent({ content }: { content: string }) {
  return (
    <div className="assistant-markdown">
      {content.split(/(```[\s\S]*?```)/g).map((part, index) => {
        const block = part.match(/^```([^\r\n]*)\r?\n([\s\S]*?)\r?\n```$/);
        if (block) {
          const language = block[1].trim() || 'code';
          const code = block[2];
          return (
            <pre className="message-code-block" key={index}>
              <div className="message-code-toolbar">
                <span>{language}</span>
                <CopyCodeButton code={code} />
              </div>
              <code>{code}</code>
            </pre>
          );
        }
        if (!part.trim()) return null;

        return (
          <p key={index}>
            {part.split(/(`[^`\n]+`)/g).map((segment, segmentIndex) => {
              if (segment.startsWith('`') && segment.endsWith('`')) {
                return <code className="inline-code" key={segmentIndex}>{segment.slice(1, -1)}</code>;
              }
              return segment;
            })}
          </p>
        );
      })}
    </div>
  );
}

async function fetchModelHealth(): Promise<ModelHealth> {
  const response = await fetch('/api/health');
  if (!response.ok) throw new Error('Backend health check failed');
  return response.json();
}

async function persistLearningExample(prompt: string, answer: string): Promise<number> {
  const response = await fetch('/api/learning', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, answer }),
  });
  const result: { error?: unknown; count?: unknown } = await response.json();
  if (!response.ok) {
    throw new Error(typeof result.error === 'string' ? result.error : 'AXOM could not save this example.');
  }
  if (typeof result.count !== 'number') {
    throw new Error('AXOM saved the example but returned an invalid learning count.');
  }
  return result.count;
}

function App() {
  const [modelProfile, setModelProfile] = useState<ModelProfile>(() => {
    const stored = window.localStorage.getItem('axom-model-profile');
    return stored === 'AXOM1.1' ? 'AXOM1.1' : 'AXOM1.0';
  });
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [prompt, setPrompt] = useState('');
  const [activeMode, setActiveMode] = useState<ModeKey>('build');
  const [files, setFiles] = useState<FileItem[]>([]);
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [modelStatus, setModelStatus] = useState<ModelStatus>('checking');
  const [modelName, setModelName] = useState('Local SLM');
  const [modelProgress, setModelProgress] = useState<number | null>(null);
  const [modelError, setModelError] = useState('');
  const [learningCount, setLearningCount] = useState(0);
  const [autoLearnEnabled, setAutoLearnEnabled] = useState(
    () => window.localStorage.getItem('axom-auto-learn') !== 'false',
  );
  const [autoLearnedMessageIds, setAutoLearnedMessageIds] = useState<string[]>([]);
  const [teachingMessageId, setTeachingMessageId] = useState<string | null>(null);
  const [teachingAnswer, setTeachingAnswer] = useState('');
  const [savedLearningIds, setSavedLearningIds] = useState<string[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;
    fetch('/api/model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profile: modelProfile }),
    })
      .then((response) => {
        if (!response.ok && response.status !== 202) {
          throw new Error('Model selection request failed');
        }
        return fetchModelHealth();
      })
      .then((health) => {
        if (!active) return;
        setModelProfile(health.profile);
        setModelStatus(health.status);
        setModelName(health.model);
        setModelProgress(health.progress);
        setModelError(health.error ?? '');
      })
      .catch(() => {
        if (active) setModelStatus('offline');
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
          setModelName(health.model);
          setModelProgress(health.progress);
          setModelError(health.error ?? '');
        })
        .catch(() => {
          if (active) setModelStatus('offline');
        });
    }, 1200);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [modelStatus]);

  useEffect(() => {
    let active = true;
    fetch('/api/learning')
      .then(async (response) => {
        if (!response.ok) throw new Error('Could not load saved examples.');
        return response.json() as Promise<{ count: number }>;
      })
      .then((data) => {
        if (active) setLearningCount(data.count);
      })
      .catch((error) => {
        console.error('Could not load AXOM local learning examples:', error);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages, isLoading]);

  const selectedFile = useMemo(
    () => files.find((file) => file.id === selectedFileId) ?? null,
    [files, selectedFileId],
  );

  const statusLabel = modelStatus === 'loading' && modelProgress !== null
    ? `Loading model ${modelProgress}%`
    : modelStatus === 'loading'
      ? 'Loading model'
      : modelStatus === 'switching'
        ? `Switching to ${modelProfile.replace('AXOM', 'AXOM ')}`
      : modelStatus === 'not-loaded'
        ? 'Model ready to load'
        : modelStatus === 'unavailable'
          ? 'Model unavailable'
          : modelStatus === 'offline'
            ? 'AXOM backend offline'
            : modelStatus === 'checking'
              ? 'Connecting'
              : 'Local model ready';

  const selectModelProfile = async (profile: ModelProfile) => {
    setModelProfile(profile);
    window.localStorage.setItem('axom-model-profile', profile);
    setModelStatus('switching');
    setModelProgress(null);
    setModelError('');
    try {
      const response = await fetch('/api/model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile }),
      });
      if (!response.ok && response.status !== 202) {
        const result: { error?: unknown } = await response.json();
        throw new Error(typeof result.error === 'string' ? result.error : 'AXOM could not switch models.');
      }
    } catch (error) {
      setModelError(error instanceof Error ? error.message : 'AXOM could not switch models.');
      setModelStatus('unavailable');
    }
  };

  const saveLearningExample = async (message: ChatMessage, messageIndex: number) => {
    const question = messages.slice(0, messageIndex).reverse().find((item) => item.role === 'user');
    const answer = teachingAnswer.trim();
    if (!question || !answer) return;
    try {
      setLearningCount(await persistLearningExample(question.content, answer));
      setSavedLearningIds((current) => [...current, message.id]);
      setTeachingMessageId(null);
      setTeachingAnswer('');
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'AXOM could not save this example.');
    }
  };

  const clearLearningExamples = async () => {
    if (!window.confirm('Delete all locally saved AXOM learning examples?')) return;
    try {
      const response = await fetch('/api/learning', { method: 'DELETE' });
      const result: { error?: unknown; count?: unknown } = await response.json();
      if (!response.ok) {
        throw new Error(typeof result.error === 'string' ? result.error : 'AXOM could not clear its saved examples.');
      }
      setLearningCount(typeof result.count === 'number' ? result.count : 0);
      setSavedLearningIds([]);
      setAutoLearnedMessageIds([]);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'AXOM could not clear its saved examples.');
    }
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const content = prompt.trim();
    if (!content || isLoading) return;

    const lastMessage = messages[messages.length - 1];
    const conversation = lastMessage?.role === 'user' && lastMessage.content === content
      ? messages
      : [...messages, { id: newId(), role: 'user' as const, content }];
    setMessages(conversation);
    setIsLoading(true);
    setErrorMessage('');
    setModelStatus('loading');
    setModelProgress(0);

    try {
      const creatingProject = activeMode === 'project' || activeMode === 'improve';
      const improvingProject = activeMode === 'improve';
      const response = await fetch(
        creatingProject
          ? improvingProject ? '/api/project/update' : '/api/project'
          : '/api/agent',
        {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(creatingProject
          ? {
            description: content,
            files: files.map(({ name, value }) => ({ path: name, content: value })),
          }
          : {
            messages: conversation.map(({ role, content: messageContent }) => ({
              role,
              content: messageContent,
            })),
            mode: activeMode,
            files: files.map(({ name, value }) => ({ path: name, content: value })),
          }),
        },
      );
      let result: {
        error?: unknown;
        message?: unknown;
        files?: unknown;
        sources?: unknown;
        learningWarning?: unknown;
      };
      try {
        const payload: unknown = await response.json();
        if (typeof payload !== 'object' || payload === null) {
          throw new Error('Invalid response');
        }
        result = payload;
      } catch {
        throw new Error('AXOM received an invalid response from its local backend. Start the backend with `npm run dev:server`, then try again.');
      }

      if (!response.ok) {
        throw new Error(typeof result.error === 'string' ? result.error : 'AXOM could not complete the request.');
      }
      if (typeof result.message !== 'string' || !result.message.trim()) {
        throw new Error('The local model returned an empty response. Please try again.');
      }
      const assistantMessage = result.message;
      const sources = Array.isArray(result.sources)
        ? result.sources.filter(isLearningSource)
        : [];
      const learningWarning = typeof result.learningWarning === 'string'
        ? result.learningWarning
        : undefined;

      if (creatingProject) {
        if (
          !Array.isArray(result.files) ||
          result.files.length === 0 ||
          !result.files.every(isGeneratedProjectFile)
        ) {
          throw new Error('AXOM returned an invalid project file list. Please try again with a more specific request.');
        }
        const generatedFiles = result.files.map((projectFile) => {
          return {
            id: newId(),
            name: projectFile.path,
            language: projectFile.path.split('.').pop() ?? 'text',
            value: projectFile.content,
          };
        });
        setFiles((current) => {
          const updated = new Map(current.map((file) => [file.name, file]));
          generatedFiles.forEach((file) => updated.set(file.name, file));
          return [...updated.values()];
        });
        setSelectedFileId(generatedFiles[generatedFiles.length - 1].id);
        setWorkspaceOpen(true);
      }

      const assistantReply = {
        id: newId(),
        role: 'assistant' as const,
        content: assistantMessage,
        sources,
        learningWarning,
      };
      setMessages((current) => [...conversation, assistantReply]);
      setPrompt('');
      if (!creatingProject && autoLearnEnabled) {
        try {
          setLearningCount(await persistLearningExample(content, assistantMessage));
          setAutoLearnedMessageIds((current) => [...current, assistantReply.id]);
        } catch (error) {
          const detail = error instanceof Error ? error.message : 'AXOM could not save this example.';
          setErrorMessage(`AXOM replied, but could not save this example locally: ${detail}`);
        }
      }
      setModelStatus('loaded');
      setModelProgress(100);
      requestAnimationFrame(() => textareaRef.current?.focus());
    } catch (error) {
      const message = error instanceof TypeError
        ? 'Could not connect to AXOM’s local model backend. Start the backend with `npm run dev:server`, then try again.'
        : error instanceof Error
          ? error.message
          : 'AXOM could not run the local model.';
      setErrorMessage(message);
      setModelStatus(error instanceof TypeError || message.toLowerCase().includes('backend') ? 'offline' : 'unavailable');
    } finally {
      setIsLoading(false);
    }
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setPrompt('');
    setErrorMessage('');
    setActiveMode('build');
    textareaRef.current?.focus();
  };

  const addFile = () => {
    const file: FileItem = {
      id: newId(),
      name: `untitled-${files.length + 1}.tsx`,
      language: 'tsx',
      value: 'export default function Component() {\n  return <main />;\n}\n',
    };
    setFiles((current) => [...current, file]);
    setSelectedFileId(file.id);
    setWorkspaceOpen(true);
  };

  const updateFile = (value: string) => {
    if (!selectedFile) return;
    setFiles((current) => current.map((file) =>
      file.id === selectedFile.id ? { ...file, value } : file,
    ));
  };

  const downloadFile = () => {
    if (!selectedFile) return;
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([selectedFile.value], { type: 'text/plain;charset=utf-8' }));
    link.download = selectedFile.name.split('/').pop() || 'axom-file.txt';
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  };

  const downloadProjectZip = () => {
    if (files.length === 0) return;
    const archive = zipSync(Object.fromEntries(
      files.map(({ name, value }) => [name, strToU8(value)]),
    ));
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([archive], { type: 'application/zip' }));
    link.download = 'axom-project.zip';
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  };

  return (
    <div className="axom-app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon" aria-hidden="true">A</div>
          <span>AXOM</span>
        </div>

        <button className="new-chat-button" onClick={startNewChat}>
          <span aria-hidden="true">＋</span>
          New chat
        </button>

        <div className="sidebar-section">
          <p className="sidebar-heading">Workspace</p>
          <button
            className={`sidebar-link ${workspaceOpen ? 'selected' : ''}`}
            onClick={() => setWorkspaceOpen((open) => !open)}
          >
            <span className="sidebar-icon" aria-hidden="true">{'{ }'}</span>
            Files &amp; code
          </button>
          <div className="learning-summary">
            <div className="learning-count">
              <span>{learningCount} local example{learningCount === 1 ? '' : 's'}</span>
              {learningCount > 0 && (
                <button type="button" onClick={() => void clearLearningExamples()}>
                  Clear
                </button>
              )}
            </div>
            <label className="learning-setting">
              <input
                type="checkbox"
                checked={autoLearnEnabled}
                onChange={(event) => {
                  const enabled = event.target.checked;
                  setAutoLearnEnabled(enabled);
                  window.localStorage.setItem('axom-auto-learn', String(enabled));
                }}
              />
              <span>Learn from chats automatically</span>
            </label>
            <small>Examples stay on this device.</small>
          </div>
        </div>

        <div className="sidebar-section recent-section">
          <p className="sidebar-heading">Recent</p>
          {messages.length > 0 ? (
            <button className="recent-chat" onClick={() => transcriptRef.current?.scrollTo({ top: 0, behavior: 'smooth' })}>
              {messages.find((message) => message.role === 'user')?.content ?? 'New conversation'}
            </button>
          ) : (
            <p className="empty-recent">Your conversations will appear here.</p>
          )}
        </div>

        <div className="sidebar-bottom">
          <div className={`connection-card ${modelStatus === 'loaded' ? 'connected' : ''}`}>
            <span className="connection-dot" />
            <div>
              <strong>{statusLabel}</strong>
              <small>Local model · {modelName}</small>
            </div>
          </div>
          {modelError && <p className="model-warning">{modelError}</p>}
          <div className="sidebar-user">
            <span className="user-avatar">A</span>
            <span>Workspace</span>
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="model-title">
            <span>AXOM</span>
            <span className="model-chevron" aria-hidden="true">⌄</span>
            <label className="model-select-label">
              <span className="visually-hidden">Select AXOM model profile</span>
              <select
                className="model-select"
                value={modelProfile}
                onChange={(event) => {
                  if (isModelProfile(event.target.value)) {
                    void selectModelProfile(event.target.value);
                  }
                }}
                disabled={modelStatus === 'switching'}
              >
                <option value="AXOM1.0">AXOM 1.0 · Faster</option>
                <option value="AXOM1.1">AXOM 1.1 · More capable</option>
              </select>
            </label>
          </div>
          <div className="topbar-right">
            <button className="top-new-chat" onClick={startNewChat}>＋ New chat</button>
            <span className={`connection-label ${modelStatus === 'loaded' ? 'connected' : ''}`}>
              <span className="connection-dot" />
              {statusLabel}
            </span>
            <button className="workspace-toggle" onClick={() => setWorkspaceOpen((open) => !open)}>
              {workspaceOpen ? 'Hide workspace' : 'Workspace'}
            </button>
          </div>
        </header>

        <div className={`conversation-layout ${workspaceOpen ? 'with-workspace' : ''}`}>
          <section className="chat-column" aria-label="AXOM conversation">
            <div className="transcript" ref={transcriptRef} aria-live="polite">
              {messages.length === 0 ? (
                <div className="welcome">
                  <div className="welcome-mark" aria-hidden="true">A</div>
                  <h1>What can I help you build?</h1>
                  <p>Ask AXOM to write, debug, explain, or improve your code.</p>
                  <div className="suggestion-grid">
                    {suggestions.map((suggestion) => (
                      <button
                        key={suggestion}
                        className="suggestion-card"
                        onClick={() => setPrompt(suggestion)}
                      >
                        <span>{suggestion}</span>
                        <span className="suggestion-arrow" aria-hidden="true">↗</span>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="message-list">
                  {messages.map((message) => (
                    <article className={`message-row ${message.role}`} key={message.id}>
                      {message.role === 'assistant' && <span className="assistant-avatar" aria-hidden="true">A</span>}
                      <div className="message-content">
                        {message.role === 'user' && <span className="message-author">You</span>}
                        {message.role === 'assistant'
                          ? <AssistantContent content={message.content} />
                          : <p>{message.content}</p>}
                        {message.role === 'assistant' && (message.sources?.length || message.learningWarning) && (
                          <div className="github-learning">
                            {message.sources?.length ? (
                              <>
                                <span>Public GitHub references</span>
                                <ul>
                                  {message.sources.map((source) => (
                                    <li key={source.url}>
                                      <a href={source.url} target="_blank" rel="noreferrer">
                                        {source.name}
                                      </a>
                                      {source.language && <small>{source.language}</small>}
                                    </li>
                                  ))}
                                </ul>
                              </>
                            ) : null}
                            {message.learningWarning && (
                              <small className="github-learning-warning">
                                GitHub lookup unavailable: {message.learningWarning}
                              </small>
                            )}
                          </div>
                        )}
                        {message.role === 'assistant' && (
                          <div className="learning-actions">
                            {autoLearnedMessageIds.includes(message.id) && (
                              <span className="learning-auto-saved">Saved automatically</span>
                            )}
                            {savedLearningIds.includes(message.id) ? (
                              <span className="learning-saved">Saved as a local example</span>
                            ) : (
                              <button
                                type="button"
                                className="learning-toggle"
                                onClick={() => {
                                  setTeachingMessageId((current) =>
                                    current === message.id ? null : message.id,
                                  );
                                  setTeachingAnswer(message.content);
                                }}
                              >
                                Teach AXOM with a correction
                              </button>
                            )}
                          </div>
                        )}
                        {message.role === 'assistant' && teachingMessageId === message.id && (
                          <form
                            className="learning-form"
                            onSubmit={(event) => {
                              event.preventDefault();
                              void saveLearningExample(message, messages.indexOf(message));
                            }}
                          >
                            <label htmlFor={`learning-${message.id}`}>
                              Edit the answer into a useful example. It stays on this device.
                            </label>
                            <textarea
                              id={`learning-${message.id}`}
                              value={teachingAnswer}
                              onChange={(event) => setTeachingAnswer(event.target.value)}
                              maxLength={12000}
                              rows={6}
                            />
                            <div>
                              <button type="button" onClick={() => setTeachingMessageId(null)}>
                                Cancel
                              </button>
                              <button type="submit" disabled={!teachingAnswer.trim()}>
                                Save example
                              </button>
                            </div>
                          </form>
                        )}
                      </div>
                    </article>
                  ))}
                  {isLoading && (
                    <article className="message-row assistant">
                      <span className="assistant-avatar" aria-hidden="true">A</span>
                      <div className="typing-indicator" aria-label="AXOM is thinking">
                        <span /><span /><span />
                      </div>
                    </article>
                  )}
                </div>
              )}
            </div>

            <div className="composer-area">
              {errorMessage && (
                <div className="error-banner" role="alert">
                  <span>{errorMessage}</span>
                  <button onClick={() => setErrorMessage('')} aria-label="Dismiss error">×</button>
                </div>
              )}
              <div className="mode-picker" aria-label="Assistant mode">
                {modes.map((mode) => (
                  <button
                    type="button"
                    className={activeMode === mode.id ? 'active' : ''}
                    key={mode.id}
                    onClick={() => setActiveMode(mode.id)}
                  >
                    {mode.label}
                  </button>
                ))}
              </div>
              <form className="composer" onSubmit={handleSubmit}>
                <textarea
                  ref={textareaRef}
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  onKeyDown={handleComposerKeyDown}
                  placeholder={
                    activeMode === 'project'
                      ? 'Describe the project and files AXOM should create'
                      : activeMode === 'improve'
                        ? 'Describe what you want AXOM to improve in the workspace'
                        : 'Message AXOM'
                  }
                  aria-label="Message AXOM"
                  rows={1}
                  maxLength={16000}
                />
                <div className="composer-footer">
                  <span className="composer-hint">
                    Local SLM · {modes.find((mode) => mode.id === activeMode)?.label} mode
                    {activeMode !== 'project' && files.length > 0 ? ' · workspace included' : ''}
                  </span>
                  <button
                    type="submit"
                    className="send-button"
                    disabled={isLoading || !prompt.trim()}
                    aria-label={isLoading ? 'AXOM is responding' : 'Send message'}
                  >
                    {isLoading ? <span className="send-spinner" /> : <span aria-hidden="true">↑</span>}
                  </button>
                </div>
              </form>
              <p className="disclaimer">
                {activeMode === 'project'
                  ? 'Project generation creates up to 12 editable files locally. Review the code before running it.'
                  : activeMode === 'improve'
                    ? 'AXOM uses your current workspace files and returns focused changes for review.'
                    : files.length > 0
                      ? 'AXOM can use your current workspace files as coding context. GitHub lookup sends only your latest prompt, not these files.'
                      : `AXOM runs locally. ${learningCount} saved example${learningCount === 1 ? '' : 's'} can guide relevant answers. GitHub lookup sends only your latest prompt; check code before relying on it.`}
              </p>
            </div>
          </section>

          {workspaceOpen && (
            <aside className="workspace-panel">
              <div className="workspace-heading">
                <div>
                  <span className="eyebrow">AXOM WORKSPACE</span>
                  <h2>Files &amp; code</h2>
                </div>
                <button className="close-workspace" onClick={() => setWorkspaceOpen(false)} aria-label="Close workspace">×</button>
              </div>
              <div className="file-toolbar">
                <span>{files.length} {files.length === 1 ? 'file' : 'files'}</span>
                <button onClick={addFile}>＋ New file</button>
              </div>
              {files.length > 0 ? (
                <>
                  <div className="file-tabs">
                    {files.map((file) => (
                      <button
                        key={file.id}
                        className={file.id === selectedFileId ? 'active' : ''}
                        onClick={() => setSelectedFileId(file.id)}
                      >
                        {file.name}
                      </button>
                    ))}
                  </div>
                  {selectedFile && (
                    <>
                      <textarea
                        className="code-editor"
                        value={selectedFile.value}
                        onChange={(event) => updateFile(event.target.value)}
                        spellCheck={false}
                        aria-label={`Edit ${selectedFile.name}`}
                      />
                      <div className="file-actions">
                        <button onClick={downloadFile}>Download file</button>
                        <button onClick={downloadProjectZip}>Download project ZIP</button>
                      </div>
                    </>
                  )}
                </>
              ) : (
                <div className="workspace-empty">
                  <span className="empty-code-icon" aria-hidden="true">{'{ }'}</span>
                  <strong>No files yet</strong>
                  <p>Create a file here while you work with AXOM.</p>
                  <button onClick={addFile}>Create a file</button>
                </div>
              )}
            </aside>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
