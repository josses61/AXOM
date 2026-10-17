# AXOM

AXOM is a local-first coding assistant with conversation history, one natural-language chat, and an editable workspace for generated code projects.

## Run locally

```bash
npm install
npm run dev
```

Open http://localhost:5173/.

Use the single natural-language prompt to ask AXOM to explain, fix, improve, or build something. For apps and other projects, it generates coordinated files one at a time in the editable workspace. There is no fixed project file-count setting; generation is still constrained by model context/output limits and available memory. Code execution requires the explicit Run code action and is limited to the isolated runner. AXOM does not train its own weights or provide datasets. Large model training needs suitable hardware, properly licensed data, storage, and infrastructure.

## Run code in the isolated runner

Install Docker, then build the local runner image once:

```bash
npm run runner:build
```

The image includes Node.js, Python, GCC/G++, Rust, Go, and Java toolchains and requires about 2.4 GB of local Docker storage. Select a JavaScript, Python, C, C++, Rust, Go, or Java source file and press **Run code**. AXOM sends the current workspace to the local runner and displays stdout, stderr, exit status, and runtime/compiler errors. Use **Ask AXON to debug this result** to put the output into the chat prompt. External package downloads and network access are unavailable during a run. Individual files are limited to 12,000 characters, the workspace to 24,000 characters and 64 files, program input to 4,000 characters, output to 48,000 characters, and execution to 8 seconds per compile/run step (20 seconds total). The runner uses a read-only container filesystem, ephemeral temporary storage, no network, a non-root user, dropped Linux capabilities, and CPU, memory, and process limits. Containers are removed after each run.

This is defense-in-depth, not a formally verified hostile-code sandbox. Docker and its daemon remain part of the trusted computing base. Keep AXOM bound to localhost, do not expose its API or Docker daemon to untrusted users, and do not run code you do not trust.

When you ask to change existing workspace files, AXOM selects files for coordinated updates and preserves unrelated files. Relevant workspace files are prioritized in the bounded model context. The workspace supports HTML, CSS, JavaScript, TypeScript, JSON, Python, C, C++, Rust, Go, Java, C#, Ruby, PHP, Swift, Kotlin, SQL, shell, YAML, TOML, and common text files. Import text files, create files by language, edit them, and export the project as a ZIP. Workspace edits and conversation history are held in the current page and are not restored after a reload; export a ZIP to keep a copy. Review generated code before running it.

AXOM uses a local Q4 Qwen2.5-Coder model: **AXOM 1.1** (1.5B) by default, or **AXOM 1.0** (0.5B) by setting `AXOM_MODEL_PROFILE=AXOM1.0`. On app startup the backend downloads and loads the selected model into its local cache if it is not already present. The larger model is more capable but still a small model and cannot be expected to match professional frontier systems.

The default cache directory is `.cache/huggingface`. Set `AXOM_MODEL_CACHE_DIR` to use another location, `AXOM_MODEL` to override AXOM 1.0, `AXOM_MODEL_1_1` to override AXOM 1.1, `AXOM_MODEL_PROFILE=AXOM1.1` to start with the larger profile (the default) or `AXOM_MODEL_PROFILE=AXOM1.0` for the smaller profile, `AXOM_MODEL_DTYPE` to change the quantization (default `q4`), `AXOM_TEMPERATURE` to tune sampling (default `0.2`), or `AXOM_MAX_NEW_TOKENS` to adjust the per-generation limit (64–4096 tokens; default 3072). Normal chat replies are capped at 512 tokens by default; tune the chat cap with `AXOM_CHAT_MAX_NEW_TOKENS`. Low-temperature top-p sampling aims to avoid brittle, repetitive outputs while staying relatively consistent for coding. Both models can make mistakes. Workspace context is bounded to 24,000 characters, and individual project files to 12,000 characters per generation request.

AXOM can retrieve public web search snippets and GitHub README excerpts for relevant requests and save references with their source links to `.cache/axom-knowledge.json` (override with `AXOM_KNOWLEDGE_FILE`). Locally saved coding examples and public references are retrieval memory, not model-weight training: AXOM's parameters do not change. Saved references are bounded to keep local storage manageable. Do not treat retrieved snippets as verified or current without checking their sources.

The team drawer lets you define up to eight specialist agents from a guided form with role templates for research, building, review, design, security, and writing; role instructions remain editable. Assign teams, group leads, reporting lines, and locally saved work items. Team group chats call the local model with the enabled specialists; later agents receive earlier agents' reports, and chats are stored in this browser. Public web pages can be read by URL; AXOM extracts page metadata and section headings while limiting requests to public HTTP(S) hosts and enforcing redirect, byte, and time limits. GitHub access uses the local `gh` CLI session (`gh auth login`), not an in-app OAuth flow. You can import repositories, create private or public repositories, and publish workspace files to a selected repository's default branch, one GitHub commit per file. Trusted-author labels are stored in this browser only and do not replace code review. Review the selected repository and workspace before publishing; environment files and common private-key files are blocked. Email is a locally saved contact address that opens a prefilled draft in your configured mail app; AXOM does not read mailboxes or send messages.

When running alongside another local AXOM backend, set `AXOM_API_URL` to configure the Vite `/api` proxy target (for example, `AXOM_API_URL=http://127.0.0.1:3002`).

AXOM's optional local profile and example data are stored under `.cache/` by the backend. This is retrieval context, not model-weight training. Never put secrets or sensitive personal data in the profile or workspace.

The API binds to `127.0.0.1` by default and accepts browser API requests only from the local Vite origins `http://localhost:5173` and `http://127.0.0.1:5173`. To use another trusted development origin, set `AXOM_ALLOWED_ORIGINS` to a comma-separated exact origin allowlist; when set, it replaces the defaults. Set `AXOM_HOST` only when you deliberately need a non-loopback bind. The origin allowlist is not authentication—do not expose the API to an untrusted network.

Generated code is never run automatically. Review it first, then explicitly select a supported source file and start a run. The runner is for checking program behavior and test output; it does not attempt to escape its container or teach escape techniques.

The app contains the model-loading, prompt, project-planning, and inference code; large model weights are downloaded separately rather than committed to the source tree. If local inference fails or the model returns an invalid project plan, AXOM shows an error instead of returning fabricated code. The backend root (`/`) and `/api/health` report model status.

## Build

```bash
npm run build
```
