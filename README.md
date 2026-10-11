# AXOM

AXOM is a local-first coding assistant with conversation history, focused build/fix/refactor/explain/design modes, and an editable workspace for generated code projects.

## Run locally

```bash
npm install
npm run dev
```

Open http://localhost:5173/.

## Local small language model


Choose **Create files** above the message composer to ask AXOM to plan and generate a coordinated project of up to twelve files. AXOM generates the files one at a time so later files can follow the earlier ones, checks for common unfinished placeholders, then opens them in the editable workspace.

Choose **Improve files** to request a focused change to the current workspace. AXOM chooses the files that need coordinated updates and preserves other files. In the other chat modes, the current workspace source is included as context, so you can ask questions about your code without pasting it into chat. Use **Download project ZIP** to export the workspace with its directory structure. Review generated code before running it.

Use the model picker to select **AXOM 1.0** (faster 0.5B) or **AXOM 1.1** (more capable 1.5B). AXOM disposes the active model before loading the other one to avoid keeping both large model sessions in memory. Switching models therefore takes time on the first load; weights are cached for later switches. The larger model requires about 1.9 GB of disk for Q4 weights.

The default cache directory is `.cache/huggingface`. Set `AXOM_MODEL_CACHE_DIR` to use another location, `AXOM_MODEL` to override AXOM 1.0, `AXOM_MODEL_1_1` to override AXOM 1.1, `AXOM_MODEL_PROFILE=AXOM1.1` to start with the larger profile, `AXOM_MODEL_DTYPE` to change the quantization (default `q4`), `AXOM_TEMPERATURE` to tune sampling (default `0.2`), or `AXOM_MAX_NEW_TOKENS` to adjust the overall generation limit (64–4096 tokens; default 3072). Normal chat replies are capped at 512 tokens by default, with tighter caps for explain/design modes, to avoid slow, unnecessarily long output; tune the chat cap with `AXOM_CHAT_MAX_NEW_TOKENS`. Low-temperature top-p sampling aims to avoid brittle, repetitive outputs while staying relatively consistent for coding. Both models can make mistakes. Workspace context is bounded to 24,000 characters, and individual project files to 12,000 characters per generation request.

AXOM does not fine-tune Qwen's weights. By default, it automatically saves each normal chat request and reply as a local example; turn **Learn from chats automatically** off in the sidebar to opt out. You can also choose **Teach AXOM with a correction** under an answer to save an edited example. Relevant examples guide future prompts through retrieval; they do not change model weights and may not help unrelated tasks. Examples are stored by the backend in `.cache/axom-learning.json` (override with `AXOM_LEARNING_FILE`), capped at 100, and can be cleared in the sidebar. Project generation requests and workspace files are not automatically saved. If you run AXOM on another machine, its examples are stored on that backend machine.

For current public references, AXOM searches GitHub repositories for each chat or project-generation request and may use a small excerpt from matching repository READMEs. The response shows links to repositories it found. Only the latest request text is sent to GitHub; conversation history, workspace files, and generated projects are not included in the search. GitHub retrieval is best-effort and subject to GitHub's public API rate limits. This provides fresh reference context, not background crawling or model training.

The app contains the model-loading, prompt, project-planning, and inference code; large model weights are downloaded separately rather than committed to the source tree. If local inference fails or the model returns an invalid project plan, AXOM shows an error instead of returning fabricated code. The backend root (`/`) and `/api/health` report model status.

## Build

```bash
npm run build
```
