# Quickstart — Run Claude Code against Ollama (local or Windows)

This repo is **Free Claude Code (FCC)**: a local proxy that lets Claude Code talk
to free/local models (Ollama, LM Studio, llama.cpp, and many cloud providers).
Everything runs on `127.0.0.1` — the proxy only forwards Claude Code's API calls
to the provider URL you configure. No external network calls.

## Pick a target

| You want…                              | Claude Code via FCC | opencode (this session) |
|----------------------------------------|---------------------|-------------------------|
| Ollama on **this Mac** (local)         | `/run-local`        | `/connect-local`        |
| Ollama on your **Windows PC** (Tailscale) | `/run-windows`  | `/connect-windows`      |

### Claude Code via FCC (one-shot)
The `/run-local` and `/run-windows` commands do everything:
1. Configure the repo `.env` (`OLLAMA_BASE_URL` + `MODEL*`) and opencode's model list.
2. Start `fcc-server` in the background and wait for it to be healthy.
3. Launch Claude Code via `fcc-claude`.

Behind the scenes they run:
```bash
uv run scripts/run_claude_ollama.py --local          # local Mac Ollama
uv run scripts/run_claude_ollama.py                  # Windows PC via Tailscale
```
Both accept an optional model-name filter, e.g. `uv run scripts/run_claude_ollama.py --local coder`.

Prefer to manage the proxy yourself? Run the connect command, then:
```bash
uv run fcc-server      # start proxy (Admin UI at http://127.0.0.1:8082/admin)
uv run fcc-claude      # launch Claude Code against the proxy
```

### Pinning your own models
Your `.env` maps four tiers (`MODEL`, `MODEL_OPUS`, `MODEL_SONNET`, `MODEL_HAIKU`).
The connect/run commands auto-pick a sensible default **only when a tier's
current model is missing on the target** (Windows GPU can run big models; the
Mac is capped at ~14B-class since it has 16GB RAM and `-cloud` models are
remote API endpoints, not local compute). To choose explicitly, pass:
```bash
uv run scripts/connect_ollama.py --local --opus qwen2.5-coder:7b --sonnet qwen3:14b --haiku qwen3:8b
```
or through the slash commands: `/connect-local --opus qwen2.5-coder:7b ...`.
Any explicit pin is validated against the target and kept; a manual edit to
`.env` is also respected as long as that model exists on the target.

### opencode (this session)
`/connect-local` and `/connect-windows` point opencode's Ollama provider at the
right host and refresh the model list. Then pick a model with opencode's
`/model` (restart opencode only if it doesn't hot-reload the config).

## Your normal Claude Code skills
Your global skills in `~/.claude/skills` (agent-reach, codebase-memory,
omc-reference, gsd, worktree-parallel) are **inherited automatically** — `fcc-claude`
just runs the real `claude` binary, and opencode already loads those same global
skills. No extra setup is needed.

## Notes
- `fcc-server` keeps running in the background after you quit Claude Code. Stop it
  with `/stop-fcc` (or `pkill -f fcc-server`) when you're done.
- `/run-local` (and `/connect-local`) auto-start `ollama serve` if the Mac's Ollama
  isn't already running, then retry — so you don't have to start it by hand.
- The Windows PC must be signed into the same Tailscale account and running Ollama
  listening on all interfaces (`OLLAMA_HOST=0.0.0.0`).
- `OLLAMA_BASE_URL` and `MODEL*` live in the repo `.env` (gitignored).
