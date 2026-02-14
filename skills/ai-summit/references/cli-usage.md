# AI Summit Script Usage

## Quick Start

```bash
SCRIPT_PATH="${CODEX_HOME:-$HOME/.codex}/skills/ai-summit/scripts/summit.ts"
bun run "$SCRIPT_PATH" \
  --question "Review this architecture and propose an implementation plan" \
  --providers claude,codex,gemini \
  --rounds 2
```

## Provider Selection

Supported providers:

- `claude`
- `codex`
- `gemini`
- `deepseek`
- `mistral`

The script only uses providers whose CLIs are currently installed in `PATH`. If fewer than two providers are available, the run fails.

## Output Modes

`--output markdown` (default): human-readable result with summary and final answer.

`--output json`: machine-readable output for scripting.

## Save Full Logs

```bash
SCRIPT_PATH="${CODEX_HOME:-$HOME/.codex}/skills/ai-summit/scripts/summit.ts"
bun run "$SCRIPT_PATH" \
  --question "..." \
  --save-log /tmp/ai-summit-result.json \
  --output json
```

## Standard Input

You can pipe the question instead of using `--question`.

```bash
SCRIPT_PATH="${CODEX_HOME:-$HOME/.codex}/skills/ai-summit/scripts/summit.ts"
echo "Assess this migration strategy" | bun run "$SCRIPT_PATH"
```
