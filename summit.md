Run an AI Summit on the user's question using the TypeScript skill runtime.

## Instructions

1. Resolve script path.

```bash
SCRIPT_PATH="${CODEX_HOME:-$HOME/.codex}/skills/ai-summit/scripts/summit.ts"
```

2. Execute summit with the user's question (`$ARGUMENTS`).

```bash
bun run "$SCRIPT_PATH" \
  --question "$ARGUMENTS" \
  --providers claude,codex,gemini \
  --rounds 2
```

3. For architecture/complex decisions, use 3 rounds.

```bash
bun run "$SCRIPT_PATH" \
  --question "$ARGUMENTS" \
  --rounds 3
```

4. Save machine-readable output when needed.

```bash
bun run "$SCRIPT_PATH" \
  --question "$ARGUMENTS" \
  --output json \
  --save-log /tmp/ai-summit-result.json
```

5. Summarize:
- Which providers participated
- Consensus trend by round
- Final recommended plan and risks
