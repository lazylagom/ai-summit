---
name: ai-summit
description: Run structured multi-LLM debate and cross-validation using local AI CLIs (claude, codex, gemini, deepseek, mistral), then synthesize a decision-ready answer. Use when the user asks for multi-model comparison, independent second opinions, architecture/code plan validation, risk stress-testing, or consensus-driven actionable recommendations. This skill is TypeScript runtime only and does not use MCP/Python tools.
---

# AI Summit

Use this skill to execute repeatable multi-model review rounds and produce a single actionable synthesis.

## Execute Workflow

1. Normalize input.
- Confirm the exact question.
- Confirm provider scope if the user restricts providers.
- Require at least two available providers.

2. Run the TypeScript orchestrator first.
- Prefer deterministic execution via `scripts/summit.ts`.
- Use the installed skill path:

```bash
SCRIPT_PATH="${CODEX_HOME:-$HOME/.codex}/skills/ai-summit/scripts/summit.ts"
bun run "$SCRIPT_PATH" \
  --question "<user question>" \
  --providers claude,codex,gemini \
  --rounds 2
```

- If the user needs logs, add `--output json --save-log <path>`.

3. Fallback to manual orchestration when the script cannot run.
- Ask one provider for initial draft.
- Ask all selected providers to review the same draft in parallel.
- Run 1-3 rounds until consensus is sufficient.
- Ask the lead provider to synthesize a final answer.

4. Enforce quality gate in the final response.
- Mark unverifiable claims as `[Unverified]`.
- Include explicit risks and limitations.
- Provide numbered action steps for execution-oriented requests.
- Preserve the user's language in the final answer.

## Command Shortcuts

- Help:

```bash
SCRIPT_PATH="${CODEX_HOME:-$HOME/.codex}/skills/ai-summit/scripts/summit.ts"
bun run "$SCRIPT_PATH" --help
```

- Read question from stdin:

```bash
SCRIPT_PATH="${CODEX_HOME:-$HOME/.codex}/skills/ai-summit/scripts/summit.ts"
echo "Review this rollout plan" | bun run "$SCRIPT_PATH"
```

## Resources

- Use `scripts/summit.ts` for automated debate/synthesis.
- Use `references/cli-usage.md` for argument examples and output modes.
