# CLAUDE.md

This repository is **TypeScript Skill only**.

## Development Commands

```bash
# Summit script help
bun run summit:help

# Example local run
bun run summit --question "Review this design" --rounds 2
```

## Skill Structure

- `skills/ai-summit/SKILL.md`: skill instructions and workflow
- `skills/ai-summit/agents/openai.yaml`: UI metadata
- `skills/ai-summit/scripts/summit.ts`: TypeScript orchestration script
- `skills/ai-summit/references/cli-usage.md`: command examples

## Runtime Notes

- AI Summit uses local CLI tools (`claude`, `codex`, `gemini`, `deepseek`, `mistral`)
- At least 2 providers must be installed in `PATH`
- No Python/MCP runtime is supported in this repository
