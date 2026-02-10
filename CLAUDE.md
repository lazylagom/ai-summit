# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Summit is an MCP (Model Context Protocol) server that orchestrates multi-LLM debates. It sends a question through multiple AI CLI tools (Claude, Codex, Gemini, DeepSeek, Mistral) in iterative review rounds, then synthesizes a final answer from the debate.

## Development Commands

```bash
# Install dependencies (editable mode)
pip install -e .

# Run the MCP server directly
python server.py

# Register with Claude Code (global)
claude mcp add --scope user ai-summit -- python /absolute/path/ai-summit/server.py
```

No test framework or linter is configured.

## Architecture

This is a single-file MCP server (`server.py`) built with FastMCP + Pydantic. All LLM calls go through local CLI tools (no API keys or HTTP calls).

### Provider System

Each LLM provider is a `Provider` class registered in the `PROVIDERS` dict with: name, emoji, cli_cmd, and an async `call_fn`. All calls are via local CLI subprocess execution. Adding a new provider requires only an async call function and a `PROVIDERS` dict entry.

Provider availability is determined at runtime by checking if the CLI command exists in PATH (`shutil.which`).

### Available Providers

| Key | CLI Command | Description |
|-----|-------------|-------------|
| `claude` | `claude` | Claude Code CLI |
| `codex` | `codex` | OpenAI Codex CLI |
| `gemini` | `gemini` | Google Gemini CLI |
| `deepseek` | `deepseek` | DeepSeek CLI |
| `mistral` | `mistral` | Mistral CLI |

### Debate Flow (summit_run)

1. First provider generates initial solution (Round 0)
2. All other providers + first provider review sequentially for N rounds using `REVIEW_SYSTEM_PROMPT`
3. First provider synthesizes final answer using `SYNTHESIS_SYSTEM_PROMPT` with full debate history
4. Results stored in-memory (`_summit_store`) with `summit_YYYYMMDD_HHMMSS` IDs

### MCP Tools

| Tool | Purpose |
|------|---------|
| `summit_run` | Full multi-round debate with synthesis |
| `summit_ask` | Single provider query (optionally with review context) |
| `summit_providers` | List configured/available providers |
| `summit_history` | Retrieve past summit results from session |

### Input Validation

Pydantic models (`SummitInput`, `AskProviderInput`, `SummitHistoryInput`) validate all tool inputs. Questions are 1-100k chars, rounds are 1-5, provider keys are validated against `PROVIDERS`.

## Environment Variables

Model overrides (optional): `CLAUDE_MODEL`, `CODEX_MODEL`, `GEMINI_MODEL`, `DEEPSEEK_MODEL`, `MISTRAL_MODEL`

## Key Constraints

- stdout is reserved for MCP stdio transport — all logging goes to stderr
- Session storage is in-memory only (lost on server restart)
- CLI call timeout is 300 seconds (5 minutes) per provider
- Prompts instruct providers to respond in the same language as the original question
