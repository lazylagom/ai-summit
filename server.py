"""
AI Summit — MCP Server
======================
Multi-LLM debate platform that orchestrates cross-validation and
iterative refinement across multiple AI models via local CLI tools.

Designed for extensibility: add new LLM providers by registering them
in the PROVIDERS dict.

Usage:
  1. Install CLI tools (claude, codex, gemini, etc.)
  2. Add to Claude Code's MCP settings
  3. Use tools directly or via /summit slash command
"""

import asyncio
import json
import os
import shutil
import sys
import logging
import time
from typing import Optional, List, Dict, Any, Callable, Awaitable
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field, ConfigDict, field_validator

# ---------------------------------------------------------------------------
# Logging (stderr only – stdout reserved for stdio transport)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("ai_summit")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_ROUNDS = 2
MAX_ROUNDS = 5
CLI_TIMEOUT = 300  # 5 minutes default for CLI calls
CLAUDE_CLI_TIMEOUT = 600  # 10 minutes for Claude (tool use takes longer)

RESEARCH_RULES = """
## Research Rules (MANDATORY)
- You MUST search the web or look up actual sources before making any factual claim.
- NEVER speculate, guess, or rely solely on training data for factual questions.
- For every major claim, provide a source (URL, file path, documentation link).
- If you cannot verify something, explicitly mark it as: **[미검증/Unverified]**
- Clearly separate verified facts from your own analysis/opinions.
- When asked about source code, search for and cite the ACTUAL code — do not reconstruct from memory.
"""

INITIAL_SYSTEM_PROMPT = f"""You are an expert AI participating in an AI Summit — a multi-LLM debate.
Your role is to provide the initial solution to the given question.

{RESEARCH_RULES}
Instructions:
1. Research the topic thoroughly before answering
2. Provide a comprehensive, well-structured solution
3. Cite sources for all factual claims
4. Be specific and actionable

Respond in the same language as the original question."""

REVIEW_SYSTEM_PROMPT = f"""You are an expert AI reviewer participating in an AI Summit — a multi-LLM debate.
Your role is to critically evaluate the previous AI's solution and improve it.

{RESEARCH_RULES}
Instructions:
1. Verify the previous AI's claims — search and fact-check before agreeing or disagreeing
2. Identify flaws, unsourced claims, edge cases, or missed considerations
3. Propose concrete improvements with reasoning and sources
4. If the solution is already optimal, confirm it with evidence
5. Preserve what works well from the previous solution

Respond in the same language as the original question."""

SYNTHESIS_SYSTEM_PROMPT = f"""You are the final synthesizer in an AI Summit — a multi-LLM debate.
Multiple AI models have debated and refined a solution through several rounds.

{RESEARCH_RULES}

You MUST structure your response in the following format:

---

## 📊 Provider별 의견 정리

For EACH participating AI, write a concise summary of their key arguments,
unique insights, and final position. Use this format for each:

### {emoji} {Provider Name}
- **핵심 주장**: (their main argument in 1-2 sentences)
- **고유 인사이트**: (unique points only this provider raised)
- **최종 입장**: (their final position after all rounds)

## 🤝 합의점 (Points of Agreement)
List the points where all or most providers agreed.

## ⚡ 쟁점 (Points of Disagreement)
List the points where providers disagreed, with brief explanation of each side.

## 🏆 최종 결론 (Final Conclusion)
Provide the definitive, production-ready answer that synthesizes the best ideas.
Resolve all disagreements with clear reasoning for your choices.
If the answer involves code, provide complete, working implementation.

---

IMPORTANT:
- The Provider summary section must cover EVERY participating AI individually.
- The Final Conclusion must be comprehensive and actionable, not just a summary.
- Respond in the same language as the original question."""


# ---------------------------------------------------------------------------
# Provider Registry (extensible — add new LLMs here)
# ---------------------------------------------------------------------------
class Provider:
    """Represents a CLI-based LLM provider."""

    def __init__(
        self,
        name: str,
        emoji: str,
        cli_cmd: str,
        call_fn: Callable[..., Awaitable[str]],
        model_env_var: str = "",
    ):
        self.name = name
        self.emoji = emoji
        self.cli_cmd = cli_cmd
        self.call_fn = call_fn
        self.model_env_var = model_env_var

    @property
    def model(self) -> str:
        if self.model_env_var:
            return os.environ.get(self.model_env_var, "")
        return ""

    @property
    def is_configured(self) -> bool:
        return shutil.which(self.cli_cmd) is not None


# ---------------------------------------------------------------------------
# CLI Call Functions (local CLI tools — no API keys needed)
# ---------------------------------------------------------------------------
async def _run_cli(
    cmd: List[str],
    input_data: Optional[bytes] = None,
    timeout: int = CLI_TIMEOUT,
) -> str:
    """Run a CLI command and return stdout, with timeout handling."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if input_data else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input_data), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        return f"Error: CLI timed out after {timeout}s."
    if proc.returncode != 0:
        return f"Error (exit {proc.returncode}): {stderr.decode()[:500]}"
    return stdout.decode().strip()


async def _call_claude_cli(
    prompt: str,
    system_prompt: str = "",
    model: str = "",
    max_tokens: int = 4096,
) -> str:
    cmd = ["claude", "-p", "--no-session-persistence", "--max-turns", "3"]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    if model:
        cmd.extend(["--model", model])
    return await _run_cli(cmd, input_data=prompt.encode(), timeout=CLAUDE_CLI_TIMEOUT)


async def _call_codex_cli(
    prompt: str,
    system_prompt: str = "",
    model: str = "",
    max_tokens: int = 4096,
) -> str:
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    cmd = ["codex", "exec"]
    if model:
        cmd.extend(["-m", model])
    return await _run_cli(cmd, input_data=full_prompt.encode())


async def _call_gemini_cli(
    prompt: str,
    system_prompt: str = "",
    model: str = "",
    max_tokens: int = 4096,
) -> str:
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    cmd = ["gemini", "-p", full_prompt]
    if model:
        cmd.extend(["-m", model])
    return await _run_cli(cmd)


async def _call_deepseek_cli(
    prompt: str,
    system_prompt: str = "",
    model: str = "",
    max_tokens: int = 4096,
) -> str:
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    cmd = ["deepseek"]
    if model:
        cmd.extend(["-m", model])
    return await _run_cli(cmd, input_data=full_prompt.encode())


async def _call_mistral_cli(
    prompt: str,
    system_prompt: str = "",
    model: str = "",
    max_tokens: int = 4096,
) -> str:
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    cmd = ["mistral"]
    if model:
        cmd.extend(["-m", model])
    return await _run_cli(cmd, input_data=full_prompt.encode())


# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------
PROVIDERS: Dict[str, Provider] = {
    "claude": Provider("Claude", "🟣", "claude", _call_claude_cli, model_env_var="CLAUDE_MODEL"),
    "codex": Provider("Codex", "🟢", "codex", _call_codex_cli, model_env_var="CODEX_MODEL"),
    "gemini": Provider("Gemini", "🔵", "gemini", _call_gemini_cli, model_env_var="GEMINI_MODEL"),
    "deepseek": Provider("DeepSeek", "🟡", "deepseek", _call_deepseek_cli, model_env_var="DEEPSEEK_MODEL"),
    "mistral": Provider("Mistral", "🟠", "mistral", _call_mistral_cli, model_env_var="MISTRAL_MODEL"),
}

DEFAULT_DEBATE_ORDER = ["claude", "codex", "gemini", "deepseek", "mistral"]


def _get_active_providers(requested: Optional[List[str]] = None) -> List[Provider]:
    """Return providers that have CLI tools installed, in requested order."""
    if requested:
        keys = [k for k in requested if k in PROVIDERS and PROVIDERS[k].is_configured]
    else:
        keys = [k for k in DEFAULT_DEBATE_ORDER if PROVIDERS[k].is_configured]

    if not keys:
        raise ValueError(
            "No LLM providers configured. Install at least 2 CLI tools. "
            f"Supported: {', '.join(f'{k} (CLI: {PROVIDERS[k].cli_cmd})' for k in PROVIDERS)}"
        )
    return [PROVIDERS[k] for k in keys]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _build_review_prompt(question: str, previous_response: str, ai_name: str) -> str:
    return f"""## Original Question
{question}

## Previous AI's ({ai_name}) Response
{previous_response}

---
Please review the above response. Identify any issues and provide an improved solution."""


def _build_parallel_review_prompt(question: str, previous_responses: List[Dict[str, str]]) -> str:
    """Build review prompt containing all responses from the previous round."""
    parts = [f"## Original Question\n{question}\n"]
    for i, resp in enumerate(previous_responses, 1):
        parts.append(f"## Response {i} — {resp['ai_name']}\n{resp['response']}\n")
    parts.append(
        "---\n"
        "Please review all responses above. Identify the best ideas, "
        "point out any issues, and provide an improved comprehensive solution."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP("ai_summit")


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------
class AskProviderInput(BaseModel):
    """Input for asking a specific LLM provider."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    provider: str = Field(
        ...,
        description="Provider key: 'claude', 'codex', 'gemini', 'deepseek', 'mistral'",
    )
    question: str = Field(
        ...,
        description="The question or prompt to send",
        min_length=1,
        max_length=100000,
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional context from a previous AI's response for review",
    )
    context_ai_name: Optional[str] = Field(
        default=None,
        description="Name of the AI that produced the context",
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in PROVIDERS:
            raise ValueError(f"Unknown provider '{v}'. Available: {', '.join(PROVIDERS.keys())}")
        return v


class SummitInput(BaseModel):
    """Input for running a full AI Summit debate."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(
        ...,
        description="The question or problem to debate across multiple LLMs",
        min_length=1,
        max_length=100000,
    )
    rounds: int = Field(
        default=DEFAULT_ROUNDS,
        description="Number of debate rounds (each round cycles through all active providers)",
        ge=1,
        le=MAX_ROUNDS,
    )
    providers: Optional[List[str]] = Field(
        default=None,
        description="Ordered list of provider keys to use. If not set, uses all configured providers. e.g. ['claude', 'codex', 'gemini']",
    )
    include_synthesis: bool = Field(
        default=True,
        description="Whether to include a final synthesis step",
    )

    @field_validator("providers")
    @classmethod
    def validate_providers(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            for p in v:
                if p.lower().strip() not in PROVIDERS:
                    raise ValueError(f"Unknown provider '{p}'. Available: {', '.join(PROVIDERS.keys())}")
            return [p.lower().strip() for p in v]
        return v


class SummitHistoryInput(BaseModel):
    """Input for retrieving summit history."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    summit_id: Optional[str] = Field(
        default=None,
        description="Specific summit ID to retrieve. If not provided, lists all summits.",
    )


# ---------------------------------------------------------------------------
# In-memory storage
# ---------------------------------------------------------------------------
_summit_store: Dict[str, Dict[str, Any]] = {}


def _generate_summit_id() -> str:
    return f"summit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool(
    name="summit_ask",
    annotations={
        "title": "Ask a specific LLM",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def summit_ask(params: AskProviderInput) -> str:
    """Ask a specific LLM provider a question, optionally with context for review.

    Supported providers: claude, codex, gemini, deepseek, mistral.
    Only providers with installed CLI tools will work.

    Args:
        params (AskProviderInput): Provider key, question, and optional review context.

    Returns:
        str: The provider's response in markdown format.
    """
    provider = PROVIDERS[params.provider]
    if not provider.is_configured:
        return f"Error: {provider.name} is not configured. Install `{provider.cli_cmd}` CLI tool."

    if params.context and params.context_ai_name:
        prompt = _build_review_prompt(params.question, params.context, params.context_ai_name)
        system = REVIEW_SYSTEM_PROMPT
    else:
        prompt = params.question
        system = ""

    logger.info(f"Calling {provider.name} ({provider.model})...")
    response = await provider.call_fn(prompt, system_prompt=system, model=provider.model)

    return f"## {provider.emoji} {provider.name} ({provider.model}) Response\n\n{response}"


@mcp.tool(
    name="summit_providers",
    annotations={
        "title": "List available LLM providers",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def summit_providers() -> str:
    """List all supported LLM providers and their configuration status.

    Returns:
        str: Markdown table of providers with status (configured/not configured).
    """
    lines = ["# 🏔️ AI Summit — Available Providers\n"]
    lines.append("| Provider | Model | Status | CLI |")
    lines.append("|----------|-------|--------|-----|")
    for key, p in PROVIDERS.items():
        status = "✅ Ready" if p.is_configured else "❌ Not installed"
        model_display = f"`{p.model}`" if p.model else "*(default)*"
        lines.append(f"| {p.emoji} {p.name} (`{key}`) | {model_display} | {status} | `{p.cli_cmd}` |")

    active = [p for p in PROVIDERS.values() if p.is_configured]
    lines.append(f"\n**{len(active)}/{len(PROVIDERS)}** providers ready for summit.")
    return "\n".join(lines)


@mcp.tool(
    name="summit_run",
    annotations={
        "title": "Run AI Summit (Multi-LLM Debate)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def summit_run(params: SummitInput, ctx: Context) -> str:
    """Run a full AI Summit: multiple LLMs debate and refine a solution through parallel rounds.

    Flow:
    1. First provider gives initial solution
    2. All providers review in parallel per round
    3. Cycle repeats for N rounds
    4. Final synthesis combines the best ideas

    Only providers with installed CLI tools participate.

    Args:
        params (SummitInput): Question, rounds, optional provider list.
        ctx (Context): MCP context for progress reporting.

    Returns:
        str: Complete summit transcript with final synthesis.
    """
    active = _get_active_providers(params.providers)
    if len(active) < 2:
        return "Error: At least 2 configured providers are required for a summit. Use `summit_providers` to check status."

    summit_id = _generate_summit_id()
    history: List[Dict[str, Any]] = []
    transcript: List[str] = []

    provider_names = " / ".join(f"{p.emoji}{p.name}" for p in active)
    transcript.append(f"# 🏔️ AI Summit: {summit_id}")
    transcript.append(f"**Question:** {params.question}")
    transcript.append(f"**Rounds:** {params.rounds} | **Participants:** {provider_names}")
    transcript.append("---")

    total_steps = 1 + params.rounds + (1 if params.include_synthesis else 0)
    step = 0
    summit_start = time.time()

    # --- Announce plan ---
    synthesis_label = " → 최종 합성" if params.include_synthesis else ""
    await ctx.info(
        f"🏔️ AI Summit 시작! 참가자: {provider_names}\n"
        f"📋 계획: Round 0 (초기 답변) → Round 1~{params.rounds} (리뷰){synthesis_label}"
    )

    # --- Round 0: Initial solution from first provider ---
    first = active[0]
    transcript.append("## Round 0 — Initial Solution")
    await ctx.report_progress(step, total_steps)
    step_start = time.time()
    await ctx.info(f"⏳ [1/{total_steps}] Round 0: {first.emoji} {first.name} 초기 답변 생성 중...")
    logger.info(f"Round 0: {first.name} initial solution...")
    initial = await first.call_fn(
        params.question, system_prompt=INITIAL_SYSTEM_PROMPT, model=first.model
    )
    step_elapsed = time.time() - step_start
    await ctx.info(f"✅ [1/{total_steps}] Round 0 완료 — {first.emoji} {first.name} ({step_elapsed:.0f}초)")
    history.append({"ai": first.name, "round": 0, "role": "initial", "response": initial})
    transcript.append(f"### {first.emoji} {first.name}\n\n{initial}\n")
    step += 1

    previous_round_responses = [{"ai_name": first.name, "response": initial}]

    # --- Parallel Debate Rounds ---
    async def _call_reviewer(provider: Provider, prompt: str, system: str):
        t0 = time.time()
        resp = await provider.call_fn(prompt, system_prompt=system, model=provider.model)
        elapsed = time.time() - t0
        await ctx.info(f"  ✓ {provider.emoji} {provider.name} 리뷰 완료 ({elapsed:.0f}초)")
        return provider, resp

    for r in range(1, params.rounds + 1):
        transcript.append(f"## Round {r}")
        await ctx.report_progress(step, total_steps)
        step_start = time.time()

        reviewer_names = " + ".join(f"{p.emoji}{p.name}" for p in active)
        await ctx.info(f"⏳ [{step+1}/{total_steps}] Round {r}: {reviewer_names} 병렬 리뷰 중...")
        logger.info(f"Round {r}: parallel review by {', '.join(p.name for p in active)}...")

        review_prompt = _build_parallel_review_prompt(params.question, previous_round_responses)
        tasks = [_call_reviewer(p, review_prompt, REVIEW_SYSTEM_PROMPT) for p in active]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        round_responses = []
        for i, result in enumerate(results):
            provider = active[i]
            if isinstance(result, Exception):
                resp = f"Error: {type(result).__name__}: {str(result)[:300]}"
                logger.warning(f"Provider {provider.name} failed in round {r}: {resp}")
            else:
                _, resp = result
            history.append({"ai": provider.name, "round": r, "role": "reviewer", "response": resp})
            transcript.append(f"### {provider.emoji} {provider.name} Review\n\n{resp}\n")
            round_responses.append({"ai_name": provider.name, "response": resp})

        step_elapsed = time.time() - step_start
        await ctx.info(f"✅ [{step+1}/{total_steps}] Round {r} 완료 ({step_elapsed:.0f}초)")
        previous_round_responses = round_responses
        transcript.append("---")
        step += 1

    # --- Final Synthesis ---
    final_answer = ""
    if params.include_synthesis:
        transcript.append("## 🏆 Final Synthesis")
        await ctx.report_progress(step, total_steps)
        step_start = time.time()

        synthesizer = active[0]
        await ctx.info(f"⏳ [{step+1}/{total_steps}] 최종 합성: {synthesizer.emoji} {synthesizer.name} 토론 종합 중...")
        logger.info("Final synthesis...")

        debate_summary = "\n\n".join(
            f"**{e['ai']}** (Round {e['round']}, {e['role']}):\n{e['response']}"
            for e in history
        )
        participant_list = "\n".join(f"- {p.emoji} {p.name}" for p in active)
        synthesis_prompt = f"""## Original Question
{params.question}

## Participating Providers
{participant_list}

## Complete Summit Debate History
{debate_summary}

---
Synthesize this summit. You MUST include a separate summary for each provider listed above."""

        final_answer = await synthesizer.call_fn(
            synthesis_prompt, system_prompt=SYNTHESIS_SYSTEM_PROMPT, model=synthesizer.model
        )
        step_elapsed = time.time() - step_start
        await ctx.info(f"✅ [{step+1}/{total_steps}] 최종 합성 완료 — {synthesizer.emoji} {synthesizer.name} ({step_elapsed:.0f}초)")
        history.append({"ai": synthesizer.name, "round": params.rounds + 1, "role": "synthesizer", "response": final_answer})
        transcript.append(f"\n{final_answer}\n")
        step += 1

    total_elapsed = time.time() - summit_start
    await ctx.report_progress(total_steps, total_steps)
    await ctx.info(f"🏁 Summit 완료! 총 {len(history)}회 교환, 소요 시간: {total_elapsed:.0f}초")

    # --- Store ---
    _summit_store[summit_id] = {
        "id": summit_id,
        "question": params.question,
        "rounds": params.rounds,
        "providers": [p.name for p in active],
        "history": history,
        "final_answer": final_answer,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    result = "\n\n".join(transcript)
    logger.info(f"Summit {summit_id} completed. {len(history)} exchanges across {len(active)} providers.")
    return result


@mcp.tool(
    name="summit_history",
    annotations={
        "title": "Get Summit History",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def summit_history(params: SummitHistoryInput) -> str:
    """Retrieve past summit results from the current session.

    Args:
        params (SummitHistoryInput): Optional summit ID.

    Returns:
        str: Summit history in markdown or JSON.
    """
    if params.summit_id:
        summit = _summit_store.get(params.summit_id)
        if not summit:
            available = ", ".join(_summit_store.keys()) if _summit_store else "none"
            return f"Error: Summit '{params.summit_id}' not found. Available: {available}"
        return json.dumps(summit, indent=2, ensure_ascii=False)

    if not _summit_store:
        return "No summits have been run in this session yet."

    lines = ["# 🏔️ Summit History\n"]
    for sid, s in _summit_store.items():
        providers = ", ".join(s["providers"])
        lines.append(f"- **{sid}** — {s['question'][:80]}... ({s['rounds']} rounds, {providers})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()