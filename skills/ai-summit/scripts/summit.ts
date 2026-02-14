#!/usr/bin/env bun

import { spawn, spawnSync } from "node:child_process";
import { writeFile } from "node:fs/promises";

const DEFAULT_ROUNDS = 2;
const MAX_ROUNDS = 5;
const DEFAULT_CONSENSUS_THRESHOLD = 0.75;
const DEFAULT_OUTPUT = "markdown";
const DEFAULT_TIMEOUT_MS = 300_000;
const CLAUDE_TIMEOUT_MS = 600_000;

type OutputMode = "markdown" | "json";

const PROVIDER_KEYS = ["claude", "codex", "gemini", "deepseek", "mistral"] as const;
type ProviderKey = (typeof PROVIDER_KEYS)[number];

interface Invocation {
  args: string[];
  stdinText?: string;
}

interface ProviderSpec {
  key: ProviderKey;
  name: string;
  emoji: string;
  command: string;
  modelEnvVar?: string;
  timeoutMs: number;
  buildInvocation: (prompt: string, systemPrompt: string, model: string) => Invocation;
}

interface CliOptions {
  question: string;
  requestedProviders?: ProviderKey[];
  rounds: number;
  consensusThreshold: number;
  earlyStop: boolean;
  output: OutputMode;
  saveLogPath?: string;
}

interface ReviewLog {
  provider: ProviderKey;
  response: string;
  isError: boolean;
}

interface RoundLog {
  round: number;
  consensus: number;
  reviews: ReviewLog[];
}

interface SummitResult {
  question: string;
  providers: ProviderKey[];
  leadProvider: ProviderKey;
  roundsRequested: number;
  roundsExecuted: number;
  stoppedEarly: boolean;
  consensusThreshold: number;
  finalConsensus: number;
  timestamp: string;
  finalAnswer: string;
  roundLogs: RoundLog[];
}

const AGREEMENT_HINTS = [
  "agree",
  "agreed",
  "consensus",
  "aligned",
  "same conclusion",
  "동의",
  "합의",
  "일치",
];

const DISAGREEMENT_HINTS = [
  "strongly disagree",
  "cannot endorse",
  "critical flaw",
  "fundamentally different",
  "major concern",
  "동의하기 어렵",
  "치명적 문제",
  "근본적으로 다름",
];

const PROVIDERS: Record<ProviderKey, ProviderSpec> = {
  claude: {
    key: "claude",
    name: "Claude",
    emoji: "🟣",
    command: "claude",
    modelEnvVar: "CLAUDE_MODEL",
    timeoutMs: CLAUDE_TIMEOUT_MS,
    buildInvocation(prompt, systemPrompt, model) {
      const args = ["-p", "--no-session-persistence", "--max-turns", "3"];
      if (systemPrompt) {
        args.push("--system-prompt", systemPrompt);
      }
      if (model) {
        args.push("--model", model);
      }
      return { args, stdinText: prompt };
    },
  },
  codex: {
    key: "codex",
    name: "Codex",
    emoji: "🟢",
    command: "codex",
    modelEnvVar: "CODEX_MODEL",
    timeoutMs: DEFAULT_TIMEOUT_MS,
    buildInvocation(prompt, systemPrompt, model) {
      const fullPrompt = systemPrompt ? `${systemPrompt}\n\n${prompt}` : prompt;
      const args = ["exec"];
      if (model) {
        args.push("-m", model);
      }
      return { args, stdinText: fullPrompt };
    },
  },
  gemini: {
    key: "gemini",
    name: "Gemini",
    emoji: "🔵",
    command: "gemini",
    modelEnvVar: "GEMINI_MODEL",
    timeoutMs: DEFAULT_TIMEOUT_MS,
    buildInvocation(prompt, systemPrompt, model) {
      const fullPrompt = systemPrompt ? `${systemPrompt}\n\n${prompt}` : prompt;
      const args = ["-p", fullPrompt];
      if (model) {
        args.push("-m", model);
      }
      return { args };
    },
  },
  deepseek: {
    key: "deepseek",
    name: "DeepSeek",
    emoji: "🟡",
    command: "deepseek",
    modelEnvVar: "DEEPSEEK_MODEL",
    timeoutMs: DEFAULT_TIMEOUT_MS,
    buildInvocation(prompt, systemPrompt, model) {
      const fullPrompt = systemPrompt ? `${systemPrompt}\n\n${prompt}` : prompt;
      const args: string[] = [];
      if (model) {
        args.push("-m", model);
      }
      return { args, stdinText: fullPrompt };
    },
  },
  mistral: {
    key: "mistral",
    name: "Mistral",
    emoji: "🟠",
    command: "mistral",
    modelEnvVar: "MISTRAL_MODEL",
    timeoutMs: DEFAULT_TIMEOUT_MS,
    buildInvocation(prompt, systemPrompt, model) {
      const fullPrompt = systemPrompt ? `${systemPrompt}\n\n${prompt}` : prompt;
      const args: string[] = [];
      if (model) {
        args.push("-m", model);
      }
      return { args, stdinText: fullPrompt };
    },
  },
};

function printHelp(): void {
  const help = [
    "AI Summit CLI (TypeScript)",
    "",
    "Usage:",
    "  bun run skills/ai-summit/scripts/summit.ts --question \"...\" [options]",
    "",
    "Options:",
    "  -q, --question <text>            Main question (or provide via stdin)",
    "  -p, --providers <csv>            e.g. claude,codex,gemini",
    "  -r, --rounds <1-5>               Debate rounds (default: 2)",
    "      --consensus-threshold <0-1>  Early-stop threshold (default: 0.75)",
    "      --no-early-stop              Disable early-stop",
    "  -o, --output <markdown|json>     Output format (default: markdown)",
    "      --save-log <path>            Save full JSON result to a file",
    "  -h, --help                       Show this help",
    "",
    "Example:",
    "  bun run skills/ai-summit/scripts/summit.ts \\",
    "    --question \"Review this architecture and propose an implementation plan\" \\",
    "    --providers claude,codex,gemini --rounds 2",
  ];
  console.log(help.join("\n"));
}

function parseProviderList(raw: string): ProviderKey[] {
  const keys = raw
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter((item) => item.length > 0);

  const unknown = keys.filter((item) => !PROVIDER_KEYS.includes(item as ProviderKey));
  if (unknown.length > 0) {
    throw new Error(`Unknown provider(s): ${unknown.join(", ")}`);
  }

  return unique(keys as ProviderKey[]);
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function parseNumber(raw: string, label: string): number {
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Invalid ${label}: ${raw}`);
  }
  return parsed;
}

async function readStdinText(): Promise<string> {
  if (process.stdin.isTTY) {
    return "";
  }

  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    process.stdin.on("data", (chunk: Buffer) => chunks.push(chunk));
    process.stdin.on("error", reject);
    process.stdin.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
  });
}

async function parseCliOptions(argv: string[]): Promise<CliOptions> {
  let question = "";
  let requestedProviders: ProviderKey[] | undefined;
  let rounds = DEFAULT_ROUNDS;
  let consensusThreshold = DEFAULT_CONSENSUS_THRESHOLD;
  let earlyStop = true;
  let output: OutputMode = DEFAULT_OUTPUT;
  let saveLogPath: string | undefined;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];

    const takeValue = (flag: string): string => {
      const value = argv[i + 1];
      if (!value || value.startsWith("-")) {
        throw new Error(`Missing value for ${flag}`);
      }
      i += 1;
      return value;
    };

    if (arg === "-h" || arg === "--help") {
      printHelp();
      process.exit(0);
    }

    if (arg === "-q" || arg === "--question") {
      question = takeValue(arg);
      continue;
    }

    if (arg === "-p" || arg === "--providers") {
      requestedProviders = parseProviderList(takeValue(arg));
      continue;
    }

    if (arg === "-r" || arg === "--rounds") {
      rounds = Math.trunc(parseNumber(takeValue(arg), "rounds"));
      continue;
    }

    if (arg === "--consensus-threshold") {
      consensusThreshold = parseNumber(takeValue(arg), "consensus-threshold");
      continue;
    }

    if (arg === "--no-early-stop") {
      earlyStop = false;
      continue;
    }

    if (arg === "-o" || arg === "--output") {
      const value = takeValue(arg).toLowerCase();
      if (value !== "markdown" && value !== "json") {
        throw new Error(`Invalid output mode: ${value}`);
      }
      output = value;
      continue;
    }

    if (arg === "--save-log") {
      saveLogPath = takeValue(arg);
      continue;
    }

    if (arg.startsWith("-")) {
      throw new Error(`Unknown option: ${arg}`);
    }

    question = `${question} ${arg}`.trim();
  }

  if (!question.trim()) {
    question = (await readStdinText()).trim();
  }

  if (!question.trim()) {
    throw new Error("Question is required. Pass --question or pipe text via stdin.");
  }

  if (rounds < 1 || rounds > MAX_ROUNDS) {
    throw new Error(`Rounds must be between 1 and ${MAX_ROUNDS}.`);
  }

  if (consensusThreshold < 0 || consensusThreshold > 1) {
    throw new Error("Consensus threshold must be between 0 and 1.");
  }

  return {
    question,
    requestedProviders,
    rounds,
    consensusThreshold,
    earlyStop,
    output,
    saveLogPath,
  };
}

function isBinaryAvailable(binary: string): boolean {
  const check = spawnSync("which", [binary], { stdio: "ignore" });
  return check.status === 0;
}

function resolveActiveProviders(requested?: ProviderKey[]): ProviderSpec[] {
  const keys = requested && requested.length > 0 ? requested : [...PROVIDER_KEYS];

  const available = keys
    .map((key) => PROVIDERS[key])
    .filter((provider) => isBinaryAvailable(provider.command));

  if (available.length < 2) {
    const installed = available.map((provider) => provider.key).join(", ") || "none";
    throw new Error(
      `Need at least 2 installed provider CLIs. Detected: ${installed}. Supported: ${PROVIDER_KEYS.join(", ")}`,
    );
  }

  return available;
}

async function runCommand(
  command: string,
  args: string[],
  stdinText: string | undefined,
  timeoutMs: number,
): Promise<string> {
  return new Promise((resolve) => {
    const child = spawn(command, args, {
      stdio: [stdinText ? "pipe" : "ignore", "pipe", "pipe"],
      env: process.env,
    });

    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];
    let didTimeout = false;

    const timer = setTimeout(() => {
      didTimeout = true;
      child.kill("SIGKILL");
    }, timeoutMs);

    child.stdout.on("data", (chunk: Buffer) => stdoutChunks.push(chunk));
    child.stderr.on("data", (chunk: Buffer) => stderrChunks.push(chunk));

    child.on("error", (error: Error) => {
      clearTimeout(timer);
      resolve(`Error: failed to run ${command}: ${error.message}`);
    });

    child.on("close", (code: number | null) => {
      clearTimeout(timer);

      if (didTimeout) {
        resolve(`Error: CLI timed out after ${Math.round(timeoutMs / 1000)}s.`);
        return;
      }

      const stdoutText = Buffer.concat(stdoutChunks).toString("utf8").trim();
      const stderrText = Buffer.concat(stderrChunks).toString("utf8").trim();

      if (code !== 0) {
        const detail = (stderrText || stdoutText || "Unknown CLI error").slice(0, 500);
        resolve(`Error (exit ${code ?? "?"}): ${detail}`);
        return;
      }

      resolve(stdoutText);
    });

    if (stdinText && child.stdin) {
      child.stdin.write(stdinText);
      child.stdin.end();
    }
  });
}

function modelFor(provider: ProviderSpec): string {
  if (!provider.modelEnvVar) {
    return "";
  }
  return (process.env[provider.modelEnvVar] || "").trim();
}

async function callProvider(provider: ProviderSpec, prompt: string, systemPrompt = ""): Promise<string> {
  const invocation = provider.buildInvocation(prompt, systemPrompt, modelFor(provider));
  return runCommand(provider.command, invocation.args, invocation.stdinText, provider.timeoutMs);
}

function buildInitialPrompt(question: string): string {
  return [
    "You are preparing the first draft answer.",
    "Provide a structured response with clear assumptions and implementation details.",
    "",
    `Question: ${question}`,
  ].join("\n");
}

function buildReviewPrompt(question: string, currentDraft: string, round: number): string {
  return [
    `Round ${round} peer review request`,
    "Review the draft critically.",
    "Include:",
    "1) What you agree with",
    "2) What you disagree with",
    "3) Missing facts or verification points (mark unknowns as [Unverified])",
    "4) Concrete improvement suggestions",
    "",
    `Question: ${question}`,
    "",
    "Current draft:",
    currentDraft,
  ].join("\n");
}

function buildRevisionPrompt(
  question: string,
  currentDraft: string,
  reviews: ReviewLog[],
  round: number,
): string {
  const reviewText = reviews
    .map((entry) => `### ${entry.provider}\n${entry.response}`)
    .join("\n\n");

  return [
    `You are revising the working draft after round ${round}.`,
    "Update the draft using the peer reviews.",
    "Keep useful parts, fix weak points, and make the answer more actionable.",
    "",
    `Question: ${question}`,
    "",
    "Current draft:",
    currentDraft,
    "",
    "Peer reviews:",
    reviewText,
  ].join("\n");
}

function buildFinalPrompt(question: string, roundLogs: RoundLog[]): string {
  const logs = roundLogs
    .map((round) => {
      const reviews = round.reviews
        .map((review) => `- ${review.provider}: ${review.response}`)
        .join("\n");
      return `Round ${round.round} (consensus=${round.consensus.toFixed(2)})\n${reviews}`;
    })
    .join("\n\n");

  return [
    "Synthesize the final answer from the full debate history.",
    "Respond in the same language as the question.",
    "Use the following structure:",
    "- Consensus",
    "- Open Issues",
    "- Recommended Plan (numbered steps)",
    "- Risks and Limitations",
    "",
    `Question: ${question}`,
    "",
    "Debate history:",
    logs,
  ].join("\n");
}

function hasAny(text: string, patterns: string[]): boolean {
  const lowered = text.toLowerCase();
  return patterns.some((pattern) => lowered.includes(pattern.toLowerCase()));
}

function isErrorResponse(text: string): boolean {
  return text.trim().toLowerCase().startsWith("error");
}

function calculateConsensus(reviews: ReviewLog[]): number {
  const valid = reviews.filter((review) => !review.isError);
  if (valid.length === 0) {
    return 0;
  }

  let agree = 0;
  let disagree = 0;

  for (const review of valid) {
    if (hasAny(review.response, AGREEMENT_HINTS)) {
      agree += 1;
    }
    if (hasAny(review.response, DISAGREEMENT_HINTS)) {
      disagree += 1;
    }
  }

  if (agree === 0 && disagree === 0) {
    return 0.5;
  }

  return agree / (agree + disagree);
}

async function runSummit(options: CliOptions): Promise<SummitResult> {
  const activeProviders = resolveActiveProviders(options.requestedProviders);
  const providerKeys = activeProviders.map((provider) => provider.key);
  const leadProvider = activeProviders[0];

  let currentDraft = await callProvider(
    leadProvider,
    buildInitialPrompt(options.question),
    "You are an expert assistant producing a high-quality initial draft.",
  );

  if (isErrorResponse(currentDraft)) {
    throw new Error(`Lead provider failed on initial draft: ${currentDraft}`);
  }

  const roundLogs: RoundLog[] = [];
  let stoppedEarly = false;
  let finalConsensus = 0;

  for (let round = 1; round <= options.rounds; round += 1) {
    const reviewPromises = activeProviders.map(async (provider) => {
      const response = await callProvider(
        provider,
        buildReviewPrompt(options.question, currentDraft, round),
        "You are a strict reviewer. Prioritize factual rigor and executable recommendations.",
      );

      return {
        provider: provider.key,
        response,
        isError: isErrorResponse(response),
      } satisfies ReviewLog;
    });

    const reviews = await Promise.all(reviewPromises);
    finalConsensus = calculateConsensus(reviews);

    roundLogs.push({
      round,
      consensus: finalConsensus,
      reviews,
    });

    if (options.earlyStop && finalConsensus >= options.consensusThreshold) {
      stoppedEarly = true;
      break;
    }

    const revisedDraft = await callProvider(
      leadProvider,
      buildRevisionPrompt(options.question, currentDraft, reviews, round),
      "You are the synthesis lead. Produce a cleaner and stronger next draft.",
    );

    if (!isErrorResponse(revisedDraft)) {
      currentDraft = revisedDraft;
    }
  }

  const finalAnswer = await callProvider(
    leadProvider,
    buildFinalPrompt(options.question, roundLogs),
    "You are the final synthesizer. Output a practical, decision-ready result.",
  );

  if (isErrorResponse(finalAnswer)) {
    throw new Error(`Lead provider failed on final synthesis: ${finalAnswer}`);
  }

  return {
    question: options.question,
    providers: providerKeys,
    leadProvider: leadProvider.key,
    roundsRequested: options.rounds,
    roundsExecuted: roundLogs.length,
    stoppedEarly,
    consensusThreshold: options.consensusThreshold,
    finalConsensus,
    timestamp: new Date().toISOString(),
    finalAnswer,
    roundLogs,
  };
}

function formatMarkdown(result: SummitResult): string {
  const providerSummary = result.providers.map((key) => `${PROVIDERS[key].emoji} ${key}`).join(", ");

  const lines = [
    "# AI Summit Result",
    "",
    `- Lead provider: ${result.leadProvider}`,
    `- Providers: ${providerSummary}`,
    `- Rounds: ${result.roundsExecuted}/${result.roundsRequested}`,
    `- Early stop: ${result.stoppedEarly ? "yes" : "no"}`,
    `- Final consensus: ${result.finalConsensus.toFixed(2)} (threshold: ${result.consensusThreshold.toFixed(2)})`,
    "",
    "## Final Answer",
    "",
    result.finalAnswer,
    "",
    "## Consensus by Round",
    ...result.roundLogs.map((round) => `- Round ${round.round}: ${round.consensus.toFixed(2)}`),
  ];

  return lines.join("\n");
}

async function main(): Promise<void> {
  const options = await parseCliOptions(process.argv.slice(2));
  const result = await runSummit(options);

  if (options.saveLogPath) {
    await writeFile(options.saveLogPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  }

  if (options.output === "json") {
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  console.log(formatMarkdown(result));
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Error: ${message}`);
  process.exit(1);
});
