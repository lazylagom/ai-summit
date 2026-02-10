# 🏔️ AI Summit

Claude Code에서 **여러 LLM CLI를 교차 검증하며 토론시키는** MCP 서버입니다.

```
질문 → Claude(초기안) → [Codex + Gemini + Claude 병렬 리뷰] × N라운드 → 최종 합성
```

## 지원 모델

| Provider | CLI 명령어 | 환경변수 (모델 오버라이드) |
|----------|-----------|--------------------------|
| 🟣 Claude | `claude` | `CLAUDE_MODEL` |
| 🟢 Codex | `codex` | `CODEX_MODEL` |
| 🔵 Gemini | `gemini` | `GEMINI_MODEL` |
| 🟡 DeepSeek | `deepseek` | `DEEPSEEK_MODEL` |
| 🟠 Mistral | `mistral` | `MISTRAL_MODEL` |

> CLI가 설치된 모델만 자동으로 참여합니다. API 키 불필요. **새 모델 추가는 `PROVIDERS` dict에 등록하면 끝.**

## 제공 Tools

| Tool | 설명 |
|------|------|
| `summit_run` | 전체 토론 자동 실행 (병렬 N라운드 + 합성) |
| `summit_ask` | 특정 LLM에 개별 질문/검증 요청 |
| `summit_providers` | 현재 사용 가능한 LLM 목록 확인 |
| `summit_history` | 현재 세션의 토론 기록 조회 |

## 설치

### 1. 의존성

```bash
cd ai-summit
pip install -e .
```

### 2. CLI 도구 설치

최소 2개 이상의 CLI 도구가 필요합니다:

```bash
# Claude Code (https://docs.anthropic.com/en/docs/claude-code)
# Codex (https://github.com/openai/codex)
# Gemini CLI (https://github.com/google-gemini/gemini-cli)
```

### 3. Claude Code에 등록

```bash
# 글로벌 등록 (모든 프로젝트에서 사용)
claude mcp add --scope user ai-summit -- python /절대경로/ai-summit/server.py

# 프로젝트 등록 (현재 프로젝트에서만 사용)
claude mcp add ai-summit -- python /절대경로/ai-summit/server.py
```

## 사용법

### 자연어 요청

```
이 설계에 대해 다른 AI들 의견도 듣고 싶어. summit_run으로 토론해줘.
```

### 특정 모델만 지정

```
summit_run으로 Claude, Gemini 2개만 토론시켜줘.
```

### 수동 오케스트레이션

Claude Code가 직접 흐름 제어:
```
1. 내가 먼저 답변
2. summit_ask로 Codex에 검증 → Gemini에 검증
3. "합의됐으니 여기서 멈추자" 판단
4. 최종 답변 작성
```

## 병렬 실행

라운드 내 모든 리뷰어가 **동시에** 실행됩니다:

```
Round 0: Claude 초기안 생성          ← 순차
Round 1: Claude + Codex + Gemini     ← 병렬 (asyncio.gather)
Round 2: Claude + Codex + Gemini     ← 병렬
Synthesis: Claude 최종 합성          ← 순차
```

3개 프로바이더 + 2라운드 기준: 순차 8단계 → **병렬 4단계** (약 2배 빠름)

실행 중 MCP 진행 상황 알림으로 현재 누가 무엇을 하는지 확인할 수 있습니다.

## 새 CLI 프로바이더 추가하기

`server.py`에서 2곳만 수정:

```python
# 1. CLI 호출 함수 추가
async def _call_newmodel_cli(prompt, system_prompt="", model="", max_tokens=4096):
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    cmd = ["newmodel"]
    if model:
        cmd.extend(["-m", model])
    return await _run_cli(cmd, input_data=full_prompt.encode())

# 2. PROVIDERS dict에 등록
PROVIDERS["newmodel"] = Provider(
    name="NewModel",
    emoji="🔴",
    cli_cmd="newmodel",
    call_fn=_call_newmodel_cli,
    model_env_var="NEWMODEL_MODEL",
)
```

끝! CLI만 설치하면 다음 summit부터 자동 참여합니다.

## 호출 횟수 참고

Provider 3개, 2라운드 기준:

- 초기 답변 1회 + 라운드당 3회(병렬) × 2 + 합성 1회 = **8회 CLI 호출 (벽시계 4단계)**
- Provider 5개, 3라운드: 초기 1 + 라운드당 5(병렬) × 3 + 합성 1 = **17회 (벽시계 5단계)**

간단한 질문은 1라운드, 아키텍처 결정은 2-3라운드 권장.
