# 🏔️ AI Summit

Claude Code에서 **여러 LLM을 교차 검증하며 토론시키는** MCP 서버입니다.

```
질문 → Claude(초기안) → OpenAI(검증) → Gemini(검증) → DeepSeek(검증) → ... → 최종 합성
```

## 지원 모델

| Provider | 기본 모델 | 환경변수 |
|----------|-----------|----------|
| 🟣 Claude | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| 🟢 OpenAI | `o3-mini` | `OPENAI_API_KEY` |
| 🔵 Gemini | `gemini-2.5-pro` | `GEMINI_API_KEY` |
| 🟡 DeepSeek | `deepseek-reasoner` | `DEEPSEEK_API_KEY` |
| 🟠 Mistral | `mistral-large-latest` | `MISTRAL_API_KEY` |

> API 키가 설정된 모델만 자동으로 참여합니다. **새 모델 추가는 `PROVIDERS` dict에 등록하면 끝.**

## 제공 Tools

| Tool | 설명 |
|------|------|
| `summit_run` | 전체 토론 자동 실행 (N라운드 + 합성) |
| `summit_ask` | 특정 LLM에 개별 질문/검증 요청 |
| `summit_providers` | 현재 사용 가능한 LLM 목록 확인 |
| `summit_history` | 현재 세션의 토론 기록 조회 |

## 설치

### 1. 의존성

```bash
cd ai-summit
pip install -e .
```

### 2. API 키

```bash
cp env.example .env
# .env 편집하여 사용할 API 키 입력 (최소 2개)
```

### 3. Claude Code에 등록

```bash
claude mcp add ai-summit -- python /절대경로/ai-summit/server.py
```

또는 `.claude/settings.json`:

```json
{
  "mcpServers": {
    "ai-summit": {
      "command": "python",
      "args": ["/절대경로/ai-summit/server.py"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "GEMINI_API_KEY": "...",
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "DEEPSEEK_API_KEY": "...",
        "MISTRAL_API_KEY": "..."
      }
    }
  }
}
```

### 4. Slash Command (선택)

```bash
mkdir -p .claude/commands
cp ai-summit/claude-commands/summit.md .claude/commands/summit.md
```

## 사용법

### `/summit` 커맨드

```
/summit Redis vs Memcached for session storage in a 10M DAU app?
```

### 자연어 요청

```
이 설계에 대해 다른 AI들 의견도 듣고 싶어. summit_run으로 토론해줘.
```

### 특정 모델만 지정

```
summit_run으로 Claude, Gemini, DeepSeek 3개만 토론시켜줘. OpenAI는 빼고.
```

### 수동 오케스트레이션

Claude Code가 직접 흐름 제어:
```
1. 내가 먼저 답변
2. summit_ask로 OpenAI에 검증 → Gemini에 검증
3. "합의됐으니 여기서 멈추자" 판단
4. 최종 답변 작성
```

## 새 LLM 추가하기

`server.py`에서 2곳만 수정:

```python
# 1. API 호출 함수 추가
async def _call_newmodel(prompt, system_prompt="", model="", max_tokens=4096):
    ...

# 2. PROVIDERS dict에 등록
PROVIDERS["newmodel"] = Provider(
    name="NewModel",
    emoji="🔴",
    env_key="NEWMODEL_API_KEY",
    default_model="newmodel-v1",
    model_env_var="NEWMODEL_MODEL",
    call_fn=_call_newmodel,
)
```

끝! API 키만 설정하면 다음 summit부터 자동 참여합니다.

## 비용 참고

Provider 3개, 2라운드 기준:

- 초기 답변 1회 + 라운드당 3회 × 2 + 합성 1회 = **8회 API 호출**
- Provider 5개, 3라운드: 초기 1 + 라운드당 5 × 3 + 합성 1 = **17회**

간단한 질문은 1라운드, 아키텍처 결정은 2-3라운드 권장.# ai-summit
