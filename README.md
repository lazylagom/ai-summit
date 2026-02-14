# AI Summit

여러 AI CLI를 병렬 토론시켜 합의점과 실행안을 만드는 TypeScript Skill입니다.

- Runtime: TypeScript (`bun`)
- Mode: Skill only
- Not supported: Python/MCP server

## 설치

```bash
# add-skill CLI
npx add-skill lazylagom/ai-summit --skill ai-summit

# skills CLI
npx skills add lazylagom/ai-summit --skill ai-summit
```

설치 후 Codex/Claude Code를 재시작하세요.

## 요구사항

- `bun`
- 아래 CLI 중 2개 이상 설치
  - `claude`
  - `codex`
  - `gemini`
  - `deepseek`
  - `mistral`

## 빠른 시작

```bash
SCRIPT_PATH="${CODEX_HOME:-$HOME/.codex}/skills/ai-summit/scripts/summit.ts"

bun run "$SCRIPT_PATH" \
  --question "이 아키텍처를 검토하고 실행 계획을 제시해줘" \
  --providers claude,codex,gemini \
  --rounds 2
```

JSON 로그 저장:

```bash
SCRIPT_PATH="${CODEX_HOME:-$HOME/.codex}/skills/ai-summit/scripts/summit.ts"

bun run "$SCRIPT_PATH" \
  --question "..." \
  --output json \
  --save-log /tmp/ai-summit-result.json
```

## 옵션

```bash
bun run "$SCRIPT_PATH" --help
```

주요 옵션:

- `--question <text>`: 질문 본문
- `--providers <csv>`: 참여 provider 목록 (`claude,codex,gemini`)
- `--rounds <1-5>`: 토론 라운드 수
- `--consensus-threshold <0-1>`: 조기 종료 임계치
- `--no-early-stop`: 조기 종료 비활성화
- `--output <markdown|json>`: 출력 형식
- `--save-log <path>`: JSON 결과 파일 저장

## 스킬 사용 예시

- `ai-summit으로 이 설계를 다중 모델 검증해줘.`
- `ai-summit으로 2라운드 토론 후 실행 계획까지 정리해줘.`
- `claude,codex,gemini만 써서 리스크 중심으로 비교해줘.`

## 로컬 개발

```bash
bun run summit:help
bun run summit --question "Review this design" --rounds 2
```

## 문제 해결

- `Need at least 2 installed provider CLIs` 오류
  - 최소 2개 provider CLI를 설치하고, 터미널에서 `which <cli>`로 PATH 인식 여부 확인
- `Error (exit ...)` 오류
  - 각 CLI 인증 상태, 모델 옵션, 네트워크 상태를 provider별로 점검

## 프로젝트 구조

- `skills/ai-summit/SKILL.md`: 스킬 본문 지침
- `skills/ai-summit/agents/openai.yaml`: UI 메타데이터
- `skills/ai-summit/scripts/summit.ts`: 오케스트레이터
- `skills/ai-summit/references/cli-usage.md`: 실행 레퍼런스

## 버전

- `v2.x`: TypeScript Skill 전용
