Run an AI Summit on the user's question — a multi-LLM debate where AI models cross-validate and refine each other's solutions.

## Instructions

1. First call `summit_providers` to see which LLMs are available
2. Call `summit_run` with the user's question: "$ARGUMENTS"
   - Default 2 rounds for most questions
   - Use 3 rounds for complex architecture or design decisions
   - Optionally specify `providers` to control which LLMs participate
3. Review the final synthesis from the summit transcript
4. If the synthesis includes code, implement it in the project
5. Summarize what each AI contributed and what the final consensus was

## Manual orchestration (for more control)

Use `summit_ask` to call individual LLMs:

1. Form your own answer first
2. Send it to OpenAI: `summit_ask(provider="openai", question=..., context=your_answer, context_ai_name="Claude")`
3. Send OpenAI's review to Gemini: `summit_ask(provider="gemini", ...)`
4. Continue with any other available providers
5. Synthesize everything yourself

This gives you full control — you can stop early when consensus is reached,
or drill deeper into specific points of disagreement.