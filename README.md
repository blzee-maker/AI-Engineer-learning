# AI Engineer Learning Journey

Building AI systems from scratch to get placed as an AI Engineer.

## Projects
| # | Topic | What I built |
|---|-------|-------------|
| [01](01-tokenization/) | Tokenization | BPE exploration with tiktoken |
| [02](02-embeddings/) | Embeddings | Semantic similarity with sentence-transformers |
| [03](03-transformers/) | Transformers | MiniGPT from scratch in PyTorch |
| [04](04-training/) | Training | MiniGPT training loop with loss curve on Shakespeare text |
| [05](05-bert-vs-gpt/) | BERT vs GPT | Comparing encoder and decoder models with HuggingFace pipelines |
| [06](06-prompting/) | Prompting | Few-shot, chain-of-thought, system/user roles, prefill, and structured-JSON tone classification with a local Llama 3.2 |
| [07](Projects/reusable_prompts/) | Reusable Prompts | Prompt-as-data engine: YAML prompt configs (character extraction, dialogue detection, speaker attribution, pronunciation flagging, tone classification) run by one generic loader with JSON schema validation, on local Llama 3.2 |
| [08](07-api-mechanics/) | API Mechanics | Same prompt through a cloud (Gemini 2.5 Flash) and a local (Ollama Llama 3.2) API with temperature / max_tokens / determinism experiments, plus tool use / function calling on both, where the model decides when to call a dialogue-counting tool vs. answer directly |

## Points
**[06-prompting/CoT.py](06-prompting/CoT.py)** — Combined few-shot+CoT still returned detached on a near-miss passage — diagnosed as the example not matching the test pattern, and reconsidered whether my own label was correct.

**[Projects/reusable_prompts/](Projects/reusable_prompts/)** — Pronunciation flagging passed JSON parsing and schema validation yet still gave a wrong phonetic hint for "Saoirse" ("shur-SHEE-rah" instead of "SEER-sha") — structure ≠ correctness; catching wrong-but-well-formed output needs an eval set with ground-truth labels, not a schema check.

**[07-api-mechanics/api-mechanics.py](07-api-mechanics/api-mechanics.py)** — temperature=0 gives deterministic sampling, but not necessarily deterministic output on cloud infrastructure, because floating-point execution order and serving variability can flip near-tied tokens.

**[07-api-mechanics/tool-use-llama.py](07-api-mechanics/tool-use-llama.py)** — Passing bulk source text to a tool as a model-supplied argument corrupts it — the model retypes the chapter and silently drops opening quotes, so `count_dialogue_lines` returned 2 instead of 4. Fix: pass identifiers through the model, not data; have the tool read the authoritative text locally (the in-scope `CHAPTER`), so the model can never mangle the payload.

**[07-api-mechanics/tool-use.py](07-api-mechanics/tool-use.py)** — Letting the model fall back to its own knowledge is just function-calling mode: AUTO (the default) lets it skip the tool and answer directly, while ANY forces a call every time. Making that fallback reliable took two fixes — scan *every* response part for the `function_call` (the model can return a text part alongside the call, so `parts[0]` alone misses it), and a system instruction telling it to only use the tool for dialogue-counting and answer everything else directly.

## Stack
Python, PyTorch, HuggingFace Transformers, tiktoken, sentence-transformers, Ollama, PyYAML, Gemini API (google-genai), python-dotenv
