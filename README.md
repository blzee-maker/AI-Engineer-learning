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
| [06](05-prompting/) | Prompting | Few-shot, chain-of-thought and roles System and user tone classification with a local Llama 3.2 |
| [07](Projects/reusable_prompts/) | Reusable Prompts | Prompt-as-data engine: YAML prompt configs (character extraction, dialogue detection, speaker attribution, pronunciation flagging, tone classification) run by one generic loader with JSON schema validation, on local Llama 3.2 |
| [08](05-api-mechanics/) | API Mechanics | Same prompt through a cloud (Gemini 2.5 Flash) and a local (Ollama Llama 3.2) API, with temperature / max_tokens / determinism experiments side by side |

## Points
Combined few-shot+CoT still returned detached on a near-miss passage — diagnosed as the example not matching the test pattern, and reconsidered whether my own label was correct.

Pronunciation flagging passed JSON parsing and schema validation yet still gave a wrong phonetic hint for "Saoirse" ("shur-SHEE-rah" instead of "SEER-sha") — structure ≠ correctness; catching wrong-but-well-formed output needs an eval set with ground-truth labels, not a schema check.

temperature=0 gives deterministic sampling, but not necessarily deterministic output on cloud infrastructure, because floating-point execution order and serving variability can flip near-tied tokens.

## Stack
Python, PyTorch, HuggingFace Transformers, tiktoken, sentence-transformers, Ollama, PyYAML, Gemini API (google-genai), python-dotenv
