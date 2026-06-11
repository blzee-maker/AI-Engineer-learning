# Prompt Design Decisions

Why each prompt in `prompts/` is written the way it is. All prompts share the
audiobook-pipeline framing from `05-prompting` and run on local `llama3.2`
via Ollama at `temperature: 0` (deterministic, reproducible — a classifier
should give the same answer twice).

## The architecture: prompt as data

The prompt lives in a YAML file; `loader.py` is a generic engine that turns
any config into an Ollama chat call. Adding a task means adding a YAML file
and one entry in `schemas.py` — the loader never changes. Each config declares:

| key | what it does |
|-----|--------------|
| `system` | role + task + output contract ("JSON only, these keys") |
| `examples` | few-shot pairs, injected as fake user/assistant turns |
| `prefill` | optional assistant-message seed (usually `"{"`) |
| `valid_keys` | quick documentation of the expected top-level keys |
| `model`, `temperature` | runtime settings, kept with the prompt they were tuned for |

## Two hardening strategies (and why both exist)

1. **Prefill** — end the message list with an assistant turn containing `"{"`.
   The model continues from there, so it can't open with "Sure! Here's your
   JSON:". Caveat: some models *repeat* the prefill instead of continuing
   from it, so the loader checks before stitching it back.
2. **JSON mode** — `format="json"` makes Ollama grammar-constrain the output
   to valid JSON. Stronger guarantee of *parseable* output, but it says
   nothing about the *right* output.

They don't combine well (JSON mode ignores the seeded prefix), so each prompt
picks one. `character_extraction` and `tone_classification` use prefill;
the other three use JSON mode — partly on merit, partly so the repo
demonstrates both techniques.

## Validation: structure ≠ correctness

`json.loads` succeeding only proves the output is JSON.
`{"tone": "banana", "confidence": 7}` parses fine and is still wrong.
So `schemas.py` checks in two layers:

1. **Structural** — required keys exist with the right Python types.
2. **Semantic** — values make sense: tone is one of the five allowed labels,
   confidence is in [0, 1], every list item carries its required keys.

A failed check returns `{"error": ..., "raw": ...}` instead of raising —
the raw output is kept so you can see *what* the model actually said.

## Per-prompt decisions

### character_extraction (few-shot + prefill)
- "Named character who appears **or speaks**" — without that, the model
  drops characters who are mentioned but silent.
- A negative example (`{"characters": []}`) teaches that an empty result is
  legal; otherwise small models invent a character rather than return nothing.
- Items are objects (`{"name": ...}`) not bare strings, so the schema can
  grow later (aliases, gender for voice casting) without breaking callers.

### dialogue_detection (few-shot + JSON mode)
- The trap is narration vs. speech: "he thought about leaving" is not
  dialogue. The system prompt says thoughts/narration don't count.
- Returns both a boolean and the quotes list. The boolean is the cheap
  signal a pipeline branches on; the quotes are the evidence, and they feed
  straight into speaker_attribution.
- Example output deliberately keeps the trailing comma inside the quote
  ("We leave at dawn,") to teach exact copying, not paraphrase.

### speaker_attribution (few-shot + JSON mode)
- Hardest task of the five: tags ("said Mira") are easy, untagged
  turn-taking is not.
- The key design decision is the **"unknown" escape hatch**, taught by its
  own example. Without it, the model guesses a name — and a wrong speaker
  is far worse for an audiobook than an honest unknown.
- "Never guess a name that is not in the passage" pins attribution to the
  text instead of the model's imagination.

### pronunciation_flagging (few-shot + JSON mode)
- The failure mode is over-flagging, so the prompt states the negative rule
  ("common everyday words must NOT be flagged") and includes a
  nothing-to-flag example.
- Each flag carries `word` + `reason` + `hint`. The hint is a plain phonetic
  respelling ("shiv-AWN"), not IPA — the consumer is a human narrator, and
  small models butcher IPA anyway. The reason makes flags auditable.

### tone_classification (closed label set + prefill)
- Ported from `05-prompting` into the config format; the design lesson is
  the **closed label set**. Free-text tone ("melancholic yet hopeful") is
  unusable downstream; five fixed labels make the output checkable, and
  `schemas.py` rejects anything outside the set.
- `confidence` is the model's self-report, not a calibrated probability —
  useful as a relative signal ("review the low-confidence ones"), nothing more.

## Known limitations

- llama3.2 (3B) still misattributes speakers in long untagged exchanges and
  occasionally paraphrases quotes instead of copying them.
- Self-reported confidence values cluster around 0.8–0.95 regardless of
  actual difficulty.
- The schema check catches malformed output, not wrong-but-well-formed
  output (a wrong speaker with valid structure passes). Catching that would
  need an eval set with ground-truth labels — a natural next project.
