# Running ATM-Bench with Harbor

ATM-Bench-Hard (schema-guided memory) is published as a
[Harbor](https://github.com/harbor-framework/harbor) dataset, so you can
benchmark any Harbor-supported agent without setting up the `agent_systems/`
harness.

The two paths measure the same thing and use the same scoring code. Pick one:

| | `agent_systems/` | Harbor |
|---|---|---|
| isolation | bwrap sandbox on the host | one Docker container per question |
| agents | the five CLIs wired up in this repo | any Harbor agent |
| setup | install + authenticate each CLI locally | Docker + the `harbor` CLI |
| scope | full benchmark, all memory modes | ATM-Bench-Hard, SGM only (31 questions) |

Use `agent_systems/` for the full benchmark or to reproduce the published
numbers. Use Harbor to evaluate an agent quickly, or to run somewhere Docker is
easier to get than a bubblewrap sandbox.

## Quickstart

Install the [Harbor CLI](https://www.harborframework.com/docs), make sure Docker
is running, then:

```bash
harbor run -d atm-bench/atm-bench-hard-sgm -a claude-code -m claude-opus-4-8
```

That pulls the 31 tasks, runs each in its own container, scores them, and prints
a mean reward. Nothing else to download — each task carries its own memory store.

Swap the agent and model freely:

```bash
harbor run -d atm-bench/atm-bench-hard-sgm -a codex    -m gpt-5.5
harbor run -d atm-bench/atm-bench-hard-sgm -a opencode -m <provider>/<model>
harbor run -d atm-bench/atm-bench-hard-sgm -a pi       -m <provider>/<model>
```

### Judge key

13 of the 31 questions are `open_end` and are graded by an LLM judge, so the
verifier needs an OpenAI key. Without one those questions score 0:

```bash
harbor run -d atm-bench/atm-bench-hard-sgm -a codex -m gpt-5.5 \
  --ve OPENAI_API_KEY="$OPENAI_API_KEY"
```

`--ve` passes the variable to the **verifier**, not the agent — the agent never
sees the judge key.

### Useful flags

```bash
-n 1                          # serial; use this if your endpoint dislikes concurrency
-k 3                          # 3 attempts per task
-l 5                          # first 5 tasks only (smoke test)
-i atm-bench-hard-sgm-0011    # one task; note the bare name, no org prefix
-o jobs/ --job-name my-run    # where results land
```

## What a task looks like

The agent starts in `/app` with the question and that question's memory store
under `/app/data/`:

```text
/app/data/question.txt
/app/data/question.json
/app/data/memory/{emails,image_metadata,video_metadata,memory_variant}.json
/app/data/prompts/{qa_schema.json,system_prompt.txt}
```

It writes its answer to `/logs/artifacts/answer.json`:

```json
{
  "id": "question-id",
  "question": "question text",
  "answer": "natural language answer"
}
```

For `list_recall` questions the answer must be a **comma-separated string**, not
a JSON array — the scorer splits on commas, so an array scores 0.

## Scoring

Same routing as this repo's evaluator, running the same vendored code:

| qtype | n | metric |
|---|---|---|
| `number` | 6 | exact match, 0/1 |
| `list_recall` | 12 | Jaccard over the comma-separated items, [0,1] |
| `open_end` | 13 | LLM judge (`gpt-5-mini`), 0/1 |

Headline score is the mean over all 31 questions. `number` and `list_recall` are
deterministic and need no API key.

The verifier runs in a **separate container** from the agent, with the gold
answer baked into the verifier image only. The agent cannot read it — Harbor's
default shared-verifier mode would expose it, so these tasks opt out.

## Scope and caveats

- SGM (text) memory only. Raw image/video memory is not packaged for Harbor;
  use `agent_systems/` for that.
- ATM-Bench-Hard only — 31 questions, the hard split.
- Agent CLI versions differ between the two paths. Harbor installs the current
  release of each agent; `agent_systems/runner_versions.md` records the versions
  behind the published numbers. Compare within a path, not across.
- One known issue is inherited from this repo: `prompts/system_prompt.txt`
  illustrates the `list_recall` format with three real image IDs that happen to
  be the first three gold items for one question. It affects both paths equally.

## See also

- Dataset: <https://huggingface.co/datasets/Jingbiao/ATM-Bench>
- Harbor docs: <https://www.harborframework.com/docs>
- `agent_systems/README.md` — the native harness
