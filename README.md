**English** | [Русский](README.ru.md)

# Human-like continual learning for a small language model

The main result of the project is **a small model that keeps learning**: Qwen2.5-0.5B-Instruct with LoRA. It learns in portions, from easy to hard, replays old examples, learns from "surprise" and periodically "sleeps". You can run it and teach it with new data or in a chat. It saves its state and continues where it stopped, and it exports as an ordinary Hugging Face model.

**The final model is on Hugging Face: [makltd1/continual-lm](https://huggingface.co/makltd1/continual-lm)** (an ordinary `transformers` model, Apache-2.0).

The control experiments E0–E6 check which techniques actually reduce forgetting (hypotheses H1–H4). The literature review is in [reports/research.md](reports/research.md) (in Russian).

This is a research project, not a product: small model, English-only educational tasks, one training run for the final model.

## Results

Details are in [reports/conclusions.md](reports/conclusions.md) (in Russian); tables and plots are in [reports/summary.md](reports/summary.md). 29 runs on Kaggle, three seeds per configuration. The raw run logs (`runs/`) are not published in this repository; the reports are built from them.

| Hypothesis | Verdict |
|---|---|
| H1: replay reduces forgetting | **Confirmed** for stories (11% of the gain lost vs 28%) and passage QA (5% vs 19%); not enough data for arithmetic |
| H2: learning from surprise gives the same quality in fewer steps | **Partly**: 31% fewer steps and the same quality on short-answer tasks, slightly worse on text tasks; not faster in wall time |
| H3: easy-to-hard speeds up learning | **Not confirmed**: slower (137 steps to target vs 78) |
| H4: sleep keeps old tasks | **Not enough data**: no visible effect |

Training on narrow tasks always wears away instruction following (task 0): 81–88% of the original model remains. Replaying the model's own answers to general requests (self-replay) fixes this, bringing it to 94–95%.

**The final model is E5_final_v2**: all techniques plus replay of 2000 of the model's own answers. Task 0 stays at 95% of the original model, so the spec's criterion (≥ 90%) is met. On its own tasks it beats the original: passage QA 0.60 vs 0.37, arithmetic 0.80 vs 0.41. Perplexity on general text rose by ~7%.

### Download and run

The model is published as an ordinary Hugging Face model; this project's code is not needed to run it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("makltd1/continual-lm")
model = AutoModelForCausalLM.from_pretrained("makltd1/continual-lm")
messages = [{"role": "user", "content": "What is 17 + 25? Reply with the answer only."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

It runs on a CPU: an answer takes from a few seconds to half a minute.

The **learner state** (adapters, replay buffer, counters) that `chat.py` and `live.py` continue from is **not published**, only the exported weights. To teach the model with your corrections or your own data you need a state, and you create one by training it yourself with the command in the next section (a 16 GB GPU, see "Resources"). Then:

1. Chat with learning from your corrections: `python chat.py --state states/ewai-e5`. With little RAM (8 GB), add `--low-memory` to halve the model's memory. Write each message on one line.
2. Keep training it on your own data: `python live.py --state states/ewai-e5 --inbox inbox`.

## The final model (E5)

The final model is configuration E5, all techniques together, at the spec's full sizes: [configs/final/E5_final.yaml](configs/final/E5_final.yaml). The continuous mode itself trains it:

```bash
# Learn the task stream A→B→C→D; after a stop it continues from the last portion
python live.py --config configs/final/E5_final.yaml --state states/ewai-e5 --stream builtin \
               --export exported/ewai-e5
```

| Need | How |
|---|---|
| Run it without this project's code (FR-15) | `exported/ewai-e5` is an ordinary Hugging Face model; its `README.md` has an example. Export any state with `python export.py --state states/ewai-e5 --out exported/ewai-e5` |
| Keep teaching it new data (FR-13) | `python live.py --state states/ewai-e5 --inbox inbox`. Drop `.jsonl` files into `inbox/`, one example per line: `{"prompt": "request", "response": "the answer it should give"}`. Every `live.portion_size` examples (32 by default) form a portion: selection, replay, update, sleep when due |
| Chat and learn from feedback (FR-16) | `python chat.py --state states/ewai-e5`. After an answer, `/good` learns that answer, and `/fix <right answer>` learns the right answer plus a bad→good pair for DPO. Learning runs every 8 pieces of feedback or on `/learn` |
| Save and continue (FR-14) | After every portion the state is written to the state folder: adapters, buffer, counters, random state. Running the same command again continues where it was. In the chat: `/save`, `/quit` |
| Checkpoints and rollback (FR-17) | `python live.py --state states/ewai-e5 --list` and `--rollback ckpt-000012`. The newest 3 are kept (`checkpoint.keep`) |

The spec's acceptance criterion: instruction following (task 0) of the final model stays at or above 90% of the original Instruct model (`eval.retention_min`). The report checks it automatically.

**Replay of general instructions.** The very first E0 run lost ~18% of its instruction following, because the model gets used to the narrow tasks. Replay in E1–E5 covers tasks A–D only, so the final model also replays the original model's own answers to 300 general requests. The requests come from Dolly; people wrote them, and they do not overlap with the task 0 test. This is the self-generated replay from the literature review (source 3). In the config it is `self_replay`, and the control run `configs/extra/E5_selfreplay.yaml` measures its effect.

What to expect: this is a small model trained further on four narrow tasks. It writes short stories, answers questions about a passage, does simple arithmetic, writes ML paper abstracts and follows simple instructions. It will not converse like a large model. The chat answers each message on its own, without the earlier messages, because the training examples are single turns too.

## Experiments E0–E6

| Config | What is on | Tests |
|---|---|---|
| E0 | Sequential fine-tuning, no protection | Baseline forgetting |
| E1 | E0 + replay (30%) | H1: replay reduces forgetting |
| E2 | E1 + selection by surprise | H2: same quality in fewer steps |
| E3 | E1 + curriculum | H3: easy-to-hard speeds up learning |
| E4 | E1 + sleep phase | H4: sleep keeps old tasks |
| E5 | Everything together | Combined effect (the final model is the same at full sizes) |
| E6 | Joint training on all tasks | Upper bound on quality |

Each configuration is a separate training run from the original model, on three seeds.

**Tasks.** Every task is a dialogue in the model's chat template, a user request and the assistant's answer. The loss is computed on the answer only.

- **A**: "Write a short story for young children that uses the words …" → a TinyStories story.
- **B**: a question about a passage (SQuAD) → a short answer.
- **C**: arithmetic and logic (generated) → the answer.
- **D**: "Write the abstract of a machine learning research paper titled …" → an arXiv abstract.

Each of A–D has 5000 training, 625 validation and 625 test examples. Two more tasks are measured but never trained:

- **T0**, task 0, instruction following: 152 instructions of 17 types with automatically checked conditions (format, length, language, keywords). It works like IFEval, but is shorter.
- **G**, the general language test: perplexity on WikiText-2.

**Control base.** The Instruct model itself is measured before any training, and the report shows every configuration's difference from it.

**Reduced workload.** The workload fits free GPU quotas, as the spec's risk table allows ("smaller portions"):
- a portion is 500 examples instead of 1000, with 5 portions per task;
- each check after a portion uses 200 examples instead of 300.

The final model uses the full sizes. For full-size experiments use `--preset configs/presets/full_tz.yaml` with a separate output folder.

## How to run

### On a GPU (needs 16 GB)

The experiments of this project were run on two T4 GPUs in Kaggle (about 30 GPU hours a week). Any machine with a 16 GB GPU works the same way: install the dependencies as in "Locally" below and use the commands in the "Commands" section. The run scripts skip finished runs (those with `final.json`), so a stopped job can be started again with the same command. The notebooks used for Kaggle and Colab are not part of this repository.

### Locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install torch               # for a GPU, pick the build for your CUDA at pytorch.org
pip install -r requirements.txt
```

On a CPU, the chat and the export are practical. Training also works, but slowly.

## Commands

```bash
python run.py --all                               # all experiments E0–E6, 3 seeds each
python run.py configs/E1.yaml --seeds 0           # one experiment, chosen seeds
python run.py configs/E5.yaml --preset configs/presets/smoke.yaml --seeds 0   # check in minutes on CPU
python report.py                                  # report: tables, plots, H1–H4 and the E5 criterion
python live.py --state DIR ...                    # continuous mode (see above)
python chat.py --state DIR                        # chat
python export.py --state DIR --out DIR            # export the model
python -m pytest tests                            # tests
```

Useful options:
- `--set key=value` changes any setting without editing a file;
- `--force` reruns a finished run;
- finished runs (those with `final.json`) are skipped.

Extra configs in `configs/extra/`:
- `E5_dpo.yaml`: DPO on task C, using a script that checks the answers (FR-11);
- `E0_full.yaml`: full fine-tuning instead of LoRA.

## How the loop works

A learner ([src/train_loop.py](src/train_loop.py)) is a model with its buffer, counters and random state. Each portion goes through five steps:

1. **Take the portion.** With the curriculum on, the task is sorted from easy to hard in advance. Difficulty is the original model's loss ([src/curriculum.py](src/curriculum.py)).
2. **Surprise.** The easiest 30% and the 2% with the highest loss are dropped; the latter are more often noise ([src/surprise.py](src/surprise.py)). Samples of dropped and kept examples go to `surprise_samples.jsonl` for manual review.
3. **Replay.** The buffer adds 30% examples from past tasks ([src/buffer.py](src/buffer.py)). Every task gets an equal quota, each task keeps a reservoir sample, and high-loss examples are replayed more often.
4. **Update.** LoRA training steps (r=16, lr 1e-4), plus DPO when enabled ([src/dpo.py](src/dpo.py)).
5. **Evaluate.** Every task, task 0 and the general test are measured. The current task is also checked every 20 steps ([src/eval.py](src/eval.py)).

**Sleep** runs every 10 portions ([src/sleep.py](src/sleep.py)):
1. Replay examples from the buffer.
2. Merge the adapter into the weights and start a new adapter.
3. Measure before and after. If learnt tasks or task 0 dropped by more than 2%, roll the sleep back.
4. Clean the buffer by loss and diversity.

**State** ([src/state.py](src/state.py)) holds the adapters merged by sleep phases (~35 MB each), the current adapter, the buffer, the counters and the random generators' state. The base model is not copied: it is downloaded at its pinned revision and the adapters are merged into it again. A test checks that a run stopped by a crash and restarted gives the same result as an uninterrupted run.

## Metrics

- **Task quality:** EM for B and C, perplexity for A, D and G, share of instructions followed for T0.
- **Difference from the control base:** end of stream minus the original model, for every task.
- **Forgetting** (Chaudhry et al., 2018): best score on a task minus its final score. A–C are compared by *normalised* forgetting, the share of the gain that was lost. It is left out when the gain is below 0.01, where it would be noise divided by noise.
- **Task 0 retention:** final / base and the worst value during the stream / base. The E5 criterion is at least 90%.
- **BWT / FWT** (Lopez-Paz & Ranzato, 2017).
- **Steps to target quality:** 90% of the gain E1 reaches on the task.
- **Quality after the first portion:** data efficiency.
- **Cost:** steps, time, peak GPU memory.

Significance follows the spec: mean ± std intervals over seeds must not overlap.

## Outputs

| Where | What |
|---|---|
| `runs/<config>/seed<k>/config.yaml` | the resolved config |
| `.../meta.json` | seed, code version (git commit and source hash), package versions, GPU, data revisions and SHA-256 |
| `.../metrics.jsonl` | one record per portion and per sleep phase |
| `.../curve.jsonl` | learning curves of the current task |
| `.../surprise_samples.jsonl` | samples for manual review of the selection |
| `.../final.json` | test scores and summary metrics |
| state folder (`--state`) | `config.yaml`, `merged/`, `ckpt-*/`, `latest`: everything needed to continue |

`report.py` turns these into `reports/summary.md`, `reports/summary.csv` and `reports/figures/`.

## Reproducibility

- The model and every dataset are pinned to a Hugging Face revision in `configs/base.yaml`.
- Pools and the 80/10/10 split depend only on `data.seed` and are cached in `data/cache/`. The cache key includes the loader code.
- The run seed sets the LoRA initialisation, shuffling and replay sampling.
- CUDA kernels are not fully deterministic, which is one reason every config runs on three seeds.

## Resources

One 16 GB GPU. Weights stay in fp32 and compute runs under autocast (bf16 or fp16). On a T4 this is about 4–5 minutes per portion and 1.5 hours per E0 run. All Kaggle launches together take about 25–30 GPU hours. Each run logs its exact time and memory.

## Implementation decisions

- **Qwen's chat template** adds a default system message. It is used the same way everywhere: training, evaluation, the chat and the exported model.
- **Task 0 uses its own set, not IFEval.** IFEval has ~500 prompts, many needing long answers. That is too slow to measure after every portion and too hard for a 0.5B model. The check is strict: an answer counts only if every condition holds. Answers are greedy, with the model's own `repetition_penalty` of 1.1.
- **Exact match for B and C** is computed without generation: every answer token must be the most likely one given the correct prefix. This is the same as greedy-decoding EM (a test checks it), and much cheaper.
- **Forgetting** is computed on validation data; final quality is on test data. The maximum over many checkpoints biases forgetting slightly upwards, so the report also has BWT.
- **Replay** covers past tasks only. Examples from the continuous mode and the chat are task `U`, with their own buffer quota.
- **The optimizer** is created afresh for every portion, because after a sleep phase the adapter is new.
- **DPO** runs inside the loop with the loss from TRL and Rafailov et al. TRL's `DPOTrainer` is built for a standalone training run.
- **E4 and E5 take more training steps than E1**, because sleep adds steps on the buffer. Keep this in mind when reading H4.
- **The buffer is in-house.** The review found no Hugging Face model support in Avalanche; the buffer borrows Avalanche's ideas of reservoir sampling and per-task quotas.

## Layout

```
configs/        # base.yaml + E0–E6, final/E5_final.yaml, extra/, presets/ (smoke, cpu_check, full_tz)
data/           # tasks A–D (dialogues), generators for task C and task 0, general test
src/
  buffer.py     # replay buffer
  surprise.py   # selection by loss
  curriculum.py # difficulty ordering
  sleep.py      # consolidation phase
  train_loop.py # the learner and the main loop
  state.py      # save, restore, rollback, export
  eval.py       # evaluation and forgetting metrics
  dpo.py, model.py, scoring.py, config.py, logger.py, report.py
reports/        # research.md, summary tables and plots
tests/          # tests
run.py, report.py, live.py, chat.py, export.py
```
