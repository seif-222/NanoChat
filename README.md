# NanoChat

From-scratch GPT: custom BPE → pretrain → SFT → RL (optional) → chat demo.

Inspired by Karpathy’s [nanochat](https://github.com/karpathy/nanochat) and the GPT speedrun. Same idea — plain PyTorch, no trainer frameworks — not a fork.

**Demo:** [seif-222--nanochat-chat-serve.modal.run](https://seif-222--nanochat-chat-serve.modal.run) (~20 seconds cold start)

**W&B:** [seif-222-student/Nanochat](https://wandb.ai/seif-222-student/Nanochat)

253.2M params. ~1.97B pretrain tokens. Modal’s $30/mo free tier ($50 total, $40 useful GPU after experimentations and errors).

| Stage | Status |
| --- | --- |
| Tokenizer | Done. Vocab 65,536 (rustbpe to train, tiktoken to serve). |
| Pretrain | Done. Step 3750, val 3.121, ~7.8 tok/param. |
| SFT | Done. 1 epoch, ~164k conversations, best val 1.5527. **This is the chat model.** |
| RL (GRPO / GSM8K) | Wired. Skipped at this version. See Reinforcement learning. |
| Demo | Live on Modal (CPU, scale-to-zero). |

---

## Model

253.2M trainable. `n_embd=896`, `n_layer=14`, `n_head=7`, context 1024, vocab 65,536.

* Pre-norm, weightless RMSNorm
* RoPE (base 100k)
* ReLU² MLP (4×)
* Untied language-model head, zero-init output projections
* Logit softcap
* Sliding-window attention (`SSSL`, window 256; every 4th layer and the last layer are full) (cheaper attention, periodic full layers keep long range)
* SmearGate (mix in the previous token)
* Value embeddings (token identity on attention values; one small shared table)
* Residual-stream skip + mid-stack backout
* Muon + AdamW (Muon on matrices, AdamW on embeddings / head)

---

## Inference engine

Custom `Engine` — this architecture does not go through vLLM or Hugging Face pipelines.

* KV cache (sliding-window + SmearGate; checked against a full forward pass)
* Streaming generation
* Sampling: temperature, top-p, repetition penalty, top-k (current setup: 0.8, 0.95, 1.05, no top-k respectively)
* Generation clipped to the remaining room in the 1024 context

---

## Pretraining

* FineWeb-Edu `sample-10BT`, 40 shards × 100M tokens (≈ 1.97B tokens used for this version)
* Custom BPE on 2B characters, GPT-4-style split
* Chat special tokens put in the vocab up front

Planned 3100-step warmup + cosine. The run died mid-way (Modal server cutoff). Last clean checkpoint was step 1500. Horizon was cut to 2200 to finish on budget — cosine is locked to `max_steps`, so that cut dropped the learning rate at the resume point (~61% of peak → ~34%), an intentional trade-off to finish annealing on budget. 1500→2199 then fully annealed (val 3.167, ~1.15B tokens). (WSD prevents this: you can train and cut without pre-planning the step count, but in this case training was already 1500 steps into cosine, so that schedule was kept.)

Extra budget (a fresh $30). Val was still falling at 2199. Two options: rewind to step 1500 and finish the original 3100-step cosine, or continue from 2200 at the floor learning rate through step 3750. I chose the second — 700 steps were already trained, so more tokens seen over a cleaner schedule. `max_steps` stayed at 2200 (a new cosine aimed at 3750 would have jumped the learning rate ~4.5×). Val **3.167 → 3.121**, ~1.97B tokens ≈ **7.8 tok/param**.

The nanochat speedrun with Muon puts compute-optimal around **8–12 tok/param**, not Chinchilla-20. This run sits on that line.

---

## SFT

~164k conversations:

* `everyday-conversations-llama3.1-2k`
* `no_robots` (CC-BY-NC-4.0, non-commercial)
* `databricks-dolly-15k`
* `smol-smoltalk` — the smaller, simpler mix (no function-calling / heavy rewrite / hard math), capped at 300k, then longer conversations dropped (~3,200 characters ≈ 1,024 tokens, this model's block size)

Loss only on assistant tokens. One epoch. Best checkpoint is step 9633, val **1.5527** (this is the served model in the demo).

---

## Reinforcement learning

GRPO on GSM8K is in the repo. **Not launched** (mainly because of the scale). SFT best scored 0/20 on a last-number probe, so RL would not be effective at this scale. Karpathy’s d20 only went 4.55% → 7.58% after mid-training that already included GSM8K. RL can make it better at math, not necessarily a better chatbot.

* GSM8K preprocessing
* Grouped sampling
* Group-relative advantages
* KL regularization
* Bounded generation
* Greedy validation
* Modal execution

---

## Infra

* Shards on Hugging Face: `seif-222/Tokenized_FineWebEdu_Shards`
* Train on Modal (`nanochat-training`, volume `nanochat-data`): data, pretrain, SFT, RL
* Serve on a separate Modal app (`nanochat-chat`)

Load `output/Models/SFT/model_checkpoint_best.pt`.

[seif-222/NanoChat](https://github.com/seif-222/NanoChat)

---

## More information

### Samples

```
User: Hello
Assistant: Hello! How can I help you today?

User: What is the capital of United States?
Assistant: The capital of the United States is Washington, D.C.

User: What is the Photosynthesis process.
Assistant: The Photosynthesis process is a process used by plants, algae, and some bacteria to convert light energy into chemical energy.

User: Is this process important?
Assistant: Yes, the process of photosynthesis is important because it provides the energy needed for all living organisms to function properly.
```

* For information outside the training distribution, the model can make things up or claim false information. Expected at this scale with this training.

### HellaSwag

Full validation split (10,042 examples):

| Checkpoint           | Correct | Accuracy |
| -------------------- | ------- | -------- |
| Pretrain step 2199   | 3068    | 0.3055   |
| Pretrain step 3750   | 3137    | 0.3124   |
| SFT best (step 9633) | 2914    | 0.2902   |

* SFT focuses on chat format, so the raw language-modelling score can drop as the model is conditioned into conversation.

## Files

```
NanoChat/
├── Config.py              # model + training configuration
├── Model.py               # GPT architecture + KV cache
├── optimizer.py           # MuonAdamW
├── Engine.py              # inference / sampling / streaming
├── Tokenizer.py           # BPE tokenizer + chat formatting
├── DataLoader.py          # pretraining + SFT loaders
├── train_utils.py         # shared training utilities
├── Train.py               # pretraining
├── train_sft.py           # supervised fine-tuning
├── train_rl.py            # GRPO / GSM8K
├── prepare_sft_data.py    # SFT dataset preparation
├── gsm8k_data.py          # GSM8K preparation
├── Train_tokenizer.py     # tokenizer training
├── Shard_tokenization.py  # FineWeb-Edu tokenization
├── HellaSwag.py           # evaluation
├── Modal_app.py           # cloud training stages
└── output/tokenizer/rustbpe_tokenizer.pkl  # tokenizer file
```

## Acknowledgements

Andrej Karpathy (nanoGPT, nanochat), Hugging Face (FineWeb-Edu, SmolTalk), Modal.
