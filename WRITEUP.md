# NanoChat: Training a 253M Chat Model From Scratch on $30 a Month

Before this project I built nanoGPT. I understood every line of it, but it was a paper recreation, and it was only pretraining: no tokenizer training, no SFT, no RL, no inference engine, and none of the newer architecture ideas. So I wanted to go deeper and build the full pipeline of a real chat model.

NanoChat is the full pipeline, built from scratch in plain PyTorch with no trainer frameworks:

- a custom BPE tokenizer
- pretraining
- supervised fine-tuning (SFT)
- a GRPO stage (implemented and smoke-tested, not launched)
- a custom inference engine with a KV cache, since this architecture doesn't run through vLLM or Hugging Face pipelines

It's inspired by Karpathy's nanochat and the modded-nanoGPT speedrun, but it's not a fork.

**253.2M parameters · ~1.97B pretraining tokens · ~13 hours on one A100-80GB · ~$40 of useful GPU compute on Modal.**

Demo: seif-222--nanochat-chat-serve.modal.run (about 20 seconds cold start) · Repo: seif-222/NanoChat · W&B: seif-222-student/Nanochat

---

## 1. Budget

I have about $30 a month for compute, and most of the decisions below come from that.

### Why I didn't experiment

Experiments on a language model have to run for a while to show anything, so they cost real money. Any budget spent on them would have meant fewer training tokens and a more undertrained final model.

So I chose to spend the budget on training rather than exploration, so the model could see more tokens. Instead of testing ideas myself, I followed an architecture that nanochat and the speedrun community had already tested with far more compute. The goal was to understand every part of a modern pipeline by building and training it myself, not to invent new parts.

### Why one GPU, even though the scripts support DDP

The training scripts support multi-GPU training, but the budget fixes the total compute. More GPUs would finish the run faster, but the same money buys the same amount of GPU time either way, so it wouldn't mean more training tokens.

So I picked the model size that gives a reasonable tokens-per-parameter ratio for my budget. That was 253M, which also fits on a single A100 at the full batch size. Each step is 524,288 tokens, made of 32 micro-batches of 16 × 1,024 tokens.

### Why FineWeb-Edu

With a small token budget, data quality matters more, so I used FineWeb-Edu. It's a filtered, educational subset of FineWeb, so it's high quality and still very large.

I tokenized about 4B tokens into 40 shards of 100M tokens, and this run used about 1.97B of them. The shards are on Hugging Face: `seif-222/Tokenized_FineWebEdu_Shards`.

---

## 2. Tokenizer

The BPE tokenizer has a 65,536-token vocabulary, trained on 2B characters with a GPT-4-style split pattern. I trained it with rustbpe and serve it with tiktoken.

The special tokens were added to the vocab from the start:

- a beginning-of-sequence token
- `user_start` / `user_end`
- `assistant_start` / `assistant_end`
- `python_start` / `python_end`
- `output_start` / `output_end`

The Python and output tokens are for tool use. They exist in the vocab, but this model isn't trained on them, because I removed tool-calling conversations from the SFT data (see section 6). That way, a larger version could use them without rebuilding the tokenizer.

---

## 3. Model

The model has 14 layers, width 896 and 7 heads (head dim 128), with a context of 1,024 and a vocab of 65,536. The config has close to a hundred settings, so most choices can be changed without touching the model code. I also added flags for some of my own ideas so I could experiment with them later (see "What I'd try next").

**Core:**

- **Normalization:** pre-norm with weightless RMSNorm, plus QK-Norm on queries and keys (scaled ×1.2).
- **Positions:** RoPE with base 100k.
- **Attention:** GQA is supported, but this run uses 7 KV heads, so it's plain multi-head attention.
- **MLP:** ReLU² activation with 4× expansion.
- **Init:** zero-initialized output projections in attention and MLP.
- **Head:** untied LM head with a logit softcap of 15.
- **Optimizer:** Muon (5 Newton–Schulz steps, Nesterov) on the block matrices. AdamW on embeddings, the head and scalars, each group with its own learning rate.

**Speedrun-style additions:**

- **Sliding-window attention.** The layers follow an `SSSL` pattern with a window of 256, and the last layer is always full, so 10 of the 14 layers only attend to the last 256 tokens. This makes attention cheaper and keeps the inference KV cache small, while the full layers keep long-range information.
- **SmearGate.** Each token's embedding gets a gated mix of the previous token's embedding. The gate reads 24 features, and its strength is a learned scalar initialized at zero.
- **Value embeddings.** A small shared table (65,536 × 12) adds token identity to the attention values. Seven layers use it (every other layer, plus the last), each with its own projection to full width and a per-head gate.
- **Per-layer residual scalars.** Before each layer, the hidden state is scaled and mixed with the first hidden state `x0` using learned scalars.
- **Backout.** The hidden state after layer 7 is saved, and a learned fraction of it (initialized at 0.2) is subtracted from the final hidden state before the head.

The embedding table and LM head together are 117.4M parameters, about 46% of the model. This is why weight tying is one of the ideas in "What I'd try next".

---

## 4. Pretraining

### The crash

The plan was a 150-step warmup, then cosine decay to 10% of peak over 3,100 steps, checkpointing every 500 steps. At step 1,873 the Modal server cut the run off, so the last checkpoint was step 1,500 (to avoid this in the future, I now checkpoint every 200 steps).

### The fix: a shorter horizon

A cosine schedule sets the learning rate based on how far you are toward `max_steps`. The full 3,100 steps no longer fit the budget, so I cut it to 2,200.

Cutting the horizon changed the learning rate at the resume point: at step 1,500 it dropped from **61% of peak to 34%**. I accepted that trade-off to finish annealing on budget. Steps 1,500–2,199 annealed fully and reached val loss 3.167 at about 1.15B tokens.

### What to do with a new $30

Val loss was still falling at step 2,199 when the next $30 came in. So there was room to keep training, and I had three options:

1. **Rewind** to step 1,500 and run the original 3,100-step cosine. This throws away 700 trained steps.
2. **Re-warm the learning rate,** hold it, and decay again at the end (WSD-style). But I found in Ibrahim et al., 2024 that re-warming isn't free: it raises validation loss for a while, even on the same data, and more so the higher you re-warm. At my budget, any experiment like this is risky, because if it turned out unstable, I'd lose steps.
3. **Continue at the floor learning rate** from step 2,200 to 3,750.

I also ruled out simply setting a new cosine to end at 3,750, because it would have raised the learning rate from the 10% floor to about 45% of peak, roughly 4.5×.

I took option 3, the safe one. I kept `max_steps` at 2,200, so every step after that ran at the floor learning rate. Val went from 3.167 to **3.121** at about **1.97B tokens**.

### Tokens per parameter

1.97B tokens over 253.2M parameters is **7.8 tokens/param**, well below Chinchilla's 20. Karpathy's nanochat scaling sweeps first put compute-optimal at about 8, and newer versions of the repo use about 10.5, so this run is a bit under-trained, but close.

The number depends on what you count, though. Since almost half the parameters are embeddings, counting only non-embedding parameters (about 135M) gives roughly 14.6 tokens/param.

### WSD

WSD (warmup–stable–decay) would have avoided the schedule problems above. The learning rate stays flat and only decays at the end, so you can stop, extend or resume without fixing the step count in advance. With cosine, you have to pick the end point before you start, which is what caused the problems here. For that reason, I'd use WSD in my next run.

---

## 5. Skipping mid-training

nanochat has a mid-training stage between pretraining and SFT. It's a broad mix that teaches the conversation format, plus multiple-choice answering and tool use. I skipped it for two reasons:

- **Tool use** was out of scope, and I removed it from the SFT data anyway. Multiple-choice formatting wasn't a goal either.
- **The chat format** was already covered by my SFT mix, which was diverse enough.

At this scale, a separate stage that overlaps this much with SFT wasn't worth the budget.

---

## 6. SFT

I fine-tuned on about 164k conversations from four sources:

- everyday-conversations-llama3.1-2k
- no_robots (CC-BY-NC-4.0, non-commercial)
- databricks-dolly-15k
- smol-smoltalk, capped at 300k before filtering

smol-smoltalk is the simplified version of SmolTalk, without function calling, heavy rewriting or hard math.

I removed tool calling, system prompts, and conversations longer than about 3,200 characters (roughly the 1,024-token context). The goal was a model that handles short, general assistant chats well. A 253M model doesn't have the capacity for system instructions and tool calls, and training on them would take capacity away from basic chat.

Training details:

- **Loss:** assistant tokens only.
- **Length:** one epoch.
- **Learning rates:** base rates scaled to 10%, with a 7.5% warmup.
- **Validation:** 1% of the data held out, checked every 5% of training.

The best checkpoint was **step 9,633, val loss 1.5527**, near the end of the epoch. That's the model in the demo.

---

## 7. Why I didn't run RL

GRPO on GSM8K is implemented and smoke-tested:

- **Sampling:** 4 questions per step, 8 completions each, temperature 0.8, top-p 0.95.
- **Learning:** group-relative advantages and a KL penalty (β = 0.02) to the frozen SFT model.
- **Limits:** a 256-token generation limit and greedy validation.

I didn't run it, for two reasons.

First, the SFT model scored **0/20** on a simple "is the last number correct" check. That matters because GRPO learns by comparing the 8 completions in each group. If all 8 get reward 0, every advantage is 0, so there's no gradient.

Sampling might produce a correct answer once in a while and give a small signal, but most steps would be all zeros. Training would be very inefficient, and most of the compute wouldn't teach the model anything.

For comparison, Karpathy's d20 nanochat model has about 560M parameters, was trained on about 11B tokens, and saw GSM8K in mid-training, and RL still only took it from 4.55% to 7.58% on GSM8K. My model is smaller and trained on far fewer tokens, so RL would be even less effective here.

Second, RL on GSM8K improves math answers, not general chat, and general chat is what this model is for. So it wasn't worth the budget.

---

## 8. Results

**HellaSwag**, full validation split, 10,042 examples (random chance is 25%):

| Checkpoint | Correct | Accuracy |
|---|---|---|
| Pretrain step 2199 | 3068 | 30.55% |
| Pretrain step 3750 | 3137 | 31.24% |
| SFT best (step 9633) | 2914 | 29.02% |

The extra tokens at the floor learning rate added about 0.7 points (30.55% → 31.24%). The drop after SFT is expected, since SFT trains the model for conversation, not sentence completion.

**Samples:**

> **User:** What is the capital of United States?
> **Assistant:** The capital of the United States is Washington, D.C.

> **User:** What is the Photosynthesis process.
> **Assistant:** The Photosynthesis process is a process used by plants, algae, and some bacteria to convert light energy into chemical energy.
> **User:** Is this process important?
> **Assistant:** Yes, the process of photosynthesis is important because it provides the energy needed for all living organisms to function properly.

It handles simple chats well, but for topics outside its training data it can make things up, which is expected at this scale.

### W&B runs

The W&B project has seven runs:

- **Smoke tests:** a few dummy runs to check that everything works.
- **Pretraining run 1:** the run that crashed (last checkpoint at step 1,500).
- **Pretraining run 2:** steps 1,500 → 2,199.
- **Pretraining run 3:** steps 2,200 → 3,750, at the floor learning rate.
- **SFT:** the final fine-tuning run.

Each run logs:

- training and validation loss
- tokens/sec and training time
- learning rate and LR multiplier
- gradient norm
- HellaSwag accuracy (capped at 1,000 examples during training)
- sample generations every 500 steps

---

## 9. What I'd try next

### Scale and schedule

- **More compute:** a bigger model trained on more tokens.
- **WSD** and other schedulers, including the re-warm option from section 4.
- **A hyperparameter sweep** over the config.

### My own ideas (already flags, not tested yet)

- **Value embeddings.**
  - *Two or three shared tables instead of one,* each serving part of the depth. At my current 12 dimensions each table is under 1M parameters, so this is cheap, but it gets expensive if the table dimension is increased.
  - *A separate full-width table per layer.* Each would be 65,536 × 896 ≈ 58.7M parameters, so memory is the limit.
- **Per-layer scalars.** Use several scalars per layer instead of one, for example 4, each covering a quarter of the dimensions.
- **SmearGate.** Change the number of features the gate reads (currently 24).
- **Backout.**
  - *Layer choice:* save the hidden state from a different layer instead of layer 7 (the middle).
  - *Learned projection:* pass the saved state through a linear layer (initialized to identity) before subtracting it.
- **Weight tying.** Share the token embedding and LM head weights, as the original Transformer and GPT-2 did. That would save 58.7M parameters, about 23% of the model. I haven't tested whether it hurts quality at this size.

### Speed and inference

**Training.** Custom Triton kernels could speed up training. The main candidate is a fused cross-entropy that avoids creating the full logits tensor.

**Inference.** Custom Triton kernels would make inference more efficient when the model is served on GPU (the current model is served on CPU, but a bigger version would be served on GPU).

### Mixture of experts

I think MoE could be a great idea to try. Its main benefit is efficiency: the model has a lot of capacity, but each token only uses a few experts.

My intuition is that at 253M the model is too small for it. I'd rather have all the parameters working on every token than split an already-small model into experts. It would make more sense at a larger scale, and I'd like to try it then.

### Multimodality

One direction is getting the model to understand images. The approach I have in mind is roughly the LLaVA recipe:

1. Take a pretrained image encoder and remove its classification head.
2. Freeze the encoder and the language model.
3. Train a small projector that maps image features into the LLM's token-embedding space, using image–caption pairs.

A simpler version would keep the classifier and pass its predicted label to the LLM as text. Either way, a 253M model would be very limited with images, so this fits a larger version better.

---

*Thanks to Andrej Karpathy (nanoGPT, nanochat), the modded-nanoGPT speedrun community, Hugging Face (FineWeb-Edu, SmolTalk), and Modal.*
