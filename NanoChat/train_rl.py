"""
RL entrypoint. Loads the SFT checkpoint, samples K completions per GSM8K
question, scores them 0/1, and takes a GRPO policy-gradient step with a KL
penalty against a frozen SFT copy.

Run with:  python train_rl.py
"""
import os
import re
import json
import glob
import copy
import random

import torch
import torch.nn.functional as F

from Config import GPT_config
from Model import GPT
from optimizer import MuonAdamW
from Tokenizer import RustTokenizer
from Engine import InferenceEngine
from train_utils import save_checkpoint


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(): return "mps"
    return "cpu"


###_________________________ DEVICE _____________________________________________________

device = pick_device()
device_type = 'cuda' if device.startswith('cuda') else 'cpu'
torch.set_float32_matmul_precision('high')
print(f'[INFO] Using device: {device}')


###_____________________________________  INSTANCES  ______________________________________

# 1. ---- Load SFT Checkpoint's Config + Override for RL ----
_base_cfg = GPT_config()
sft_ckpt_path = os.environ.get('RL_SFT_CKPT', _base_cfg.rl_sft_ckpt)
ckpt = torch.load(sft_ckpt_path, map_location=device, weights_only=False)
config = copy.deepcopy(ckpt['config'])       # start from the exact config the checkpoint was trained with
config.device        = device
config.process_rank  = 0
config.num_processes = 1

# 2. ---- Add RL attributes to Config ----
# overlay from current Config.py so an SFT (or later RL) pickle doesn't pin stale knobs
rl_attrs = (
    'rl_sft_ckpt', 'rl_gsm8k_train_path', 'rl_gsm8k_test_path', 'rl_log_dir',
    'rl_prompts_per_step', 'rl_k_samples', 'rl_max_new_tokens',
    'rl_temperature', 'rl_top_p',
    'rl_lr_scale', 'rl_kl_beta', 'rl_clip_grad_norm',
    'rl_max_steps', 'rl_checkpoint_every', 'rl_val_every', 'rl_val_questions',
)
for f in rl_attrs:
    setattr(config, f, getattr(_base_cfg, f))
config.rl_gsm8k_train_path = os.environ.get('RL_GSM8K_TRAIN', config.rl_gsm8k_train_path)  # Modal override, falls back to Config.py default
config.rl_gsm8k_test_path  = os.environ.get('RL_GSM8K_TEST',  config.rl_gsm8k_test_path)

# 3. ---- Load Tokenizer ----
enc = RustTokenizer.from_directory(config.tokenizer_dir)
assert enc.get_vocab_size() == config.vocab_size, "tokenizer/checkpoint vocab size mismatch"
assistant_end_id = enc.encode_special("<|assistant_end|>")

# 4. ---- Policy + Reference ----
policy = GPT(config)
policy.load_state_dict(ckpt['model'])
policy.to(device)

# frozen SFT copy -- KL pulls the policy back toward this, never toward an RL ckpt
reference = GPT(copy.deepcopy(config))
reference.load_state_dict(ckpt['model'])
reference.to(device)
reference.eval()
for p in reference.parameters(): p.requires_grad_(False)

engine = InferenceEngine(policy)   # samples from the live policy as it updates

# 5. ---- Scale LRs down for RL ----
# SFT-scaled lrs from the ckpt, then rl_lr_scale -- RL grads are higher-variance than SFT
config.matrix_lr      *= config.rl_lr_scale
config.unembedding_lr *= config.rl_lr_scale
config.embedding_lr   *= config.rl_lr_scale
config.scalar_lr      *= config.rl_lr_scale

# 6. ---- Optimizer ----
optimizer = policy.optimizers_config(device_type, MuonAdamW)

# 7. ---- Data ----
with open(config.rl_gsm8k_train_path) as f:
    train_problems = [json.loads(line) for line in f if line.strip()]
print(f'[INFO] Loaded {len(train_problems)} GSM8K training problems')

# 8. ---- Logging / Resume ----
os.makedirs(config.rl_log_dir, exist_ok=True)
log_file = os.path.join(config.rl_log_dir, 'log.txt')

# only policy + optimizer resume. reference is always the original SFT weights (the KL anchor)
def try_resume(path):
    try:
        rl_ckpt = torch.load(path, map_location=device, weights_only=False)
        policy.load_state_dict(rl_ckpt['model'])
        optimizer.load_state_dict(rl_ckpt['optimizer'])
        return rl_ckpt['step'] + 1
    except Exception as e:
        print(f"[INFO] Skipping unloadable checkpoint {path}: {e}")
        return None

start_step = 0
for candidate in sorted(glob.glob(os.path.join(config.rl_log_dir, 'model_checkpoint_step_*.pt')), reverse=True):
    result = try_resume(candidate)
    if result is not None:
        start_step = result
        print(f"[INFO] Resumed RL from {candidate} at step {start_step}")
        break


###_________________________ REWARD _____________________________________________________

NUMBER_RE = re.compile(r"-?[\d,]+\.?\d*")

def extract_answer(text):
    """Last number anywhere in the completion. None if there isn't one (counts as wrong)."""
    matches = NUMBER_RE.findall(text)
    if not matches: return None
    try: return float(matches[-1].replace(',', ''))
    except ValueError: return None


def reward_fn(completion_text, ground_truth):
    """1.0 if the final number matches the gold (float tolerance), else 0.0."""
    pred = extract_answer(completion_text)
    if pred is None: return 0.0
    return 1.0 if abs(pred - ground_truth) < 1e-3 else 0.0


###_________________________ HELPERS ____________________________________________________

def generate_bounded(prompt_tokens, temperature, top_p):
    """Sample from stream_generation but stop at rl_max_new_tokens (not block_size)."""
    gen = engine.stream_generation(prompt_tokens, eos_token=assistant_end_id, pad_token_id=0,
                                    temperature=temperature, top_p=top_p)
    collected = []
    # @torch.no_grad on a generator only covers construction, not each yield -- wrap the loop
    with torch.no_grad():
        for next_token in gen:
            collected.append(next_token)
            if len(collected) >= config.rl_max_new_tokens: break
    if not collected:
        return torch.zeros((prompt_tokens.shape[0], 0), dtype=torch.long, device=device)
    return torch.cat(collected, dim=1)


def make_completion_mask(full_tokens, prompt_len, eos_id):
    """1 on generated tokens through first EOS, 0 on the prompt and on pad after EOS."""
    completion = full_tokens[:, prompt_len:]
    T = completion.shape[1]
    is_eos = (completion == eos_id)
    idx = torch.arange(T, device=completion.device).unsqueeze(0).expand_as(is_eos)
    first_eos = torch.where(is_eos, idx, torch.full_like(idx, T)).min(dim=1).values
    comp_mask = (idx <= first_eos.unsqueeze(1)).long()
    prompt_mask = torch.zeros(full_tokens.shape[0], prompt_len, dtype=torch.long, device=full_tokens.device)
    return torch.cat([prompt_mask, comp_mask], dim=1)


def sequence_logprobs(model, tokens, completion_mask):
    """Mean log-prob of completion tokens (averaged, not summed, so length doesn't dominate)."""
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits, _ = model(tokens)                                  # (B, T, V), no targets -- CE math is below
    shift_logits = logits[:, :-1]                                   # position t predicts token t+1
    shift_labels = tokens[:, 1:]
    shift_mask = completion_mask[:, 1:].float()

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_logprobs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    token_logprobs = token_logprobs * shift_mask

    denom = shift_mask.sum(-1).clamp(min=1)
    seq_logprobs = token_logprobs.sum(-1) / denom
    return seq_logprobs, token_logprobs, shift_mask


def run_group(problem):
    """K samples for one question -> pg + KL loss and logging metrics."""
    K = config.rl_k_samples
    # Message -> Completion format
    convo = {'messages': [{'role': 'user', 'content': problem['question']}, {'role': 'assistant', 'content': ''}]}
    prompt_ids = enc.render_for_completion(convo)
    prompt_len = len(prompt_ids)
    prompt_tokens = torch.tensor(prompt_ids, device=device).unsqueeze(0).repeat(K, 1)
    # Completion
    completion_tokens = generate_bounded(prompt_tokens, config.rl_temperature, config.rl_top_p)
    out = torch.cat([prompt_tokens, completion_tokens], dim=1)
    # Decode Completion & Reward & Advantage
    completions = [enc.decode(out[i, prompt_len:].tolist()) for i in range(K)]
    rewards = torch.tensor([reward_fn(c, problem['answer']) for c in completions], device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)    # zero-mean / unit-variance within this question's K. all-same reward -> std~0 -> pg term ~0

    # Completion mask
    completion_mask = make_completion_mask(out, prompt_len, assistant_end_id)
    # Run Policy & Reference
    policy_seq_lp, policy_tok_lp, shift_mask = sequence_logprobs(policy, out, completion_mask)
    with torch.no_grad():   _, ref_tok_lp, _ = sequence_logprobs(reference, out, completion_mask)

    # Loss_1
    pg_loss = -(advantages.detach() * policy_seq_lp).mean()
    # Loss_2
    log_ratio = ref_tok_lp - policy_tok_lp
    kl_per_token = torch.exp(log_ratio) - log_ratio - 1
    kl_loss = (kl_per_token * shift_mask).sum() / shift_mask.sum().clamp(min=1)
    # Combined_Loss
    loss = pg_loss + config.rl_kl_beta * kl_loss

    # Info Dict & Return
    metrics = dict(reward=rewards.mean().item(), kl=kl_loss.item(), pg_loss=pg_loss.item())
    return loss, metrics


@torch.no_grad()
def evaluate(n_questions):
    """Greedy accuracy on a held-out slice -- one sample per question."""
    policy.eval()
    # Load Problems
    with open(config.rl_gsm8k_test_path) as f:
        problems = [json.loads(line) for line in f if line.strip()][:n_questions]
    # Solve
    correct = 0
    for problem in problems:
        convo = {'messages': [{'role': 'user', 'content': problem['question']}, {'role': 'assistant', 'content': ''}]}
        prompt_ids = enc.render_for_completion(convo)
        prompt_tokens = torch.tensor(prompt_ids, device=device).unsqueeze(0)
        completion_tokens = generate_bounded(prompt_tokens, temperature=0.0, top_p=1.0)
        completion = enc.decode(completion_tokens[0].tolist())
        correct += reward_fn(completion, problem['answer'])

    policy.train()
    # Return Accuracy
    return correct / len(problems)


##_____________________________________  TRAINING  ______________________________________

for step in range(start_step, config.rl_max_steps + 1):   # +1 -> final step index == rl_max_steps
    problems = random.sample(train_problems, config.rl_prompts_per_step)
    # Train
    policy.train()
    optimizer.zero_grad()
    step_metrics = []
    for problem in problems:
        loss, metrics = run_group(problem)
        loss /= config.rl_prompts_per_step
        loss.backward()
        step_metrics.append(metrics)

    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.rl_clip_grad_norm)
    optimizer.step(1.0)  # flat lr for now

    mean_reward = sum(m['reward'] for m in step_metrics) / len(step_metrics)
    mean_kl = sum(m['kl'] for m in step_metrics) / len(step_metrics)
    mean_pg = sum(m['pg_loss'] for m in step_metrics) / len(step_metrics)
    print(f"Step {step:4d} | reward={mean_reward:.3f} | pg_loss={mean_pg:.4f} | kl={mean_kl:.4f} | grad_norm={grad_norm:.4f}")
    with open(log_file, 'a') as f: f.write(f"{step} train reward={mean_reward:.4f} kl={mean_kl:.4f}\n")

    # Validate
    if step % config.rl_val_every == 0 and step > 0:
        acc = evaluate(config.rl_val_questions)
        print(f"[INFO] val accuracy: {acc:.3f}")
        with open(log_file, 'a') as f: f.write(f"{step} val acc={acc:.4f}\n")
    # Checkpoint
    if step % config.rl_checkpoint_every == 0 and step > 0:
        save_checkpoint(config.rl_log_dir, step,
                         model=policy.state_dict(), optimizer=optimizer.state_dict(),
                         config=policy.config)

print(f"[INFO] RL done. Checkpoints in {config.rl_log_dir}")
