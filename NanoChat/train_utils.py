"""A File tha Contain the train Functions"""
import os
import math
import torch

from torch.nn import functional as F
import torch.distributed as dist

from HellaSwag import iterate_examples, render_examples_batch, get_most_likely_rows_batch


# 1. ------- LR (Ratios) ---------
def get_lr_ratio(step, config):
    """Warmup-then-cosine-decay ratio in [min_lr_ratio, 1], multiplied onto each
    optimizer group's own base lr every step."""
    if step < config.warmup_steps: return (step + 1) / config.warmup_steps
    if step > config.max_steps: return config.min_lr_ratio
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    assert 0 <= decay_ratio <= 1, f'There is something wrong with max steps:{config.max_steps}, step:{step}, warmup_steps:{config.warmup_steps}'
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return config.min_lr_ratio + coeff * (1 - config.min_lr_ratio)


# 2. -------- Get LR ----------
def get_lr(step, config):
    """Absolute matrix/muon lr, derived from the ratio -- kept only for the printed log line.
    Since every group now has its own base lr, this number is representative, not literal."""
    return config.matrix_lr * get_lr_ratio(step, config)

# 3. -------- Validate ----------
def validate(model, val_dl, device, device_type, val_loss_accum_steps, ddp, master_process, step, log_file, print_flag=True, log=True):
    """Run val_loss_accum_steps validation mini-batches and return the (DDP-averaged) mean loss."""
    model.eval()
    with torch.no_grad():
        val_accum_loss = 0.
        for _ in range(val_loss_accum_steps):
            x, y = val_dl.get_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss /= val_loss_accum_steps
            val_accum_loss += loss.detach()
    if ddp: dist.all_reduce(val_accum_loss, op=dist.ReduceOp.AVG)
    val_dl.reset()
    if master_process:
        if print_flag: print(f"Validation loss: {val_accum_loss.item():.4f}")
        if log:
            with open(log_file, "a") as f: f.write(f"{step} val {val_accum_loss.item():.4f}\n")
    return val_accum_loss


# 4. -------- Save Checkpoint ----------
def save_checkpoint(log_dir, step, **kwargs):
        """Save model/optimizer/dataloader state (plus any extra kwargs) to a step-numbered checkpoint file."""
        checkpoint_path = os.path.join(log_dir, f'model_checkpoint_step_{step:05d}.pt')
        checkpoint = {'step': step, **kwargs}
        torch.save(checkpoint, checkpoint_path)

# 5. --------- Sampling -----------
def sample(model, enc, num_sequence, max_length, device, device_type, process_rank, beginning_str='I am crazy man,'):
    """Autoregressively sample num_sequence continuations of beginning_str, print and return them."""
    model.eval()
    tokens = enc.encode(beginning_str)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.repeat(num_sequence, 1)
    x_gen = tokens.to(device)

    while x_gen.shape[1] < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x_gen)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)  # take the highest 50 probs
            ix = torch.multinomial(topk_probs, num_samples=1)  # take one sample
            ids = torch.gather(topk_indices, dim=-1, index=ix)
            x_gen = torch.cat((x_gen, ids), dim=1)

    decoded_samples = []
    for i in range(num_sequence):
        decoded = enc.decode(x_gen[i, :max_length].tolist())
        decoded_samples.append(decoded)
        print(f'Rank: {process_rank} | Sample{i + 1}: {decoded} ')
    return decoded_samples


# 6. ------- Hellaswag Validation ---------
def validation_hellaswag(model, enc, device, device_type, ddp, ddp_world_size, ddp_rank, master_process, step, log_file, max_examples=None, batch_size=32):
    """Score the model on HellaSwag val, DDP-reduce the counts, print/log and return accuracy.
    Examples are grouped into `batch_size`-sized forward passes instead of one forward pass
    (+ one host-device sync) per example -- that per-example sync was the actual bottleneck,
    since it serializes GPU work behind Python on a networked GPU."""
    # first collect this rank's share of examples (cheap: max_examples is small, plain python objects)
    rank_examples = []
    for i, example in enumerate(iterate_examples("val")):
        if max_examples is not None and i >= max_examples: break
        # only keep examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size == ddp_rank: rank_examples.append(example)

    num_correct_norm = 0
    num_total = 0
    for start in range(0, len(rank_examples), batch_size):
        batch = rank_examples[start: start + batch_size]
        tokens, mask, labels = render_examples_batch(batch, enc)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits for the whole batch in one forward pass
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(tokens)
            preds = get_most_likely_rows_batch(tokens, mask, logits, len(batch))
        labels_t = torch.tensor([ex['label'] for ex in batch], device=preds.device)
        num_total += len(batch)
        num_correct_norm += (preds == labels_t).sum().item()   # one sync per batch, not one per example
        if master_process and (start // batch_size) % 5 == 0:  # light progress ping so this never looks hung again
            print(f"  hellaswag: {num_total}/{len(rank_examples)} scored so far...")
    # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, "a") as f: f.write(f"{step} hella {acc_norm:.4f}\n")
    return acc_norm

# 7. ------ Train -------
def train_model(model, train_dl, step, device, device_type, optimizer, lr_ratio_getter, clip_grad_norm_value, ddp):
    """Run one full grad-accum training step (all mini-batches) and return (loss, lr_mult, grad_norm)."""
    model.train()
    accum_loss = 0
    for mini_step in range(train_dl.tot_mini_batches):
        x, y = train_dl.get_batch()
        x, y = x.to(device), y.to(device)
        if ddp: model.require_backward_grad_sync = (mini_step == train_dl.tot_mini_batches - 1)  # So that we avoid unnecessary communication during backward () unless it is the last step
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            loss /= train_dl.tot_mini_batches
            accum_loss += loss.detach()
        loss.backward()
    if ddp: dist.all_reduce(accum_loss, op=dist.ReduceOp.AVG)  # Averaging the Loss Across all the Processes
    lr_mult = lr_ratio_getter(step)
    # for param_group in optimizer.param_groups:  -----> Because we are using the MuonAdamW optimizer, we don't need to set the lr for each param group, we just pass it to the step function
    #     param_group['lr'] = lr
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_value)  # This Makes Normalization for the Norm of the Gradients proportionally, so that -> root(sum(grads**2)) <= 1
    optimizer.step(lr_mult)
    optimizer.zero_grad()
    if device_type == 'cuda': torch.cuda.synchronize()  # so that the cpu don't run the next command while the GPU still hasn't Finished
    return accum_loss, lr_mult, grad_norm



# ----- Chat-format sampling (for SFT/RL,etc...) ------
def sample_chat(model, enc, prompts, max_new_tokens, device, device_type, process_rank):
    """Like `sample()`, but renders each prompt through the real chat template
    (render_for_completion) and greedy-decodes, stopping at <|assistant_end|>,
    instead of raw string continuation to a fixed length. This is what actually
    shows whether SFT learned the format -- val loss and HellaSwag don't."""
    model.eval()
    assistant_end_id = enc.encode_special("<|assistant_end|>")
    decoded_samples = []
    for prompt in prompts:
        convo = {"messages": [{"role": "user", "content": prompt},
                               {"role": "assistant", "content": ""}]}  # dummy turn -- render_for_completion pops it before rendering
        ids = enc.render_for_completion(convo)
        x_gen = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        start_len = x_gen.shape[1]
        for _ in range(max_new_tokens):
            if x_gen.shape[1] >= model.config.block_size: break  # context-full guard
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _ = model(x_gen)
                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x_gen = torch.cat((x_gen, next_id), dim=1)
            if next_id.item() == assistant_end_id: break
        gen_ids = x_gen[0, start_len:].tolist()
        if gen_ids and gen_ids[-1] == assistant_end_id: gen_ids = gen_ids[:-1]
        decoded = enc.decode(gen_ids)
        decoded_samples.append((prompt, decoded))
        print(f"Rank: {process_rank} | Q: {prompt!r} -> A: {decoded!r}")
    return decoded_samples

