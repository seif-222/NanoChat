"""
Hybrid optimizer: Muon (Newton-Schulz orthogonalization) for 2D+ matrix
parameters, AdamW-style updates for everything else (embeddings, lm_head,
per-layer scalars, biases). Each param group carries its own hyperparameters
(lr/beta1/beta2/eps/weight_decay/use_muon); anything a group doesn't specify
falls back to the optimizer-level default passed at construction time.
"""

import torch


class MuonAdamW:
    """Hybrid optimizer: Muon (Newton-Schulz) for matrices + AdamW-style for the rest.
    NOTE: lr here is max_lr at the schedule."""

    def __init__(self, params, adamw_lr, adamw_wd, muon_lr, muon_wd, red_dim, steps=5,
                 adamw_beta1=0.9, adamw_beta2=0.97, muon_beta1=0.9, muon_beta2=0.97,
                 Nesterov=True, eps=1e-7):
        self.params = params
        self.adamw_lr = adamw_lr
        self.muon_lr = muon_lr
        self.adamw_wd = adamw_wd
        self.muon_wd = muon_wd
        self.adamw_beta1 = adamw_beta1
        self.adamw_beta2 = adamw_beta2
        self.muon_beta1 = muon_beta1
        self.muon_beta2 = muon_beta2
        self.eps = eps
        self.red_dim = red_dim
        self.steps = steps         # Newton_Sched steps
        self.Nesterov = Nesterov
        self.i = 0

    def state_dict(self):
        """Flatten each parameter's raw momentum attributes (grad_avg/grad_sqr_avg/v_mean_avg) into a picklable dict, in group/param order."""
        state = []
        for g in self.params:
            for p in g['params']:
                s = {}
                if hasattr(p, 'grad_avg'): s['grad_avg'] = p.grad_avg
                if hasattr(p, 'grad_sqr_avg'): s['grad_sqr_avg'] = p.grad_sqr_avg
                if hasattr(p, 'v_mean_avg'): s['v_mean_avg'] = p.v_mean_avg
                state.append(s)
        return {'i': self.i, 'state': state}

    def load_state_dict(self, state_dict):
        """Restore each param's momentum, matching saved entries to params by
        shape whenever the positional order doesn't line up (the Muon group is
        built from a set, so its order can differ between processes)."""
        self.i = state_dict['i']
        params_flat = [p for g in self.params for p in g['params']]
        state_list  = state_dict['state']
        if len(params_flat) != len(state_list): raise ValueError(f"optimizer state has {len(state_list)} entries but the model has {len(params_flat)} params")

        def _apply_state(p, s):
            if 'grad_avg' in s: p.grad_avg = s['grad_avg']
            if 'grad_sqr_avg' in s: p.grad_sqr_avg = s['grad_sqr_avg']
            if 'v_mean_avg' in s: p.v_mean_avg = s['v_mean_avg']

        # fast path: positional order matches (state restores exactly)
        aligned = all(
            ('grad_avg' not in s) or tuple(s['grad_avg'].shape) == tuple(p.shape)
            for p, s in zip(params_flat, state_list)
        )
        if aligned:
            for p, s in zip(params_flat, state_list):  _apply_state(p, s)
            return

        # fallback: give each entry to the first same-shaped param not yet taken
        used = set()
        for p in params_flat:
            for i, s in enumerate(state_list):
                if i in used: continue
                if 'grad_avg' in s and tuple(s['grad_avg'].shape) == tuple(p.shape):
                    _apply_state(p, s)
                    used.add(i)
                    break

    def step(self, lr_mult=1.):
        """Run one optimizer step across every param group, scaling each group's
        base lr by lr_mult (the schedule ratio for this training step)."""
        # Update params
        with torch.no_grad():
            for g in self.params:
                use_muon = g.get('use_muon', None)
                # Get the hyperparams in group (if None -> set default)
                lr = g.get('lr', self.muon_lr if use_muon else self.adamw_lr) * lr_mult    # base_lr * lr_mult
                weight_decay = g.get('weight_decay', self.muon_wd if use_muon else self.adamw_wd)
                beta1 = g.get('beta1', self.muon_beta1 if use_muon else self.adamw_beta1)
                beta2 = g.get('beta2', self.muon_beta2 if use_muon else self.adamw_beta2)
                eps = g.get('eps', self.eps)
                if use_muon:
                    red_dim = g.get('red_dim', self.red_dim)
                    Nesterov = g.get('Nesterov', self.Nesterov)
                    steps = g.get('steps', self.steps)
                else: red_dim, Nesterov, steps = None, None, None

                for p in g['params']:
                    self.opt_step(p, lr, beta1, beta2, eps, weight_decay, red_dim, Nesterov, steps, use_muon)
        self.i += 1

    def zero_grad(self):
        """Zero every parameter's gradient in-place across all groups."""
        for g in self.params:
            for p in g['params']:
                if p.grad is not None: p.grad.data.zero_()

    def opt_step(self, p, lr, beta1, beta2, eps, wd, red_dim, Nesterov, steps, use_muon):
        """Apply one parameter's update: AdamW branch for 1D/embedding-like params,
        Muon (Nesterov momentum -> row-norm -> Newton-Schulz orthogonalization ->
        variance reduction) branch for 2D+ matrix params."""
        if use_muon is None:
            use_muon = (p.dim() >= 2)   # fallback for old-style groups

        ### 1. -------- AdamW -----------
        if not use_muon:
            if not hasattr(p, 'grad_avg'): p.grad_avg = torch.zeros_like(p.grad.data)
            if not hasattr(p, 'grad_sqr_avg'): p.grad_sqr_avg = torch.zeros_like(p.grad.data)
            p.grad_avg.lerp_(p.grad, 1 - beta1)
            p.grad_sqr_avg.lerp_(p.grad.square(), 1 - beta2)
            unbiased_grad_avg = p.grad_avg / (1 - beta1 ** (self.i+1))
            unbiased_grad_sqr_avg = p.grad_sqr_avg / (1 - beta2 ** (self.i+1))
            update = unbiased_grad_avg / (unbiased_grad_sqr_avg.sqrt() + eps)
            if wd: update += wd * p.data
            p.data.sub_(lr * update)

        ### 2. -------- Muon ------------
        else:
            # Save grad at g
            g = p.grad.data

            # Nesterov momentum
            if not hasattr(p, 'grad_avg'): p.grad_avg = torch.zeros_like(g)
            p.grad_avg.lerp_(g, 1 - beta1)
            unbiased_grad_avg = p.grad_avg / (1 - beta1 ** (self.i+1))
            g = g.lerp(unbiased_grad_avg, beta1) if Nesterov else unbiased_grad_avg

            # Row Normalization ->  To make all rows norms equal (with the same matrix norm), nudge singular numbers slightly (each row diff factor), but it is minor and the speed-up is worth it.
            target = g.norm(dim=(-1, -2), keepdim=True) * (g.size(-2)**-0.5)
            row_norm = g.norm(dim=(-1), keepdim=True)
            g = g * (target / (row_norm + eps))

            # Frobenius norm -> Shrink the whole matrix down so the biggest direction is safely -1 < k < 1,  Newton-Schulz's converges correctly if every direction below 1 at start — feed it something too big and it diverges instead of converging.
            g /= (g.norm(dim=(-1, -2), keepdim=True) * 1.01 + eps)  # 1.01 is just safety so that everything is <1 and <-1 and not =

            # Newton sched
            a, b, c = 3.4445, -4.7750, 2.0315
            for _ in range(steps):
                A = g @ g.mT                # mT transposes the last 2 dims
                B = b * A + c * (A @ A)
                g = a * g + B @ g

            # Muon+ normalization -> Make it same as the SVD where the norm is going to be root of smallest dim
            targ_norm = min(g.size(-2), g.size(-1))**0.5
            current_norm = g.norm(dim=(-1, -2), keepdim=True)
            g = g * (targ_norm / (current_norm + eps))

            # Variance Reduction -> To normalize the rows or columns using EMA as Adam, and at the end it makes the norm of the whole matrix the same as before the variance reduction (same norm that we approximated the matrix to be in Muon+ norm)
            v_mean = g.square().mean(dim=red_dim, keepdim=True)
            red_dim_sz = g.size(red_dim)
            v_norm_sq = v_mean.sum(dim=(-1, -2), keepdim=True) * red_dim_sz
            v_norm = v_norm_sq.sqrt()
            if not hasattr(p, 'v_mean_avg'): p.v_mean_avg = torch.zeros_like(v_mean)
            p.v_mean_avg.lerp_(v_mean, 1 - beta2)
            unbiased_v_mean_avg = p.v_mean_avg / (1 - beta2 ** (self.i+1))
            step_sz = (unbiased_v_mean_avg + eps).rsqrt()
            scaled_sq_sum = (v_mean * red_dim_sz) * step_sz.square()
            v_norm_new = scaled_sq_sum.sum(dim=(-1, -2), keepdim=True).sqrt()
            final_scale = step_sz * (v_norm / (v_norm_new + eps))
            g = g * final_scale

            # Update
            mask = (g * unbiased_grad_avg) >= 0
            update = g + wd * p.data * mask if wd else g
            p.data.sub_(lr * update)  # Make it in place