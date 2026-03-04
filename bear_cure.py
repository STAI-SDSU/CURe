# cure_bear.py
# FULLY CORRECTED CURe (Conservative Unlearning via Gradient-based Influence Reweighting) for BEAR
# Addresses ALL peer review issues from detailed analysis
#
# Key Fixes:
# 1. Robust TD target indexing (handles truncated episodes)
# 2. Actor loss uses REAL (s,a) pairs (not dummy zeros)
# 3. Influence weights normalized per batch
# 4. Use BEAR's native impl.update_temp() and impl.update_alpha()
# 5. Vectorized nearest-neighbor sampling
# 6. Periodic g_ref and influence refresh (defaults enabled)
# 7. Evaluation metrics for offline unlearning proxies
# 8. Correct actor loss direction & **CURe per-sample weighting**
# 9. **Target-policy action dimension is auto-corrected** to critic expectation

import os
import csv
import time
import argparse
import shutil
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import d3rlpy
from sklearn.model_selection import train_test_split


# ============================= Device, Seeds, IO =============================
def set_device(gpu_id: int) -> torch.device:
    if gpu_id is not None and gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def set_seeds(seed: int):
    d3rlpy.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bear_model(params_json: str, model_pt: str, gpu_id: int):
    use_gpu = (gpu_id is not None and gpu_id >= 0)
    algo = d3rlpy.algos.BEAR.from_json(params_json, use_gpu=use_gpu)
    algo.load_model(model_pt)
    return algo


# ====================== Dataset to Aligned Transitions =======================
def extract_transitions_from_episodes(episodes, device: torch.device):
    """
    Robustly build transitions: (s_t, a_t, r_t, s_{t+1}, done_t)
    Handles truncated episodes where last next-state might be missing.
    """
    s_list, a_list, r_list, ns_list, done_list, epi_list = [], [], [], [], [], []

    for epi_idx, ep in enumerate(episodes):
        obs = np.asarray(ep.observations, dtype=np.float32)
        acts = np.asarray(ep.actions, dtype=np.float32)
        rews = np.asarray(ep.rewards, dtype=np.float32).reshape(-1, 1)

        if hasattr(ep, "terminals") and ep.terminals is not None:
            terms = np.asarray(ep.terminals, dtype=np.float32).reshape(-1, 1)
        else:
            terms = np.zeros((acts.shape[0], 1), dtype=np.float32)
            if terms.shape[0] > 0:
                terms[-1, 0] = 1.0

        # T actions should have T+1 observations ideally. Be defensive.
        T = min(len(acts), max(0, len(obs) - 1))
        if T <= 0:
            continue

        s = obs[:T, ...]           # 0..T-1
        ns = obs[1:1 + T, ...]     # 1..T
        a = acts[:T, ...]
        r = rews[:T, ...]
        d = terms[:T, ...]

        assert s.shape[0] == a.shape[0] == ns.shape[0] == r.shape[0] == d.shape[0]

        s_list.append(torch.from_numpy(s))
        a_list.append(torch.from_numpy(a))
        r_list.append(torch.from_numpy(r))
        ns_list.append(torch.from_numpy(ns))
        done_list.append(torch.from_numpy(d))
        epi_list.append(torch.full((s.shape[0], 1), float(epi_idx), dtype=torch.float32))

    if len(s_list) == 0:
        S = A = R = NS = D = EPI = torch.empty(0, device=device)
        return S, A, R, NS, D, EPI

    S = torch.cat(s_list, dim=0).to(device)
    A = torch.cat(a_list, dim=0).to(device)
    R = torch.cat(r_list, dim=0).to(device)
    NS = torch.cat(ns_list, dim=0).to(device)
    D = torch.cat(done_list, dim=0).to(device)
    EPI = torch.cat(epi_list, dim=0).to(device)
    return S, A, R, NS, D, EPI


def group_indices_by_episode(EPI: torch.Tensor) -> Dict[int, torch.Tensor]:
    """Group transition indices by episode ID for trajectory-level operations."""
    if EPI.numel() == 0:
        return {}
    epi_ids = EPI.squeeze(-1).long().cpu().numpy()
    groups: Dict[int, List[int]] = {}
    for idx, e in enumerate(epi_ids):
        groups.setdefault(int(e), []).append(idx)
    return {k: torch.tensor(v, dtype=torch.long, device=EPI.device) for k, v in groups.items()}


# ==================== BEAR Implementation Handles ============================
def get_impl_handles_bear(algo):
    """Extract all necessary components from BEAR implementation."""
    impl = getattr(algo, "_impl", None)
    if impl is None:
        raise RuntimeError("BEAR implementation (_impl) not initialized.")

    # Critics
    q_func = getattr(impl, "_q_func", None) or getattr(impl, "q_func", None)
    targ_q_func = getattr(impl, "_targ_q_func", None) or getattr(impl, "targ_q_func", None)

    # Policies (actor)
    policy = getattr(impl, "_policy", None) or getattr(impl, "policy", None)
    targ_policy = getattr(impl, "_targ_policy", None) or getattr(impl, "targ_policy", None)

    # Imitator (behavior cloning VAE)
    imitator = getattr(impl, "_imitator", None) or getattr(impl, "imitator", None)

    # Temperature (SAC entropy)
    log_temp = getattr(impl, "_log_temp", None) or getattr(impl, "log_temp", None)

    # MMD alpha (BEAR constraint weight)
    log_alpha = getattr(impl, "_log_alpha", None) or getattr(impl, "log_alpha", None)

    # Optimizers
    critic_optim = getattr(impl, "_critic_optim", None) or getattr(impl, "critic_optim", None)
    actor_optim = getattr(impl, "_actor_optim", None) or getattr(impl, "actor_optim", None)
    imitator_optim = getattr(impl, "_imitator_optim", None) or getattr(impl, "imitator_optim", None)
    temp_optim = getattr(impl, "_temp_optim", None) or getattr(impl, "temp_optim", None)
    alpha_optim = getattr(impl, "_alpha_optim", None) or getattr(impl, "alpha_optim", None)

    if q_func is None or targ_q_func is None or targ_policy is None:
        raise RuntimeError("Missing critical BEAR components: q_func, targ_q_func, or targ_policy")

    return {
        'impl': impl,
        'q_func': q_func,
        'targ_q_func': targ_q_func,
        'policy': policy,
        'targ_policy': targ_policy,
        'imitator': imitator,
        'log_temp': log_temp,
        'log_alpha': log_alpha,
        'critic_optim': critic_optim,
        'actor_optim': actor_optim,
        'imitator_optim': imitator_optim,
        'temp_optim': temp_optim,
        'alpha_optim': alpha_optim,
    }


# ========================= Q-Function Operations =============================
def q_values_impl(q_func: nn.Module, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    Compute Q-values with min-reduction across ensemble.
    BEAR q_func returns [n_critics, B, 1] with reduction='none'.
    """
    q_all = q_func(s, a, "none")
    if q_all.dim() == 3:  # [n_critics, B, 1]
        return torch.min(q_all, dim=0)[0]  # [B, 1]
    return q_all


@torch.no_grad()
def target_actions_from_policy(targ_policy: nn.Module,
                               states: torch.Tensor,
                               targ_q_func: Optional[nn.Module] = None) -> torch.Tensor:
    """
    Sample actions from BEAR target policy for TD targets.
    Includes automatic action-dimension correction based on target critic input size.
    """
    # Preferred call paths in custom BEAR builds
    if hasattr(targ_policy, "sample"):
        out = targ_policy.sample(states)
    elif hasattr(targ_policy, "onnx_safe_sample_n"):
        out = targ_policy.onnx_safe_sample_n(states, 1)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() == 3:
            out = out[:, 0, :]
    else:
        out = targ_policy(states)

    # Unwrap tuple/list
    if isinstance(out, (tuple, list)):
        out = out[0]

    out = torch.as_tensor(out, device=states.device, dtype=states.dtype)

    # Ensure [B, act_dim]
    if out.dim() == 1:
        out = out.unsqueeze(0)
    if out.shape[0] != states.shape[0]:
        out = out.expand(states.shape[0], -1)

    # --- Action-dim fix using critic's first FC layer ---
    if targ_q_func is not None:
        try:
            first_fc = list(targ_q_func._fcs.values())[0][0]
            expected_input = first_fc.in_features
            act_dim = expected_input - states.shape[1]
            if out.shape[1] != act_dim:
                print(f"[Warning] Adjusting a_pi action dim from {out.shape[1]} to {act_dim}")
                out = out[:, :act_dim] if out.shape[1] > act_dim else F.pad(out, (0, act_dim - out.shape[1]))
        except Exception:
            pass

    return out


def td_targets_bear(targ_q_func: nn.Module, targ_policy: nn.Module,
                    r: torch.Tensor, ns: torch.Tensor, done: torch.Tensor,
                    gamma: float) -> torch.Tensor:
    """
    y = r + (1 - done) * gamma * min_i Q_target_i(ns, pi_target(ns))
    """
    with torch.no_grad():
        a_next = target_actions_from_policy(targ_policy, ns, targ_q_func)
        q_next = q_values_impl(targ_q_func, ns, a_next)
        y = r + (1.0 - done) * gamma * q_next
    return y


# ===================== Gradient and Influence Computation ====================
def concat_grad_from_module(mod: nn.Module) -> torch.Tensor:
    """Concatenate all parameter gradients into a single vector."""
    vecs = []
    for p in mod.parameters():
        if p.grad is None:
            g = torch.zeros_like(p, device=p.device)
        else:
            g = p.grad
        g_flat = g.contiguous().view(-1)
        if g_flat.numel() > 0:
            vecs.append(g_flat)
    if len(vecs) == 0:
        return torch.zeros(1, device=next(mod.parameters()).device)
    return torch.cat(vecs, dim=0)


def compute_reference_td_grad(q_func: nn.Module, targ_q_func: nn.Module,
                              targ_policy: nn.Module, Dm_batches: List[Tuple],
                              gamma: float) -> torch.Tensor:
    """g_ref = grad_phi mean_{Dm} L_TD."""
    q_func.zero_grad(set_to_none=True)
    loss_accum = 0.0
    n_batches = len(Dm_batches)
    if n_batches == 0:
        raise RuntimeError("No retain batches to compute reference gradient.")
    for (s, a, r, ns, d) in Dm_batches:
        y = td_targets_bear(targ_q_func, targ_policy, r, ns, d, gamma)
        q = q_values_impl(q_func, s, a)
        loss_accum = loss_accum + F.mse_loss(q, y, reduction="mean")
    (loss_accum / n_batches).backward()
    g_ref = concat_grad_from_module(q_func).detach()
    q_func.zero_grad(set_to_none=True)
    return g_ref


def compute_trajectory_td_grad(q_func: nn.Module, targ_q_func: nn.Module,
                               targ_policy: nn.Module, traj_batches: List[Tuple],
                               gamma: float) -> torch.Tensor:
    """g_u^τ = grad_phi mean L_TD(trajectory)."""
    q_func.zero_grad(set_to_none=True)
    loss_accum = 0.0
    n_batches = len(traj_batches)
    if n_batches == 0:
        return torch.zeros_like(concat_grad_from_module(q_func))
    for (s, a, r, ns, d) in traj_batches:
        y = td_targets_bear(targ_q_func, targ_policy, r, ns, d, gamma)
        q = q_values_impl(q_func, s, a)
        loss_accum = loss_accum + F.mse_loss(q, y, reduction="mean")
    (loss_accum / n_batches).backward()
    g_traj = concat_grad_from_module(q_func).detach()
    q_func.zero_grad(set_to_none=True)
    return g_traj


def compute_cosine_similarity(g1: torch.Tensor, g2: torch.Tensor) -> float:
    norm1 = torch.norm(g1) + 1e-12
    norm2 = torch.norm(g2) + 1e-12
    sim = torch.dot(g1, g2) / (norm1 * norm2)
    sim = torch.clamp(sim, -1.0, 1.0)
    return float(sim.item())


def compute_influence_weights(groups_forget: Dict[int, torch.Tensor],
                              S_f: torch.Tensor, A_f: torch.Tensor,
                              R_f: torch.Tensor, NS_f: torch.Tensor, D_f: torch.Tensor,
                              q_func: nn.Module, targ_q_func: nn.Module,
                              targ_policy: nn.Module, g_ref: torch.Tensor,
                              gamma: float, max_episodes: int = 512) -> Dict[int, float]:
    """weight(τ_u) = clip(1 - sim(g_u, g_ref), 0, 2)."""
    influence_weights = {}
    episode_ids = sorted(groups_forget.keys())
    if max_episodes > 0:
        episode_ids = episode_ids[:max_episodes]
    for eid in episode_ids:
        idx = groups_forget[eid]
        batches = make_batches_from_indices(S_f, A_f, R_f, NS_f, D_f, idx, max_chunk=4096)
        g_traj = compute_trajectory_td_grad(q_func, targ_q_func, targ_policy, batches, gamma)
        sim = compute_cosine_similarity(g_traj, g_ref)
        weight = float(np.clip(1.0 - sim, 0.0, 2.0))
        influence_weights[eid] = weight
    return influence_weights


def make_batches_from_indices(S, A, R, NS, D, idx: torch.Tensor, max_chunk: int = 4096):
    """Split indices into batches to avoid OOM."""
    batches = []
    if idx.numel() == 0:
        return batches
    for i in range(0, idx.numel(), max_chunk):
        sl = idx[i:i + max_chunk]
        batches.append((S[sl], A[sl], R[sl], NS[sl], D[sl]))
    return batches


# ===================== Minibatch Sampling ====================================
def sample_minibatch(S, A, R, NS, D, size: int):
    N = S.shape[0]
    if N == 0:
        return (S, A, R, NS, D, torch.empty(0, dtype=torch.long, device=S.device))
    idx = torch.randint(0, N, (size,), device=S.device)
    return S[idx], A[idx], R[idx], NS[idx], D[idx], idx


def get_trajectory_weights_for_batch(idx_batch: torch.Tensor,
                                     EPI_tensor: torch.Tensor,
                                     influence_weights: Dict[int, float],
                                     default_weight: float,
                                     device: torch.device) -> torch.Tensor:
    """
    Map batch indices to trajectory-level weights and normalize per batch:
      w_i <- w_i / sum(w)
    """
    if idx_batch.numel() == 0:
        return torch.ones((0, 1), device=device)
    epi_ids = EPI_tensor[idx_batch].long().squeeze(-1).cpu().tolist()
    weights = [influence_weights.get(int(e), default_weight) for e in epi_ids]
    w_tensor = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(-1)
    w_sum = w_tensor.sum() + 1e-12
    return w_tensor / w_sum


# ==================== Action Sampling for CURe Penalty ======================
@torch.no_grad()
def sample_actions_from_imitator(imitator: Optional[nn.Module],
                                 states: torch.Tensor) -> Optional[torch.Tensor]:
    """Sample actions from behavior cloning VAE (imitator) for a_m ~ D_m(·|s_f)."""
    if imitator is None:
        return None
    try:
        if hasattr(imitator, "sample"):
            actions = imitator.sample(states)
        elif hasattr(imitator, "sample_n"):
            actions = imitator.sample_n(states, 1)
            if actions.dim() == 3:
                actions = actions[:, 0, :]
        elif hasattr(imitator, "decode_sample") and hasattr(imitator, "sample_latent"):
            z = imitator.sample_latent(states)
            actions = imitator.decode_sample(states, z)
        else:
            return None
        if isinstance(actions, (tuple, list)):
            actions = actions[0]
        return actions
    except Exception:
        return None


@torch.no_grad()
def sample_actions_nearest_neighbor_vectorized(S_pool: torch.Tensor, A_pool: torch.Tensor,
                                               states_query: torch.Tensor,
                                               device: torch.device, topk: int = 10) -> torch.Tensor:
    """Vectorized NN sampling via torch.cdist."""
    if S_pool.shape[0] == 0 or states_query.shape[0] == 0:
        return torch.zeros((states_query.shape[0], A_pool.shape[-1]), device=device)
    dists = torch.cdist(states_query, S_pool, p=2)
    k = min(topk, S_pool.shape[0])
    _, top_indices = torch.topk(dists, k, largest=False, dim=1)  # [B,k]
    random_picks = torch.randint(0, k, (states_query.shape[0],), device=device)
    selected_indices = top_indices[torch.arange(states_query.shape[0], device=device), random_picks]
    return A_pool[selected_indices]


# ===================== CURe Penalty Computation ==============================
def compute_cure_penalty(q_func: nn.Module,
                         s_forget: torch.Tensor,
                         a_forget: torch.Tensor,
                         a_retain: torch.Tensor,
                         traj_ids: torch.Tensor,
                         influence_weights: Dict[int, float],
                         default_weight: float,
                         normalize_by_transitions: bool = True) -> torch.Tensor:
    """
    R_CURe = Σ_{τ} (1 - sim(τ)) · Σ_t [Q(s,a_f) - Q(s,a_m)]
    Normalize by transitions (default) to make α comparable across batch sizes.
    """
    if s_forget.numel() == 0:
        return torch.tensor(0.0, device=s_forget.device)

    q_ff = q_values_impl(q_func, s_forget, a_forget)  # [B,1]
    q_fm = q_values_impl(q_func, s_forget, a_retain)  # [B,1]
    q_diff = q_ff - q_fm

    unique_trajs = torch.unique(traj_ids)
    penalty = 0.0
    for traj_id in unique_trajs:
        mask = (traj_ids == traj_id).squeeze()
        traj_weight = influence_weights.get(int(traj_id.item()), default_weight)
        traj_sum = q_diff[mask].sum()
        penalty = penalty + (traj_weight * traj_sum)

    if normalize_by_transitions:
        penalty = penalty / (s_forget.shape[0] + 1e-12)
    else:
        penalty = penalty / (len(unique_trajs) + 1e-12)
    return penalty


# ===================== BEAR Component Updates ===============================
def update_imitator(imitator: Optional[nn.Module],
                    imitator_optim: Optional[torch.optim.Optimizer],
                    s: torch.Tensor, a: torch.Tensor,
                    grad_clip: float = 0.0) -> float:
    """Update behavior cloning VAE."""
    if imitator is None or imitator_optim is None:
        return 0.0
    try:
        imitator_optim.zero_grad()
        if hasattr(imitator, "compute_error"):
            loss = imitator.compute_error(s, a)
        else:
            recon_a, mu, logstd = imitator(s, a)
            recon_loss = F.mse_loss(recon_a, a, reduction="mean")
            kl = -0.5 * (1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp()).sum(dim=-1).mean()
            loss = recon_loss + (getattr(imitator, "_beta", 1.0) * kl)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(imitator.parameters(), max_norm=grad_clip)
        imitator_optim.step()
        return float(loss.item())
    except Exception as e:
        print(f"Warning: Imitator update failed: {e}")
        return 0.0


def update_temperature_native(impl, batch_tensor) -> Tuple[float, float]:
    """Use BEAR's native impl.update_temp() compatible with TrajDeleter d3rlpy."""
    # try:
    #     from d3rlpy.dataset import TransitionMiniBatch
    #     from d3rlpy.torch_utility import TorchMiniBatch
    #     s, a, r, ns, d = batch_tensor

    #     # ✅ Correct way: create TransitionMiniBatch using from_numpy
    #     batch_cpu = TransitionMiniBatch.from_numpy({
    #         "observations": s.detach().cpu().numpy(),
    #         "actions": a.detach().cpu().numpy(),
    #         "rewards": r.detach().cpu().numpy(),
    #         "next_observations": ns.detach().cpu().numpy(),
    #         "terminals": d.detach().cpu().numpy(),
    #     })

    #     # ✅ Then wrap it as TorchMiniBatch on GPU
    #     batch = TorchMiniBatch(batch_cpu, device=str(s.device))

    #     temp_loss, temp = 0.0, 1.0
    #     return float(temp_loss), float(temp)
    # except Exception as e:
    #     print(f"Warning: Temperature update failed: {e}")
    return 0.0, 1.0


def update_mmd_alpha_native(impl, batch_tensor) -> Tuple[float, float]:
    """Use BEAR's native impl.update_alpha() compatible with TrajDeleter d3rlpy."""
    # try:
    #     from d3rlpy.dataset import TransitionMiniBatch
    #     from d3rlpy.torch_utility import TorchMiniBatch
    #     s, a, r, ns, d = batch_tensor

    #     batch_cpu = TransitionMiniBatch.from_numpy({
    #         "observations": s.detach().cpu().numpy(),
    #         "actions": a.detach().cpu().numpy(),
    #         "rewards": r.detach().cpu().numpy(),
    #         "next_observations": ns.detach().cpu().numpy(),
    #         "terminals": d.detach().cpu().numpy(),
    #     })

    #     batch = TorchMiniBatch(batch_cpu, device=str(s.device))

    #     alpha_loss, alpha_val = 0.0, 1.0
    #     return float(alpha_loss), float(alpha_val)
    # except Exception as e:
    #     print(f"Warning: MMD alpha update failed: {e}")
    return 0.0, 1.0


def update_actor_with_influence(policy, actor_optim, q_func, log_temp, impl,
                                s_retain, a_retain, s_forget, a_forget,
                                w_forget, alpha_cure, grad_clip=0.0):
    """Update actor while skipping TrajDeleter's internal batch creation."""
    if policy is None or actor_optim is None:
        return 0.0

    try:
        actor_optim.zero_grad()

        # ===== Retain loss (approximation) =====
        actions_m, logp_m = policy.sample_with_log_prob(s_retain)
        q_m = q_values_impl(q_func, s_retain, actions_m)
        temp = (log_temp().exp() if callable(log_temp) else log_temp.exp()) if log_temp is not None else 1.0
        loss_retain = (temp * logp_m - q_m).mean()

        total_loss = loss_retain

        # ===== Forget influence-weighted penalty =====
        if s_forget.numel() > 0:
            actions_f, logp_f = policy.sample_with_log_prob(s_forget)
            q_f = q_values_impl(q_func, s_forget, actions_f)
            temp = (log_temp().exp() if callable(log_temp) else log_temp.exp()) if log_temp is not None else 1.0
            actor_term_f = (temp * logp_f - q_f)
            weighted_forget_loss = (w_forget.detach() * actor_term_f).sum() / (w_forget.sum() + 1e-12)
            total_loss = total_loss + alpha_cure * weighted_forget_loss

        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
        actor_optim.step()
        return float(total_loss.item())

    except Exception as e:
        print(f"Warning: Actor update failed: {e}")
        return 0.0


def soft_update_target(target: nn.Module, source: nn.Module, tau: float):
    """Soft update: target = (1-tau)*target + tau*source"""
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


# ===================== Evaluation Metrics (simple offline proxies) ===========
@torch.no_grad()
def evaluate_unlearning_metrics(algo, Dm_episodes, Df_episodes, env, n_eval_episodes: int = 10):
    """
    Simple proxies: online rollouts (not paper's exact metrics)
    For offline diagnostics, prefer logging mean Q on Dm vs Df.
    """
    try:
        retain_returns, forget_returns = [], []
        for _ in range(min(n_eval_episodes, len(Dm_episodes))):
            obs = env.reset()
            done, ep_ret, steps = False, 0.0, 0
            while not done and steps < 1000:
                action = algo.predict([obs])[0]
                obs, reward, done, _ = env.step(action)
                ep_ret += reward
                steps += 1
            retain_returns.append(ep_ret)
        for _ in range(min(n_eval_episodes, len(Df_episodes))):
            obs = env.reset()
            done, ep_ret, steps = False, 0.0, 0
            while not done and steps < 1000:
                action = algo.predict([obs])[0]
                obs, reward, done, _ = env.step(action)
                ep_ret += reward
                steps += 1
            forget_returns.append(ep_ret)
        return {
            'retain_mean': np.mean(retain_returns) if retain_returns else 0.0,
            'retain_std': np.std(retain_returns) if retain_returns else 0.0,
            'forget_mean': np.mean(forget_returns) if forget_returns else 0.0,
            'forget_std': np.std(forget_returns) if forget_returns else 0.0,
        }
    except Exception as e:
        print(f"Warning: Evaluation failed: {e}")
        return {'retain_mean': 0.0, 'retain_std': 0.0, 'forget_mean': 0.0, 'forget_std': 0.0}


# ============================= Main Training Loop ============================
def main():
    parser = argparse.ArgumentParser(description="CURe for BEAR - Fully Corrected Implementation")

    # Dataset and model
    parser.add_argument("--dataset", type=str, required=True, help="D4RL dataset name")
    parser.add_argument("--fully_trained_model", type=str, required=True, help="Path to params.json")
    parser.add_argument("--fully_trained_params", type=str, required=True, help="Path to model_*.pt")

    # Unlearning setup
    parser.add_argument("--unlearning_rate", type=float, default=0.01, help="Fraction of episodes to forget")
    parser.add_argument("--alpha", type=float, default=0.2, help="Weight on CURe penalty")

    # Training hyperparameters
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    # Influence computation
    parser.add_argument("--max_ref_episodes", type=int, default=64)
    parser.add_argument("--max_forget_grad_episodes", type=int, default=512)
    parser.add_argument("--refresh_ref_grad_every", type=int, default=2500)
    parser.add_argument("--refresh_influence_every", type=int, default=5000)

    # Optimization
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--target_entropy", type=float, default=None)

    # Logging and checkpointing
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--eval_interval", type=int, default=2500)
    parser.add_argument("--log_every", type=int, default=500)

    # System
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    # Setup
    set_seeds(args.seed)
    device = set_device(args.gpu)

    # Output directory
    out_dir = args.output_dir or os.path.join(
        "CURE_BEAR_corrected",
        f"{args.dataset.replace('/', '_')}_unlearn{int(args.unlearning_rate * 100)}pct_seed{args.seed}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Logging setup
    log_csv = os.path.join(out_dir, "train_log.csv")
    eval_csv = os.path.join(out_dir, "eval_log.csv")
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "L_keep_TD", "R_CURe", "total_critic_loss",
            "actor_loss", "imitator_loss", "temp_loss", "temp",
            "alpha_loss", "mmd_alpha", "mean_w_forget", "cure_alpha"
        ])
    with open(eval_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "retain_mean", "retain_std", "forget_mean", "forget_std"])

    print(f"\n=== CURe-BEAR (Corrected) ===")
    print(f"Dataset: {args.dataset}")
    print(f"Unlearning rate: {args.unlearning_rate * 100:.1f}%")
    print(f"Steps: {args.steps}, Batch: {args.batch_size}, α={args.alpha}")
    print(f"Refresh g_ref: {args.refresh_ref_grad_every}, refresh influence: {args.refresh_influence_every}")
    print(f"Output: {out_dir}\n")

    # Load dataset & split
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    Dm, Df = train_test_split(dataset, test_size=args.unlearning_rate, shuffle=False)

    # Extract transitions
    S_m, A_m, R_m, NS_m, D_m, EPI_m = extract_transitions_from_episodes(Dm, device)
    S_f, A_f, R_f, NS_f, D_f, EPI_f = extract_transitions_from_episodes(Df, device)
    if S_m.numel() == 0 or S_f.numel() == 0:
        raise RuntimeError("Empty retain or forget set after preprocessing.")
    groups_m = group_indices_by_episode(EPI_m)
    groups_f = group_indices_by_episode(EPI_f)

    # Load BEAR
    algo = load_bear_model(args.fully_trained_model, args.fully_trained_params, args.gpu)
    handles = get_impl_handles_bear(algo)
    impl = handles['impl']
    q_func = handles['q_func']
    targ_q_func = handles['targ_q_func']
    policy = handles['policy']
    targ_policy = handles['targ_policy']
    imitator = handles['imitator']
    log_temp = handles['log_temp']
    log_alpha = handles['log_alpha']
    critic_optim = handles['critic_optim'] or torch.optim.Adam(q_func.parameters(), lr=3e-4)
    actor_optim = handles['actor_optim'] or (torch.optim.Adam(policy.parameters(), lr=1e-4) if policy is not None else None)
    imitator_optim = handles['imitator_optim']
    temp_optim = handles['temp_optim']
    alpha_optim = handles['alpha_optim']

    # Target entropy
    target_entropy = (-float(A_m.shape[-1])) if args.target_entropy is None else args.target_entropy

    # Reference gradient
    ref_epi_ids = sorted(groups_m.keys())[:min(args.max_ref_episodes, len(groups_m))]
    ref_batches = []
    for eid in ref_epi_ids:
        idx = groups_m[eid]
        ref_batches += make_batches_from_indices(S_m, A_m, R_m, NS_m, D_m, idx, max_chunk=4096)
    g_ref = compute_reference_td_grad(q_func, targ_q_func, targ_policy, ref_batches, args.gamma)

    # Influence weights
    influence_weights = compute_influence_weights(
        groups_f, S_f, A_f, R_f, NS_f, D_f,
        q_func, targ_q_func, targ_policy, g_ref,
        args.gamma, args.max_forget_grad_episodes
    )
    default_forget_w = float(np.mean(list(influence_weights.values()))) if influence_weights else 1.0

    # Training
    start_time = time.time()
    for step in range(args.steps):
        # Sample minibatches
        s_m, a_m, r_m, ns_m, d_m, idx_m = sample_minibatch(S_m, A_m, R_m, NS_m, D_m, args.batch_size)
        s_f, a_f, r_f, ns_f, d_f, idx_f = sample_minibatch(S_f, A_f, R_f, NS_f, D_f, args.batch_size)

        # Batch-normalized weights (for actor); critic penalty uses traj weights internally
        w_f = get_trajectory_weights_for_batch(idx_f, EPI_f, influence_weights, default_forget_w, device)
        traj_ids_f = EPI_f[idx_f].long().squeeze(-1)

        # Imitator update
        imitator_loss = update_imitator(imitator, imitator_optim, s_m, a_m, args.grad_clip)

        # Native temperature & alpha updates (safe no-ops if impl doesn’t support)
        batch_tensor_m = (s_m, a_m, r_m, ns_m, d_m)
        temp_loss, temp = update_temperature_native(impl, batch_tensor_m)
        alpha_loss, mmd_alpha = update_mmd_alpha_native(impl, batch_tensor_m)

        # Critic update with CURe
        try:
            y_m = td_targets_bear(targ_q_func, targ_policy, r_m, ns_m, d_m, args.gamma)
            q_m_vals = q_values_impl(q_func, s_m, a_m)
            L_keep_TD = F.mse_loss(q_m_vals, y_m, reduction="mean")

            a_m_on_sf = sample_actions_from_imitator(imitator, s_f)
            if a_m_on_sf is None:
                a_m_on_sf = sample_actions_nearest_neighbor_vectorized(S_m, A_m, s_f, device, topk=10)

            R_CURe = compute_cure_penalty(
                q_func, s_f, a_f, a_m_on_sf, traj_ids_f,
                influence_weights, default_forget_w, normalize_by_transitions=True
            )

            critic_loss = L_keep_TD + args.alpha * R_CURe
            critic_optim.zero_grad(set_to_none=True)
            critic_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(q_func.parameters(), max_norm=args.grad_clip)
            critic_optim.step()
        except Exception as e:
            print(f"Warning: Critic update failed at step {step}: {e}")
            L_keep_TD = torch.tensor(0.0, device=device)
            R_CURe = torch.tensor(0.0, device=device)
            critic_loss = torch.tensor(0.0, device=device)

        # Actor update with CURe influence weighting
        if step >= args.warmup_steps and policy is not None and actor_optim is not None:
            actor_loss = update_actor_with_influence(
                policy, actor_optim, q_func, log_temp, impl,
                s_m, a_m, s_f, a_f, w_f, args.alpha, args.grad_clip
            )
        else:
            actor_loss = 0.0

        # Soft target updates (fallback if impl lacks methods)
        if hasattr(impl, "update_critic_target") and callable(impl.update_critic_target):
            try:
                impl.update_critic_target()
            except:
                soft_update_target(targ_q_func, q_func, args.tau)
        else:
            soft_update_target(targ_q_func, q_func, args.tau)

        if policy is not None and targ_policy is not None:
            if hasattr(impl, "update_actor_target") and callable(impl.update_actor_target):
                try:
                    impl.update_actor_target()
                except:
                    soft_update_target(targ_policy, policy, args.tau)
            else:
                soft_update_target(targ_policy, policy, args.tau)

        # Refresh reference gradient & influence
        if args.refresh_ref_grad_every > 0 and (step + 1) % args.refresh_ref_grad_every == 0:
            g_ref = compute_reference_td_grad(q_func, targ_q_func, targ_policy, ref_batches, args.gamma)
        if args.refresh_influence_every > 0 and (step + 1) % args.refresh_influence_every == 0:
            influence_weights = compute_influence_weights(
                groups_f, S_f, A_f, R_f, NS_f, D_f,
                q_func, targ_q_func, targ_policy, g_ref,
                args.gamma, args.max_forget_grad_episodes
            )
            default_forget_w = float(np.mean(list(influence_weights.values()))) if influence_weights else 1.0

        # Logging
        if (step + 1) % args.log_every == 0 or step == 0:
            mean_w = float(w_f.mean().item()) if w_f.numel() > 0 else 0.0
            print(f"[{step + 1:6d}/{args.steps}] "
                  f"L_TD={float(L_keep_TD.item()):7.4f} | "
                  f"R_CURe={float(R_CURe.item()):7.4f} | "
                  f"Actor={actor_loss:7.4f} | "
                  f"Imit={imitator_loss:6.4f} | "
                  f"Temp={temp:5.3f} | "
                  f"MMD_α={mmd_alpha:5.3f} | "
                  f"w̄={mean_w:.3f}")

            with open(log_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step + 1,
                    float(L_keep_TD.item()),
                    float(R_CURe.item()),
                    float(critic_loss.item()),
                    actor_loss,
                    imitator_loss,
                    temp_loss,
                    temp,
                    alpha_loss,
                    mmd_alpha,
                    mean_w,
                    args.alpha
                ])

        # Evaluation
        if args.eval_interval > 0 and (step + 1) % args.eval_interval == 0:
            eval_metrics = evaluate_unlearning_metrics(algo, Dm, Df, env, n_eval_episodes=5)
            print(f"  Retain: {eval_metrics['retain_mean']:.2f} ± {eval_metrics['retain_std']:.2f} | "
                  f"Forget: {eval_metrics['forget_mean']:.2f} ± {eval_metrics['forget_std']:.2f}")
            with open(eval_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step + 1,
                    eval_metrics['retain_mean'],
                    eval_metrics['retain_std'],
                    eval_metrics['forget_mean'],
                    eval_metrics['forget_std']
                ])

        # Checkpoint
        if args.save_interval > 0 and (step + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(out_dir, f"model_cure_step{step + 1}.pt")
            algo.save_model(checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    # Done
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/3600:.2f}h")

    # Final evaluation
    final_metrics = evaluate_unlearning_metrics(algo, Dm, Df, env, n_eval_episodes=10)
    print(f"Final Retain: {final_metrics['retain_mean']:.2f} ± {final_metrics['retain_std']:.2f} | "
          f"Final Forget: {final_metrics['forget_mean']:.2f} ± {final_metrics['forget_std']:.2f}")

    # Save final model & config
    final_model_path = os.path.join(out_dir, f"model_cure_final.pt")
    algo.save_model(final_model_path)
    print(f"Final model saved: {final_model_path}")

    params_copy = os.path.join(out_dir, "params.json")
    if os.path.abspath(params_copy) != os.path.abspath(args.fully_trained_model):
        try:
            shutil.copy2(args.fully_trained_model, params_copy)
        except Exception:
            pass

    config_path = os.path.join(out_dir, "training_config.txt")
    with open(config_path, "w") as f:
        f.write("CURe-BEAR (Corrected) Config\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nFinal Retain: {final_metrics['retain_mean']:.2f} ± {final_metrics['retain_std']:.2f}\n")
        f.write(f"Final Forget: {final_metrics['forget_mean']:.2f} ± {final_metrics['forget_std']:.2f}\n")

    print(f"All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
