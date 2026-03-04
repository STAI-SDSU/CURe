# cure_bcq.py
# CURe unlearning for BCQ (critic-only by default), TrajDeleter custom d3rlpy
# - TD targets use BCQ's own target estimator (impl.compute_target) when available
# - CURe penalty computed on the SAME forget states: Q(s_f, a_f) - Q(s_f, a_m), a_m ~ Dm(·|s_f) via NN sampler
# - Influence weights from cosine similarity of TD gradients (paper spec)
# - No actor update (paper's optional note for actor-critic; BCQ has specialized objectives)
#
# Usage (paper-exact critic-only):
#   python cure_bcq.py \
#     --dataset hopper-medium-expert-v0 \
#     --fully_trained_model ./Fully_trained/hopper-medium-expert-v0/BCQ/params.json \
#     --fully_trained_params ./Fully_trained/hopper-medium-expert-v0/BCQ/model_1000000.pt \
#     --unlearning_rate 0.01 --gpu 0 --steps 10000 --save_interval 1000 \
#     --output_dir ./CURE_BCQ_runs/hopper-medium-expert-v0/

import os
import argparse
import shutil
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import d3rlpy
from sklearn.model_selection import train_test_split
import time


# ----------------------------- utils: device & IO -----------------------------
def set_device(gpu_id: int) -> torch.device:
    if gpu_id is not None and gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def load_bcq_model(params_json: str, model_pt: str, gpu_id: int):
    use_gpu = (gpu_id is not None and gpu_id >= 0)
    algo = d3rlpy.algos.BCQ.from_json(params_json, use_gpu=use_gpu)
    algo.load_model(model_pt)
    return algo


# ----------------------- dataset to aligned transitions -----------------------
def extract_transitions_from_episodes(episodes, device: torch.device):
    """
    Build T-1 SARSA-style transitions per episode:
      s_t = obs[t],       a_t  = acts[t],
      r_t = rews[t],      d_t  = terms[t],
      s_{t+1} = obs[t+1], a_{t+1} = acts[t+1]

    Returns: S, A, R, NS, NA, D, EPI  (all concatenated across episodes)
    """
    s_list, a_list, r_list, ns_list, na_list, done_list, epi_list = [], [], [], [], [], [], []

    for epi_idx, ep in enumerate(episodes):
        obs = np.asarray(ep.observations, dtype=np.float32)              # [T+1, ...]
        acts = np.asarray(ep.actions, dtype=np.float32)                  # [T, act_dim]
        rews = np.asarray(ep.rewards, dtype=np.float32).reshape(-1, 1)   # [T, 1]
        if hasattr(ep, "terminals") and ep.terminals is not None:
            terms = np.asarray(ep.terminals, dtype=np.float32).reshape(-1, 1)  # [T,1]
        else:
            terms = np.zeros((acts.shape[0], 1), dtype=np.float32)
            if terms.shape[0] > 0:
                terms[-1, 0] = 1.0

        T = acts.shape[0]
        if T < 2:
            continue

        # T-1 aligned rows using same T base
        s   = obs[:T-1, ...]   # 0 .. T-2
        ns  = obs[1:T,  ...]   # 1 .. T-1
        a   = acts[:T-1, ...]
        na  = acts[1:T,  ...]  # next action a_{t+1}
        r   = rews[:T-1, ...]
        d   = terms[:T-1, ...]

        assert s.shape[0] == a.shape[0] == ns.shape[0] == na.shape[0] == r.shape[0] == d.shape[0], \
            "Alignment bug in episode slicing."

        s_list.append(torch.from_numpy(s))
        a_list.append(torch.from_numpy(a))
        r_list.append(torch.from_numpy(r))
        ns_list.append(torch.from_numpy(ns))
        na_list.append(torch.from_numpy(na))
        done_list.append(torch.from_numpy(d))
        epi_list.append(torch.full((s.shape[0], 1), float(epi_idx), dtype=torch.float32))

    if len(s_list) == 0:
        S = A = R = NS = NA = D = EPI = torch.empty(0, device=device)
        return S, A, R, NS, NA, D, EPI

    S  = torch.cat(s_list,  dim=0).to(device)
    A  = torch.cat(a_list,  dim=0).to(device)
    R  = torch.cat(r_list,  dim=0).to(device)
    NS = torch.cat(ns_list, dim=0).to(device)
    NA = torch.cat(na_list, dim=0).to(device)
    D  = torch.cat(done_list, dim=0).to(device)
    EPI= torch.cat(epi_list,  dim=0).to(device)
    return S, A, R, NS, NA, D, EPI


def group_indices_by_episode(EPI: torch.Tensor) -> Dict[int, torch.Tensor]:
    if EPI.numel() == 0:
        return {}
    epi_ids = EPI.squeeze(-1).long().cpu().numpy()
    groups: Dict[int, List[int]] = {}
    for idx, e in enumerate(epi_ids):
        groups.setdefault(int(e), []).append(idx)
    return {k: torch.tensor(v, dtype=torch.long, device=EPI.device) for k, v in groups.items()}


# ------------------- BCQ impl handles and core primitives --------------------
def get_impl_handles_bcq(algo):
    impl = getattr(algo, "_impl", None)
    if impl is None:
        raise RuntimeError("BCQ implementation (_impl) not initialized after load_model().")

    q_func = getattr(impl, "_q_func", None) or getattr(impl, "q_func", None)
    targ_q_func = getattr(impl, "_targ_q_func", None) or getattr(impl, "targ_q_func", None)

    # These exist in BCQ but we don't strictly need them unless we build targets manually
    policy = getattr(impl, "_policy", None) or getattr(impl, "policy", None)
    imitator = getattr(impl, "_imitator", None) or getattr(impl, "imitator", None)

    if q_func is None or targ_q_func is None:
        raise RuntimeError("Missing _q_func/_targ_q_func on BCQ impl.")

    return impl, q_func, targ_q_func, policy, imitator


def q_values_impl(q_func: nn.Module, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    BCQ (custom d3rlpy) uses a single q_func that returns [n_critics, B, 1] if reduction='none'.
    We min-reduce across critics to match standard conservative usage.
    """
    q_all = q_func(s, a, "none")
    return torch.min(q_all, dim=0)[0] if q_all.dim() == 3 else q_all


# TD target via BCQ's own target builder when available
class _MiniBatch:
    __slots__ = ("next_observations",)
    def __init__(self, next_observations: torch.Tensor):
        self.next_observations = next_observations


def td_targets_bcq_impl(impl,
                        targ_q_func: nn.Module,
                        r: torch.Tensor, ns: torch.Tensor, na: torch.Tensor,
                        done: torch.Tensor, gamma: float,
                        prefer_impl_target: bool = True) -> torch.Tensor:
    """
    y = r + (1 - done) * gamma * Q_target(ns, a'), where a' is chosen via BCQ's target logic.
    Prefer calling impl.compute_target if available (handles VAE + perturbation + max-over-samples).
    Fallback to SARSA with dataset next-action.
    """
    with torch.no_grad():
        if prefer_impl_target and hasattr(impl, "compute_target"):
            # impl.compute_target returns target Q-values (no r/d yet)
            mb = _MiniBatch(ns)
            q_next = impl.compute_target(mb)  # [B,1]
        else:
            # Fallback: SARSA on dataset next action
            q_next = q_values_impl(targ_q_func, ns, na)
        y = r + (1.0 - done) * gamma * q_next
    return y


# ---------------------- influence & CURe penalty utils -----------------------
def concat_grad_from_module(mod: nn.Module) -> torch.Tensor:
    vecs = []
    for p in mod.parameters():
        g = p.grad if (p.grad is not None) else torch.zeros_like(p)
        vecs.append(g.view(-1))
    return torch.cat(vecs) if len(vecs) > 0 else torch.tensor([])


def compute_reference_td_grad_bcq(impl, q_func: nn.Module, targ_q_func: nn.Module,
                                  Dm_batches: List[Tuple[torch.Tensor, ...]],
                                  gamma: float) -> torch.Tensor:
    """
    g_ref = grad_phi sum_{Dm} MSE(Q_phi(s,a), y_bcq(ns))  with y_bcq via impl.compute_target
    """
    q_func.zero_grad(set_to_none=True)
    total = 0.0
    for (s, a, r, ns, na, d) in Dm_batches:
        y = td_targets_bcq_impl(impl, targ_q_func, r, ns, na, d, gamma, prefer_impl_target=True)
        q = q_values_impl(q_func, s, a)
        total = total + F.mse_loss(q, y, reduction="sum")
    total.backward()
    gref = concat_grad_from_module(q_func).detach()
    q_func.zero_grad(set_to_none=True)
    return gref


def compute_forget_td_grad_bcq(impl, q_func: nn.Module, targ_q_func: nn.Module,
                               batch_list: List[Tuple[torch.Tensor, ...]],
                               gamma: float) -> torch.Tensor:
    q_func.zero_grad(set_to_none=True)
    total = 0.0
    for (s, a, r, ns, na, d) in batch_list:
        y = td_targets_bcq_impl(impl, targ_q_func, r, ns, na, d, gamma, prefer_impl_target=True)
        q = q_values_impl(q_func, s, a)
        total = total + F.mse_loss(q, y, reduction="sum")
    total.backward()
    g = concat_grad_from_module(q_func).detach()
    q_func.zero_grad(set_to_none=True)
    return g


def make_batches_from_indices(S, A, R, NS, NA, D, idx: torch.Tensor, max_chunk: int = 4096):
    batches = []
    if idx.numel() == 0:
        return batches
    for i in range(0, idx.numel(), max_chunk):
        sl = idx[i:i+max_chunk]
        batches.append((S[sl], A[sl], R[sl], NS[sl], NA[sl], D[sl]))
    return batches


def sample_minibatch(S, A, R, NS, NA, D, size: int):
    N = S.shape[0]
    if N == 0:
        return (S, A, R, NS, NA, D, torch.empty(0, dtype=torch.long, device=S.device))
    idx = torch.randint(0, N, (size,), device=S.device)
    return S[idx], A[idx], R[idx], NS[idx], NA[idx], D[idx], idx


@torch.no_grad()
def sample_actions_from_retain_for_states(S_m: torch.Tensor, A_m: torch.Tensor,
                                          states_query: torch.Tensor, device: torch.device,
                                          topk: int = 10) -> torch.Tensor:
    """
    Approximate a_m ~ Dm(·|s_f) by picking an action from a nearest state in Dm (top-k NN for diversity).
    """
    B = states_query.shape[0]
    sampled = []
    for i in range(B):
        dists = torch.norm(S_m - states_query[i:i+1], dim=-1)  # [N_m]
        k = min(topk, S_m.shape[0])
        _, top_idx = torch.topk(dists, k, largest=False)
        pick = top_idx[torch.randint(0, k, (1,), device=device)]
        sampled.append(A_m[pick])
    return torch.cat(sampled, dim=0)


def compute_cure_penalty_same_state(q_func: nn.Module,
                                    s_forget: torch.Tensor,
                                    a_forget: torch.Tensor,
                                    a_retain: torch.Tensor,
                                    w_forget: torch.Tensor) -> torch.Tensor:
    """
    R_CURe = E_{s_f}[ w(τ) * ( Q(s_f, a_f) - Q(s_f, a_m) ) ], weight-normalized mean.
    """
    q_ff = q_values_impl(q_func, s_forget, a_forget)      # [B,1]
    q_fm = q_values_impl(q_func, s_forget, a_retain)      # [B,1]
    penalty = (w_forget * (q_ff - q_fm)).sum() / (w_forget.sum() + 1e-12)
    return penalty


# ----------------------------------- main ------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--fully_trained_model", type=str, required=True, help="Path to params.json")
    parser.add_argument("--fully_trained_params", type=str, required=True, help="Path to model_*.pt/.pth")
    parser.add_argument("--unlearning_rate", type=float, default=0.01, help="Fraction of episodes to forget (Df)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight on CURe penalty")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005, help="Kept for completeness; impl has its own tau")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--max_ref_episodes", type=int, default=64)
    parser.add_argument("--max_forget_grad_episodes", type=int, default=512)
    args = parser.parse_args()

    d3rlpy.seed(args.seed)
    device = set_device(args.gpu)

    out_dir = args.output_dir or os.path.join("CURE_BCQ_runs", args.dataset.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    # Data split
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    Dm, Df = train_test_split(dataset, test_size=args.unlearning_rate, shuffle=False)

    # Transitions
    S_m, A_m, R_m, NS_m, NA_m, D_m, EPI_m = extract_transitions_from_episodes(Dm, device)
    S_f, A_f, R_f, NS_f, NA_f, D_f, EPI_f = extract_transitions_from_episodes(Df, device)
    if S_m.numel() == 0 or S_f.numel() == 0:
        raise RuntimeError("Empty retain or forget set after preprocessing — check dataset and split rate.")

    # Episode groups
    groups_m = group_indices_by_episode(EPI_m)
    groups_f = group_indices_by_episode(EPI_f)

    # Model
    algo = load_bcq_model(args.fully_trained_model, args.fully_trained_params, args.gpu)
    impl, q_func, targ_q_func, policy, imitator = get_impl_handles_bcq(algo)

    # Optimizer
    critic_optim = (getattr(impl, "_critic_optim", None) or getattr(impl, "critic_optim", None)
                    or torch.optim.Adam(list(q_func.parameters()), lr=3e-4))

    # Reference gradient (keep)
    ref_epi_ids = sorted(groups_m.keys())[:min(args.max_ref_episodes, len(groups_m))]
    ref_batches = []
    for eid in ref_epi_ids:
        idx = groups_m[eid]
        ref_batches += make_batches_from_indices(S_m, A_m, R_m, NS_m, NA_m, D_m, idx)
    if len(ref_batches) == 0:
        raise RuntimeError("No retain transitions available to form a reference gradient.")

    g_ref = compute_reference_td_grad_bcq(impl, q_func, targ_q_func, ref_batches, args.gamma)
    g_ref_norm = torch.norm(g_ref) + 1e-12

    # Influence weights (forget)
    forget_epi_ids = sorted(groups_f.keys())
    if args.max_forget_grad_episodes > 0:
        forget_epi_ids = forget_epi_ids[:min(args.max_forget_grad_episodes, len(forget_epi_ids))]

    influence_weights: Dict[int, float] = {}
    for eid in forget_epi_ids:
        idx = groups_f[eid]
        batch_list = make_batches_from_indices(S_f, A_f, R_f, NS_f, NA_f, D_f, idx)
        g_u = compute_forget_td_grad_bcq(impl, q_func, targ_q_func, batch_list, args.gamma)
        sim = torch.dot(g_u, g_ref) / (torch.norm(g_u) * g_ref_norm + 1e-12)
        influence_weights[eid] = 1.0 - float(torch.clamp(sim, -1.0, 1.0).item())

    default_forget_w = float(np.mean(list(influence_weights.values()))) if len(influence_weights) > 0 else 1.0

    def w_for_indices(idx_tensor: torch.Tensor, epi_tensor: torch.Tensor) -> torch.Tensor:
        if len(influence_weights) == 0 or idx_tensor.numel() == 0:
            return torch.ones((idx_tensor.shape[0], 1), device=device)
        eids = epi_tensor[idx_tensor].long().squeeze(-1).tolist()
        ww = [influence_weights.get(int(e), default_forget_w) for e in eids]
        return torch.tensor(ww, dtype=torch.float32, device=device).unsqueeze(-1)

    # Action pool for a_m ~ Dm(·|s_f)
    action_pool_states = S_m
    action_pool_actions = A_m

    # Training loop (critic-only CURe)
    steps = args.steps
    bs = args.batch_size
    gamma = args.gamma
    alpha_w = args.alpha

    print("\nStarting CURe training...\n")
    start_time = time.time()

    for t in range(steps):
        # TD on retain set
        s_m, a_m, r_m, ns_m, na_m, d_m, idx_m = sample_minibatch(S_m, A_m, R_m, NS_m, NA_m, D_m, bs)
        y_m = td_targets_bcq_impl(impl, targ_q_func, r_m, ns_m, na_m, d_m, gamma, prefer_impl_target=True)
        q_m = q_values_impl(q_func, s_m, a_m)
        L_keep_TD = F.mse_loss(q_m, y_m, reduction="mean")

        # R_CURe on SAME forget states
        s_f, a_f, r_f, ns_f, na_f, d_f, idx_f = sample_minibatch(S_f, A_f, R_f, NS_f, NA_f, D_f, bs)
        w_f = w_for_indices(idx_f, EPI_f)
        a_m_on_s_f = sample_actions_from_retain_for_states(action_pool_states, action_pool_actions,
                                                           s_f, device, topk=10)
        R_CURe = compute_cure_penalty_same_state(q_func, s_f, a_f, a_m_on_s_f, w_f)

        loss = L_keep_TD + alpha_w * R_CURe

        critic_optim.zero_grad(set_to_none=True)
        loss.backward()
        critic_optim.step()

        # Target update via impl (uses its internal tau)
        impl.update_critic_target()

        # Periodic saves
        if args.save_interval > 0 and (t + 1) % args.save_interval == 0:
            out_model_step = os.path.join(out_dir, f"model_cure_{t+1}.pt")
            algo.save_model(out_model_step)
            print(f"[{t+1}/{steps}] checkpoint saved: {out_model_step}")

        # Logging
        if (t + 1) % 1000 == 0:
            print(f"[{t+1}/{steps}] L_keep_TD={float(L_keep_TD.item()):.4f}  R_CURe={float(R_CURe.item()):.4f}")

    # Final save
    out_model = os.path.join(out_dir, f"model_cure_{steps}.pt")
    algo.save_model(out_model)
    params_copy = os.path.join(out_dir, "params.json")
    if os.path.abspath(params_copy) != os.path.abspath(args.fully_trained_model):
        try:
            shutil.copy2(args.fully_trained_model, params_copy)
        except Exception:
            pass
    print(f"Saved final CURe-BCQ checkpoint: {out_model}")
    print(f"Params file copied to: {params_copy}")


if __name__ == "__main__":
    main()
