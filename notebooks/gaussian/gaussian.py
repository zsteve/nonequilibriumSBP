import os

num_threads = "4"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import torch
from torchdiffeq import odeint
import torchsde
import numpy as np
import scipy as sp
import sys

sys.path.append("../../src")
import fm
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--N", type=int, default=256)
parser.add_argument("--d", type=int, default=2)
parser.add_argument("--sigma", type=float, default=1.0)
parser.add_argument("--N_validation", type=int, default=1024)
parser.add_argument("--train_otfm", action="store_true")
parser.add_argument("--otfm_iter", type=int, default=2_500)
parser.add_argument("--otfm_print_iter", type=int, default=100)
parser.add_argument("--otfm_batch", type=int, default=64)
parser.add_argument("--hidden_sizes_flow", nargs="+", type=int, default=[64, 64, 64])
parser.add_argument("--hidden_sizes_score", nargs="+", type=int, default=[64, 64, 64])
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--T", type=int, default=25)
parser.add_argument("--outdir_weights", type=str, default="weights")
parser.add_argument("--outdir_plots", type=str, default="plots")
parser.add_argument("--outdir_eval", type=str, default="eval")
parser.add_argument("--suffix", type=str, default="default")
args = parser.parse_args()

# set seed
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

## setup problem instance
# reference dynamics
U = torch.linalg.svd(torch.randn(args.d, args.d)).U[:, range(2)]
A = U @ (-1.5 * torch.tensor([[0, 1], [-2.5, 0]], dtype=torch.float32)) @ U.T
# mu = torch.zeros(args.d, dtype = torch.float32)
mu = torch.tensor([1, -1], dtype=torch.float32) @ U.T
# marginals
mean0, mean1 = torch.tensor([-2.5, -0.5]) @ U.T, torch.tensor([0.5, 2.5]) @ U.T
var0, var1 = (
    torch.tensor([[0.1, 0.005], [0.005, 0.1]]),
    torch.tensor([[1.1, -2.0], [-2.0, 4.5]]),
)
var0 = U @ var0 @ U.T + 0.1 * torch.eye(args.d)
var1 = U @ var1 @ U.T + 0.1 * torch.eye(args.d)
# sample data
x0 = torch.randn(args.N, args.d) @ fm._sqrtm(var0) + mean0
x1 = torch.randn(args.N, args.d) @ fm._sqrtm(var1) + mean1
t = torch.rand(args.N)

torch.save(
    {
        "d": args.d,
        "A": A,
        "mu": mu,
        "sigma": args.sigma,
        "mean0": mean0,
        "mean1": mean1,
        "var0": var0,
        "var1": var1,
        "x0": x0,
        "x1": x1,
        "U": U,
    },
    f"data_{args.suffix}.pkl",
)

ts = torch.tensor([0.0, 1.0])

# OU-OTFM
otfm = fm.LinearEntropicOTFM(
    torch.vstack([x0, x1]),
    torch.hstack([torch.full((x0.shape[0],), 0), torch.full((x0.shape[0],), 1)]),
    ts=ts,
    sigma=args.sigma,
    A=A,
    mu=mu,
    T=2,
    dim=args.d,
    device=torch.device("cpu"),
)

# BM-OTFM
otfm_null = fm.EntropicOTFM(
    torch.vstack([x0, x1]),
    torch.hstack([torch.full((x0.shape[0],), 0), torch.full((x0.shape[0],), 1)]),
    ts=ts,
    sigma=args.sigma,
    T=2,
    dim=args.d,
    device=torch.device("cpu"),
)

alpha = 0.5

print("Training OU-OTFM")
s_model = fm.MLP(
    d=args.d,
    hidden_sizes=args.hidden_sizes_score,
    time_varying=True,
    activation=torch.nn.ReLU,
)
v_model = fm.MLP(
    d=args.d,
    hidden_sizes=args.hidden_sizes_flow,
    time_varying=True,
    activation=torch.nn.ReLU,
)
if args.train_otfm:
    trace_model = []
    optim = torch.optim.AdamW(
        list(s_model.parameters()) + list(v_model.parameters()), args.lr
    )
    for i in tqdm(range(args.otfm_iter)):
        _x, _s, _u, _t, _t_orig = otfm.sample_bridges_flows(batch_size=args.otfm_batch)
        optim.zero_grad()
        s_fit = s_model(_t, _x)
        v_fit = v_model(_t, _x)
        L_score = (
            torch.mean((_t_orig * (1 - _t_orig) * (s_fit - _s)) ** 2) * args.sigma**2
        )
        L_flow = torch.mean(_t_orig * (1 - _t_orig) * (v_fit - _u) ** 2)
        # L_flow = torch.mean((v_fit * otfm.dt - _u)**2)
        L = (1 - alpha) * L_score + alpha * L_flow
        trace_model.append(L.item())
        if i % args.otfm_print_iter == 0:
            print(
                f"Iteration {i}, L_score = {L_score.item()}, L_flow = {L_flow.item()}"
            )
        L.backward()
        optim.step()
    torch.save(
        s_model.state_dict(),
        os.path.join(args.outdir_weights, f"otfm_score_{args.suffix}.pt"),
    )
    torch.save(
        v_model.state_dict(),
        os.path.join(args.outdir_weights, f"otfm_flow_{args.suffix}.pt"),
    )
    # plot trace
    plt.figure(figsize=(3, 3))
    plt.plot(trace_model)
    plt.title("OU-OTFM")
    plt.savefig(os.path.join(args.outdir_plots, f"otfm_{args.suffix}.pdf"))
else:
    s_model.load_state_dict(
        torch.load(os.path.join(args.outdir_weights, f"otfm_score_{args.suffix}.pt"))
    )
    v_model.load_state_dict(
        torch.load(os.path.join(args.outdir_weights, f"otfm_flow_{args.suffix}.pt"))
    )

print("Training BM-OTFM")
s_model_null = fm.MLP(
    d=args.d,
    hidden_sizes=args.hidden_sizes_score,
    time_varying=True,
    activation=torch.nn.ReLU,
)
v_model_null = fm.MLP(
    d=args.d,
    hidden_sizes=args.hidden_sizes_flow,
    time_varying=True,
    activation=torch.nn.ReLU,
)
if args.train_otfm:
    trace_null = []
    optim = torch.optim.AdamW(
        list(s_model_null.parameters()) + list(v_model_null.parameters()), args.lr
    )
    for i in tqdm(range(args.otfm_iter)):
        _x, _s, _u, _t, _t_orig = otfm_null.sample_bridges_flows(
            batch_size=args.otfm_batch
        )
        optim.zero_grad()
        s_fit = s_model_null(_t, _x)
        v_fit = v_model_null(_t, _x)
        L_score = (
            torch.mean(((_t_orig * (1 - _t_orig)) * (s_fit - _s)) ** 2) * args.sigma**2
        )
        L_flow = torch.mean((_t_orig * (1 - _t_orig) * (v_fit - _u)) ** 2)
        # L_flow = torch.mean((v_fit * otfm_null.dt - _u)**2)
        L = (1 - alpha) * L_score + alpha * L_flow
        trace_null.append(L.item())
        if i % args.otfm_print_iter == 0:
            print(
                f"Iteration {i}, L_score = {L_score.item()}, L_flow = {L_flow.item()}"
            )
        L.backward()
        optim.step()
    torch.save(
        s_model_null.state_dict(),
        os.path.join(args.outdir_weights, f"otfm_null_score_{args.suffix}.pt"),
    )
    torch.save(
        v_model_null.state_dict(),
        os.path.join(args.outdir_weights, f"otfm_null_flow_{args.suffix}.pt"),
    )
    # plot trace
    plt.figure(figsize=(3, 3))
    plt.plot(trace_null)
    plt.title("BM-OTFM")
    plt.savefig(os.path.join(args.outdir_plots, f"otfm_null_{args.suffix}.pdf"))
else:
    s_model_null.load_state_dict(
        torch.load(
            os.path.join(args.outdir_weights, f"otfm_null_score_{args.suffix}.pt")
        )
    )
    v_model_null.load_state_dict(
        torch.load(
            os.path.join(args.outdir_weights, f"otfm_null_flow_{args.suffix}.pt")
        )
    )

## Compare to closed form
gsb = fm.GaussianOUSB(otfm.bm, otfm)
t = torch.linspace(0, 1, args.T)
sb_means, sb_vars, S_t, d_sb_means = gsb.evaluate(t, mean0, mean1, var0, var1)
# Forward simulation
sde = fm.SDE(lambda t, x: v_model(t, x) + args.sigma**2 / 2 * s_model(t, x), args.sigma)
sde_null = fm.SDE(
    lambda t, x: v_model_null(t, x) + args.sigma**2 / 2 * s_model_null(t, x), args.sigma
)
# with torch.no_grad():
#     xs_sim = torchsde.sdeint(sde, torch.tensor(x0), t, method = "euler")
#     xs_sim_null = torchsde.sdeint(sde_null, torch.tensor(x0), t, method = "euler")
with torch.no_grad():
    xs_sim = odeint(v_model, torch.tensor(x0), t)
    xs_sim_null = odeint(v_model_null, torch.tensor(x0), t)
# Validation of flow field
xs_sample_gt = [
    torch.randn((args.N_validation, args.d)) @ fm._sqrtm(sb_vars[i]) + sb_means[i].T
    for i in range(len(t))
]
vs_gt = [
    d_sb_means[i]
    + (S_t[i].T @ torch.linalg.pinv(sb_vars[i])) @ (xs_sample_gt[i].T - sb_means[i])
    for i in range(len(t))
]
with torch.no_grad():
    vs_estim = [sde.f(t[i], xs_sample_gt[i]).T for i in range(len(t))]
    vs_estim_null = [sde_null.f(t[i], xs_sample_gt[i]).T for i in range(len(t))]

## Compare to Vargas et al.
sys.path.append("../../tools/GP_Sinkhorn/")
from gp_sinkhorn.SDE_solver import solve_sde_RK
from gp_sinkhorn.utils import plot_trajectories_2
from gp_sinkhorn.MLE_drift import *
from gp_sinkhorn import MLE_drift
from functools import partial
import pyro
from pyro.contrib.gp.kernels import (
    Exponential,
    Matern32,
    RBF,
    Brownian,
    Combination,
    Product,
    Sum,
)

print("Running GP_sinkhorn")
prior_drift = lambda x: (x[:, range(args.d)] - mu) @ A.double().T


def kern_mix_1(input_dim, variance=None):
    return pyro.contrib.gp.kernels.Exponential(
        input_dim=args.d + 1,
        lengthscale=torch.tensor(2.5),
        active_dims=list(range(args.d + 1)),
    )


# If you find the results of kern_mix_1 a bit unstable try this kernel instead
# that is change kernel=kern_mix_1 to kernel=kern_mix_stable in ML_IPFP
def kern_mix_stable(input_dim, variance=None):
    return pyro.contrib.gp.kernels.RBF(
        input_dim=args.d + 1,
        lengthscale=torch.tensor(2.5),
        active_dims=list(range(args.d + 1)),
    )


result, drift_forward, drift_backward = MLE_IPFP(
    x0,
    x1,
    N=len(t) - 1,
    sigma=args.sigma,
    prior_drift=prior_drift,
    prior_X_0=None,
    iteration=10,
    decay_sigma=1,
    gp_mean_prior_flag=True,
    kernel=kern_mix_1,
    nn=False,
)
v_ipfp_fwd = [
    drift_forward(fm.cat_tx(_t, _x).double()).float().T
    for (_t, _x) in zip(t, xs_sample_gt)
]
v_ipfp_rev = [
    -drift_backward(fm.cat_tx(1 - _t, _x).double()).float().T
    for (_t, _x) in zip(t, xs_sample_gt)
]

# # Double check forward and reverse drifts
# import torchsde
# _sde = fm.SDE(lambda t, x: drift_forward(fm.cat_tx(t, x).double()).float())
# y_fwd = torchsde.sdeint(_sde, x0, t, dt = 0.05)
# _sde = fm.SDE(lambda t, x: drift_backward(fm.cat_tx(t, x).double()).float())
# y_rev = torchsde.sdeint(_sde, x1, t, dt = 0.05)
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.plot(y_fwd[..., 0], y_fwd[..., 1], c = 'grey', alpha = 0.1)
# plt.scatter(x0[:, 0], x0[:, 1], c = 'r'); plt.scatter(x1[:, 0], x1[:, 1], c = 'b')
# plt.subplot(1, 2, 2)
# plt.plot(y_rev[..., 0], y_rev[..., 1], c = 'grey', alpha = 0.1)
# plt.scatter(x0[:, 0], x0[:, 1], c = 'r'); plt.scatter(x1[:, 0], x1[:, 1], c = 'b')
# plt.savefig("sde_test.pdf")

torch.save(result, os.path.join(args.outdir_weights, f"gp_sinkhorn_{args.suffix}.pkl"))

M = result[-1][1]
M2 = result[-1][3]

pd.DataFrame(
    {
        "t": t,
        "OU_EOT_BW": [
            fm.bures_wasserstein(
                torch.mean(xs_sim[i], 0).flatten(),
                sb_means[i].flatten(),
                torch.cov(xs_sim[i].T),
                sb_vars[i],
            ).item()
            for i in range(len(t))
        ],
        "EOT_BW": [
            fm.bures_wasserstein(
                torch.mean(xs_sim_null[i], 0).flatten(),
                sb_means[i].flatten(),
                torch.cov(xs_sim_null[i].T),
                sb_vars[i],
            ).item()
            for i in range(len(t))
        ],
        "OU_EOT_vf": [
            (torch.linalg.norm(vs_estim[i] - vs_gt[i], axis=0) ** 2)
            .mean()
            .sqrt()
            .item()
            for i in range(len(t))
        ],
        "EOT_vf": [
            (torch.linalg.norm(vs_estim_null[i] - vs_gt[i], axis=0) ** 2)
            .mean()
            .sqrt()
            .item()
            for i in range(len(t))
        ],
        "GP_sinkhorn_fwd_BW": [
            fm.bures_wasserstein(
                torch.mean(M[:, i, range(args.d)], 0).flatten().float(),
                sb_means[i].flatten(),
                torch.cov(M[:, i, range(args.d)].T).float(),
                sb_vars[i],
            ).item()
            for i in range(len(t))
        ],
        "GP_sinkhorn_rev_BW": [
            fm.bures_wasserstein(
                torch.mean(M2[:, -(i + 1), range(args.d)], 0).flatten().float(),
                sb_means[i].flatten(),
                torch.cov(M2[:, -(i + 1), range(args.d)].T).float(),
                sb_vars[i],
            ).item()
            for i in range(len(t))
        ],
        "GP_sinkhorn_vf": [
            (torch.linalg.norm(v_ipfp_fwd[i] - vs_gt[i], axis=0) ** 2)
            .mean()
            .sqrt()
            .item()
            for i in range(len(t))
        ],
    }
).to_csv(os.path.join(args.outdir_eval, f"eval_{args.suffix}.csv"))
