import os
num_threads = "8"
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
import sklearn as sk
from sklearn import linear_model
torch.set_default_dtype(torch.float32)

view_params = {'elev' : 45, 'azim' : -60, 'roll' : 0}

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--sigma', type=float, default=0.3)
parser.add_argument('--train_otfm', action='store_true')
parser.add_argument('--make_plots', action='store_true')
parser.add_argument('--make_evals', action='store_true')
parser.add_argument('--outer_iter', type=int, default=5)
parser.add_argument('--holdout', type=int, default=-1)
parser.add_argument('--otfm_iter', type=int, default=1000)
parser.add_argument('--otfm_print_iter', type=int, default=100)
parser.add_argument('--otfm_batch', type=int, default=64)
parser.add_argument('--hidden_sizes_flow', nargs='+', type=int, default=[64, 64, 64])
parser.add_argument('--hidden_sizes_score', nargs='+', type=int, default=[64, 64, 64])
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--outdir_weights', type=str, default='weights')
parser.add_argument('--outdir_plots', type=str, default='plots')
parser.add_argument('--outdir_eval', type=str, default='eval')
parser.add_argument('--suffix', type=str, default='default')
args = parser.parse_args()

# set seed
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

args.train_otfm = True
args.make_plots = True
args.make_evals = True

x = np.load("../../data/mESC/X_pca.npy")[:, range(10)]
x = torch.tensor(sk.preprocessing.StandardScaler().fit_transform(x), dtype = torch.float32)

df_time = pd.read_csv("../../data/mESC/time.txt", sep = "\t", header = None)
t_idx = torch.tensor([{0 : 0, 12 : 1, 24 : 2, 48 : 3, 72 : 4}[x] for x in df_time.iloc[:, 0]])

# setup problem parameters
T = len(np.unique(t_idx))
# ts = torch.linspace(0, 1, T)
ts = torch.tensor(np.unique(df_time.iloc[:, 0].values) / 72.0, dtype = torch.float32) * T
# ts = torch.arange(T, dtype = torch.float32)

data = {'x' : x, 't_idx' : t_idx, 'sigma' : args.sigma, 'ts' : ts}
d = data['x'].shape[-1]
torch.save(data, "mESC.pkl")

# data = torch.load("moon_data.pkl")
# T = len(np.unique(data['t_idx']))
# ts = data['ts']
# d = data['x'].shape[1]

A = torch.zeros((d, d), dtype = torch.float32)
mu = torch.zeros(d, dtype = torch.float32)
A_hist, mu_hist = [A, ], [mu, ]
d_marginal_hist = []

import copy
s_model = fm.MLP(d = d, hidden_sizes = args.hidden_sizes_score, time_varying=True, activation = torch.nn.ReLU)
v_model = fm.MLP(d = d, hidden_sizes = args.hidden_sizes_flow, time_varying=True, activation = torch.nn.ReLU)
optim = torch.optim.AdamW(list(s_model.parameters()) + list(v_model.parameters()), args.lr)
params_hist = []

import importlib
importlib.reload(fm)

print(f"hold out: {args.holdout}")
_ts = torch.cat([ts[:args.holdout], ts[args.holdout+1:]]) if args.holdout > -1 else ts
_idx = (data['t_idx'] != args.holdout)
_X = data['x'][_idx, :]
_d = {**{i : i for i in range(args.holdout)}, **{i : i-1 for i in range(args.holdout+1, T)}}
_t_idx = torch.tensor([_d[i.item()] for i in data['t_idx'][_idx]]) if args.holdout > -1 else data['t_idx']
_T = T-1 if args.holdout > -1 else T

for it in range(args.outer_iter):
    if args.train_otfm:
        print(f"Iteration {it}: Fitting SB drift")
        otfm = fm.LinearEntropicOTFM(_X, 
                            _t_idx, 
                            ts = _ts,
                            sigma = args.sigma,
                            A = A,
                            mu = mu,
                            T = _T,
                            dim = d,
                            device = torch.device('cpu')
                            )
        s_model = fm.MLP(d = d, hidden_sizes = args.hidden_sizes_score, time_varying=True, activation = torch.nn.ReLU)
        v_model = fm.MLP(d = d, hidden_sizes = args.hidden_sizes_flow, time_varying=True, activation = torch.nn.ReLU)
        optim = torch.optim.AdamW(list(s_model.parameters()) + list(v_model.parameters()), args.lr)
        alpha = 0.95
        for i in tqdm(range(args.otfm_iter)):
            _x, _s, _u, _t, _t_orig = otfm.sample_bridges_flows(batch_size = args.otfm_batch)
            optim.zero_grad()
            s_fit = s_model(_t, _x)
            v_fit = v_model(_t, _x)
            L_score = torch.mean(((_t_orig * (1-_t_orig)) * (s_fit - _s))**2) * args.sigma**2
            L_flow = torch.mean((_t_orig * (1-_t_orig)*(v_fit - _u))**2)
            # L_flow = torch.mean((v_fit - _u)**2)
            L = (1-alpha)*L_score + alpha*L_flow
            if i % args.otfm_print_iter == 0:
                print(L_score.item(), L_flow.item())
            L.backward()
            optim.step()
        print(f"Iteration {it}: saving weights")
        torch.save(s_model.state_dict(), os.path.join(args.outdir_weights, f"otfm_score_iter_{it}_{args.suffix}.pt"))
        torch.save(v_model.state_dict(), os.path.join(args.outdir_weights, f"otfm_flow_iter_{it}_{args.suffix}.pt"))
        print(f"Iteration {it}: updating reference")
        with torch.no_grad():
            vs_t = [v_model(torch.scalar_tensor(_ts[i]), _X[_t_idx == i, :]) + args.sigma**2/2 * s_model(torch.scalar_tensor(_ts[i]), _X[_t_idx == i, :]) for i in range(_T)]
        lr = linear_model.RidgeCV()
        lr.fit(_X, torch.vstack(vs_t))
        A, b = lr.coef_, lr.intercept_
        # mu = -torch.tensor(sp.linalg.solve(A + 0.1*np.eye(d), b), dtype = torch.float32)
        mu = -torch.tensor(np.linalg.pinv(A.T @ A + 0.01*np.eye(d)) @ A.T @ b, dtype = torch.float32) 
        A = torch.tensor(A, dtype = torch.float32)
        # save new reference parameters
        torch.save({"A" : A, "mu" : mu}, os.path.join(args.outdir_weights, f"reference_iter_{it}_{args.suffix}.pt"))
        A_hist.append(A)
        mu_hist.append(mu)
        params_hist.append({'s': copy.deepcopy(s_model).state_dict(), 'v' : copy.deepcopy(v_model).state_dict()})
    else:
        _ref_params=torch.load(os.path.join(args.outdir_weights, f"reference_iter_{it}_{args.suffix}.pt"))
        A_hist.append(_ref_params["A"]); mu_hist.append(_ref_params["mu"])
        _score_params=torch.load(os.path.join(args.outdir_weights, f"otfm_score_iter_{it}_{args.suffix}.pt"))
        _flow_params=torch.load(os.path.join(args.outdir_weights, f"otfm_flow_iter_{it}_{args.suffix}.pt"))
        s_model.load_state_dict(_score_params)
        v_model.load_state_dict(_flow_params)
        params_hist.append({'s': _score_params,
                            'v' : _flow_params})


if args.make_plots:
    # Generate plots for each iteration
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    xlims = (-5.5, 5.5)
    ylims = (-5.5, 5.5)
    for k in range(len(params_hist)):
        _v, _s = copy.deepcopy(v_model), copy.deepcopy(s_model)
        _v.load_state_dict(params_hist[k]['v'])
        _s.load_state_dict(params_hist[k]['s'])
        sde = fm.SDE(lambda t, x: _v(t, x) + args.sigma**2 / 2 * _s(t, x), args.sigma)
        x0 = _X[_t_idx == 0]
        _T = 100
        with torch.no_grad():
            xs_sde = torchsde.sdeint(sde, torch.tensor(x0), torch.linspace(0, T, _T), method = "euler")
            xs_ode = odeint(lambda t, x: _v(t, x), torch.tensor(x0), torch.linspace(0, T, _T))
        fig = plt.figure(figsize = (15, 5))
        ax=fig.add_subplot(131, projection='3d')
        ax.view_init(**view_params)
        ax.scatter(_X[:, 0], _X[:, 1], _X[:, 2], c = _t_idx)
        ax=fig.add_subplot(132, projection='3d')
        ax.view_init(**view_params)
        ax.scatter(_X[:, 0], _X[:, 1], _X[:, 2], c = 'grey')
        for i in range(5):
            ax.plot(xs_sde[:, i, 0], xs_sde[:, i, 1], xs_sde[:, i, 2], alpha = 0.8, zorder = 100)
        plt.title("SDE")
        ax=fig.add_subplot(133, projection='3d')
        ax.view_init(**view_params)
        ax.scatter(_X[:, 0], _X[:, 1], _X[:, 2], c = 'grey')
        for i in range(5):
            ax.plot(xs_ode[:, i, 0], xs_ode[:, i, 1], xs_ode[:, i, 2], alpha = 0.8, zorder = 100)
        plt.title("ODE")
        plt.suptitle(f"Reference iteration {k}")
        plt.savefig(os.path.join(args.outdir_plots, f"trajectory_plot_iteration_{k}_{args.suffix}.pdf"))

    plt.figure(figsize = (8, 3))
    plt.subplot(1, 2, 1)
    plt.scatter(_X[..., 0], _X[..., 1], alpha = 1, marker = "^", c = _t_idx)
    V = torch.vstack(vs_t)
    plt.quiver(_X[..., 0], _X[..., 1], V[..., 0], V[..., 1], scale = 25)
    plt.title("SB vector field")
    plt.subplot(1, 2, 2)
    i = -1
    A, mu = A_hist[i], mu_hist[i]
    plt.scatter(_X[..., 0], _X[..., 1], alpha = 1, marker = "^", c = _t_idx)
    V = (_X.reshape(-1, d) - mu) @ A.T
    plt.quiver(_X[..., 0], _X[..., 1], V[..., 0], V[..., 1], scale = 25)
    plt.title("Reference")
    plt.savefig(os.path.join(args.outdir_plots, f"vectorfield_plot_iteration_last_{args.suffix}.pdf"))

    plt.figure(figsize = (8, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(torch.vstack([x.flatten() for x in A_hist]))
    plt.title("$A$")
    plt.subplot(1, 2, 2)
    plt.imshow(torch.vstack(mu_hist))
    plt.title("$\mu$")
    plt.savefig(os.path.join(args.outdir_plots, f"ref_params_all_iters_{args.suffix}.pdf"))

    i = -1
    A, mu = A_hist[i], mu_hist[i]
    ax = plt.figure(figsize = (5, 5)).add_subplot(projection='3d')
    ax.view_init(**view_params)
    ax.scatter(_X[..., 0], _X[..., 1], _X[..., 2], c = _t_idx)
    _X = _X.reshape(-1, d)
    V = (_X.reshape(-1, d) - mu) @ A.T
    ax.quiver(_X[..., 0], _X[..., 1], _X[..., 2], V[..., 0], V[..., 1], V[..., 2], normalize = True, length = 0.5, color = 'k', alpha = 0.5, arrow_length_ratio=1)
    ax.scatter(mu[None, 0], mu[None, 1], mu[None, 2], c = 'red', s = 100)
    plt.savefig(os.path.join(args.outdir_plots, f"ref_vectorfield_last_{args.suffix}.pdf"))


# evaluations
# marginal validation

data['x_val'] = data['x']
data['t_idx_val'] = data['t_idx']

import evals
if args.make_evals:
    _v, _s = copy.deepcopy(v_model), copy.deepcopy(s_model)
    energy_dists_val = {}
    emd_val = {}
    for i in range(len(params_hist)):
        _v.load_state_dict(params_hist[i]['v'])
        _s.load_state_dict(params_hist[i]['s'])
        x0_val = data['x_val'][data['t_idx_val'] == 0, :]
        with torch.no_grad():
            xs_ode_val = odeint(_v, x0_val, ts*1.0)
        energy_dists_val[i]=[evals.energy_distance(xs_ode_val[i, ...], data['x_val'][data['t_idx_val'] == i, :]).item() for i in range(T)]
        emd_val[i]=[evals.emd_samples(xs_ode_val[i, ...], data['x_val'][data['t_idx_val'] == i, :]).item() for i in range(T)]
    pd.concat([pd.DataFrame(energy_dists_val), pd.DataFrame(emd_val)], keys = ["energy", "emd"]).to_csv(f"evals/marginal_validation_{args.suffix}.csv")

    # marginal validation - adjacent timepoints
    energy_dists_val = {}
    emd_val = {}
    for i in range(len(params_hist)):
        _v.load_state_dict(params_hist[i]['v'])
        _s.load_state_dict(params_hist[i]['s'])
        energy_dists_val[i] = [0, ]
        emd_val[i] = [0, ]
        for j in range(T-1):
            x0_val = data['x_val'][data['t_idx_val'] == j, :]
            with torch.no_grad():
                xs_ode_val = odeint(_v, x0_val, 1.0*ts[j:j+2])
            energy_dists_val[i].append(evals.energy_distance(xs_ode_val[-1, ...], data['x_val'][data['t_idx_val'] == j+1, :]).item())
            emd_val[i].append(evals.emd_samples(xs_ode_val[-1, ...], data['x_val'][data['t_idx_val'] == j+1, :]).item())
    pd.concat([pd.DataFrame(energy_dists_val), pd.DataFrame(emd_val)], keys = ["energy", "emd"]).to_csv(f"evals/marginal_adjacent_validation_{args.suffix}.csv")
