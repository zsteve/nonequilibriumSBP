import os

num_threads = "4"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import torch
import numpy as np
import scipy as sp
import sys
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
import json
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../../tools/nlsb/")
from utils import EarlyStopping

torch.set_default_dtype(torch.float32)

from model import ODENet, SDENet, SDE_MODEL_NAME, ODE_MODEL_NAME, LAGRANGIAN_NAME
from dataset import (
    OrnsteinUhlenbeckSDE_Dataset,
    BalancedBatchSampler,
    PotentialSDE_Dataset,
    UniformDataset,
    scRNASeq,
    TrajectoryInferenceDataset,
)

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--T", type=int, default=25)
parser.add_argument("--epochs", type=int, default=2_500)
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--N_validation", type=int, default=1024)
parser.add_argument("--outdir_weights", type=str, default="weights")
parser.add_argument("--outdir_plots", type=str, default="plots")
parser.add_argument("--outdir_eval", type=str, default="eval")
parser.add_argument("--suffix", type=str, default="default")
args = parser.parse_args()

# set seed
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

data = torch.load(f"data_{args.suffix}.pkl")

ts = torch.tensor([0.0, 1.0])

cfg = {
    "seed": 0,
    "epochs": args.epochs,
    "LMT": -1,
    "train_size": data["x0"].shape[0],
    "val_size": data["x0"].shape[0],
    "model_name": "ito",
    "model": {
        "noise_type": "diagonal",
        "sigma_type": "const",
        "input_dim": data["d"],
        "brownian_size": data["d"],
        "drift_cfg": {"nTh": 2, "m": 32, "use_t": True},
        "diffusion_cfg": {"sigma": data["sigma"]},
        "criterion_cfg": {
            "alpha_D": 1.0,
            "alpha_L": [0.3, 0.1, 0.01, 0.0001],
            "alpha_R": [0.2, 0.01, 0.0001, 0.0001],
            "p": 2,
            "blur": 0.05,
        },
        "solver_cfg": {
            "adjoint": False,
            "dt": 0.01,
            "method": "euler",
            "adaptive": False,
        },
    },
    "optim": {"lr": 0.001},
}


class OUSBDataset(TrajectoryInferenceDataset):
    def __init__(self, device, data):
        super().__init__()
        self.data = data
        self.dim = data["d"]
        self.device = device
        t_size = 1
        self.t_0, self.t_T = 0, 1
        self.ts = torch.linspace(self.t_0, self.t_T, t_size + 1, device=device)
        self.X = torch.vstack([data["x0"], data["x1"]]).to(device)
        self.labels = self.ts.cpu().repeat_interleave(data["x0"].shape[0])
        self.ncells = self.X.shape[0]
        self.t_set = sorted(list(set(self.labels.cpu().numpy())))


class OULagrangian:
    def __init__(self, A, mu, device="cpu"):
        self.A = A.to(device)
        self.mu = mu.to(device)

    def L(self, t, x, u, v=None):
        v = (x - self.mu) @ self.A.T
        return 0.5 * torch.sum(torch.pow(u - v, 2), 1, keepdims=True)

    def inv_L(self, t, x, f, v=None):
        if v is None:
            return f
        else:
            return (v + f) / (1 + 1)


ds = OUSBDataset(device=device, data=data)

t_set = ds.ts[1:]
train_t_set = t_set[:]
ts = train_t_set

batch_sampler_tr = BalancedBatchSampler(ds, args.batchsize)
dl = DataLoader(ds, batch_sampler=batch_sampler_tr)

model_name = cfg["model_name"].lower()
L = OULagrangian(data["A"], data["mu"], device=device)
net = SDE_MODEL_NAME[model_name](**cfg["model"], lagrangian=L)
model = SDENet(net, device)
MODEL = "sde"
model.to(device)

optimizer = optim.Adam(model.parameters_lr(), lr=cfg["optim"]["lr"])

for i in tqdm(range(args.epochs)):
    outputs = []
    for batch_idx, train_batch in enumerate(dl):
        train_batch["base"] = {"X": ds.data["x0"]}
        optimizer.zero_grad()
        out = model.training_step(train_batch, batch_idx, train_t_set, ds.T0)
        outputs.append(out)
        loss = out["loss"]
        loss.backward()
        del train_batch
        optimizer.step()
        if hasattr(model, "clamp_parameters"):
            model.clamp_parameters()
        # scheduler.step()
    train_result = model.training_epoch_end(outputs)

sys.path.append("../../src")
import fm

otfm = fm.LinearEntropicOTFM(
    torch.vstack([data["x0"], data["x1"]]),
    torch.hstack(
        [torch.full((data["x0"].shape[0],), 0), torch.full((data["x0"].shape[0],), 1)]
    ),
    ts=torch.tensor([0.0, 1.0], dtype=torch.float32),
    sigma=data["sigma"],
    A=data["A"],
    mu=data["mu"],
    T=2,
    dim=data["d"],
    device=torch.device("cpu"),
)
## Compare to closed form
gsb = fm.GaussianOUSB(otfm.bm, otfm)
t = torch.linspace(0, 1, args.T)
sb_means, sb_vars, S_t, d_sb_means = gsb.evaluate(
    t, data["mean0"], data["mean1"], data["var0"], data["var1"]
)

xs_sim = model.sample(data["x0"], t).cpu().squeeze()
xs_sim = torch.permute(xs_sim, [1, 0, 2])

matplotlib.use("Agg")
U = data["U"]
x0_proj, x1_proj = data["x0"] @ U, data["x1"] @ U
plt.scatter(x0_proj[:, 0], x0_proj[:, 1])
plt.scatter(x1_proj[:, 0], x1_proj[:, 1])
y = model.sample(data["x0"], t).cpu().squeeze()
y_proj = torch.einsum("ijk,kl->ijl", y, U)
for i in range(len(t)):
    plt.scatter(
        y_proj[:, i, 0],
        y_proj[:, i, 1],
        marker="x",
        c=matplotlib.colormaps["viridis"](i / len(t)),
    )
plt.savefig(os.path.join(args.outdir_plots, f"scatter_nlsb_{args.suffix}.pdf"))

xs_sample_gt = [
    torch.randn((args.N_validation, data["d"])) @ fm._sqrtm(sb_vars[i]) + sb_means[i].T
    for i in range(len(t))
]
vs_gt = [
    d_sb_means[i]
    + (S_t[i].T @ torch.linalg.pinv(sb_vars[i])) @ (xs_sample_gt[i].T - sb_means[i])
    for i in range(len(t))
]

with torch.no_grad():
    vs_estim = [
        model.net.f(t[i], xs_sample_gt[i].to(device)).cpu().T for i in range(len(t))
    ]


pd.DataFrame(
    {
        "t": t,
        "NLSB_BW": [
            fm.bures_wasserstein(
                torch.mean(xs_sim[i], 0).flatten(),
                sb_means[i].flatten(),
                torch.cov(xs_sim[i].T),
                sb_vars[i],
            ).item()
            for i in range(len(t))
        ],
        "NLSB_vf": [
            (torch.linalg.norm(vs_estim[i] - vs_gt[i], axis=0) ** 2)
            .mean()
            .sqrt()
            .item()
            for i in range(len(t))
        ],
    }
).to_csv(os.path.join(args.outdir_eval, f"eval_nlsb_{args.suffix}.csv"))
