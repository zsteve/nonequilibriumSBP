import torch
from torch import nn, optim
import copy
import numpy as np
import sys
import math
import ot
from torchdiffeq import odeint

_eps = 1e-5


class MLP(nn.Module):
    """Minimal MLP with optional time-dependence"""

    def __init__(
        self,
        d=2,
        hidden_sizes=[
            100,
        ],
        activation=nn.ReLU,
        time_varying=True,
    ):
        super(MLP, self).__init__()
        self.net = nn.Sequential()
        self.time_varying = time_varying
        assert len(hidden_sizes) > 0
        hidden_sizes = copy.copy(hidden_sizes)
        if time_varying:
            hidden_sizes.insert(0, d + 1)
        else:
            hidden_sizes.insert(0, d)
        hidden_sizes.append(d)
        for i in range(len(hidden_sizes) - 1):
            self.net.add_module(
                name=f"L{i}", module=nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            )
            if i < len(hidden_sizes) - 2:
                self.net.add_module(name=f"A{i}", module=activation())
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0)

    def forward(self, t, x):
        if self.time_varying:
            return self.net(torch.hstack([x, t.expand(*x.shape[:-1], 1)]))
        else:
            return self.net(x)


def _get_coupling(l):
    """Calculate Sinkhorn coupling from dual potentials"""
    T = l.u[:, None] * l.K * l.v[None, :]
    return T / T.sum()


class BridgeMatcher:
    """Helper class for sampling from Brownian Schrodinger bridges"""

    def __init__(self):
        pass

    def sample_map(self, pi, batch_size, replace=True):
        p = pi.flatten()
        p = p / p.sum()
        choices = torch.multinomial(p, num_samples=batch_size, replacement=replace)
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1, pi, batch_size, replace=True):
        i, j = self.sample_map(pi, batch_size, replace=replace)
        return x0[i], x1[j]

    def sample_bridge_and_flow(self, x0, x1, ts, sigma):
        """Sample Brownian bridges between paired entries of [x0, x1] at times ts \in [0, 1] and diffusivity \sigma^2"""
        means = (1 - ts) * x0 + ts * x1
        vars = (sigma**2) * ts * (1 - ts)
        x = means + torch.sqrt(vars) * torch.randn_like(x0)
        s = (-1 / vars) * (x - means)
        u = (1 - 2 * ts) / (2 * ts * (1 - ts)) * (x - means) + x1 - x0
        return means, vars, x, s, u


def _sqrtm(x):
    """ "
    Matrix square root and its batched variant
    """
    decomp = torch.linalg.eigh(x)
    # numerically unstable
    # return decomp.eigenvectors @ torch.diag(decomp.eigenvalues**0.5) @ decomp.eigenvectors.T
    return (
        decomp.eigenvectors
        @ torch.diag(torch.clamp_min(decomp.eigenvalues, 0) ** 0.5)
        @ decomp.eigenvectors.T
    )


_bsqrtm = torch.vmap(_sqrtm)


class LinearBridgeMatcher:
    """Sampler from Schrodinger bridge with linear reference dynamics on unit time interval.

    Assumes a reference process described by the linear SDE:
    dX_t = A(X_t - mu) dt + sigma dB_t
    """

    def __init__(self, A, mu, T=25):
        """
        Initialize class and pre-compute several quantities

        A: drift matrix
        mu: bias
        T: number of timesteps to compute matrix integrals
        """
        self.A = A
        self.mu = mu
        self.T = T
        self.ts = torch.linspace(0, 1.0, self.T)
        self.dt = 1.0 / (self.T - 1)
        # precompute all the Phi_t (for variance)
        self.Phi_t = odeint(
            lambda s, x: torch.linalg.matrix_exp(s * self.A)
            @ torch.linalg.matrix_exp(s * self.A.T),
            torch.zeros_like(self.A),
            self.ts,
            method="dopri5",
        )
        # precompute all the Lambda_t (for control)
        self.Lambda_t = odeint(
            lambda s, x: torch.linalg.matrix_exp(-s * self.A)
            @ torch.linalg.matrix_exp(-s * self.A.T),
            torch.zeros_like(self.A),
            self.ts,
            method="dopri5",
        )
        self.epsI = torch.eye(self.A.shape[0]) * _eps
        return None

    def interp(self, x_t, t):
        """Linear interpolation helper function"""
        i, f = (t // self.dt).int(), (t % self.dt) / self.dt
        _x = (1 - f[:, None, None]) * x_t[i, ...]
        _idx = f > 1e-7
        _x[_idx, ...] += f[_idx, None, None] * x_t[i[_idx] + 1, ...]
        return _x

    def sample_map(self, pi, batch_size, replace=True):
        p = pi.flatten()
        p = p / p.sum()
        choices = torch.multinomial(p, num_samples=batch_size, replacement=replace)
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1, pi, batch_size, replace=True):
        """Sample `batch_size` pairs from transport plan"""
        i, j = self.sample_map(pi, batch_size, replace=replace)
        return x0[i], x1[j]

    def _bridge_ctrl(self, x, t, x0, x1):
        """Calculate mvOU bridge control term between (x0, x1) at (x, t), see Theorem 1"""
        _Lambda_inv_t = torch.linalg.pinv(self.interp(self.Lambda_t, 1 - t) + self.epsI)
        _k_t = (
            torch.bmm(
                torch.linalg.matrix_exp(-(1 - t[:, None, None]) * self.A),
                (x1 - self.mu)[:, :, None],
            ).squeeze()
            + self.mu
        )
        return -torch.bmm(_Lambda_inv_t, (x - _k_t)[:, :, None]).squeeze()

    def _bridge_cov(self, t, x0, x1):  # doesn't depend on (x0, x1)
        """Calculate mvOU bridge covariance at time t \in [0, 1], see Theorem 2"""
        _A = self.interp(self.Phi_t, t)
        _B = torch.bmm(_A, torch.linalg.matrix_exp((1 - t[:, None, None]) * self.A.T))
        _C = self.Phi_t[-1, ...]
        return _A - torch.bmm(
            _B, torch.linalg.solve(_C + self.epsI, _B.permute((0, 2, 1)))
        )

    def _bridge_mean(self, t, x0, x1):
        """Calculate mvOU bridge mean at time t \in [0, 1], see Theorem 2"""
        _Phi_t = self.interp(self.Phi_t, t)
        _Phi_T = self.Phi_t[-1, ...]
        mu_t = (
            torch.bmm(
                torch.linalg.matrix_exp(t[:, None, None] * self.A),
                (x0 - self.mu[None, :]).unsqueeze(2),
            ).squeeze(2)
            + self.mu[None, :]
        )
        mu_T = (x0 - self.mu[None, :]) @ torch.linalg.matrix_exp(self.A.T) + self.mu[
            None, :
        ]
        return mu_t + torch.bmm(
            _Phi_t,
            torch.bmm(
                torch.linalg.matrix_exp((1 - t[:, None, None]) * self.A.T),
                torch.linalg.solve(_Phi_T, (x1 - mu_T).T).T.unsqueeze(2),
            ),
        ).squeeze(2)

    def sample_bridge_and_flow(self, x0, x1, ts, sigma):
        """
        Sample from mvOU bridges between (x0, x1) at time ts.

        Returns a tuple of (means, cov, x, s, u) where (means, cov) are the parameters
        of the bridge marginal between (x0, x1).
        (x, s, u) are samples from each bridge along with the score and flow evaluated at sampled points.
        """
        # Sample Brownian bridges between paired entries of [x0, x1] at times ts \in [0, 1].
        means = self._bridge_mean(ts, x0, x1)
        cov = sigma**2 * self._bridge_cov(ts, x0, x1)
        x = means + torch.bmm(_bsqrtm(cov), torch.randn_like(x0)[:, :, None]).squeeze()
        # s = -torch.bmm(torch.linalg.pinv(cov + self.epsI), (x - means)[:, :, None]).squeeze()
        s = -torch.linalg.solve(
            cov + self.epsI, (x - means)[:, :, None]
        ).squeeze()  # more stable than pinv?
        ctrl = self._bridge_ctrl(x, ts, x0, x1)
        u = torch.matmul(x - self.mu, self.A.T) + ctrl - sigma**2 / 2 * s
        return means, cov, x, s, u


class LinearEntropicOTFM:
    """Sampler for mvOU-OTFM"""

    def __init__(
        self,
        x,
        t_idx,
        ts,
        sigma,
        A,
        mu,
        T,
        dim,
        device,
        bm_kwargs={},
    ):
        """
        Initialize and pre-compute EOT plans between time steps

        x: data points (N, d)
        t_idx: time indices for each data point (N,)
        ts: time steps (T,)
        sigma: square-root diffusivity
        A: drift matrix (see Eq. 1)
        mu: drift bias (see Eq. 1)
        T: total number of timesteps
        dim: dimension d
        """

        def entropic_ot_plan(x0, x1, eps):
            y0 = (x0 - self.mu) @ torch.linalg.matrix_exp(self.A.T) + self.mu
            y1 = x1
            _Sigma_inv = torch.linalg.pinv(self.bm.Phi_t[-1] * eps + self.epsI)
            C = (
                torch.diagonal(y0 @ _Sigma_inv @ y0.T)[:, None]
                + torch.diagonal(y1 @ _Sigma_inv @ y1.T)[None, :]
                - 2 * y0 @ _Sigma_inv @ y1.T
            ) / 2
            p, q = (
                torch.full((x0.shape[0],), 1 / x0.shape[0]),
                torch.full((x1.shape[0],), 1 / x1.shape[0]),
            )
            # use epsilon scaling to improve stability
            return ot.bregman.sinkhorn_epsilon_scaling(
                p.double(), q.double(), C.double(), 1.0
            ).float()

        self.sigma = sigma
        self.A = A
        self.epsI = torch.eye(self.A.shape[0]) * _eps
        self.mu = mu
        self.bm = LinearBridgeMatcher(A, mu, **bm_kwargs)
        self.x = x
        self.t_idx = t_idx
        self.ts = ts
        self.dts = ts[1:] - ts[:-1]
        self.T = T
        self.dim = dim
        self.device = device
        self.Ts = []
        # construct EOT plans
        for i in range(self.T - 1):
            self.Ts.append(
                entropic_ot_plan(
                    self.x[self.t_idx == i, :],
                    self.x[self.t_idx == i + 1, :],
                    self.dts[i] * self.sigma**2,
                )
            )

    def sample_bridges_flows(self, batch_size=64):
        _x = []
        _t = []
        _t_orig = []
        _s = []
        _u = []
        for i in range(self.T - 1):
            with torch.no_grad():
                x0, x1 = self.bm.sample_plan(
                    self.x[self.t_idx == i, :],
                    self.x[self.t_idx == i + 1, :],
                    self.Ts[i],
                    batch_size,
                )
            ts = torch.rand_like(x0[:, 0])
            _, _, x, s, u = self.bm.sample_bridge_and_flow(
                x0, x1, ts, (self.sigma**2 * self.dts[i]) ** 0.5
            )
            _x.append(x)
            _s.append(s)
            # _t.append((i + ts)*self.dt)
            # print(((i + ts)*self.dts[0])-(self.ts[i] + self.dts[i]*ts))
            _t.append(self.ts[i] + self.dts[i] * ts)
            _t_orig.append(ts)
            _u.append(u / self.dts[i])
        return (
            torch.vstack(_x),
            torch.vstack(_s),
            torch.vstack(_u),
            torch.hstack(_t)[:, None],
            torch.hstack(_t_orig)[:, None],
        )


class EntropicOTFM:
    """
    Sampler for OTFM
    """

    def __init__(self, x, t_idx, ts, sigma, T, dim, device):
        """
        Initialize and pre-compute EOT plans between time steps

        x: data points (N, d)
        t_idx: time indices for each data point (N,)
        ts: time steps (T,)
        sigma: square-root diffusivity
        T: total number of timesteps
        dim: dimension d
        """

        def entropic_ot_plan(x0, x1, eps):
            C = ot.utils.euclidean_distances(x0, x1, squared=True) / 2
            p, q = (
                torch.full((x0.shape[0],), 1 / x0.shape[0]),
                torch.full((x1.shape[0],), 1 / x1.shape[0]),
            )
            return ot.bregman.sinkhorn_epsilon_scaling(
                p.double(), q.double(), C.double(), eps
            ).float()

        self.sigma = sigma
        self.bm = BridgeMatcher()
        self.x = x
        self.t_idx = t_idx
        self.ts = ts
        self.dts = ts[1:] - ts[:-1]
        self.T = T
        self.dim = dim
        self.device = device
        self.Ts = []
        # construct EOT plans
        for i in range(self.T - 1):
            self.Ts.append(
                entropic_ot_plan(
                    self.x[self.t_idx == i, :],
                    self.x[self.t_idx == i + 1, :],
                    self.dts[i] * self.sigma**2,
                )
            )

    def sample_bridges_flows(self, batch_size=64):
        _x = []
        _t = []
        _t_orig = []
        _s = []
        _u = []
        for i in range(self.T - 1):
            with torch.no_grad():
                x0, x1 = self.bm.sample_plan(
                    self.x[self.t_idx == i, :],
                    self.x[self.t_idx == i + 1, :],
                    self.Ts[i],
                    batch_size,
                )
            ts = torch.rand_like(x0[:, :1])
            _, _, x, s, u = self.bm.sample_bridge_and_flow(
                x0, x1, ts, (self.sigma**2 * self.dts[i]) ** 0.5
            )
            _x.append(x)
            _s.append(s)
            # _t.append((i + ts)*self.dt)
            _t.append(self.ts[i] + self.dts[i] * ts)
            _t_orig.append(ts)
            # _u.append(u)
            _u.append(u / self.dts[i])
        return (
            torch.vstack(_x),
            torch.vstack(_s),
            torch.vstack(_u),
            torch.vstack(_t),
            torch.vstack(_t_orig),
        )


def cat_tx(t, x):
    return torch.hstack(
        [x, torch.scalar_tensor(t).to(x.device).expand(*x.shape[:-1], 1)]
    )


class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, drift, sigma=1.0):
        super().__init__()
        self.drift = drift
        self.sigma = sigma

    def f(self, t, y):
        return self.drift(t, y)

    def g(self, t, y):
        return torch.ones_like(y) * self.sigma


### The following are code for computing the closed form mvOU-GSB between Gaussian marginal distributions (see Theorem 3)
# Assumes t0 = 0, t1 = 1.


def gaussian_EOT_batch(A, B, sigma):
    """Variant of gaussian_EOT batched along first dimension"""
    d = A.shape[-1]
    _eye = torch.eye(d).unsqueeze(0).expand(A.shape)
    sqrtA = _bsqrtm(A)
    sqrtA_inv = torch.linalg.pinv(sqrtA)
    Q = torch.bmm(sqrtA, torch.bmm(B, sqrtA)) + sigma**4 / 4 * _eye
    C = torch.bmm(sqrtA, torch.bmm(_bsqrtm(Q), sqrtA_inv)) - sigma**2 / 2 * _eye
    return C


def gaussian_EOT(A, B, sigma):
    """Closed form Gaussian EOT covariance between N(0, A) and N(0, B) with noise level sigma (see Eq. 64)."""
    d = A.shape[-1]
    _eye = torch.eye(d)
    sqrtA = _sqrtm(A)
    sqrtA_inv = torch.linalg.pinv(sqrtA)
    Q = torch.matmul(sqrtA, torch.matmul(B, sqrtA)) + sigma**4 / 4 * _eye
    C = torch.matmul(sqrtA, torch.matmul(_sqrtm(Q), sqrtA_inv)) - sigma**2 / 2 * _eye
    return C


def bures(A, B):
    """Bures metric between covariances A and B"""
    _sqrtA = _sqrtm(A)
    return (
        torch.trace(A) + torch.trace(B) - 2 * torch.trace(_sqrtm(_sqrtA @ B @ _sqrtA))
    )


def bures_wasserstein(a, b, A, B):
    """Bures-Wasserstein distance between Gaussians N(a, A) and N(b, B)"""
    return torch.norm(a - b) ** 2 + bures(A, B)


class GaussianOUSB:
    """Class for computing Gaussian mvOU-SB parameters using exact formulas (Theorem 3)"""

    def __init__(self, bm: LinearBridgeMatcher, otfm: LinearEntropicOTFM):
        """Initialize from a LinearBridgeMatcher object"""
        self.bm = bm
        self.otfm = otfm
        # Precompute a few things
        self._Sigma = (
            bm.Phi_t[-1] * otfm.dts[0] * otfm.sigma**2
        )  # covariance in the transition kernel
        self._sqrtSigma = _sqrtm(self._Sigma)
        self._sqrtSigma_inv = torch.linalg.pinv(self._sqrtSigma)
        self._Phi_1 = bm.Phi_t[-1]
        self._Phi_1_inv = torch.linalg.pinv(self._Phi_1)
        self._expA = torch.linalg.matrix_exp(bm.A)

    def evaluate(self, t, a, b, A, B):
        """
        Return the SB marginal parameters and drift at time t \in [0, 1] between N(a, A) and N(b, B).

        t: time points at which to evaluate
        a, b: source and target mean
        A, B: source and target covariance
        returns: (means, vars, S_t, d_sb_means) where (means, vars) are parameters of SB marginal at time t
        and (S_t, d_sb_means) specify the SB drift (see Eq. 17)
        """
        bm = self.bm
        d = a.shape[0]
        nbatch = len(t)
        _u = torch.linalg.matrix_exp(-(1 - t[:, None, None]) * bm.A)  # exp(-(1-t)A)
        _v = torch.linalg.matrix_exp((1 - t[:, None, None]) * bm.A)  # exp((1-t)A)
        _Phi_t = bm.interp(bm.Phi_t, t)
        _Phi_1_inv_expanded = self._Phi_1_inv.unsqueeze(0).expand(_Phi_t.shape)
        _Lambda_t = torch.bmm(torch.bmm(_Phi_t, _v.mT), _Phi_1_inv_expanded)
        _Omega_t = self.otfm.sigma**2 * bm._bridge_cov(t, None, None)
        # compute coefficients for SB
        A_t = torch.bmm(
            _u - _Lambda_t, self._sqrtSigma[None, :, :].expand(len(t), d, d)
        )
        B_t = torch.bmm(_Lambda_t, self._sqrtSigma[None, :, :].expand(len(t), d, d))
        c_t = torch.bmm(
            torch.eye(d).expand(len(t), d, d) - _u,
            bm.mu.unsqueeze(1).expand(len(t), d, 1),
        )
        # transformed mean and covariances
        a_bar = self._sqrtSigma_inv @ (self._expA @ (a - bm.mu) + bm.mu)
        a_bar_expanded = a_bar.unsqueeze(0).expand((nbatch, d))
        b_bar = self._sqrtSigma_inv @ b
        b_bar_expanded = b_bar.unsqueeze(0).expand((nbatch, d))
        A_bar = (
            self._sqrtSigma_inv @ self._expA @ A @ self._expA.T @ self._sqrtSigma_inv
        )
        A_bar_expanded = A_bar.unsqueeze(0).expand(A_t.shape)
        B_bar = self._sqrtSigma_inv @ B @ self._sqrtSigma_inv
        B_bar_expanded = B_bar.unsqueeze(0).expand(B_t.shape)
        C_bar = gaussian_EOT(A_bar, B_bar, 1.0)
        C_bar_expanded = C_bar.unsqueeze(0).expand(A_t.shape)
        # mean and covariance of SB
        AAA = torch.bmm(A_t, torch.bmm(A_bar_expanded, A_t.mT))
        BBB = torch.bmm(B_t, torch.bmm(B_bar_expanded, B_t.mT))
        ACB = torch.bmm(A_t, torch.bmm(C_bar_expanded, B_t.mT))
        BCA = torch.bmm(B_t, torch.bmm(C_bar_expanded.mT, A_t.mT))
        sb_vars = AAA + BBB + ACB + BCA + _Omega_t
        sb_means = (
            torch.bmm(A_t, a_bar_expanded[..., None])
            + torch.bmm(B_t, b_bar_expanded[..., None])
            + c_t
        )
        # compute derivative of Phi_t
        _exptA = torch.linalg.matrix_exp(t[:, None, None] * bm.A)
        _dPhi_t = torch.bmm(_exptA, _exptA.mT)
        _Q_t = torch.bmm(_Phi_t, _v.mT)
        _dQ_t = torch.bmm(_dPhi_t, _v.mT) - torch.bmm(
            _Phi_t, torch.bmm(bm.A.T.unsqueeze(0).expand(_v.shape), _v.mT)
        )
        _dLambda_t = torch.bmm(_dQ_t, _Phi_1_inv_expanded)
        _dOmega_t = _dPhi_t - (
            torch.bmm(_dQ_t, torch.bmm(_Phi_1_inv_expanded, _Q_t.mT))
            + torch.bmm(_Q_t, torch.bmm(_Phi_1_inv_expanded, _dQ_t.mT))
        )
        _dOmega_ts = torch.bmm(
            _Phi_t,
            bm.A.T.unsqueeze(0).expand(_v.shape)
            - torch.bmm(_v.mT, torch.bmm(_Phi_1_inv_expanded, _dQ_t.mT)),
        )
        dA_t = torch.bmm(
            torch.bmm(bm.A.unsqueeze(0).expand(_v.shape), _u) - _dLambda_t,
            self._sqrtSigma.unsqueeze(0).expand(_v.shape),
        )
        dB_t = torch.bmm(_dLambda_t, self._sqrtSigma.unsqueeze(0).expand(_v.shape))
        dc_t = -torch.bmm(
            torch.bmm(bm.A.unsqueeze(0).expand(_u.shape), _u),
            bm.mu.unsqueeze(1).expand(len(t), d, 1),
        )
        S_t = (
            _dOmega_ts
            + torch.bmm(A_t, torch.bmm(A_bar_expanded, dA_t.mT))
            + torch.bmm(B_t, torch.bmm(B_bar_expanded, dB_t.mT))
            + torch.bmm(A_t, torch.bmm(C_bar_expanded, dB_t.mT))
            + torch.bmm(B_t, torch.bmm(C_bar_expanded.mT, dA_t.mT))
        )
        d_sb_means = torch.bmm(dA_t, a_bar_expanded[..., None]) + torch.bmm(
            dB_t, b_bar_expanded[..., None]
        )
        return sb_means, sb_vars, S_t, d_sb_means
