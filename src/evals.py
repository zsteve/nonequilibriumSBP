import geomloss
import ot
import ot.utils
from ot.backend import get_backend
import numpy as np
import torch


def emd_samples(x, y, x_w=None, y_w=None):
    """EMD between samples (x, y) with optional weights (x_w, y_w)"""
    C = ot.utils.euclidean_distances(x, y, squared=True)
    nx = get_backend(x, y)
    p = nx.full((x.shape[0],), 1 / x.shape[0]) if x_w is None else x_w / x_w.sum()
    q = nx.full((y.shape[0],), 1 / y.shape[0]) if y_w is None else y_w / y_w.sum()
    return ot.emd2(p, q, C)


def sinkhorn_divergence(x, y, x_w=None, y_w=None, reg=1.0):
    """Sinkhorn divergence between samples (x, y) with optional weights (x_w, y_w)"""
    # p = np.full((x.shape[0], ), 1/x.shape[0]) if x_w is None else x_w / x_w.sum()
    # q = np.full((y.shape[0], ), 1/y.shape[0]) if y_w is None else y_w / y_w.sum()
    # return ot.bregman.empirical_sinkhorn_divergence(x, y, reg, a = p, b = q)
    p = torch.full((x.shape[0],), 1 / x.shape[0]) if x_w is None else x_w / x_w.sum()
    q = torch.full((y.shape[0],), 1 / y.shape[0]) if y_w is None else y_w / y_w.sum()
    loss = geomloss.SamplesLoss(loss="sinkhorn")
    return loss(p, x, q, y)


def energy_distance(x, y, x_w=None, y_w=None):
    """Energy distance between samples (x, y) with optional weights (x_w, y_w)"""
    nx = get_backend(x, y)
    x_w = nx.full((x.shape[0],), 1 / x.shape[0]) if x_w is None else x_w / x_w.sum()
    y_w = nx.full((y.shape[0],), 1 / y.shape[0]) if y_w is None else y_w / y_w.sum()
    xy = nx.dot(x_w, ot.utils.euclidean_distances(x, y, squared=False) @ y_w)
    xx = nx.dot(x_w, ot.utils.euclidean_distances(x, x, squared=False) @ x_w)
    yy = nx.dot(y_w, ot.utils.euclidean_distances(y, y, squared=False) @ y_w)
    return 2 * xy - xx - yy


def energy_distance_paths(x, y):
    """Energy distance between path samples (x, y)"""
    return energy_distance(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1))


def emd_paths(x, y):
    """EMD between path samples (x, y)"""
    return emd_samples(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1))
