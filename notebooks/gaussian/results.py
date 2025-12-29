import pandas as pd
import glob
import seaborn as sb
import os

N = 128
files = glob.glob(f"eval/eval_seed_*_dim_*_N_{N}.csv")
seeds = [int(os.path.basename(f).split("_")[2]) for f in files]
dims = [int(os.path.basename(f).split("_")[4]) for f in files]
df = pd.concat(
    [
        pd.DataFrame(pd.read_csv(f, index_col=0).iloc[:, 1:].mean(0))
        .T.assign(seed=s)
        .assign(dim=d)
        .assign(file=f)
        for (f, s, d) in zip(files, seeds, dims)
    ]
)
_df = pd.concat(
    [
        pd.DataFrame(
            pd.read_csv(f"eval/eval_nlsb_seed_{s}_dim_{d}_N_{N}.csv", index_col=0)
            .iloc[:, 1:]
            .mean(0)
        ).T
        for (f, s, d) in zip(files, seeds, dims)
    ]
)
df = pd.concat([df, _df], axis=1)


_df1 = df.melt(
    id_vars=["seed", "dim", "file"],
    value_vars=[
        "OU_EOT_BW",
        "EOT_BW",
        "GP_sinkhorn_fwd_BW",
        "GP_sinkhorn_rev_BW",
        "NLSB_BW",
    ],
)
_df1 = _df1.replace(
    {
        "variable": {
            "OU_EOT_BW": "OU-GSB",
            "EOT_BW": "BM-GSB",
            "GP_sinkhorn_fwd_BW": "IPML(→)",
            "GP_sinkhorn_rev_BW": "IPML(←)",
            "NLSB_BW": "NLSB",
        }
    }
)
_df1 = _df1.rename(
    columns={
        "variable": "method",
    }
)
_df1.dim = _df1.dim.astype("category")

_df2 = df.melt(
    id_vars=["seed", "dim", "file"],
    value_vars=[
        "OU_EOT_vf",
        "EOT_vf",
        "GP_sinkhorn_vf",
        "NLSB_vf",
    ],
)
_df2 = _df2.replace(
    {
        "variable": {
            "OU_EOT_vf": "OU-GSB",
            "EOT_vf": "BM-GSB",
            "GP_sinkhorn_vf": "IPML",
            "NLSB_vf": "NLSB",
        }
    }
)
_df2 = _df2.rename(
    columns={
        "variable": "method",
    }
)
_df2.dim = _df2.dim.astype("category")


_make_bold = lambda x: "**" + x + "**"

from toolz import interleave
import numpy as np

_df1_mean = (
    _df1.groupby(["dim", "method"])[["value"]]
    .agg(
        {
            "value": [
                "mean",
            ]
        }
    )
    .unstack()
)
_df1_std = (
    _df1.groupby(["dim", "method"])[["value"]]
    .agg(
        {
            "value": [
                "std",
            ]
        }
    )
    .unstack()
)
_df1_mean_str = _df1_mean.applymap(
    lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
)
_df1_std_str = _df1_std.applymap(
    lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
)
_df1_str = pd.DataFrame(
    {
        _df1_mean_str.index[i]: _df1_mean_str.iloc[:, i].str.cat(
            _df1_std_str.iloc[:, i], sep=" $\\pm$ "
        )
        for i in range(_df1_mean_str.shape[1])
    }
)
_df1_str.columns = _df1_mean.columns
for i, j in enumerate(np.argmin(_df1_mean.values, 1)):
    _df1_str.iloc[i, j] = _make_bold(_df1_str.iloc[i, j])
_df1_str.columns = [x[-1] for x in _df1_str.columns]


_df2_mean = (
    _df2.groupby(["dim", "method"])[["value"]]
    .agg(
        {
            "value": [
                "mean",
            ]
        }
    )
    .unstack()
)
_df2_std = (
    _df2.groupby(["dim", "method"])[["value"]]
    .agg(
        {
            "value": [
                "std",
            ]
        }
    )
    .unstack()
)
_df2_mean_str = _df2_mean.applymap(
    lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
)
_df2_std_str = _df2_std.applymap(
    lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
)
_df2_str = pd.DataFrame(
    {
        _df2_mean_str.index[i]: _df2_mean_str.iloc[:, i].str.cat(
            _df2_std_str.iloc[:, i], sep=" $\\pm$ "
        )
        for i in range(_df2_mean_str.shape[1])
    }
)
_df2_str.columns = _df2_mean.columns
for i, j in enumerate(np.argmin(_df2_mean.values, 1)):
    _df2_str.iloc[i, j] = _make_bold(_df2_str.iloc[i, j])
_df2_str.columns = [x[-1] for x in _df2_str.columns]


# Print table of Bures-Wasserstein error
print(_df1_str.iloc[:, [4, 0, 1, 2, 3]].to_markdown())

# Print table of vector field error
print(_df2_str.iloc[:, [3, 0, 1, 2]].to_markdown())
