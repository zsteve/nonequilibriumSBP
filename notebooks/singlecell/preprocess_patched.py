from typing import Any, Dict, Optional

import numpy as np
from anndata import AnnData


import dynamo 

from dynamo.configuration import DynamoAdataConfig
from dynamo.preprocessing.pca import pca
from dynamo.preprocessing.utils import (
    del_raw_layers,
    detect_experiment_datatype,
    reset_adata_X,
)
from dynamo.tools.cell_velocities import cell_velocities
from dynamo.tools.connectivity import neighbors, normalize_knn_graph
from dynamo.tools.dimension_reduction import reduceDimension
from dynamo.tools.dynamics import dynamics
from dynamo.tools.moments import moments
from dynamo.tools.utils import get_vel_params, set_transition_genes, update_vel_params


def recipe_kin_data(
    adata: AnnData,
    tkey: Optional[str] = None,
    reset_X: bool = True,
    X_total_layers: bool = False,
    splicing_total_layers: bool = False,
    n_top_genes: int = 1000,
    keep_filtered_cells: Optional[bool] = None,
    keep_filtered_genes: Optional[bool] = None,
    keep_raw_layers: Optional[bool] = None,
    del_2nd_moments: Optional[bool] = None,
    ekey: str = "M_t",
    vkey: str = "velocity_T",
    basis: str = "umap",
    kwargs: Dict["str", Any] = {},
    pca_kwargs = {},
) -> AnnData:

    from dynamo.preprocessing import Preprocessor

    keep_filtered_cells = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_cells, DynamoAdataConfig.RECIPE_KEEP_FILTERED_CELLS_KEY
    )
    keep_filtered_genes = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_genes, DynamoAdataConfig.RECIPE_KEEP_FILTERED_GENES_KEY
    )
    keep_raw_layers = DynamoAdataConfig.use_default_var_if_none(
        keep_raw_layers, DynamoAdataConfig.RECIPE_KEEP_RAW_LAYERS_KEY
    )
    del_2nd_moments = DynamoAdataConfig.use_default_var_if_none(
        del_2nd_moments, DynamoAdataConfig.RECIPE_DEL_2ND_MOMENTS_KEY
    )

    has_splicing, has_labeling, splicing_labeling, _ = detect_experiment_datatype(adata)

    if has_splicing and has_labeling and splicing_labeling:
        layers = ["X_new", "X_total", "X_uu", "X_ul", "X_su", "X_sl"]
    elif has_labeling:
        layers = ["X_new", "X_total"]

    if not has_labeling:
        raise Exception(
            "This recipe is only applicable to kinetics experiment datasets that have "
            "labeling data (at least either with `'uu', 'ul', 'su', 'sl'` or `'new', 'total'` "
            "layers."
        )

    # Preprocessing
    preprocessor = Preprocessor(cell_cycle_score_enable=True, **kwargs)
    preprocessor.config_monocle_recipe(adata, n_top_genes=n_top_genes)
    preprocessor.size_factor_kwargs.update(
        {
            "X_total_layers": X_total_layers,
            "splicing_total_layers": splicing_total_layers,
        }
    )
    preprocessor.normalize_by_cells_function_kwargs.update(
        {
            "X_total_layers": X_total_layers,
            "splicing_total_layers": splicing_total_layers,
            "keep_filtered": keep_filtered_genes,
            "total_szfactor": "total_Size_Factor",
        }
    )
    preprocessor.filter_cells_by_outliers_kwargs["keep_filtered"] = keep_filtered_cells
    preprocessor.select_genes_kwargs["keep_filtered"] = keep_filtered_genes
    print(f"preprocessor.pca_kwargs = {preprocessor.pca_kwargs}")
    preprocessor.pca_kwargs.update(pca_kwargs)
    print(f"preprocessor.pca_kwargs = {preprocessor.pca_kwargs}")

    if reset_X:
        reset_adata_X(adata, experiment_type="kin", has_labeling=has_labeling, has_splicing=has_splicing)
    preprocessor.preprocess_adata_monocle(adata=adata, tkey=tkey, experiment_type="kin")
    if not keep_raw_layers:
        del_raw_layers(adata)

    if has_splicing and has_labeling:
        # new, total (and uu, ul, su, sl if existed) layers will be normalized with size factor calculated with total
        # layers spliced / unspliced layers will be normalized independently.

        tkey = adata.uns["pp"]["tkey"]
        # first calculate moments for labeling data relevant layers using total based connectivity graph
        moments(adata, group=tkey, layers=layers)

        # then we want to calculate moments for spliced and unspliced layers based on connectivity graph from spliced
        # data.
        # first get X_spliced based pca embedding
        CM = np.log1p(adata[:, adata.var.use_for_pca].layers["X_spliced"].A)
        cm_genesums = CM.sum(axis=0)
        valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
        valid_ind = np.array(valid_ind).flatten()

        pca(adata, CM[:, valid_ind], pca_key="X_spliced_pca")
        # then get neighbors graph based on X_spliced_pca
        neighbors(adata, X_data=adata.obsm["X_spliced_pca"], layer="X_spliced")
        # then normalize neighbors graph so that each row sums up to be 1
        conn = normalize_knn_graph(adata.obsp["connectivities"] > 0)
        # then calculate moments for spliced related layers using spliced based connectivity graph
        moments(adata, conn=conn, layers=["X_spliced", "X_unspliced"])
        # then perform kinetic estimations with properly preprocessed layers for either the labeling or the splicing
        # data
        dynamics(adata, model="deterministic", est_method="twostep", del_2nd_moments=del_2nd_moments)
        # then perform dimension reduction
        reduceDimension(adata, reduction_method=basis)
        # lastly, project RNA velocity to low dimensional embedding.
        cell_velocities(adata, enforce=True, vkey=vkey, ekey=ekey, basis=basis)
    else:
        dynamics(adata, model="deterministic", est_method="twostep", del_2nd_moments=del_2nd_moments)
        reduceDimension(adata, reduction_method=basis)
        cell_velocities(adata, basis=basis)

    return adata
