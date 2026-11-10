"""
Evaluation utilities for perturbation prediction.

Delegates metric computation to scGPT's compute_perturbation_metrics, which computes
per-condition Pearson correlation then averages across conditions. The four metrics:

  pearson         — absolute expression, all genes (misleadingly high for all models)
  pearson_delta   — delta from ctrl, all genes (more informative)
  pearson_de      — absolute expression, top-20 DE genes
  pearson_de_delta — delta from ctrl, top-20 DE genes (THE key metric: ~0.80 for scGPT)
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def compute_metrics(results: Dict, ctrl_adata) -> Dict[str, float]:
    """
    Compute all four standard perturbation metrics using scGPT's implementation.
    ctrl_adata must have uns['rank_genes_groups_cov_all'] (pre-computed by GEARS).
    'ctrl' must NOT be in results['pert_cat'].
    """
    from scgpt.utils import compute_perturbation_metrics
    return compute_perturbation_metrics(results, ctrl_adata)


def filter_ctrl_from_results(results: Dict) -> Dict:
    """Remove ctrl cells from a results dict (compute_perturbation_metrics requires this)."""
    mask = results["pert_cat"] != "ctrl"
    return {
        "pert_cat": results["pert_cat"][mask],
        "pred": results["pred"][mask],
        "truth": results["truth"][mask],
        "pred_de": results["pred_de"][mask],
        "truth_de": results["truth_de"][mask],
    }


def build_comparison_table(model_results: Dict[str, Dict],
                            ctrl_adata) -> pd.DataFrame:
    """
    Build a side-by-side metric table across models.
    model_results: {"scGPT": results_dict, "DeltaMean": results_dict, ...}
    Returns DataFrame indexed by model name.
    """
    rows = {}
    for name, res in model_results.items():
        clean = filter_ctrl_from_results(res)
        rows[name] = compute_metrics(clean, ctrl_adata)
    return pd.DataFrame(rows).T[
        ["pearson", "pearson_delta", "pearson_de", "pearson_de_delta"]
    ]


def format_baseline_results(baseline_preds: Dict[str, np.ndarray],
                             pert_data) -> Dict:
    """
    Convert {pert_name: predicted_array} into the results dict format used by
    compute_perturbation_metrics: keys pert_cat, pred, truth, pred_de, truth_de.
    Iterates through the test DataLoader to build correctly aligned arrays.
    """
    pert_cat, pred, truth, pred_de, truth_de = [], [], [], [], []

    for batch in pert_data.dataloader["test_loader"]:
        for i, pert in enumerate(batch.pert):
            if pert not in baseline_preds:
                continue
            p = baseline_preds[pert]
            t = batch.y[i].numpy()
            de_idx = batch.de_idx[i]
            pert_cat.append(pert)
            pred.append(p)
            truth.append(t)
            pred_de.append(p[de_idx])
            truth_de.append(t[de_idx])

    return {
        "pert_cat": np.array(pert_cat),
        "pred": np.stack(pred).astype(np.float32),
        "truth": np.stack(truth).astype(np.float32),
        "pred_de": np.stack(pred_de).astype(np.float32),
        "truth_de": np.stack(truth_de).astype(np.float32),
    }
