import numpy as np
from anndata import AnnData
from typing import Dict, List


class MeanControlBaseline:
    """
    Predict ctrl mean expression for every perturbation.
    Scores very high on overall Pearson (>0.99) because most genes don't change,
    but near-zero on pearson_de_delta — it predicts no perturbation effect at all.
    Use this as a sanity floor: any real model must beat it on pearson_de_delta.
    """

    def fit(self, ctrl_adata: AnnData) -> None:
        self.ctrl_mean = np.asarray(ctrl_adata.X.mean(axis=0)).flatten()

    def predict(self, perturbations: List[str]) -> Dict[str, np.ndarray]:
        return {pert: self.ctrl_mean.copy() for pert in perturbations}


class DeltaMeanBaseline:
    """
    For training perturbations: return the observed mean expression.
    For unseen test perturbations: ctrl_mean + global mean delta across all training perts.

    This is the 'Train Mean' baseline from the perturbation benchmarking literature
    and is surprisingly competitive (Adamson pearson_de_delta ~0.71).
    It represents the bar a deep model must beat to show it learned something
    beyond memorizing average perturbation direction.
    """

    def fit(self, adata: AnnData, ctrl_adata: AnnData) -> None:
        self.ctrl_mean = np.asarray(ctrl_adata.X.mean(axis=0)).flatten()

        conditions = [c for c in adata.obs["condition"].unique() if c != "ctrl"]
        per_pert_means = []
        self.pert_means: Dict[str, np.ndarray] = {}

        for cond in conditions:
            subset = adata[adata.obs["condition"] == cond]
            mean_expr = np.asarray(subset.X.mean(axis=0)).flatten()
            self.pert_means[cond] = mean_expr
            per_pert_means.append(mean_expr - self.ctrl_mean)

        self.global_delta = np.mean(per_pert_means, axis=0)

    def predict(self, perturbations: List[str]) -> Dict[str, np.ndarray]:
        predictions = {}
        for pert in perturbations:
            if pert in self.pert_means:
                predictions[pert] = self.pert_means[pert].copy()
            else:
                predictions[pert] = self.ctrl_mean + self.global_delta
        return predictions
