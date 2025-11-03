"""
Gaussian Mixture Model (GMM) overextension detector utilities.

Modules:
- data: loading processed bars, splitting days, building feature matrices
- model: tuning/training GMM, scoring log-likelihood, saving artifacts
- evaluate: evaluation metrics, persistence of candidates, plotting
"""

__all__ = [
    "data",
    "model",
    "evaluate",
]


