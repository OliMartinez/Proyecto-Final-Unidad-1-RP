"""
tomek.py
--------
Higiene de la frontera de decisión mediante el algoritmo de Tomek Links.

Definición formal:
    Dos instancias (x, y) forman un Tomek Link si y solo si:
        1) x es el vecino más cercano de y,
        2) y es el vecino más cercano de x, y
        3) x y y pertenecen a clases opuestas.
    La eliminación de estos pares purga el ruido interclase y reduce el
    sobreajuste en la frontera, sin alterar la estructura interna de cada
    clase.

Se utiliza la implementación de referencia de imbalanced-learn, que
identifica los pares mediante búsqueda de vecino más cercano vectorizada
(KD-Tree / fuerza bruta optimizada), evitando bucles Python explícitos.
"""

from __future__ import annotations

import numpy as np
from imblearn.under_sampling import TomekLinks


def apply_tomek_links(
    X: np.ndarray, y: np.ndarray, sampling_strategy: str = "all"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aplica Tomek Links y regresa (X_limpio, y_limpio, indices_conservados).

    sampling_strategy="all" permite la limpieza simétrica entre todas las
    clases (generalización multiclase de la definición binaria original).
    """
    tl = TomekLinks(sampling_strategy=sampling_strategy)
    X_res, y_res = tl.fit_resample(X, y)
    kept_idx = tl.sample_indices_
    return X_res, y_res, kept_idx


def removed_summary(y: np.ndarray, kept_idx: np.ndarray) -> dict[int, int]:
    """Número de muestras eliminadas por clase."""
    removed_mask = np.ones(len(y), dtype=bool)
    removed_mask[kept_idx] = False
    removed_classes, removed_counts = np.unique(y[removed_mask], return_counts=True)
    return {int(c): int(n) for c, n in zip(removed_classes, removed_counts)}
