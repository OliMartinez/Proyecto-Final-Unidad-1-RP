"""
geometry.py
-----------
Evaluación geométrica del espacio de características mediante seis
métricas analíticas de distancia, y formalización vectorizada de la
dispersión intraclase y la separación interclase.

Todas las distancias se calculan con scipy.spatial.distance.cdist:
ningún bucle ``for`` recorre pares de muestras.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

# Las 6 métricas analíticas exigidas por el enunciado (incluye explícitamente
# Euclidiana, Manhattan/cityblock y Mahalanobis).
METRICS: list[str] = [
    "euclidean",
    "cityblock",
    "mahalanobis",
    "chebyshev",
    "cosine",
    "minkowski",
]

_MINKOWSKI_P = 3  # p empleado cuando metric == "minkowski" (distinto de Euclidiana/Manhattan)


def _cdist_kwargs(metric: str, VI: np.ndarray | None) -> dict:
    """Argumentos extra que requiere cada métrica en cdist."""
    if metric == "mahalanobis":
        return {"VI": VI}
    if metric == "minkowski":
        return {"p": _MINKOWSKI_P}
    return {}


def pooled_within_class_covariance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matriz de covarianza combinada (pooled) Sigma_W, usada como Sigma^-1
    (vía pseudo-inversa) en la distancia de Mahalanobis.

    Sigma_W = (1 / (n - K)) * sum_k sum_{x in clase k} (x - mu_k)(x - mu_k)^T
    """
    n, d = X.shape
    classes = np.unique(y)
    Sw = np.zeros((d, d))
    for c in classes:
        Xc = X[y == c]
        Xc_centered = Xc - Xc.mean(axis=0, keepdims=True)
        Sw += Xc_centered.T @ Xc_centered
    Sw /= max(n - len(classes), 1)
    return Sw


def class_centroids(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Regresa (centroides, clases_ordenadas). centroides.shape = (K, d)."""
    classes = np.unique(y)
    centroids = np.array([X[y == c].mean(axis=0) for c in classes])
    return centroids, classes


def intraclass_dispersion(
    X: np.ndarray, y: np.ndarray, metric: str, VI: np.ndarray | None = None
) -> dict[int, float]:
    """Dispersión intraclase: distancia promedio de cada muestra x_i hacia
    el centroide mu_k de su propia clase (vectorizado con cdist, una
    llamada por clase -- nunca por muestra).

        D_intra(k) = (1/|C_k|) * sum_{x in C_k} d(x, mu_k)
    """
    classes = np.unique(y)
    kwargs = _cdist_kwargs(metric, VI)
    dispersion = {}
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(axis=0, keepdims=True)
        d = cdist(Xc, mu, metric=metric, **kwargs).ravel()
        dispersion[int(c)] = float(d.mean())
    return dispersion


def interclass_separation(
    centroids: np.ndarray, metric: str, VI: np.ndarray | None = None
) -> np.ndarray:
    """Matriz K x K de distancias entre centroides de clase (una sola
    llamada vectorizada a cdist)."""
    kwargs = _cdist_kwargs(metric, VI)
    return cdist(centroids, centroids, metric=metric, **kwargs)


def evaluate_all_metrics(X: np.ndarray, y: np.ndarray) -> dict:
    """Corre las 6 métricas y regresa un diccionario con:
        - dispersión intraclase por clase
        - matriz de separación interclase
        - razón de separabilidad J = separación_min_interclase / dispersión_media_intraclase
    """
    Sw = pooled_within_class_covariance(X, y)
    VI = np.linalg.pinv(Sw)
    centroids, classes = class_centroids(X, y)

    results = {}
    for metric in METRICS:
        vi = VI if metric == "mahalanobis" else None
        intra = intraclass_dispersion(X, y, metric, VI=vi)
        inter = interclass_separation(centroids, metric, VI=vi)

        mean_intra = float(np.mean(list(intra.values())))
        off_diag = inter[~np.eye(inter.shape[0], dtype=bool)]
        min_inter = float(off_diag.min()) if off_diag.size else float("nan")

        results[metric] = {
            "intraclass": intra,
            "interclass_matrix": inter,
            "classes": classes.tolist(),
            "mean_intraclass": mean_intra,
            "min_interclass": min_inter,
            "separability_ratio": min_inter / mean_intra if mean_intra > 0 else float("nan"),
        }
    return results, Sw, VI


def per_sample_intra_inter_distances(
    X: np.ndarray, y: np.ndarray, metric: str = "euclidean", VI: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Para cada muestra x_i, regresa (dist_intra, dist_inter):
        dist_intra(x_i) = d(x_i, mu_{clase de x_i})
        dist_inter(x_i) = min_{k != clase de x_i} d(x_i, mu_k)
    Una sola llamada vectorizada a cdist (matriz N x K); el mínimo sobre
    clases distintas se obtiene enmascarando la columna propia con +inf,
    sin ningún bucle sobre muestras.

    Estas dos distribuciones (agregadas sobre todas las muestras de una
    clase) son la contraparte "por muestra" de D_intra(k) y separación
    interclase de la Ec. (1)-(2) del reporte, y permiten visualizar la
    separabilidad sin recurrir a una proyección 2D que puede distorsionar
    la geometría real de un espacio de alta dimensión.
    """
    centroids, classes = class_centroids(X, y)
    kwargs = _cdist_kwargs(metric, VI)
    D = cdist(X, centroids, metric=metric, **kwargs)  # (n, K)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    own_idx = np.array([class_to_idx[c] for c in y])
    n = len(y)
    intra = D[np.arange(n), own_idx]
    D_other = D.copy()
    D_other[np.arange(n), own_idx] = np.inf
    inter = D_other.min(axis=1)
    return intra, inter


def class_covariance_contributions(X: np.ndarray, y: np.ndarray) -> dict[int, float]:
    """Cuantifica qué proporción de la traza de Sigma_W (la covarianza
    combinada que alimenta la distancia de Mahalanobis, Ec. 8) proviene de
    cada clase:

        contribucion(k) = traza(sum_{x in C_k} (x-mu_k)(x-mu_k)^T) / traza(Sigma_W * (n-K))

    Es el diagnóstico directo del riesgo de "secuestro": si una clase
    mayoritaria aporta una fracción desproporcionada de esta traza (muy por
    encima de su proporción muestral n_k/n), la geometría "blanqueada" por
    Sigma^-1 queda calibrada casi exclusivamente a la dispersión de esa
    clase, penalizando la geometría real de las clases minoritarias.
    """
    classes = np.unique(y)
    contributions: dict[int, float] = {}
    total_trace = 0.0
    for c in classes:
        Xc = X[y == c]
        Xc_centered = Xc - Xc.mean(axis=0, keepdims=True)
        trace_c = float(np.trace(Xc_centered.T @ Xc_centered))
        contributions[int(c)] = trace_c
        total_trace += trace_c
    if total_trace <= 0:
        return {int(c): float("nan") for c in classes}
    return {c: v / total_trace for c, v in contributions.items()}
