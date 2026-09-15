"""
dmin.py
-------
Clasificador Paramétrico de Mínima Distancia (D-min), con
extensión de costo asimétrico (priors ajustados por riesgo).

Fundamento probabilístico
--------------------------
Sea la verosimilitud p(x | w_k) ~ N(mu_k, Sigma_k). Por Bayes:

    p(w_k | x) proporcional a  p(x | w_k) * p(w_k)

Si Sigma_k = sigma^2 * I para toda k (covarianza isotrópica compartida) y
p(w_k) es igual para toda k, entonces:

    log p(x|w_k) = -d/2 log(2 pi sigma^2) - (1 / 2 sigma^2) * ||x - mu_k||^2

El primer término no depende de k, así que maximizar p(w_k | x) sobre k es
equivalente a minimizar ||x - mu_k||^2, es decir:

    w*(x) = argmin_k  d_euclid(x, mu_k)

Esta es exactamente la regla D-min con métrica euclidiana (Duda, Hart &
Stork, 2001, Cap. 2).

Si en cambio Sigma_k = Sigma (covarianza compartida, no necesariamente
isotrópica) para toda k, el término cuadrático x^T Sigma^-1 x se cancela
entre clases y la regla de decisión colapsa a:

    w*(x) = argmin_k  (x - mu_k)^T Sigma^-1 (x - mu_k)  =  argmin_k D_Mahalanobis(x, mu_k)

Esto generaliza el D-min euclidiano e incorpora la correlación entre
características (Bishop, 2006, Sec. 4.2.1): la métrica de Mahalanobis es
invariante a transformaciones lineales inversibles porque "blanquea" el
espacio con Sigma^-1 antes de medir distancia.

ADVERTENCIA -- vulnerabilidad de Mahalanobis bajo desbalance extremo:
Sigma (la covarianza combinada / *pooled*) se estima estrictamente a partir
de la dispersión intraclase observada. Cuando una clase domina numéricamente
el conjunto (como la clase benigna en este estudio, >96%), su dispersión
domina la suma que define a Sigma, de modo que la geometría "blanqueada"
por Sigma^-1 queda calibrada casi exclusivamente a la forma de la nube de
la clase mayoritaria. La distancia de Mahalanobis puede entonces distorsionar
sistemáticamente la geometría real de las clases minoritarias -- un riesgo
estructural del propio estimador de covarianza compartida, no un error de
implementación. Este efecto se cuantifica en la Sección IV (Tabla II) y se
observa empíricamente en el colapso de Sensibilidad de la Sección IV-C.

Extensión de costo asimétrico
--------------------------------------------
Bajo una función de riesgo con costos desiguales (costo de falso negativo
C_FN >> costo de falso positivo C_FP), la regla de decisión de riesgo
mínimo de Bayes asigna x a la clase que minimiza el riesgo esperado, lo que
--tras sustituir Bayes y cancelar los términos comunes-- equivale a
introducir un prior efectivo p(w_k) ya no uniforme, sino ponderado por el
costo. Repitiendo la derivación anterior con priors desiguales:

    w*(x) = argmin_k [ D(x, mu_k)^2  -  2 * scale * log( p(w_k) ) ]

donde ``scale = sigma^2`` (varianza combinada) para la métrica Euclidiana, y
``scale = 1`` para Mahalanobis (que ya normaliza la escala vía Sigma^-1).
Inflar p(w_k) para la clase patológica *reduce* su distancia efectiva,
sesgando la frontera de decisión a su favor -- exactamente el mecanismo
buscado para penalizar los falsos negativos.

Implementación
--------------
Todas las distancias de todas las muestras hacia todos los centroides se
calculan en una sola llamada vectorizada a `scipy.spatial.distance.cdist`
(matriz N x K). La asignación final es un `argmin` sobre el eje de clases.
No existe ningún bucle ``for`` sobre muestras.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .geometry import pooled_within_class_covariance

_MINKOWSKI_P = 3


class MinimumDistanceClassifier:
    """Clasificador de mínima distancia a centroide (D-min), multiclase.

    Parameters
    ----------
    metric : str
        Una de las 6 métricas de geometry.METRICS.
    priors : None | dict | array-like
        Si es None (default), D-min clásico sin ponderar -- idéntico al
        comportamiento original, preserva la reproducibilidad de resultados
        previos. Si se provee, un peso relativo por clase (no necesita
        sumar 1; se normaliza internamente) que implementa la regla de
        riesgo mínimo de Bayes con costo asimétrico (ver docstring del
        módulo). Inflar el peso de una clase reduce su distancia efectiva.
    """

    def __init__(self, metric: str = "euclidean", priors=None):
        self.metric = metric
        self.priors = priors
        self.centroids_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.VI_: np.ndarray | None = None
        self._scale_: float = 1.0
        self._log_priors_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MinimumDistanceClassifier":
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])

        need_cov = (self.metric == "mahalanobis") or (self.priors is not None)
        Sw = pooled_within_class_covariance(X, y) if need_cov else None

        if self.metric == "mahalanobis":
            self.VI_ = np.linalg.pinv(Sw)
            self._scale_ = 1.0  # Mahalanobis ya normaliza escala vía Sigma^-1
        elif self.priors is not None:
            d = X.shape[1]
            self._scale_ = float(np.trace(Sw) / d)  # sigma^2 combinada (caso isotrópico)
        else:
            self._scale_ = 1.0

        if self.priors is not None:
            if isinstance(self.priors, dict):
                p = np.array([self.priors[c] for c in self.classes_], dtype=float)
            else:
                p = np.asarray(self.priors, dtype=float)
            p = p / p.sum()
            self._log_priors_ = np.log(p)
        else:
            self._log_priors_ = None
        return self

    def _kwargs(self) -> dict:
        if self.metric == "mahalanobis":
            return {"VI": self.VI_}
        if self.metric == "minkowski":
            return {"p": _MINKOWSKI_P}
        return {}

    def decision_distances(self, X: np.ndarray) -> np.ndarray:
        """Matriz (n_muestras, n_clases) de distancias a cada centroide."""
        if self.centroids_ is None:
            raise RuntimeError("El clasificador no ha sido entrenado (fit).")
        return cdist(X, self.centroids_, metric=self.metric, **self._kwargs())

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """Puntuación a minimizar por clase (argmin => clase asignada).

        Sin ``priors``: coincide exactamente con ``decision_distances`` (el
        D-min clásico de la Sección 2.3). Con ``priors``: aplica la
        corrección de riesgo mínimo de Bayes de la Sección 2.5,
        score_k = D(x,mu_k)^2 - 2*scale*log(pi_k).
        """
        D = self.decision_distances(X)
        if self._log_priors_ is None:
            return D
        return D ** 2 - 2.0 * self._scale_ * self._log_priors_[np.newaxis, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        S = self.decision_scores(X)
        idx = np.argmin(S, axis=1)
        return self.classes_[idx]

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Convierte la puntuación de decisión en pseudo-probabilidades vía
        softmax de la puntuación negativa (score menor = más cerca => score
        de "pertenencia" mayor). Se usa para trazar curvas ROC por clase
        (one-vs-rest) y, con priors, para explorar el movimiento del umbral
        de decisión; no reemplaza la regla de decisión D-min
        en sí misma.
        """
        S = self.decision_scores(X)
        neg = -S
        neg = neg - neg.max(axis=1, keepdims=True)
        expd = np.exp(neg)
        return expd / expd.sum(axis=1, keepdims=True)
