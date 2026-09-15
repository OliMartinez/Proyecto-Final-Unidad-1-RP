"""
data.py
-------
Generación del dataset sintético que emula un espacio de características
de espectrometría de masas para el cribado oncológico temprano.

Se definen tres clases biológicamente interpretables:
    0 -> Perfil benigno / sano                (clase mayoritaria, ~97%)
    1 -> Perfil límite / artefacto de sensor   (clase minoritaria intermedia, ~2%)
    2 -> Perfil patológico (variante oncológica rara, clase de interés clínico, ~1%)

El generador sklearn.datasets.make_classification produce un espacio de
características x in R^d con estructura gaussiana por clúster, lo cual es
consistente con el supuesto de verosimilitud normal multivariada que se
utiliza para derivar el clasificador de mínima distancia (D-min) en la
Sección 2.3 del reporte.

Anclaje semántico (ilustrativo, no una simulación física)
-----------------------------------------------------------
Cada una de las d=12 dimensiones representa un canal de relación
masa-carga (m/z): la intensidad de un fragmento peptídico o proteico
candidato a biomarcador. De ellas, 8 se declaran informativas
(biomarcadores efectivamente asociados a la variante oncológica) y 2
redundantes (combinaciones lineales de las anteriores, análogas a
fragmentos correlacionados por vías bioquímicas compartidas). El 1% de
ruido de etiquetado (flip_y) no es un artificio numérico arbitrario:
representa errores inevitables del mundo real -un falso negativo/positivo
en la biopsia de referencia usada para etiquetar la muestra, o una
descalibración ocasional del espectrómetro- que ningún pipeline de
cómputo puede corregir después del hecho.

Esta correspondencia es ilustrativa, NO una simulación física de un
espectrómetro real: make_classification no modela la química analítica
subyacente. Las cifras de este proyecto deben leerse como una validación
controlada del flujo de Reconocimiento de Patrones sobre un problema con
la misma estructura cualitativa del caso clínico real (alta
dimensionalidad, traslape de frontera, desbalance extremo), no como
estimaciones clínicas.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification

# Nombres descriptivos de las clases, usados en tablas y gráficas.
CLASS_NAMES = {
    0: "Benigno",
    1: "Límite / artefacto",
    2: "Patológico (raro)",
}


def generate_biomedical_dataset(
    n_samples: int = 3000,
    n_features: int = 12,
    n_informative: int = 8,
    n_redundant: int = 2,
    weights: tuple[float, float, float] = (0.97, 0.02, 0.01),
    class_sep: float = 1.15,
    flip_y: float = 0.01,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simula el dataset biomédico de alta asimetría.

    Parameters
    ----------
    n_samples : int
        Número total de muestras (pacientes) simuladas.
    n_features : int
        Dimensionalidad d del espacio de características x in R^d. En la
        analogía clínica, cada dimensión es un canal m/z (masa-carga) de
        espectrometría de masas.
    n_informative : int
        Número de características realmente discriminativas (canales m/z
        que corresponden a biomarcadores efectivamente asociados a la
        variante oncológica).
    n_redundant : int
        Combinaciones lineales de las informativas (ruido geométrico
        correlacionado, análogo a fragmentos peptídicos correlacionados
        por vías bioquímicas compartidas o artefactos instrumentales).
    weights : tuple[float, float, float]
        Proporción de cada clase. La clase patológica (índice 2) se fija
        por debajo del 1% del total, tal como exige el enunciado.
    class_sep : float
        Separación entre los clústeres gaussianos de cada clase; controla
        el traslape de fronteras de decisión.
    flip_y : float
        Fracción de etiquetas invertidas aleatoriamente. Representa
        errores inevitables del mundo real (un falso negativo/positivo en
        la biopsia de referencia usada para etiquetar la muestra, o una
        descalibración ocasional del espectrómetro), no ruido numérico
        arbitrario.
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        weights=list(weights),
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state,
    )
    return X, y


def class_distribution(y: np.ndarray) -> dict[int, tuple[int, float]]:
    """Regresa {clase: (conteo, proporción)} para reportar el desbalance."""
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    return {int(c): (int(n), float(n) / total) for c, n in zip(classes, counts)}
