# ABP -- Optimización de Fronteras y Evaluación en Espacios Geométricos de Alta Asimetría

Curso: Reconocimiento de Patrones -- TecNM, Campus León.

Implementación vectorizada (NumPy / SciPy) del flujo completo de
Reconocimiento de Patrones para un problema de cribado oncológico
sintético con desbalance severo de clases (clase patológica < 1.5 %),
incluyendo integración explícita del costo asimétrico de los falsos
negativos y visualización no lineal (UMAP) del espacio de características.

## Estructura

```
project/
├── main.py                  # orquesta el pipeline completo (7 secciones)
├── requirements.txt
├── src/
│   ├── data.py               # generación del dataset sintético (make_classification)
│   ├── geometry.py           # 6 métricas de distancia + separabilidad + diagnóstico de covarianza
│   ├── tomek.py               # limpieza de frontera con Tomek Links (imblearn)
│   ├── dmin.py                # clasificador D-min + priors ajustados por costo (riesgo mínimo de Bayes)
│   ├── evaluation.py          # matriz de confusión, métricas, Holdout / K-Fold / Bootstrap, costo asimétrico
│   └── visualize.py           # gráficas: histogramas intra/inter, UMAP, paleta institucional TecNM
└── outputs/
    ├── figures/               # PNG generados por main.py
    └── tables/                # CSV generados por main.py
```

## Ejecución

```bash
pip install -r requirements.txt
python main.py
```

El script imprime un resumen en consola y escribe todas las figuras y
tablas usadas en el reporte IEEE y en la presentación de defensa dentro
de `outputs/`.

## Principios de diseño

- **Vectorización estricta**: todas las distancias (intraclase,
  interclase, y la clasificación D-min) se calculan con
  `scipy.spatial.distance.cdist`; no existe ningún bucle `for` sobre
  pares de muestras.
- **6 métricas analíticas**: Euclidiana, Manhattan (cityblock),
  Mahalanobis, Chebyshev, Coseno y Minkowski (p=3).
- **D-min como caso particular de Bayes**: `src/dmin.py` documenta en su
  docstring la derivación que colapsa la regla MAP Gaussiana a un
  clasificador de mínima distancia a centroide, tanto para covarianza
  isotrópica (Euclidiana) como para covarianza compartida general
  (Mahalanobis) -- incluyendo la advertencia de que Σ puede quedar
  "secuestrada" por la clase mayoritaria bajo desbalance extremo
  (cuantificado en `geometry.class_covariance_contributions`).
- **Costo asimétrico de los falsos negativos**: `dmin.py` extiende la
  regla D-min con priors ajustados por cociente de costo (Ec. de riesgo
  mínimo de Bayes); `evaluation.py` implementa además el movimiento de
  umbral sobre la curva ROC como mecanismo equivalente, sin reentrenar.
- **Visualización fiel de la geometría**: en vez de una proyección PCA
  lineal, se usan histogramas de distancia intra/interclase por muestra
  (en las d dimensiones originales) y proyecciones UMAP -- coherentes
  con la naturaleza de vecino-más-cercano de Tomek Links.
- **Tres esquemas de validación** comparados explícitamente: Holdout
  simple, Stratified K-Fold (k=5) y Bootstrap (200 iteraciones,
  evaluación out-of-bag), reportando la estabilidad (media ± desviación
  estándar) de la Sensibilidad sobre la clase patológica.

## Referencias

- Duda, R. O., Hart, P. E., & Stork, D. G. (2001). *Pattern Classification* (2nd ed.). Wiley.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Vapnik, V. N. (1995). *The Nature of Statistical Learning Theory*. Springer.
- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. arXiv:1802.03426.
- Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning*. IJCAI 2001.
