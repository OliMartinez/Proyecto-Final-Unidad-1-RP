"""
evaluation.py
-------------
Evaluación matemática ante la asimetría de clases e
integración del costo asimétrico de los falsos negativos.

Contiene:
    - Matriz de confusión multiclase (vectorizada, numpy puro).
    - Sensibilidad, Especificidad, Precisión y Exactitud por clase
      (esquema uno-contra-resto).
    - Curvas ROC y AUC uno-contra-resto.
    - Los tres esquemas de validación comparados en el reporte:
      Holdout simple, Stratified K-Fold y Bootstrap.
    - Costo asimétrico: barrido de priors inflados por cociente de costo
      (cost_sensitive_priors / evaluate_cost_sensitive_sweep) y movimiento
      de umbral sobre la curva ROC (evaluate_threshold_moving) -- dos
      implementaciones de la misma regla de riesgo mínimo de Bayes.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample

from .dmin import MinimumDistanceClassifier


def confusion_matrix_multiclass(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Matriz de confusión K x K calculada con un solo `np.add.at`
    vectorizado (sin bucle por muestra)."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    true_idx = np.array([class_to_idx[c] for c in y_true])
    pred_idx = np.array([class_to_idx[c] for c in y_pred])
    K = len(classes)
    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (true_idx, pred_idx), 1)
    return cm


def per_class_rates(cm: np.ndarray) -> dict:
    """Deriva Sensibilidad, Especificidad, Precisión y Exactitud por clase
    a partir de la matriz de confusión, colapsando cada clase k a un
    problema binario uno-contra-resto:

        TP_k = cm[k, k]
        FN_k = sum_j cm[k, j] - TP_k                (fila k, fuera de la diagonal)
        FP_k = sum_i cm[i, k] - TP_k                (columna k, fuera de la diagonal)
        TN_k = suma total - TP_k - FN_k - FP_k

        Sensibilidad_k = TP_k / (TP_k + FN_k)
        Especificidad_k = TN_k / (TN_k + FP_k)
        Precision_k     = TP_k / (TP_k + FP_k)
        Exactitud_k     = (TP_k + TN_k) / total
    """
    K = cm.shape[0]
    total = cm.sum()
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)

    tp = np.diag(cm).astype(float)
    fn = row_sums - tp
    fp = col_sums - tp
    tn = total - tp - fn - fp

    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        specificity = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)
        precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        accuracy = (tp + tn) / total

    global_accuracy = tp.sum() / total  # exactitud global multiclase

    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "accuracy_per_class": accuracy,
        "global_accuracy": float(global_accuracy),
    }


def roc_auc_one_vs_rest(y_true: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> dict:
    """Curvas ROC y AUC uno-contra-resto para cada clase."""
    out = {}
    for i, c in enumerate(classes):
        y_bin = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, scores[:, i])
        out[int(c)] = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
    return out


# --------------------------------------------------------------------------
# Esquemas de validación estadística
# --------------------------------------------------------------------------

def evaluate_holdout(X, y, metric="euclidean", test_size=0.3, random_state=42):
    """Holdout simple estratificado (una sola partición train/test)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    clf = MinimumDistanceClassifier(metric=metric).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    cm = confusion_matrix_multiclass(y_te, y_pred, clf.classes_)
    return per_class_rates(cm), cm, clf.classes_


def evaluate_stratified_kfold(X, y, metric="euclidean", n_splits=5, random_state=42):
    """Stratified K-Fold: preserva la proporción de clases (incluida la
    minoritaria) en cada partición. Regresa métricas agregadas (media y
    desviación estándar) y la matriz de confusión acumulada."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classes = np.unique(y)
    cm_total = np.zeros((len(classes), len(classes)), dtype=int)
    fold_sensitivities = []

    for train_idx, test_idx in skf.split(X, y):
        clf = MinimumDistanceClassifier(metric=metric).fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        cm = confusion_matrix_multiclass(y[test_idx], y_pred, classes)
        cm_total += cm
        fold_sensitivities.append(per_class_rates(cm)["sensitivity"])

    fold_sensitivities = np.array(fold_sensitivities)  # (n_splits, K)
    rates_total = per_class_rates(cm_total)
    return {
        "cm_total": cm_total,
        "rates_total": rates_total,
        "sensitivity_mean": fold_sensitivities.mean(axis=0),
        "sensitivity_std": fold_sensitivities.std(axis=0),
        "classes": classes,
    }


def evaluate_bootstrap(X, y, metric="euclidean", n_iterations=200, random_state=42):
    """Bootstrap (muestreo con reemplazo): entrena con una muestra
    bootstrap del tamaño del dataset y evalúa en las muestras
    "out-of-bag" (no seleccionadas), repitiendo n_iterations veces."""
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    n = len(y)
    sensitivities = []

    for i in range(n_iterations):
        boot_idx = resample(
            np.arange(n), replace=True, n_samples=n, random_state=rng.randint(0, 1_000_000)
        )
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(boot_idx)] = False
        oob_idx = np.where(oob_mask)[0]
        if len(oob_idx) == 0 or len(np.unique(y[boot_idx])) < len(classes):
            continue
        clf = MinimumDistanceClassifier(metric=metric).fit(X[boot_idx], y[boot_idx])
        y_pred = clf.predict(X[oob_idx])
        cm = confusion_matrix_multiclass(y[oob_idx], y_pred, classes)
        sensitivities.append(per_class_rates(cm)["sensitivity"])

    sensitivities = np.array(sensitivities)
    return {
        "sensitivity_mean": sensitivities.mean(axis=0),
        "sensitivity_std": sensitivities.std(axis=0),
        "n_valid_iterations": sensitivities.shape[0],
        "classes": classes,
    }


# --------------------------------------------------------------------------
# Integración del costo asimétrico: priors inflados por costo
# y movimiento de umbral sobre la curva ROC. Ambos mecanismos implementan
# la MISMA regla de riesgo mínimo de Bayes (ver derivación en dmin.py);
# se ofrecen los dos porque operan en escenarios distintos: el primero
# ajusta la regla D-min multiclase en sí; el segundo re-calibra un umbral
# sobre puntuaciones ya entrenadas, en el encuadre binario patológica-vs-
# resto habitual en la literatura de cribado clínico.
# --------------------------------------------------------------------------

def cost_sensitive_priors(classes, pathological_class, cost_ratio, base_priors=None):
    """Construye el vector de priors efectivos que implementa la regla de
    riesgo mínimo de Bayes para un cociente de costos
    C = costo(falso negativo) / costo(falso positivo):

        pi_patologica_efectivo  proporcional a  C * pi_patologica_base
        pi_k_efectivo           proporcional a  pi_k_base   para k != patológica

    (ver derivación completa -- equivalencia con el movimiento de umbral de
    razón de verosimilitud -- en la Sección 2.5 del reporte). Con
    ``base_priors=None`` (uniforme), C=1 reproduce exactamente el D-min
    clásico ("priors iguales").
    """
    classes = list(classes)
    if base_priors is None:
        base = np.ones(len(classes)) / len(classes)
    else:
        base = np.array([base_priors[c] for c in classes], dtype=float)
        base = base / base.sum()
    weights = base.copy()
    idx = classes.index(pathological_class)
    weights[idx] = weights[idx] * cost_ratio
    return weights / weights.sum()


def evaluate_cost_sensitive_sweep(
    X, y, cost_ratios, pathological_class, metric="euclidean", test_size=0.3, random_state=42
):
    """Barrido sobre cocientes de costo C_FN/C_FP: para cada C, entrena un
    D-min con priors inflados (cost_sensitive_priors) y evalúa Sensibilidad/
    Especificidad/Precisión/Exactitud de la clase patológica sobre el mismo
    holdout.

    La base de referencia (C=1) usa priors UNIFORMES, no las frecuencias
    empíricas de clase: es la misma condición ("priors iguales") bajo la
    cual se derivó el D-min clásico en la Sección 2.3, y por lo tanto C=1
    reproduce exactamente los resultados de la Sección IV-C (Sensibilidad
    patológica = 0.556 con métrica Euclidiana). Usar frecuencias empíricas
    como base ya sería, en sí mismo, una forma de ponderación por costo -- y
    rompería esa comparación directa con la línea base ya reportada."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    classes = np.unique(y)
    idx_path = list(classes).index(pathological_class)

    rows = []
    for ratio in cost_ratios:
        priors = cost_sensitive_priors(classes.tolist(), pathological_class, ratio, base_priors=None)
        clf = MinimumDistanceClassifier(metric=metric, priors=priors).fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        cm = confusion_matrix_multiclass(y_te, y_pred, classes)
        rates = per_class_rates(cm)
        rows.append({
            "cost_ratio": ratio,
            "sensibilidad_patologica": float(rates["sensitivity"][idx_path]),
            "especificidad_patologica": float(rates["specificity"][idx_path]),
            "precision_patologica": float(rates["precision"][idx_path]),
            "exactitud_global": rates["global_accuracy"],
        })
    return rows


def find_roc_threshold_for_sensitivity(y_true_bin, scores, target_sensitivity):
    """Busca, sobre la curva ROC empírica, el umbral que alcanza al menos
    ``target_sensitivity`` con la menor Tasa de Falsos Positivos posible
    (el punto de la curva ROC más "barato" que satisface el requisito
    clínico mínimo de Sensibilidad)."""
    fpr, tpr, thresholds = roc_curve(y_true_bin, scores)
    valid = np.where(tpr >= target_sensitivity)[0]
    idx = valid[np.argmin(fpr[valid])] if len(valid) else int(np.argmax(tpr))
    return float(thresholds[idx]), float(fpr[idx]), float(tpr[idx])


def evaluate_threshold_moving(
    X, y, pathological_class, metric="euclidean",
    target_sensitivities=(0.70, 0.80, 0.90, 0.95),
    test_size=0.3, random_state=42,
):
    """Movimiento de umbral sobre la curva ROC, en el encuadre binario
    patológica-vs-resto: entrena un D-min clásico (priors uniformes/
    empíricos) una sola vez, y para cada Sensibilidad objetivo mueve el
    umbral de decisión sobre las pseudo-probabilidades ya calculadas,
    sin reentrenar el clasificador."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    clf = MinimumDistanceClassifier(metric=metric).fit(X_tr, y_tr)
    scores = clf.predict_scores(X_te)
    idx_path = list(clf.classes_).index(pathological_class)
    y_bin = (y_te == pathological_class).astype(int)
    score_path = scores[:, idx_path]

    rows = []
    operating_points = []
    for target in target_sensitivities:
        t, fpr_t, tpr_t = find_roc_threshold_for_sensitivity(y_bin, score_path, target)
        y_pred_bin = (score_path >= t).astype(int)
        tp = int(np.sum((y_pred_bin == 1) & (y_bin == 1)))
        fn = int(np.sum((y_pred_bin == 0) & (y_bin == 1)))
        fp = int(np.sum((y_pred_bin == 1) & (y_bin == 0)))
        tn = int(np.sum((y_pred_bin == 0) & (y_bin == 0)))
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rows.append({
            "sensibilidad_objetivo": target,
            "umbral": t,
            "sensibilidad_lograda": sens,
            "especificidad_lograda": spec,
            "precision_lograda": prec,
        })
        operating_points.append((fpr_t, tpr_t, target))
    return rows, operating_points, (y_bin, score_path)
