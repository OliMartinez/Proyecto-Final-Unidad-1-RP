"""
main.py
=======
Orquesta el flujo completo:

    1. Generación del dataset biomédico sintético (>=3 clases, <1% clase
       patológica) y visualización fiel de su geometría (histogramas de
       distancia intra/interclase, sin proyección lineal).
    2. Evaluación geométrica del espacio de características con 6
       métricas de distancia, incluyendo el diagnóstico cuantitativo de
       "secuestro" de la covarianza combinada por la clase mayoritaria
       (vulnerabilidad de Mahalanobis).
    3. Higiene de la frontera con Tomek Links (antes/después, vía UMAP).
    4. Clasificador D-min (mínima distancia a centroide), comparado bajo
       las 6 métricas.
    5. Evaluación ante el desbalance: matriz de confusión, Sensibilidad,
       Especificidad, Precisión, Exactitud, curvas ROC/AUC.
    6. Integración del costo asimétrico de los falsos negativos: priors
       inflados por cociente de costo y movimiento de umbral sobre ROC.
    7. Comparación de esquemas de validación: Holdout, Stratified K-Fold
       y Bootstrap.

Todas las figuras se guardan en outputs/figures/ y las tablas en
outputs/tables/ (CSV).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import generate_biomedical_dataset, class_distribution, CLASS_NAMES
from src.geometry import (
    evaluate_all_metrics, METRICS, per_sample_intra_inter_distances,
    class_covariance_contributions,
)
from src.tomek import apply_tomek_links, removed_summary
from src.dmin import MinimumDistanceClassifier
from src.evaluation import (
    confusion_matrix_multiclass, per_class_rates, roc_auc_one_vs_rest,
    evaluate_holdout, evaluate_stratified_kfold, evaluate_bootstrap,
    evaluate_cost_sensitive_sweep, evaluate_threshold_moving,
)
from src.visualize import (
    plot_intra_inter_histogram, plot_tomek_before_after_umap, plot_confusion_matrix,
    plot_roc_curves, plot_metric_comparison, plot_validation_scheme_comparison,
    plot_cost_sensitivity_sweep,
)

FIG_DIR = "outputs/figures"
TAB_DIR = "outputs/tables"


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    rng_seed = 42

    # ------------------------------------------------------------------
    # 1. Generación del dataset
    # ------------------------------------------------------------------
    section("1. Generación del dataset biomédico sintético")
    X, y = generate_biomedical_dataset(random_state=rng_seed)
    dist = class_distribution(y)
    for c, (n, p) in dist.items():
        print(f"  Clase {c} ({CLASS_NAMES[c]:<20s}): n={n:5d}  ({p*100:5.2f}%)")

    pd.DataFrame(
        [{"clase": c, "nombre": CLASS_NAMES[c], "n": n, "proporcion": p} for c, (n, p) in dist.items()]
    ).to_csv(f"{TAB_DIR}/01_distribucion_clases.csv", index=False)

    # Visualización fiel de la geometría real (sin proyección lineal PCA):
    # distancia intraclase vs. interclase por muestra, en las d dimensiones
    # originales.
    intra_all, inter_all = per_sample_intra_inter_distances(X, y, metric="euclidean")
    plot_intra_inter_histogram(
        intra_all, inter_all, y, CLASS_NAMES,
        save_path=f"{FIG_DIR}/01_intra_inter_histograma.png",
    )

    # ------------------------------------------------------------------
    # 2. Evaluación geométrica: 6 métricas, dispersión intra/interclase
    # ------------------------------------------------------------------
    section("2. Evaluación geométrica del espacio de características")
    geo_results, Sw, VI = evaluate_all_metrics(X, y)

    rows = []
    for metric in METRICS:
        r = geo_results[metric]
        rows.append({
            "metrica": metric,
            "dispersion_intraclase_media": r["mean_intraclass"],
            "separacion_interclase_min": r["min_interclass"],
            "razon_separabilidad_J": r["separability_ratio"],
        })
        print(f"  {metric:12s}  intra={r['mean_intraclass']:.3f}  "
              f"inter_min={r['min_interclass']:.3f}  J={r['separability_ratio']:.3f}")
    geo_df = pd.DataFrame(rows)
    geo_df.to_csv(f"{TAB_DIR}/02_metricas_geometricas.csv", index=False)

    plot_metric_comparison(
        geo_df["metrica"], geo_df["razon_separabilidad_J"],
        ylabel="Razón de separabilidad J = inter_min / intra_media",
        title="Separabilidad geométrica por métrica de distancia",
        save_path=f"{FIG_DIR}/02_separabilidad_por_metrica.png",
        highlight="mahalanobis",
    )

    # -- Advertencia de Mahalanobis: ¿qué clase "secuestra" Sigma_W? --
    contrib = class_covariance_contributions(X, y)
    print("  Contribución de cada clase a la traza de Sigma_W (covarianza de Mahalanobis):")
    for c, frac in contrib.items():
        print(f"    Clase {c} ({CLASS_NAMES[c]:<20s}): {frac*100:5.2f}% de Sigma_W "
              f"(participación muestral: {dist[c][1]*100:5.2f}%)")
    pd.DataFrame([
        {"clase": c, "nombre": CLASS_NAMES[c],
         "contribucion_covarianza_pct": frac * 100,
         "participacion_muestral_pct": dist[c][1] * 100}
        for c, frac in contrib.items()
    ]).to_csv(f"{TAB_DIR}/02b_contribucion_covarianza.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Tomek Links: limpieza de frontera
    # ------------------------------------------------------------------
    section("3. Higiene de la frontera (Tomek Links)")
    X_clean, y_clean, kept_idx = apply_tomek_links(X, y)
    removed = removed_summary(y, kept_idx)
    print(f"  Muestras originales: {len(y)}   Muestras tras limpieza: {len(y_clean)}")
    print(f"  Eliminadas por clase: {removed}")

    pd.DataFrame([
        {"clase": c, "nombre": CLASS_NAMES[c], "eliminadas": removed.get(c, 0)}
        for c in CLASS_NAMES
    ]).to_csv(f"{TAB_DIR}/03_tomek_eliminadas.csv", index=False)

    plot_tomek_before_after_umap(
        X, y, kept_idx, CLASS_NAMES,
        save_path=f"{FIG_DIR}/03_tomek_antes_despues.png",
    )

    # ------------------------------------------------------------------
    # 4. Clasificador D-min bajo las 6 métricas (sobre datos limpios)
    # ------------------------------------------------------------------
    section("4. Clasificador D-min bajo las 6 métricas de distancia")
    dmin_rows = []
    for metric in METRICS:
        rates, cm, classes = evaluate_holdout(X_clean, y_clean, metric=metric, random_state=rng_seed)
        dmin_rows.append({
            "metrica": metric,
            "exactitud_global": rates["global_accuracy"],
            "sensibilidad_patologica": rates["sensitivity"][-1],
            "especificidad_patologica": rates["specificity"][-1],
            "precision_patologica": rates["precision"][-1],
        })
        print(f"  {metric:12s}  exactitud={rates['global_accuracy']:.3f}  "
              f"sens_patologica={rates['sensitivity'][-1]:.3f}")
    dmin_df = pd.DataFrame(dmin_rows)
    dmin_df.to_csv(f"{TAB_DIR}/04_dmin_por_metrica.csv", index=False)

    plot_metric_comparison(
        dmin_df["metrica"], dmin_df["exactitud_global"],
        ylabel="Exactitud global (holdout)",
        title="Desempeño del clasificador D-min por métrica",
        save_path=f"{FIG_DIR}/04_dmin_exactitud_por_metrica.png",
        highlight="euclidean",
    )

    # ------------------------------------------------------------------
    # 5. Evaluación detallada con métrica Euclidiana (D-min "canónico")
    # ------------------------------------------------------------------
    section("5. Evaluación ante el desbalance (métrica Euclidiana)")
    rates_e, cm_e, classes_e = evaluate_holdout(X_clean, y_clean, metric="euclidean", random_state=rng_seed)
    print(f"  Exactitud global: {rates_e['global_accuracy']:.3f}")
    for i, c in enumerate(classes_e):
        print(f"  Clase {c} ({CLASS_NAMES[c]:<20s}): Sens={rates_e['sensitivity'][i]:.3f}  "
              f"Espec={rates_e['specificity'][i]:.3f}  Prec={rates_e['precision'][i]:.3f}")

    plot_confusion_matrix(
        cm_e, CLASS_NAMES, classes_e,
        "Matriz de confusión -- D-min (Euclidiana, holdout)",
        save_path=f"{FIG_DIR}/05_matriz_confusion_holdout.png",
    )

    pd.DataFrame({
        "clase": classes_e, "nombre": [CLASS_NAMES[c] for c in classes_e],
        "sensibilidad": rates_e["sensitivity"], "especificidad": rates_e["specificity"],
        "precision": rates_e["precision"], "exactitud_por_clase": rates_e["accuracy_per_class"],
    }).to_csv(f"{TAB_DIR}/05_metricas_holdout.csv", index=False)

    # ROC / AUC one-vs-rest (el clasificador base, sin costo asimétrico, se
    # reutiliza en la Sección 6 para el movimiento de umbral)
    X_tr, X_te, y_tr, y_te = _holdout_split(X_clean, y_clean, rng_seed)
    clf = MinimumDistanceClassifier(metric="euclidean").fit(X_tr, y_tr)
    scores = clf.predict_scores(X_te)
    roc_dict = roc_auc_one_vs_rest(y_te, scores, clf.classes_)
    for c, d in roc_dict.items():
        print(f"  AUC clase {c} ({CLASS_NAMES[c]}): {d['auc']:.3f}")
    pd.DataFrame([{"clase": c, "nombre": CLASS_NAMES[c], "auc": d["auc"]} for c, d in roc_dict.items()]) \
        .to_csv(f"{TAB_DIR}/06_auc_por_clase.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Costo asimétrico de los falsos negativos (Sección 2.5)
    # ------------------------------------------------------------------
    section("6. Integración del costo asimétrico (priors inflados + movimiento de umbral)")
    pathological_class = int(classes_e[-1])

    cost_ratios = [1, 2, 5, 10, 20, 50, 100]
    sweep_rows = evaluate_cost_sensitive_sweep(
        X_clean, y_clean, cost_ratios, pathological_class, metric="euclidean", random_state=rng_seed
    )
    for r in sweep_rows:
        print(f"  C={r['cost_ratio']:>4}   sens_pat={r['sensibilidad_patologica']:.3f}  "
              f"espec_pat={r['especificidad_patologica']:.3f}  prec_pat={r['precision_patologica']:.3f}  "
              f"exactitud={r['exactitud_global']:.3f}")
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(f"{TAB_DIR}/07_costo_asimetrico_sweep.csv", index=False)

    plot_cost_sensitivity_sweep(
        sweep_df["cost_ratio"], sweep_df["sensibilidad_patologica"],
        sweep_df["especificidad_patologica"], sweep_df["precision_patologica"],
        save_path=f"{FIG_DIR}/07_costo_asimetrico_sweep.png",
    )

    threshold_rows, operating_points, _ = evaluate_threshold_moving(
        X_clean, y_clean, pathological_class, metric="euclidean",
        target_sensitivities=(0.70, 0.80, 0.90, 0.95), random_state=rng_seed,
    )
    for r in threshold_rows:
        print(f"  objetivo={r['sensibilidad_objetivo']:.0%}  umbral={r['umbral']:.4f}  "
              f"sens_lograda={r['sensibilidad_lograda']:.3f}  espec_lograda={r['especificidad_lograda']:.3f}  "
              f"prec_lograda={r['precision_lograda']:.3f}")
    pd.DataFrame(threshold_rows).to_csv(f"{TAB_DIR}/08_movimiento_umbral.csv", index=False)

    # Figura final de ROC/AUC, enriquecida con los puntos de operación del
    # movimiento de umbral sobre la clase patológica.
    plot_roc_curves(
        roc_dict, CLASS_NAMES, "Curvas ROC uno-contra-resto -- D-min",
        f"{FIG_DIR}/06_roc_auc.png",
        operating_points=operating_points, operating_class=pathological_class,
    )

    # ------------------------------------------------------------------
    # 7. Comparación de esquemas de validación
    # ------------------------------------------------------------------
    section("7. Comparación de esquemas de validación")
    holdout_rates, _, holdout_classes = evaluate_holdout(X_clean, y_clean, random_state=rng_seed)
    skf_result = evaluate_stratified_kfold(X_clean, y_clean, n_splits=5, random_state=rng_seed)
    boot_result = evaluate_bootstrap(X_clean, y_clean, n_iterations=200, random_state=rng_seed)

    pathological_idx = -1  # última clase = patológica
    holdout_sens = holdout_rates["sensitivity"][pathological_idx]
    skf_mean = skf_result["sensitivity_mean"][pathological_idx]
    skf_std = skf_result["sensitivity_std"][pathological_idx]
    boot_mean = boot_result["sensitivity_mean"][pathological_idx]
    boot_std = boot_result["sensitivity_std"][pathological_idx]

    print(f"  Holdout          -> Sensibilidad patológica = {holdout_sens:.3f} (una sola partición, sin std)")
    print(f"  Stratified 5-Fold -> Sensibilidad patológica = {skf_mean:.3f} +/- {skf_std:.3f}")
    print(f"  Bootstrap (200)   -> Sensibilidad patológica = {boot_mean:.3f} +/- {boot_std:.3f} "
          f"(n_validas={boot_result['n_valid_iterations']})")

    pd.DataFrame([
        {"esquema": "Holdout", "sens_media": holdout_sens, "sens_std": 0.0},
        {"esquema": "Stratified 5-Fold", "sens_media": skf_mean, "sens_std": skf_std},
        {"esquema": "Bootstrap (200)", "sens_media": boot_mean, "sens_std": boot_std},
    ]).to_csv(f"{TAB_DIR}/09_validacion_esquemas.csv", index=False)

    plot_validation_scheme_comparison(
        ["Holdout", "Stratified\n5-Fold", "Bootstrap\n(200)"],
        [holdout_sens, skf_mean, boot_mean],
        [0.0, skf_std, boot_std],
        save_path=f"{FIG_DIR}/09_comparacion_validacion.png",
    )

    section("Pipeline completo. Resultados en outputs/figures y outputs/tables")


def _holdout_split(X, y, seed):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


if __name__ == "__main__":
    main()
