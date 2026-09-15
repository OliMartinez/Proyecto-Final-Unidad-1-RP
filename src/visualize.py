"""
visualize.py
------------
Funciones de graficación. La paleta de colores sigue el Manual de
Identidad Gráfica del TecNM:
    Pantone 294 C   -> #1B396A  (azul institucional, texto/curvas primarias)
    Cool Gray 10 C  -> #807E82  (gris institucional, elementos secundarios)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import umap

TECNM_BLUE = "#1B396A"
TECNM_GRAY = "#807E82"
ACCENT = "#B08D57"  # dorado sutil, evocando el listón institucional
PALETTE = [TECNM_BLUE, ACCENT, TECNM_GRAY]

# Noto Sans es la tipografía obligatoria del MIG-TecNM 2025 para cuerpos de
# texto y documentación oficial (Sec. 1.3 del manual). Se registra
# explícitamente para que todas las figuras del reporte y las diapositivas
# compartan tipografía con el documento LaTeX y el PPTX.
for _weight in ("Regular", "Bold", "Italic", "BoldItalic"):
    try:
        fm.fontManager.addfont(f"/usr/share/fonts/truetype/noto/NotoSans-{_weight}.ttf")
    except Exception:
        pass

plt.rcParams.update({
    "font.family": "Noto Sans",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "axes.titleweight": "bold",
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def compute_umap_embedding(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.25, random_state: int = 42):
    """Proyección UMAP 2D (no lineal, basada en el grafo de vecinos más
    cercanos). Se prefiere sobre PCA para visualizar Tomek Links porque
    ambos algoritmos comparten el mismo principio geométrico -- la
    vecindad más cercana -- por lo que la proyección refleja más
    fielmente qué pares fueron eliminados y por qué."""
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                         random_state=random_state)
    return reducer.fit_transform(X)


def _scatter_embedding(ax, X2, y, class_names, title):
    classes = np.unique(y)
    for i, c in enumerate(classes):
        mask = y == c
        ax.scatter(
            X2[mask, 0], X2[mask, 1],
            s=14 if c == classes[0] else 26,
            alpha=0.55 if c == classes[0] else 0.9,
            c=PALETTE[i % len(PALETTE)],
            label=f"{class_names.get(c, c)} (n={mask.sum()})",
            edgecolors="none",
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, frameon=False, loc="best")
    return ax


def plot_tomek_before_after_umap(X, y, kept_idx, class_names, save_path):
    """Proyección UMAP ajustada UNA sola vez sobre el conjunto original
    (X, y); el panel "después" reutiliza las MISMAS coordenadas 2D
    restringidas a las muestras conservadas (kept_idx), en vez de reajustar
    UMAP por separado -- evita el artefacto de comparar dos embeddings no
    lineales independientes (y por tanto no directamente comparables)."""
    X2 = compute_umap_embedding(X)
    kept_mask = np.zeros(len(y), dtype=bool)
    kept_mask[kept_idx] = True

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    _scatter_embedding(axes[0], X2, y, class_names, "Antes de Tomek Links (UMAP)")
    _scatter_embedding(axes[1], X2[kept_mask], y[kept_mask], class_names, "Después de Tomek Links (UMAP)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_intra_inter_histogram(intra, inter, y, class_names, save_path):
    """Histogramas de densidad de distancia intraclase (a mu propio) vs.
    distancia interclase (al centroide ajeno más cercano), un panel por
    clase. A diferencia de una proyección 2D, no distorsiona ni descarta
    información: usa las distancias reales en las d dimensiones originales
    (Ec. 1-2 del reporte), por lo que es la visualización fiel de la
    separabilidad geométrica en un espacio de alta dimensión con clases
    muy desbalanceadas."""
    classes = np.unique(y)
    fig, axes = plt.subplots(1, len(classes), figsize=(4.1 * len(classes), 4.0), sharey=False)
    if len(classes) == 1:
        axes = [axes]
    for ax, c in zip(axes, classes):
        mask = y == c
        bins = min(30, max(8, mask.sum() // 2))
        ax.hist(intra[mask], bins=bins, alpha=0.65, color=TECNM_BLUE, density=True,
                label=f"Intraclase (a $\\mu_{{{class_names.get(c, c)}}}$)")
        ax.hist(inter[mask], bins=bins, alpha=0.55, color=ACCENT, density=True,
                label="Interclase (al más cercano ajeno)")
        ax.set_title(f"{class_names.get(c, c)}  (n={mask.sum()})", fontsize=11)
        ax.set_xlabel("Distancia Euclidiana")
        if ax is axes[0]:
            ax.set_ylabel("Densidad")
        ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle("Distancia intraclase vs. interclase por muestra", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, classes, title, save_path):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=11)
    labels = [class_names.get(c, str(c)) for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(roc_dict, class_names, title, save_path, operating_points=None, operating_class=None):
    """Curvas ROC uno-contra-resto. Si se proveen ``operating_points``
    (lista de (fpr, tpr, etiqueta) del movimiento de umbral, Sección 2.5)
    para la clase ``operating_class``, se marcan sobre su curva. Puntos
    que coinciden (misma fpr/tpr, frecuente con muestras de prueba muy
    escasas) se fusionan en una sola etiqueta para evitar texto encimado."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for i, (c, d) in enumerate(roc_dict.items()):
        ax.plot(
            d["fpr"], d["tpr"],
            color=PALETTE[i % len(PALETTE)], lw=2,
            label=f"{class_names.get(c, c)} (AUC={d['auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1, label="Azar (AUC=0.500)")

    if operating_points:
        merged: dict[tuple, list] = {}
        for fpr_p, tpr_p, lab in operating_points:
            key = (round(fpr_p, 4), round(tpr_p, 4))
            merged.setdefault(key, []).append(lab)
        pts_x = [k[0] for k in merged]
        pts_y = [k[1] for k in merged]
        ax.scatter(pts_x, pts_y, color=ACCENT, edgecolor="#222222", s=55, zorder=5,
                   label="Umbrales movidos (Sec. 2.5)")
        for (fpr_p, tpr_p), labs in merged.items():
            text = "/".join(f"{lab:.0%}" for lab in labs)
            ax.annotate(text, (fpr_p, tpr_p), textcoords="offset points",
                        xytext=(8, -4), fontsize=8, color="#222222")

    ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
    ax.set_title(title, fontsize=12.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_metric_comparison(metric_names, values, ylabel, title, save_path, highlight=None):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    colors = [ACCENT if m == highlight else TECNM_GRAY for m in metric_names]
    bars = ax.bar(metric_names, values, color=colors, edgecolor=TECNM_BLUE, linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=25, ha="right", fontsize=9)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_validation_scheme_comparison(scheme_names, minority_sensitivity_means, minority_sensitivity_stds, save_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    x = np.arange(len(scheme_names))
    ax.bar(x, minority_sensitivity_means, yerr=minority_sensitivity_stds, capsize=5,
           color=[TECNM_BLUE, ACCENT, TECNM_GRAY], edgecolor="#222222", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(scheme_names, fontsize=10)
    ax.set_ylabel("Sensibilidad clase patológica")
    ax.set_title("Estabilidad de la Sensibilidad según esquema de validación", fontsize=11)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_cost_sensitivity_sweep(cost_ratios, sensitivity, specificity, precision, save_path):
    """Barrido de Sensibilidad/Especificidad/Precisión de la clase
    patológica en función del cociente de costo C_FN/C_FP (Sección 2.5).
    Eje x logarítmico: el cociente de costo suele abarcar varios órdenes
    de magnitud en aplicaciones clínicas reales."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(cost_ratios, sensitivity, "o-", color=TECNM_BLUE, lw=2, label="Sensibilidad patológica")
    ax.plot(cost_ratios, specificity, "o-", color=ACCENT, lw=2, label="Especificidad patológica")
    ax.plot(cost_ratios, precision, "o-", color=TECNM_GRAY, lw=2, label="Precisión patológica")
    ax.set_xscale("log")
    ax.set_xlabel("Cociente de costo  C = costo(FN) / costo(FP)")
    ax.set_ylabel("Valor de la métrica")
    ax.set_ylim(0, 1.05)
    ax.set_title("Costo asimétrico: priors inflados por cociente de costo", fontsize=11.5)
    ax.axvline(1.0, color="#999999", ls=":", lw=1)
    ax.text(1.05, 0.03, "C=1\n(D-min\nclásico)", fontsize=7.5, color="#666666")
    ax.legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
