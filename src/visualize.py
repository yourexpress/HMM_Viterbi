"""
Visualization utilities for HMM POS-tagger analysis.

Generates and saves figures that illustrate:
  1. Confusion matrix (heat-map)
  2. Per-tag F1 scores (bar chart)
  3. Transition probability matrix (heat-map)
  4. Top emission probabilities per tag (grid of bar charts)
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ---------------------------------------------------------------------------
# Colour palette / style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.0)
CMAP_MATRIX = "Blues"
CMAP_TRANS = "YlOrRd"
BAR_COLOR = "#2196F3"
FIGURE_DPI = 150


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(confusion_matrix: dict, output_path: str):
    """
    Save a normalised confusion-matrix heat-map.

    Parameters
    ----------
    confusion_matrix : dict {gold_tag: {pred_tag: count}}
    output_path : str
        Full path (including filename) to save the figure.
    """
    tags = sorted(set(confusion_matrix.keys()))
    for inner in confusion_matrix.values():
        tags = sorted(set(tags) | set(inner.keys()))

    n = len(tags)
    matrix = np.zeros((n, n), dtype=float)
    tag_idx = {t: i for i, t in enumerate(tags)}

    for gold_tag, preds in confusion_matrix.items():
        for pred_tag, count in preds.items():
            matrix[tag_idx[gold_tag]][tag_idx[pred_tag]] += count

    # Row-normalise to get recall per tag
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix_norm = np.where(row_sums > 0, matrix / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.6)))
    sns.heatmap(
        matrix_norm,
        annot=True,
        fmt=".2f",
        xticklabels=tags,
        yticklabels=tags,
        cmap=CMAP_MATRIX,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Tag", fontsize=12)
    ax.set_ylabel("Gold Tag", fontsize=12)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrix → {output_path}")


# ---------------------------------------------------------------------------
# 2. Per-tag F1 bar chart
# ---------------------------------------------------------------------------

def plot_per_tag_f1(per_tag_f1: dict, overall_accuracy: float, output_path: str):
    """
    Bar chart of per-tag F1 scores with overall accuracy line.

    Parameters
    ----------
    per_tag_f1 : dict {tag: F1_score}
    overall_accuracy : float
    output_path : str
    """
    tags = sorted(per_tag_f1.keys())
    f1_vals = [per_tag_f1[t] for t in tags]

    fig, ax = plt.subplots(figsize=(max(8, len(tags) * 0.8), 5))
    bars = ax.bar(tags, f1_vals, color=BAR_COLOR, edgecolor="white", linewidth=0.7)

    # Colour bars by value
    cmap = plt.cm.get_cmap("RdYlGn")
    for bar, val in zip(bars, f1_vals):
        bar.set_facecolor(cmap(val))

    ax.axhline(overall_accuracy, color="red", linestyle="--", linewidth=1.5,
               label=f"Overall Accuracy = {overall_accuracy:.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("POS Tag", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Tag F1 Score", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    for bar, val in zip(bars, f1_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved per-tag F1 bar chart → {output_path}")


# ---------------------------------------------------------------------------
# 3. Transition probability matrix
# ---------------------------------------------------------------------------

def plot_transition_matrix(log_transition: dict, output_path: str):
    """
    Heat-map of (linear) transition probabilities between POS tags.

    Parameters
    ----------
    log_transition : dict {tag_i: {tag_j: log_prob}}
    output_path : str
    """
    tags = sorted(log_transition.keys())
    n = len(tags)
    tag_idx = {t: i for i, t in enumerate(tags)}

    matrix = np.zeros((n, n))
    for tag_i, inner in log_transition.items():
        for tag_j, lp in inner.items():
            if tag_j in tag_idx:
                matrix[tag_idx[tag_i]][tag_idx[tag_j]] = math.exp(lp)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.6)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        xticklabels=tags,
        yticklabels=tags,
        cmap=CMAP_TRANS,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Next Tag", fontsize=12)
    ax.set_ylabel("Current Tag", fontsize=12)
    ax.set_title("HMM Transition Probability Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved transition matrix → {output_path}")


# ---------------------------------------------------------------------------
# 4. Top-N emission probabilities per tag
# ---------------------------------------------------------------------------

def plot_top_emissions(log_emission: dict, output_path: str, top_n: int = 8):
    """
    Grid of bar charts showing the top-N most likely words for each tag.

    Parameters
    ----------
    log_emission : dict {tag: {word: log_prob}}
    output_path : str
    top_n : int
        Number of words to show per tag.
    """
    tags = sorted(log_emission.keys())
    ncols = 4
    nrows = math.ceil(len(tags) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4, nrows * 3),
        constrained_layout=True,
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, tag in enumerate(tags):
        ax = axes_flat[idx]
        emissions = {
            w: math.exp(lp)
            for w, lp in log_emission[tag].items()
            if w != "<UNK>"
        }
        top_items = sorted(emissions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        if not top_items:
            ax.set_visible(False)
            continue
        words, probs = zip(*top_items)
        ax.barh(list(words)[::-1], list(probs)[::-1], color=BAR_COLOR)
        ax.set_title(tag, fontsize=11, fontweight="bold")
        ax.set_xlabel("P(word|tag)", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)

    # Hide unused subplots
    for idx in range(len(tags), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Top-{top_n} Emission Probabilities per POS Tag",
        fontsize=14,
        fontweight="bold",
    )
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved top emissions plot → {output_path}")


# ---------------------------------------------------------------------------
# 5. Precision / Recall / F1 grouped bar chart
# ---------------------------------------------------------------------------

def plot_precision_recall_f1(
    per_tag_precision: dict,
    per_tag_recall: dict,
    per_tag_f1: dict,
    output_path: str,
):
    """
    Grouped bar chart showing precision, recall, and F1 for every tag.

    Parameters
    ----------
    per_tag_precision, per_tag_recall, per_tag_f1 : dict {tag: float}
    output_path : str
    """
    tags = sorted(per_tag_precision.keys())
    x = np.arange(len(tags))
    width = 0.25

    p_vals = [per_tag_precision[t] for t in tags]
    r_vals = [per_tag_recall[t] for t in tags]
    f_vals = [per_tag_f1[t] for t in tags]

    fig, ax = plt.subplots(figsize=(max(10, len(tags) * 0.9), 5))
    ax.bar(x - width, p_vals, width, label="Precision", color="#4CAF50")
    ax.bar(x, r_vals, width, label="Recall", color="#FF9800")
    ax.bar(x + width, f_vals, width, label="F1", color="#2196F3")

    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("POS Tag", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision, Recall, and F1 per POS Tag", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved precision/recall/F1 chart → {output_path}")


# ---------------------------------------------------------------------------
# Convenience: generate all figures at once
# ---------------------------------------------------------------------------

def generate_all_figures(model, results: dict, figures_dir: str):
    """
    Generate all standard analysis figures and save them to *figures_dir*.

    Parameters
    ----------
    model : HMMModel
        Trained model (provides transition / emission tables).
    results : dict
        Evaluation results from :func:`src.evaluate.evaluate_model`.
    figures_dir : str
        Directory where PNG files will be written.
    """
    print("\nGenerating figures …")
    _ensure_dir(figures_dir)

    plot_confusion_matrix(
        results["confusion_matrix"],
        os.path.join(figures_dir, "confusion_matrix.png"),
    )
    plot_per_tag_f1(
        results["per_tag_f1"],
        results["overall_accuracy"],
        os.path.join(figures_dir, "per_tag_f1.png"),
    )
    plot_transition_matrix(
        model.log_transition,
        os.path.join(figures_dir, "transition_matrix.png"),
    )
    plot_top_emissions(
        model.log_emission,
        os.path.join(figures_dir, "top_emissions.png"),
    )
    plot_precision_recall_f1(
        results["per_tag_precision"],
        results["per_tag_recall"],
        results["per_tag_f1"],
        os.path.join(figures_dir, "precision_recall_f1.png"),
    )
    print("All figures saved.")
