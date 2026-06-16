"""
utils.py
--------
Reusable preprocessing, evaluation, and visualisation utilities for the
Clinical NLP Drug Review XAI pipeline.

Author : Garcia Wa Nnam
Project: Clinical NLP — Drug Review Sentiment Classification with XAI & UQ
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. TEXT PREPROCESSING
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lightweight text cleaning for drug review data.
    - Removes HTML entities (&#039; etc.)
    - Strips excess whitespace
    - Lowercases
    Deliberately avoids removing stopwords or stemming so that
    transformer tokenisers receive naturalistic input.
    """
    if not isinstance(text, str):
        return ""
    # Decode common HTML entities
    text = re.sub(r"&#\d+;", "'", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def binarise_rating(rating: float, threshold: int = 7) -> int:
    """
    Convert a 1-10 drug rating into a binary sentiment label.
    ≥ threshold → 1 (Positive / Recommended)
    <  threshold → 0 (Negative / Not Recommended)

    Threshold of 7 chosen to reflect clinically meaningful satisfaction;
    aligns with common patient-reported outcome conventions.
    """
    return int(rating >= threshold)


def truncate_text(text: str, max_words: int = 128) -> str:
    """Truncate review to max_words to manage memory during tokenisation."""
    words = text.split()
    return " ".join(words[:max_words])


def load_and_prepare(train_path: str, test_path: str,
                     rating_threshold: int = 7,
                     sample_n: int = None,
                     random_state: int = 42) -> tuple:
    """
    Load raw TSV files, clean text, binarise labels, and return
    train/test DataFrames ready for modelling.

    Parameters
    ----------
    train_path     : path to drugsComTrain_raw.tsv
    test_path      : path to drugsComTest_raw.tsv
    rating_threshold: threshold for positive label (default 7)
    sample_n       : if not None, downsample train set to n rows
    random_state   : for reproducibility

    Returns
    -------
    df_train, df_test : cleaned DataFrames with columns
                        ['review_clean', 'label', 'condition', 'drugName', 'rating']
    """
    df_train = pd.read_csv(train_path, sep="\t", on_bad_lines="skip")
    df_test  = pd.read_csv(test_path,  sep="\t", on_bad_lines="skip")

    for df in [df_train, df_test]:
        df.dropna(subset=["review", "rating"], inplace=True)
        df["review_clean"] = df["review"].apply(clean_text)
        df["label"] = df["rating"].apply(lambda r: binarise_rating(r, rating_threshold))

    if sample_n is not None:
        df_train = df_train.sample(n=min(sample_n, len(df_train)),
                                   random_state=random_state)

    keep_cols = ["review_clean", "label", "condition", "drugName", "rating"]
    df_train = df_train[[c for c in keep_cols if c in df_train.columns]].reset_index(drop=True)
    df_test  = df_test[[c for c in keep_cols if c in df_test.columns]].reset_index(drop=True)

    return df_train, df_test


# ─────────────────────────────────────────────
# 2. EVALUATION
# ─────────────────────────────────────────────

def full_evaluation_report(y_true, y_pred, y_prob=None, model_name: str = "Model"):
    """
    Print a comprehensive classification report and optionally compute ROC-AUC.

    Parameters
    ----------
    y_true     : array-like of true binary labels
    y_pred     : array-like of predicted binary labels
    y_prob     : array-like of predicted probabilities for class 1 (optional)
    model_name : label for display purposes
    """
    print(f"\n{'='*55}")
    print(f"  Evaluation Report — {model_name}")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred,
                                 target_names=["Negative (0)", "Positive (1)"]))
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        ap  = average_precision_score(y_true, y_prob)
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  Avg Prec : {ap:.4f}")
    print(f"{'='*55}\n")


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model",
                          save_path: str = None):
    """Plot a labelled, normalised confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, title in zip(axes,
                                [cm, cm_norm],
                                ["Counts", "Normalised"]):
        sns.heatmap(data, annot=True, fmt=".2f" if title == "Normalised" else "d",
                    cmap="Blues", ax=ax,
                    xticklabels=["Pred Neg", "Pred Pos"],
                    yticklabels=["True Neg", "True Pos"])
        ax.set_title(f"{model_name} — Confusion Matrix ({title})")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curve(y_true, y_prob_dict: dict, save_path: str = None):
    """
    Overlay ROC curves for multiple models.

    Parameters
    ----------
    y_true       : array-like of true labels
    y_prob_dict  : dict {model_name: y_prob_array}
    """
    plt.figure(figsize=(8, 6))
    for name, y_prob in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# 3. UNCERTAINTY QUANTIFICATION HELPERS
# ─────────────────────────────────────────────

def compute_confidence_stats(y_prob: np.ndarray, y_pred: np.ndarray,
                              y_true: np.ndarray) -> pd.DataFrame:
    """
    Summarise model confidence for correct vs incorrect predictions.
    Useful for identifying overconfident or underconfident regions.
    """
    df = pd.DataFrame({
        "probability": y_prob,
        "predicted":   y_pred,
        "actual":      y_true,
        "correct":     (y_pred == y_true)
    })
    summary = df.groupby("correct")["probability"].describe()
    summary.index = ["Incorrect Predictions", "Correct Predictions"]
    return summary


def plot_calibration_curve(y_true, y_prob_dict: dict,
                           n_bins: int = 10, save_path: str = None):
    """
    Plot reliability diagrams (calibration curves) for one or more models.
    A perfectly calibrated model lies on the diagonal.

    In clinical settings, poor calibration can lead to misleading confidence
    scores — this plot surfaces that risk explicitly.
    """
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)

    for name, y_prob in y_prob_dict.items():
        fraction_pos, mean_pred = calibration_curve(y_true, y_prob,
                                                     n_bins=n_bins,
                                                     strategy="uniform")
        plt.plot(mean_pred, fraction_pos, marker="o", label=name, linewidth=2)

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve (Reliability Diagram)\n"
              "Closer to diagonal = better-calibrated confidence scores")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    ECE = weighted mean absolute difference between confidence and accuracy
    per probability bin.  Lower is better.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = (y_true[mask] == (y_prob[mask] >= 0.5).astype(int)).mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


# ─────────────────────────────────────────────
# 4. REPRODUCIBILITY
# ─────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across numpy, torch, and Python."""
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def results_to_dataframe(results_dict: dict) -> pd.DataFrame:
    """
    Convert a dictionary of {model_name: metrics_dict} to a tidy DataFrame
    for easy tabular comparison — suitable for inclusion in README or report.
    """
    rows = []
    for model, metrics in results_dict.items():
        row = {"Model": model}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")
