"""
Evaluates the trained GLSim model on the Herlev Cervical Cancer dataset.
Generates plots and full metrics in eval_output/.

Usage:
    python eval.py
    python eval.py --ckpt results_train/cervical_vit_b16_16_0/vit_b16_best.pth
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize
from types import SimpleNamespace

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "results_train", "cervical_vit_b16_16_0", "vit_b16_best.pth")
OUT_DIR   = os.path.join(BASE_DIR, "eval_output")
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate",
]
COLORS = ["#E53935", "#8E24AA", "#1E88E5", "#43A047", "#FB8C00"]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── model ──────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    sys.path.insert(0, BASE_DIR)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    args = SimpleNamespace(**cfg) if isinstance(cfg, dict) else cfg
    args.ckpt_path   = ckpt_path
    args.device      = device
    args.pretrained  = False
    args.distributed = False
    args.world_size  = args.rank = args.local_rank = 0

    from glsim.model_utils.build_model import build_model
    model = build_model(args)
    model.eval()
    return model, args


# ── data ───────────────────────────────────────────────────────────────────

def build_test_loader(args, batch_size, num_workers):
    from glsim.data_utils.build_transform import build_transform
    from glsim.data_utils.datasets import DatasetImgTarget

    transform = build_transform(args=args, split="test")
    ds     = DatasetImgTarget(args, split="test", transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    print(f"Test set: {len(ds)} images, {ds.num_classes} classes")
    return loader


# ── inference ──────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device, n_classes):
    all_labels, all_preds, all_probs = [], [], []
    n_total = len(loader.dataset)
    t0 = time.time()
    seen = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        out  = model(imgs)
        logits = out[0] if isinstance(out, tuple) else out
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = probs.argmax(axis=1)

        all_labels.append(labels.numpy())
        all_preds.append(preds)
        all_probs.append(probs)

        seen += imgs.size(0)
        elapsed = time.time() - t0
        print(f"\r  [{seen}/{n_total}]  {seen/max(elapsed,1e-6):.1f} img/s", end="")

    print()
    return (np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs, axis=0))


# ── plots ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, names, acc, out_dir):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True,
                xticklabels=names, yticklabels=names,
                linewidths=0.5, linecolor="#ddd",
                cbar_kws={"label": "# images", "shrink": 0.8}, ax=ax)
    ax.set_xlabel("Predicted", labelpad=8)
    ax.set_ylabel("True", labelpad=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fp = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp}")

    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", square=True,
                xticklabels=names, yticklabels=names,
                vmin=0, vmax=1,
                linewidths=0.5, linecolor="#ddd",
                cbar_kws={"label": "recall per class", "shrink": 0.8}, ax=ax)
    ax.set_xlabel("Predicted", labelpad=8)
    ax.set_ylabel("True", labelpad=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fp = os.path.join(out_dir, "confusion_matrix_normalized.png")
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp}")

    return cm


def plot_per_class_metrics(report_df, names, colors, out_dir):
    metrics = ["precision", "recall", "f1-score"]
    df = report_df.loc[names, metrics]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, color) in enumerate(zip(metrics, ["#1E88E5", "#43A047", "#E53935"])):
        ax.bar(x + (i - 1) * width, df[m], width, label=m.capitalize(),
               color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.legend(framealpha=0.9)
    ax.axhline(df["f1-score"].mean(), color="gray", linestyle="--",
               linewidth=1, label=f"Mean F1 = {df['f1-score'].mean():.3f}")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    fp = os.path.join(out_dir, "precision_recall_f1.png")
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp}")


def plot_roc_curves(y_true, y_probs, names, colors, n_classes, out_dir):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, color) in enumerate(zip(names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    all_fpr = np.unique(np.concatenate([
        roc_curve(y_bin[:, i], y_probs[:, i])[0] for i in range(n_classes)
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, "k--", linewidth=2,
            label=f"Macro average (AUC = {macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fp = os.path.join(out_dir, "roc_curves.png")
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp}")
    return macro_auc


def plot_precision_recall_curves(y_true, y_probs, names, colors, n_classes, out_dir):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, color) in enumerate(zip(names, colors)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f"{name} (AP = {ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fp = os.path.join(out_dir, "precision_recall_curves.png")
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp}")


def plot_per_class_accuracy(y_true, y_pred, names, colors, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, per_class_acc * 100, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.8)
    ax.axhline(np.mean(per_class_acc) * 100, color="gray", linestyle="--",
               linewidth=1.2, label=f"Mean = {np.mean(per_class_acc)*100:.1f}%")
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(out_dir, "accuracy_per_class.png")
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp}")


def plot_summary_table(summary, out_dir):
    rows = [
        ["Top-1 Accuracy",       f"{summary['acc']*100:.2f}%"],
        ["F1 (macro)",           f"{summary['f1_macro']:.4f}"],
        ["F1 (weighted)",        f"{summary['f1_weighted']:.4f}"],
        ["Precision (macro)",    f"{summary['prec_macro']:.4f}"],
        ["Recall (macro)",       f"{summary['rec_macro']:.4f}"],
        ["AUC ROC (macro)",      f"{summary['auc_macro']:.4f}"],
        ["# Images (test)",      str(summary['n'])],
        ["# Classes",            str(summary['k'])],
    ]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                   cellLoc="center", loc="center",
                   colWidths=[0.6, 0.4])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ccc")
        if r == 0:
            cell.set_facecolor("#1E3A5F")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")
    plt.tight_layout()
    fp = os.path.join(out_dir, "metrics_summary_table.png")
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=CKPT_PATH)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    cli = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Ckpt   : {cli.ckpt}\n")

    model, args = load_model(cli.ckpt, device)
    loader = build_test_loader(args, cli.batch_size, cli.num_workers)

    print("Running inference...")
    y_true, y_pred, y_probs = run_inference(model, loader, device, len(CLASSES))

    acc          = accuracy_score(y_true, y_pred)
    f1_mac       = f1_score(y_true, y_pred, average="macro",     zero_division=0)
    f1_wt        = f1_score(y_true, y_pred, average="weighted",  zero_division=0)
    prec_mac     = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    rec_mac      = recall_score(y_true, y_pred, average="macro",      zero_division=0)

    report       = classification_report(y_true, y_pred, target_names=CLASSES,
                                         zero_division=0, output_dict=True)
    report_df    = pd.DataFrame(report).T

    print("\n" + classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    print("Generating plots...")

    plot_confusion_matrix(y_true, y_pred, CLASSES, acc, OUT_DIR)
    plot_per_class_metrics(report_df, CLASSES, COLORS, OUT_DIR)
    plot_per_class_accuracy(y_true, y_pred, CLASSES, COLORS, OUT_DIR)
    auc_macro = plot_roc_curves(y_true, y_probs, CLASSES, COLORS, len(CLASSES), OUT_DIR)
    plot_precision_recall_curves(y_true, y_probs, CLASSES, COLORS, len(CLASSES), OUT_DIR)

    summary = dict(acc=acc, f1_macro=f1_mac, f1_weighted=f1_wt,
                   prec_macro=prec_mac, rec_macro=rec_mac,
                   auc_macro=auc_macro, n=len(y_true), k=len(CLASSES))
    plot_summary_table(summary, OUT_DIR)

    report_df.to_csv(os.path.join(OUT_DIR, "classification_report.csv"))

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        os.path.join(OUT_DIR, "confusion_matrix.csv"))

    pd.DataFrame({
        "y_true": y_true, "y_pred": y_pred,
        "true_name": [CLASSES[i] for i in y_true],
        "pred_name": [CLASSES[i] for i in y_pred],
        "correct": y_true == y_pred,
    }).to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

    txt = (
        f"================= Results (test split) =================\n"
        f"  # Images          : {len(y_true)}\n"
        f"  # Classes         : {len(CLASSES)}\n"
        f"  Top-1 accuracy    : {acc*100:.2f}%\n"
        f"  F1  (macro)       : {f1_mac:.4f}\n"
        f"  F1  (weighted)    : {f1_wt:.4f}\n"
        f"  Precision (macro) : {prec_mac:.4f}\n"
        f"  Recall    (macro) : {rec_mac:.4f}\n"
        f"  AUC ROC (macro)   : {auc_macro:.4f}\n"
        f"========================================================\n"
    )
    with open(os.path.join(OUT_DIR, "metrics_summary.txt"), "w") as f:
        f.write(txt)
    print("\n" + txt)
    print(f"All files saved in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
