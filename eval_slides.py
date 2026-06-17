"""
Evaluates the GLSim model on the full-slide (non-CROPPED) images.
Images are 1536x2048 px whole microscope slides, not individual cells.
The same test-time transform is applied (resize 300 → center crop 224).
Results are saved to eval_slides_output/.

Usage:
    python eval_slides.py
    python eval_slides.py --ckpt results_train/cervical_vit_b16_16_0/vit_b16_best.pth
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize
from types import SimpleNamespace

BASE_DIR  = Path(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = BASE_DIR / "results_train" / "cervical_vit_b16_16_0" / "vit_b16_best.pth"
OUT_DIR   = BASE_DIR / "eval_slides_output"
OUT_DIR.mkdir(exist_ok=True)

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


# ── dataset ────────────────────────────────────────────────────────────────

class SlideDataset(Dataset):
    """Loads full-slide BMP images (root of each class folder, not CROPPED)."""

    def __init__(self, base_dir, transform=None):
        self.samples = []
        self.transform = transform
        for class_id, cls in enumerate(CLASSES):
            folder = base_dir / f"im_{cls}" / f"im_{cls}"
            bmps = sorted(p for p in folder.iterdir()
                          if p.suffix.lower() == ".bmp")
            for p in bmps:
                self.samples.append((p, class_id))
        print(f"Slides found: {len(self.samples)} images across {len(CLASSES)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── model ──────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    sys.path.insert(0, str(BASE_DIR))
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    args = SimpleNamespace(**cfg) if isinstance(cfg, dict) else cfg
    args.ckpt_path   = str(ckpt_path)
    args.device      = device
    args.pretrained  = False
    args.distributed = False
    args.world_size  = args.rank = args.local_rank = 0

    from glsim.model_utils.build_model import build_model
    model = build_model(args)
    model.eval()
    return model, args


# ── inference ──────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
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

def plot_confusion_matrix(y_true, y_pred, acc, out_dir):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True,
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.5, linecolor="#ddd",
                cbar_kws={"label": "# images", "shrink": 0.8}, ax=ax)
    ax.set_xlabel("Predicted", labelpad=8)
    ax.set_ylabel("True", labelpad=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", square=True,
                xticklabels=CLASSES, yticklabels=CLASSES,
                vmin=0, vmax=1,
                linewidths=0.5, linecolor="#ddd",
                cbar_kws={"label": "recall per class", "shrink": 0.8}, ax=ax)
    ax.set_xlabel("Predicted", labelpad=8)
    ax.set_ylabel("True", labelpad=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_normalized.png", bbox_inches="tight")
    plt.close(fig)
    return cm


def plot_per_class_metrics(report_df, out_dir):
    metrics = ["precision", "recall", "f1-score"]
    df = report_df.loc[CLASSES, metrics]
    x = np.arange(len(CLASSES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, color) in enumerate(zip(metrics, ["#1E88E5", "#43A047", "#E53935"])):
        ax.bar(x + (i - 1) * width, df[m], width, label=m.capitalize(),
               color=color, alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.axhline(df["f1-score"].mean(), color="gray", linestyle="--",
               linewidth=1, label=f"Mean F1 = {df['f1-score'].mean():.3f}")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "precision_recall_f1.png", bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(y_true, y_probs, out_dir):
    n_classes = len(CLASSES)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, color) in enumerate(zip(CLASSES, COLORS)):
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
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", bbox_inches="tight")
    plt.close(fig)
    return macro_auc


def plot_precision_recall_curves(y_true, y_probs, out_dir):
    n_classes = len(CLASSES)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, color) in enumerate(zip(CLASSES, COLORS)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f"{name} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_dir / "precision_recall_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_per_class_accuracy(y_true, y_pred, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(CLASSES, per_class_acc * 100, color=COLORS, alpha=0.85,
                  edgecolor="white", linewidth=0.8)
    ax.axhline(np.mean(per_class_acc) * 100, color="gray", linestyle="--",
               linewidth=1.2, label=f"Mean = {np.mean(per_class_acc)*100:.1f}%")
    for bar, acc_v in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{acc_v*100:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "accuracy_per_class.png", bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar(cropped_metrics, slides_metrics, out_dir):
    """Side-by-side bar comparing CROPPED vs Full-Slide results."""
    metrics = ["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)", "AUC ROC"]
    cropped_vals = [
        cropped_metrics["acc"] * 100,
        cropped_metrics["f1"] * 100,
        cropped_metrics["prec"] * 100,
        cropped_metrics["rec"] * 100,
        cropped_metrics["auc"] * 100,
    ]
    slides_vals = [
        slides_metrics["acc"] * 100,
        slides_metrics["f1"] * 100,
        slides_metrics["prec"] * 100,
        slides_metrics["rec"] * 100,
        slides_metrics["auc"] * 100,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_c = ax.bar(x - width/2, cropped_vals, width, label="CROPPED cells",
                    color="#1E88E5", alpha=0.85, edgecolor="white")
    bars_s = ax.bar(x + width/2, slides_vals, width, label="Full slides",
                    color="#E53935", alpha=0.85, edgecolor="white")

    for bar in bars_c:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars_s:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 115)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "comparison_cropped_vs_slides.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: comparison_cropped_vs_slides.png")


def plot_summary_table(summary, out_dir):
    rows = [
        ["Top-1 Accuracy",    f"{summary['acc']*100:.2f}%"],
        ["F1 (macro)",        f"{summary['f1']:.4f}"],
        ["Precision (macro)", f"{summary['prec']:.4f}"],
        ["Recall (macro)",    f"{summary['rec']:.4f}"],
        ["AUC ROC (macro)",   f"{summary['auc']:.4f}"],
        ["# Images (slides)", str(summary['n'])],
        ["# Classes",         str(len(CLASSES))],
    ]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                   cellLoc="center", loc="center", colWidths=[0.6, 0.4])
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
    fig.savefig(out_dir / "metrics_summary_table.png", bbox_inches="tight")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(CKPT_PATH))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    cli = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Ckpt   : {cli.ckpt}")
    print(f"Mode   : Full-slide images (NOT CROPPED)\n")

    model, args = load_model(cli.ckpt, device)

    # Same transform as test split
    from glsim.data_utils.build_transform import build_transform
    transform = build_transform(args=args, split="test")
    print(f"Transform: {transform}\n")

    ds = SlideDataset(BASE_DIR, transform=transform)
    loader = DataLoader(ds, batch_size=cli.batch_size, shuffle=False,
                        num_workers=cli.num_workers, pin_memory=True)

    print("Running inference on full slides...")
    y_true, y_pred, y_probs = run_inference(model, loader, device)

    acc      = accuracy_score(y_true, y_pred)
    f1_mac   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wt    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_mac = precision_score(y_true, y_pred, average="macro",   zero_division=0)
    rec_mac  = recall_score(y_true, y_pred, average="macro",      zero_division=0)

    report    = classification_report(y_true, y_pred, target_names=CLASSES,
                                      zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).T

    print("\n" + classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    print("Generating plots...")
    plot_confusion_matrix(y_true, y_pred, acc, OUT_DIR)
    print("Saved: confusion_matrix.png / confusion_matrix_normalized.png")
    plot_per_class_metrics(report_df, OUT_DIR)
    print("Saved: precision_recall_f1.png")
    plot_per_class_accuracy(y_true, y_pred, OUT_DIR)
    print("Saved: accuracy_per_class.png")
    auc_macro = plot_roc_curves(y_true, y_probs, OUT_DIR)
    print("Saved: roc_curves.png")
    plot_precision_recall_curves(y_true, y_probs, OUT_DIR)
    print("Saved: precision_recall_curves.png")

    summary = dict(acc=acc, f1=f1_mac, prec=prec_mac, rec=rec_mac,
                   auc=auc_macro, n=len(y_true))
    plot_summary_table(summary, OUT_DIR)
    print("Saved: metrics_summary_table.png")

    # Comparison chart vs CROPPED results
    cropped_metrics = dict(acc=0.9619, f1=0.9618, prec=0.9625, rec=0.9621, auc=0.9978)
    plot_comparison_bar(cropped_metrics, summary, OUT_DIR)

    # CSVs
    report_df.to_csv(OUT_DIR / "classification_report.csv")
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        OUT_DIR / "confusion_matrix.csv")
    pd.DataFrame({
        "y_true": y_true, "y_pred": y_pred,
        "true_name":  [CLASSES[i] for i in y_true],
        "pred_name":  [CLASSES[i] for i in y_pred],
        "correct":    y_true == y_pred,
    }).to_csv(OUT_DIR / "predictions.csv", index=False)

    txt = (
        f"================= Results — Full Slides =================\n"
        f"  # Images          : {len(y_true)}\n"
        f"  # Classes         : {len(CLASSES)}\n"
        f"  Top-1 accuracy    : {acc*100:.2f}%\n"
        f"  F1  (macro)       : {f1_mac:.4f}\n"
        f"  F1  (weighted)    : {f1_wt:.4f}\n"
        f"  Precision (macro) : {prec_mac:.4f}\n"
        f"  Recall    (macro) : {rec_mac:.4f}\n"
        f"  AUC ROC (macro)   : {auc_macro:.4f}\n"
        f"\n"
        f"  --- CROPPED test set (reference) ---\n"
        f"  Top-1 accuracy    : 96.19%\n"
        f"  F1  (macro)       : 0.9618\n"
        f"  AUC ROC (macro)   : 0.9978\n"
        f"=========================================================\n"
    )
    with open(OUT_DIR / "metrics_summary.txt", "w") as f:
        f.write(txt)
    print("\n" + txt)
    print(f"All files saved in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
