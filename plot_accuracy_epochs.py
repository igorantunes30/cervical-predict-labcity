"""
Plots train and val accuracy per epoch for both models and saves to their
respective result folders:
  - eval_output/           (CROPPED model)
  - eval_slides_results/   (Full slides model)
  - accuracy_epochs.png         (side-by-side, root)
  - accuracy_epochs_overlay.png (overlay, root)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUNS = {
    "CROPPED cells": {
        "tb":    os.path.join(BASE_DIR, "results_train", "cervical_vit_b16_16_0", "tensorboard"),
        "out":   os.path.join(BASE_DIR, "eval_output"),
        "color": ("#1E88E5", "#90CAF9"),
    },
    "Full slides": {
        "tb":    os.path.join(BASE_DIR, "results_train", "cervical_vit_b16_16_1", "tensorboard"),
        "out":   os.path.join(BASE_DIR, "eval_slides_results"),
        "color": ("#E53935", "#EF9A9A"),
    },
}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_scalars(tb_path, tag):
    ea = EventAccumulator(tb_path)
    ea.Reload()
    events = ea.Scalars(tag)
    return (np.array([e.step for e in events]),
            np.array([e.value for e in events]))


def single_plot(run_name, cfg, out_path):
    col_val, col_train = cfg["color"]
    steps_tr, acc_tr = load_scalars(cfg["tb"], "train/acc1")
    steps_va, acc_va = load_scalars(cfg["tb"], "val/acc1")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps_tr, acc_tr, color=col_train, linewidth=1.5,
            alpha=0.8, label="Train Acc@1")
    ax.plot(steps_va, acc_va, color=col_val, linewidth=2.0,
            label="Val Acc@1")

    best_idx = int(np.argmax(acc_va))
    best_ep  = steps_va[best_idx]
    best_acc = acc_va[best_idx]
    ax.axvline(best_ep, color=col_val, linestyle="--", linewidth=1, alpha=0.6)
    ax.scatter([best_ep], [best_acc], color=col_val, zorder=5, s=70)
    offset_x = max(steps_va) * 0.04
    ax.annotate(f"Best: {best_acc:.1f}%\n(epoch {best_ep})",
                xy=(best_ep, best_acc),
                xytext=(best_ep + offset_x, best_acc - 10),
                fontsize=9, color=col_val,
                arrowprops=dict(arrowstyle="->", color=col_val, lw=1))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(1, int(steps_tr[-1]))
    ax.set_ylim(0, 105)
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── individual plots in each result folder ─────────────────────────────────

for run_name, cfg in RUNS.items():
    out = os.path.join(cfg["out"], "accuracy_epochs.png")
    single_plot(run_name, cfg, out)


# ── side-by-side (root) ────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (run_name, cfg) in zip(axes, RUNS.items()):
    col_val, col_train = cfg["color"]
    steps_tr, acc_tr = load_scalars(cfg["tb"], "train/acc1")
    steps_va, acc_va = load_scalars(cfg["tb"], "val/acc1")

    ax.plot(steps_tr, acc_tr, color=col_train, linewidth=1.5,
            alpha=0.8, label="Train Acc@1")
    ax.plot(steps_va, acc_va, color=col_val, linewidth=2.0,
            label="Val Acc@1")

    best_idx = int(np.argmax(acc_va))
    best_ep  = steps_va[best_idx]
    best_acc = acc_va[best_idx]
    ax.axvline(best_ep, color=col_val, linestyle="--", linewidth=1, alpha=0.6)
    ax.scatter([best_ep], [best_acc], color=col_val, zorder=5, s=70)
    offset_x = max(steps_va) * 0.04
    ax.annotate(f"Best: {best_acc:.1f}%\n(epoch {best_ep})",
                xy=(best_ep, best_acc),
                xytext=(best_ep + offset_x, best_acc - 10),
                fontsize=9, color=col_val,
                arrowprops=dict(arrowstyle="->", color=col_val, lw=1))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(1, int(steps_tr[-1]))
    ax.set_ylim(0, 105)
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_title(run_name, fontweight="bold")

plt.tight_layout()
out = os.path.join(BASE_DIR, "accuracy_epochs.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")


# ── overlay (root) ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

for run_name, cfg in RUNS.items():
    col_val, col_train = cfg["color"]
    steps_tr, acc_tr = load_scalars(cfg["tb"], "train/acc1")
    steps_va, acc_va = load_scalars(cfg["tb"], "val/acc1")

    ax.plot(steps_tr, acc_tr, color=col_train, linewidth=1.2,
            alpha=0.5, linestyle="--")
    ax.plot(steps_va, acc_va, color=col_val, linewidth=2.0,
            label=f"{run_name} — Val")

    best_idx = int(np.argmax(acc_va))
    ax.scatter([steps_va[best_idx]], [acc_va[best_idx]],
               color=col_val, zorder=5, s=70)

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 105)
ax.set_xlim(1, 50)
ax.legend(framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(BASE_DIR, "accuracy_epochs_overlay.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
