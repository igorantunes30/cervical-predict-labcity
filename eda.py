"""
EDA — Herlev Cervical Cancer Dataset
Generates complete exploratory analysis in eda_output/:
  - Class distribution
  - Sample images per class
  - Pixel statistics (RGB) per class
  - Color histograms
  - Feature boxplots
  - Feature correlation matrix
  - PCA + t-SNE + UMAP colored by class
  - Statistical test per feature (ANOVA)

Usage: python eda.py
"""

import os
import warnings
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
from PIL import Image
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ── config ─────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
OUT_DIR   = BASE_DIR / "eda_output"
OUT_DIR.mkdir(exist_ok=True)

CLASSES = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate",
]
CLASS_DIRS = [
    BASE_DIR / f"im_{c}" / f"im_{c}" / "CROPPED"
    for c in CLASSES
]
COLORS  = ["#E53935", "#8E24AA", "#1E88E5", "#43A047", "#FB8C00"]
PALETTE = dict(zip(CLASSES, COLORS))

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})


# ── image collection ───────────────────────────────────────────────────────

def collect_paths():
    paths, labels = [], []
    for cid, cdir in enumerate(CLASS_DIRS):
        bmps = sorted(p for p in cdir.iterdir() if p.suffix.lower() == ".bmp")
        paths.extend(bmps)
        labels.extend([cid] * len(bmps))
    return paths, labels


def load_image_rgb(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── feature extraction ─────────────────────────────────────────────────────

def extract_features(paths, labels):
    rows = []
    n = len(paths)
    for i, (p, lbl) in enumerate(zip(paths, labels)):
        if (i + 1) % 200 == 0 or i == n - 1:
            print(f"\r  [{i+1}/{n}]", end="")
        img = load_image_rgb(p)
        h, w = img.shape[:2]
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        rows.append({
            "class_id":   lbl,
            "class_name": CLASSES[lbl],
            "width":      w,
            "height":     h,
            "aspect":     round(w / h, 3),
            "r_mean":     r.mean(),
            "g_mean":     g.mean(),
            "b_mean":     b.mean(),
            "r_std":      r.std(),
            "g_std":      g.std(),
            "b_std":      b.std(),
            "brightness": gray.mean(),
            "contrast":   gray.std(),
            "r_min":      r.min(),  "r_max": r.max(),
            "g_min":      g.min(),  "g_max": g.max(),
            "b_min":      b.min(),  "b_max": b.max(),
            "saturation": _saturation(img),
        })
    print()
    return pd.DataFrame(rows)


def _saturation(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return hsv[:, :, 1].mean()


# ── plots ──────────────────────────────────────────────────────────────────

def plot_class_distribution(df):
    counts = df.groupby("class_name").size().reindex(CLASSES)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bars = ax.bar(CLASSES, counts, color=COLORS, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(v), ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Number of cells")
    ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        counts, labels=CLASSES, colors=COLORS,
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    plt.tight_layout()
    fp = OUT_DIR / "01_class_distribution.png"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp.name}")


def plot_sample_images(paths, labels, n_per_class=6):
    fig, axes = plt.subplots(len(CLASSES), n_per_class,
                             figsize=(n_per_class * 2, len(CLASSES) * 2.2))
    by_class = defaultdict(list)
    for p, l in zip(paths, labels):
        by_class[l].append(p)

    for cid, name in enumerate(CLASSES):
        samples = by_class[cid][:n_per_class]
        for j, p in enumerate(samples):
            ax = axes[cid][j]
            img = load_image_rgb(p)
            ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(name, fontsize=9, fontweight="bold",
                              rotation=0, labelpad=80, va="center")

    plt.tight_layout()
    fp = OUT_DIR / "02_sample_images.png"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp.name}")


def plot_image_size_distribution(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, col, xlabel in zip(axes,
                               ["width", "height", "aspect"],
                               ["Width (px)", "Height (px)", "Aspect Ratio (w/h)"]):
        for cid, name in enumerate(CLASSES):
            d = df.loc[df.class_id == cid, col]
            ax.hist(d, bins=30, alpha=0.6, color=COLORS[cid], label=name, edgecolor="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    fp = OUT_DIR / "03_image_sizes.png"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp.name}")


def plot_rgb_distributions(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    channels = ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"]
    titles   = ["Mean R", "Mean G", "Mean B", "Std R", "Std G", "Std B"]

    for ax, col, xlabel in zip(axes.flat, channels, titles):
        for cid, name in enumerate(CLASSES):
            d = df.loc[df.class_id == cid, col]
            ax.hist(d, bins=40, alpha=0.55, color=COLORS[cid], label=name, edgecolor="none")
        ax.set_xlabel(xlabel + " — Pixel value (0-255)")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)

    axes[0, 0].legend(fontsize=7)
    plt.tight_layout()
    fp = OUT_DIR / "04_rgb_distributions.png"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp.name}")


def plot_color_histograms_per_class(paths, labels, n_sample=150):
    rng = np.random.default_rng(42)
    by_class = defaultdict(list)
    for p, l in zip(paths, labels):
        by_class[l].append(p)

    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(14, len(CLASSES)*2.5))
    ch_names = ["R", "G", "B"]
    ch_colors = ["#E53935", "#43A047", "#1E88E5"]

    for cid, name in enumerate(CLASSES):
        sample = rng.choice(by_class[cid],
                            min(n_sample, len(by_class[cid])), replace=False)
        combined = [None, None, None]
        for p in sample:
            img = load_image_rgb(p)
            for ch in range(3):
                combined[ch] = (img[:,:,ch].flatten() if combined[ch] is None
                                else np.concatenate([combined[ch], img[:,:,ch].flatten()]))
        for ch in range(3):
            ax = axes[cid][ch]
            ax.hist(combined[ch], bins=64, color=ch_colors[ch], alpha=0.75,
                    range=(0, 255), edgecolor="none")
            ax.set_xlim(0, 255)
            ax.set_xlabel(f"{name} — {ch_names[ch]}", fontsize=9)
            ax.set_yticks([])
            ax.grid(alpha=0.2)
    plt.tight_layout()
    fp = OUT_DIR / "05_color_histograms.png"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp.name}")


def plot_boxplots(df):
    feat_groups = [
        (["r_mean", "g_mean", "b_mean"], "Mean RGB by Class"),
        (["brightness", "contrast", "saturation"], "Brightness Contrast Saturation by Class"),
    ]

    for feats, title in feat_groups:
        n = len(feats)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        for ax, feat in zip(np.atleast_1d(axes), feats):
            data = [df.loc[df.class_id == cid, feat].values for cid in range(len(CLASSES))]
            bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                            medianprops=dict(color="white", linewidth=2))
            for patch, color in zip(bp["boxes"], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            ax.set_xticks(range(1, len(CLASSES)+1))
            ax.set_xticklabels([c[:8] for c in CLASSES], rotation=30, ha="right", fontsize=8)
            ax.set_xlabel(feat.replace("_", " ").title())
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        safe = title[:30].replace(" ", "_").replace("·", "").replace("__", "_").strip("_")
        fp = OUT_DIR / f"06_boxplot_{safe}.png"
        fig.savefig(fp, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fp.name}")


def plot_correlation_matrix(df):
    num_cols = ["r_mean","g_mean","b_mean","r_std","g_std","b_std",
                "brightness","contrast","saturation","width","height","aspect","class_id"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, linecolor="#ddd",
                mask=mask,
                cbar_kws={"label": "Pearson Correlation", "shrink": 0.8},
                ax=ax, annot_kws={"size": 8})
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    fp = OUT_DIR / "07_correlation_matrix.png"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp.name}")

    corr_class = corr["class_id"].drop("class_id").sort_values(key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#E53935" if v > 0 else "#1E88E5" for v in corr_class]
    ax.barh(corr_class.index[::-1], corr_class.values[::-1], color=colors[::-1], alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fp = OUT_DIR / "08_correlation_with_class.png"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp.name}")

    return corr


def plot_pca(df):
    feats = ["r_mean","g_mean","b_mean","r_std","g_std","b_std",
             "brightness","contrast","saturation"]
    X = df[feats].values
    y = df["class_id"].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_sc)
    var = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for cid, name in enumerate(CLASSES):
        mask = y == cid
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=COLORS[cid], label=name, alpha=0.5, s=18, edgecolors="none")
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
    ax.legend(fontsize=8, markerscale=1.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for cid, name in enumerate(CLASSES):
        mask = y == cid
        ax.scatter(X_pca[mask, 0], X_pca[mask, 2],
                   c=COLORS[cid], label=name, alpha=0.5, s=18, edgecolors="none")
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
    ax.set_ylabel(f"PC3 ({var[2]:.1f}%)")
    ax.legend(fontsize=8, markerscale=1.5)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fp = OUT_DIR / "09_pca.png"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp.name}")

    fig, ax = plt.subplots(figsize=(7, 4))
    pca_full = PCA().fit(X_sc)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
    ax.plot(range(1, len(cumvar)+1), cumvar, marker="o", markersize=4, linewidth=1.5)
    ax.axhline(90, color="gray", linestyle="--", linewidth=1, label="90%")
    ax.axhline(95, color="orange", linestyle="--", linewidth=1, label="95%")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fp = OUT_DIR / "10_pca_variance.png"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp.name}")

    return X_sc, y


def plot_tsne(X_sc, y):
    print("  Running t-SNE (may take ~1 min)...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X_sc)

    fig, ax = plt.subplots(figsize=(9, 7))
    for cid, name in enumerate(CLASSES):
        mask = y == cid
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=COLORS[cid], label=name, alpha=0.6, s=20, edgecolors="none")
    ax.legend(fontsize=9, markerscale=1.8, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fp = OUT_DIR / "11_tsne.png"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp.name}")


def plot_umap(X_sc, y):
    try:
        import umap
        print("  Running UMAP...")
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=42)
        X_2d = reducer.fit_transform(X_sc)
        fig, ax = plt.subplots(figsize=(9, 7))
        for cid, name in enumerate(CLASSES):
            mask = y == cid
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=COLORS[cid], label=name, alpha=0.6, s=20, edgecolors="none")
        ax.legend(fontsize=9, markerscale=1.8, framealpha=0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.2)
        plt.tight_layout()
        fp = OUT_DIR / "12_umap.png"
        fig.savefig(fp, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fp.name}")
    except ImportError:
        print("  UMAP not available, skipping.")


def plot_anova_results(df):
    feats = ["r_mean","g_mean","b_mean","r_std","g_std","b_std",
             "brightness","contrast","saturation","width","height"]

    results = []
    for feat in feats:
        groups = [df.loc[df.class_id == cid, feat].values for cid in range(len(CLASSES))]
        f_stat, p_val = stats.f_oneway(*groups)
        results.append({"feature": feat, "F": f_stat, "p_value": p_val,
                        "significant": p_val < 0.05})
    df_anova = pd.DataFrame(results).sort_values("F", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.barh(df_anova["feature"][::-1], df_anova["F"][::-1],
            color=["#E53935" if s else "#9E9E9E"
                   for s in df_anova["significant"][::-1]],
            alpha=0.85)
    ax.set_xlabel("F-statistic (ANOVA)  |  red = p < 0.05")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    neg_log_p = -np.log10(df_anova["p_value"].clip(1e-300))
    ax.barh(df_anova["feature"][::-1], neg_log_p[::-1],
            color=["#E53935" if s else "#9E9E9E"
                   for s in df_anova["significant"][::-1]],
            alpha=0.85)
    ax.axvline(-np.log10(0.05), color="gray", linestyle="--",
               linewidth=1, label="p=0.05")
    ax.set_xlabel("-log₁₀(p-value)  |  red = significant")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fp = OUT_DIR / "13_anova.png"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp.name}")

    df_anova.to_csv(OUT_DIR / "anova_results.csv", index=False)
    print(f"Saved: anova_results.csv")
    return df_anova


def plot_statistics_summary(df):
    feats = ["r_mean","g_mean","b_mean","brightness","contrast","saturation"]
    rows = []
    for feat in feats:
        row = {"feature": feat}
        for cid, name in enumerate(CLASSES):
            d = df.loc[df.class_id == cid, feat]
            row[f"{name[:6]}_mean"] = round(d.mean(), 2)
            row[f"{name[:6]}_std"]  = round(d.std(), 2)
        rows.append(row)
    df_stats = pd.DataFrame(rows)
    df_stats.to_csv(OUT_DIR / "statistics_per_class.csv", index=False)

    full_stats = df.groupby("class_name")[feats].agg(["mean","std"]).round(2)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    flat = full_stats.copy()
    flat.columns = [f"{c[0]}\n({c[1]})" for c in flat.columns]
    tbl = ax.table(cellText=flat.values.round(2),
                   rowLabels=flat.index,
                   colLabels=flat.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ccc")
        if r == 0 or c == -1:
            cell.set_facecolor("#1E3A5F")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")
    plt.tight_layout()
    fp = OUT_DIR / "14_statistics_table.png"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp.name}")


def plot_pairplot(df):
    feats = ["brightness","contrast","saturation","r_mean","b_mean"]
    df_sub = df[feats + ["class_name"]].copy()
    g = sns.pairplot(df_sub, hue="class_name", palette=PALETTE,
                     diag_kind="kde", plot_kws={"alpha": 0.4, "s": 12, "edgecolor": "none"},
                     diag_kws={"alpha": 0.6})
    fp = OUT_DIR / "15_pairplot.png"
    g.savefig(str(fp), bbox_inches="tight", dpi=120)
    plt.close("all")
    print(f"Saved: {fp.name}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("=== EDA — Herlev Cervical Cancer ===\n")

    print("1. Collecting image paths...")
    paths, labels = collect_paths()
    print(f"   Total: {len(paths)} cells in {len(CLASSES)} classes\n")

    print("2. Extracting features from each image...")
    df = extract_features(paths, labels)
    df.to_csv(OUT_DIR / "features.csv", index=False)
    print(f"   Features saved to features.csv ({df.shape})\n")

    print("3. Generating plots...\n")
    plot_class_distribution(df)
    plot_sample_images(paths, labels)
    plot_image_size_distribution(df)
    plot_rgb_distributions(df)
    plot_color_histograms_per_class(paths, labels)
    plot_boxplots(df)
    plot_correlation_matrix(df)
    X_sc, y = plot_pca(df)
    plot_tsne(X_sc, y)
    plot_umap(X_sc, y)
    plot_anova_results(df)
    plot_statistics_summary(df)
    plot_pairplot(df)

    print(f"\n=== Done! All files saved in: eda_output/ ===")


if __name__ == "__main__":
    main()
