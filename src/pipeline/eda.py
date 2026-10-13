import json
import yaml
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_NAMES = ["1★", "2★", "3★", "4★", "5★"]
PALETTE = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_splits(cfg):
    splits_dir = Path(cfg["data"]["splits_dir"])
    train = pd.read_parquet(splits_dir / "train.parquet")
    val   = pd.read_parquet(splits_dir / "val.parquet")
    test  = pd.read_parquet(splits_dir / "test.parquet")
    logger.info(f"Loaded — Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test

def plot_class_distribution(train, val, test, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Class Distribution Across Splits", fontsize=13, fontweight="bold")

    for ax, (df, name) in zip(axes, [(train, "Train"), (val, "Validation"), (test, "Test")]):
        counts = df["sentiment"].value_counts().sort_index()
        bars = ax.bar(LABEL_NAMES, counts.values, color=PALETTE, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{name} ({len(df):,} samples)", fontsize=11)
        ax.set_xlabel("Sentiment Class")
        ax.set_ylabel("Count")
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 30,
                    f"{count:,}", ha="center", va="bottom", fontsize=9) 
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")

def print_class_distribution(train, val, test):
    print("\n" + "="*50)
    print("  CLASS DISTRIBUTION")
    print("="*50)
    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"\n  {name} ({len(df):,} samples)")
        print("  " + "-"*36)
        counts = df["sentiment"].value_counts().sort_index()
        max_count = counts.max()
        stars = ["1★", "2★", "3★", "4★", "5★"]
        colors = ["\033[91m", "\033[93m", "\033[93m", "\033[92m", "\033[92m"]
        reset = "\033[0m"
        for i, (label, count) in enumerate(zip(stars, counts)):
            bar_len = int((count / max_count) * 25)
            bar = "█" * bar_len
            print(f"  {label}  {colors[i]}{bar:<25}{reset}  {count:,}")
    print("\n" + "="*50 + "\n")

def plot_review_length(train, out_dir):
    train = train.copy()
    train["char_len"] = train["review_text"].str.len()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Review Length by Sentiment Class", fontsize=13, fontweight="bold")

    # Box plot
    data_by_class = [train[train["sentiment"] == i]["char_len"].values for i in range(5)]
    axes[0].boxplot(data_by_class, tick_labels=LABEL_NAMES, patch_artist=True,
                    boxprops=dict(facecolor="#85B7EB", alpha=0.7),
                    medianprops=dict(color="red", linewidth=2))
    axes[0].set_title("Box plot — length per class")
    axes[0].set_ylabel("Characters")
    axes[0].set_xlabel("Sentiment Class")
    axes[0].grid(axis="y", alpha=0.3)

    # KDE
    for i, (name, color) in enumerate(zip(LABEL_NAMES, PALETTE)):
        subset = train[train["sentiment"] == i]["char_len"]
        subset.clip(upper=3000).plot.kde(ax=axes[1], label=name, color=color, linewidth=2)
    axes[1].set_title("KDE — length density (clipped at 3000)")
    axes[1].set_xlabel("Character length")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "review_length_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")

    stats = train.groupby("sentiment")["char_len"].agg(["mean", "median"]).round(1)
    print("\n" + "="*50)
    print("  REVIEW LENGTH STATS (chars)")
    print("="*50)
    print(f"  {'Class':<8} {'Mean':>8} {'Median':>8}")
    print("  " + "-"*26)
    for i, row in stats.iterrows():
        print(f"  {LABEL_NAMES[i]:<8} {row['mean']:>8.0f} {row['median']:>8.0f}")
    print("="*50 + "\n")

def plot_token_length(train, cfg, out_dir):
    logger.info("Tokenizing 5,000 sample reviews — this takes ~2 min...")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    sample = train["review_text"].sample(5000, random_state=42).tolist()
    token_lengths = [len(tokenizer.encode(text, truncation=False)) for text in sample]

    import numpy as np
    token_lengths = np.array(token_lengths)
    p50 = int(np.percentile(token_lengths, 50))
    p90 = int(np.percentile(token_lengths, 90))
    p95 = int(np.percentile(token_lengths, 95))
    max_len = cfg["model"]["max_length"]
    coverage = round(float((token_lengths <= max_len).mean() * 100), 2)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(token_lengths, bins=60, color="#85B7EB", edgecolor="white", linewidth=0.4)
    ax.axvline(max_len, color="#e74c3c", linewidth=2, linestyle="--", label=f"max_length={max_len}")
    ax.axvline(p50, color="#f39c12", linewidth=1.5, linestyle=":", label=f"P50={p50}")
    ax.axvline(p90, color="#8e44ad", linewidth=1.5, linestyle=":", label=f"P90={p90}")
    ax.axvline(p95, color="#2c3e50", linewidth=1.5, linestyle=":", label=f"P95={p95}")
    ax.set_title("Token Length Distribution (DistilBERT tokenizer, 5k sample)", fontsize=12)
    ax.set_xlabel("Token count")
    ax.set_ylabel("Number of reviews")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "token_length_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")

    print("\n" + "="*50)
    print("  TOKEN LENGTH STATS")
    print("="*50)
    print(f"  P50  (median) : {p50} tokens")
    print(f"  P90           : {p90} tokens")
    print(f"  P95           : {p95} tokens")
    print(f"  max_length    : {max_len} tokens")
    print(f"  Coverage      : {coverage}% reviews fit within max_length")
    if p90 > max_len:
        print(f"  ⚠️  P90 exceeds max_length — consider increasing to 512")
    else:
        print(f"  ✅  max_length={max_len} is sufficient")
    print("="*50 + "\n")

    return {"p50": p50, "p90": p90, "p95": p95, "coverage_pct": coverage}

def plot_length_heatmap(train, out_dir):
    train = train.copy()
    train["char_len"] = train["review_text"].str.len()

    stats = train.groupby("sentiment")["char_len"].agg(["mean", "median", "std"]).round(1)
    stats.index = LABEL_NAMES

    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(stats.T, annot=True, fmt=".0f", cmap="Blues", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Characters"})
    ax.set_title("Review Length Stats by Sentiment Class", fontsize=12)
    ax.set_xlabel("Sentiment Class")
    plt.tight_layout()
    path = out_dir / "length_stats_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")

def save_eda_summary(token_stats, train, out_dir):
    summary = {
        "token_length": token_stats,
        "review_char_length": {
            "mean_by_class": train.assign(
                char_len=train["review_text"].str.len()
            ).groupby("sentiment")["char_len"].mean().round(1).to_dict()
        },
        "class_counts": train["sentiment"].value_counts().sort_index().to_dict()
    }
    path = out_dir / "eda_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved: {path}")

if __name__ == "__main__":
    cfg = load_config()
    out_dir = Path("reports/eda")
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = load_splits(cfg)

    plot_class_distribution(train, val, test, out_dir)
    print_class_distribution(train, val, test)
    logger.info("✅ Plot 1 done — class_distribution.png")

    plot_review_length(train, out_dir)
    logger.info("✅ Plot 2 done — review_length_distribution.png")

    token_stats = plot_token_length(train, cfg, out_dir)
    logger.info("✅ Plot 3 done — token_length_distribution.png")

    plot_length_heatmap(train, out_dir)
    logger.info("✅ Plot 4 done — length_stats_heatmap.png")

    save_eda_summary(token_stats, train, out_dir)
    logger.info("✅ EDA summary saved — eda_summary.json")

    logger.info("✅ Process 2 Complete — all EDA artifacts saved to reports/eda/")
    logger.info("Next → Process 3: python src/training/train.py")