import json
import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
    classification_report, roc_auc_score
)
from sklearn.preprocessing import label_binarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_NAMES = ["1★", "2★", "3★", "4★", "5★"]


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def load_test_data(cfg):
    splits_dir = Path(cfg["data"]["splits_dir"])
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    test_df  = pd.read_parquet(splits_dir / "test.parquet")
    logger.info(f"Test set: {len(test_df):,} samples")
    return train_df, test_df


def get_transformer_predictions(test_df, cfg):
    model_path = Path(cfg["training"]["output_dir"]) / "best"
    device = torch.device(cfg["training"]["device"])
    logger.info(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    all_preds, all_probs = [], []
    batch_size = cfg["training"]["batch_size"]
    texts = test_df["review_text"].tolist()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(
                batch,
                max_length=cfg["model"]["max_length"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = probs.argmax(axis=-1)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

            if i % (batch_size * 10) == 0:
                logger.info(f"  Processed {min(i+batch_size, len(texts)):,}/{len(texts):,}")

    logger.info("✅ Transformer inference complete")
    return np.array(all_preds), np.array(all_probs)

def get_baseline_predictions(train_df, test_df):
    logger.info("Training TF-IDF + Logistic Regression baseline...")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df["review_text"])
    X_test  = vectorizer.transform(test_df["review_text"])

    clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
    clf.fit(X_train, train_df["sentiment"])

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    logger.info("✅ Baseline inference complete")
    return np.array(preds), np.array(probs)


def compute_metrics(y_true, y_pred, y_probs):
    y_bin   = label_binarize(y_true, classes=list(range(5)))
    roc_auc = roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr")
    return {
        "accuracy":      round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1":      round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1":   round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "roc_auc_macro": round(float(roc_auc), 4),
    }


def print_metrics(name, metrics):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Macro F1    : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1 : {metrics['weighted_f1']:.4f}")
    print(f"  ROC-AUC     : {metrics['roc_auc_macro']:.4f}")
    print(f"{'='*50}")

def plot_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
                linewidths=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_model_comparison(t_metrics, b_metrics, out_dir):
    metrics = ["accuracy", "macro_f1", "weighted_f1", "roc_auc_macro"]
    labels  = ["Accuracy", "Macro F1", "Weighted F1", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, [b_metrics[m] for m in metrics],
                width, label="TF-IDF + LogReg", color="#95a5a6", alpha=0.85)
    b2 = ax.bar(x + width/2, [t_metrics[m] for m in metrics],
                width, label="DistilBERT (fine-tuned)", color="#378ADD", alpha=0.85)

    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs Fine-tuned DistilBERT", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")

def save_results(y_true, t_preds, b_preds, t_metrics, b_metrics, out_dir):
    report = classification_report(
        y_true, t_preds,
        target_names=LABEL_NAMES,
        output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report.csv")

    results = {
        "distilbert": t_metrics,
        "baseline_tfidf": b_metrics,
        "improvement": {k: round(t_metrics[k] - b_metrics[k], 4) for k in t_metrics}
    }
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: evaluation_results.json")

    print("\n" + "="*50)
    print("  PER-CLASS F1 (DistilBERT)")
    print("="*50)
    for name, row in pd.DataFrame(report).transpose().iterrows():
        if name in LABEL_NAMES:
            print(f"  {name}  →  F1: {row['f1-score']:.4f}  Precision: {row['precision']:.4f}  Recall: {row['recall']:.4f}")
    print("="*50)

if __name__ == "__main__":
    cfg = load_config()
    out_dir = Path("reports/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_test_data(cfg)
    y_true = test_df["sentiment"].values

    t_preds, t_probs = get_transformer_predictions(test_df, cfg)
    b_preds, b_probs = get_baseline_predictions(train_df, test_df)

    t_metrics = compute_metrics(y_true, t_preds, t_probs)
    b_metrics = compute_metrics(y_true, b_preds, b_probs)

    print_metrics("DistilBERT (fine-tuned)", t_metrics)
    print_metrics("TF-IDF + Logistic Regression", b_metrics)

    plot_confusion_matrix(y_true, t_preds,
                          "DistilBERT Confusion Matrix (normalized)",
                          out_dir / "confusion_matrix_distilbert.png")
    plot_confusion_matrix(y_true, b_preds,
                          "TF-IDF Baseline Confusion Matrix (normalized)",
                          out_dir / "confusion_matrix_baseline.png")
    plot_model_comparison(t_metrics, b_metrics, out_dir)
    save_results(y_true, t_preds, b_preds, t_metrics, b_metrics, out_dir)

    logger.info("✅ Process 4 Complete — all evaluation artifacts saved")
    logger.info("Next → wait for training to finish, then re-run for final scores")