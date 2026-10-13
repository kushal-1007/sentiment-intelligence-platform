import json
import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_NAMES = ["1★", "2★", "3★", "4★", "5★"]


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg):
    model_path = Path(cfg["training"]["output_dir"]) / "best"
    logger.info(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    logger.info("Model loaded on CPU (required for SHAP)")
    return model, tokenizer

def predict_fn(texts, model, tokenizer, max_length):
    encoded = tokenizer(
        list(texts),
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1).numpy()
    return probs

def sample_reviews(cfg, n_per_class=2):
    splits_dir = Path(cfg["data"]["splits_dir"])
    test_df = pd.read_parquet(splits_dir / "test.parquet")

    samples = []
    for label in range(5):
        subset = test_df[test_df["sentiment"] == label]
        chosen = subset.sample(min(n_per_class, len(subset)), random_state=42)
        for _, row in chosen.iterrows():
            samples.append({
                "text": row["review_text"],
                "true_label": int(row["sentiment"])
            })

    logger.info(f"Sampled {len(samples)} reviews ({n_per_class} per class)")
    return samples

def explain_sample(text, true_label, sample_id, model, tokenizer, cfg, out_dir):
    max_length = cfg["model"]["max_length"]

    def _predict(texts):
        return predict_fn(texts, model, tokenizer, max_length)

    masker   = shap.maskers.Text(tokenizer=tokenizer)
    explainer = shap.Explainer(_predict, masker, output_names=LABEL_NAMES)

    logger.info(f"  Explaining sample {sample_id} (true={LABEL_NAMES[true_label]})...")
    shap_values = explainer([text], fixed_context=1, max_evals=200)

    probs     = _predict([text])[0]
    pred_class = int(probs.argmax())
    confidence = round(float(probs[pred_class]), 4)

    tokens = shap_values.data[0]
    values = shap_values.values[0, :, pred_class].tolist()

    token_list = [
        {"token": tok, "shap_value": round(float(val), 6)}
        for tok, val in zip(tokens, values)
    ]
    token_list_sorted = sorted(token_list, key=lambda x: abs(x["shap_value"]), reverse=True)

    result = {
        "sample_id":       sample_id,
        "text":            text[:300],
        "true_label":      LABEL_NAMES[true_label],
        "predicted_label": LABEL_NAMES[pred_class],
        "confidence":      confidence,
        "correct":         pred_class == true_label,
        "token_importance": token_list_sorted[:15]
    }

    json_path = out_dir / f"shap_sample_{sample_id}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Sample {sample_id}: True={LABEL_NAMES[true_label]} "
          f"Pred={LABEL_NAMES[pred_class]} Conf={confidence:.3f} "
          f"{'✅' if pred_class == true_label else '❌'}")
    print(f"  Top tokens: {[(t['token'], round(t['shap_value'],3)) for t in token_list_sorted[:5]]}")

    return result
def plot_top_tokens(all_results, out_dir):
    token_scores = {}
    for r in all_results:
        for item in r["token_importance"]:
            tok = item["token"].strip()
            if len(tok) < 2:
                continue
            token_scores[tok] = token_scores.get(tok, 0) + item["shap_value"]

    sorted_tokens = sorted(token_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
    tokens = [t[0] for t in sorted_tokens]
    values = [t[1] for t in sorted_tokens]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(tokens[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top 20 Tokens by Cumulative SHAP Value (all samples)", fontsize=12)
    ax.set_xlabel("Cumulative SHAP Value")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "top_shap_tokens.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def run_explainability():
    cfg     = load_config()
    out_dir = Path("reports/explainability")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(cfg)
    samples = sample_reviews(cfg, n_per_class=2)

    all_results = []
    for i, s in enumerate(samples):
        result = explain_sample(
            text=s["text"],
            true_label=s["true_label"],
            sample_id=i,
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            out_dir=out_dir
        )
        all_results.append(result)

    plot_top_tokens(all_results, out_dir)

    summary = {
        "total_samples":   len(all_results),
        "correct":         sum(1 for r in all_results if r["correct"]),
        "accuracy":        round(sum(1 for r in all_results if r["correct"]) / len(all_results), 3),
        "samples":         all_results
    }
    with open(out_dir / "shap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  ✅ Process 5 Complete")
    print(f"  Samples explained : {summary['total_samples']}")
    print(f"  Correct           : {summary['correct']}/{summary['total_samples']}")
    print(f"  Artifacts saved to: {out_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_explainability()




# if __name__ == "__main__":
    cfg = load_config()
    out_dir = Path("reports/explainability")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(cfg)
    samples = sample_reviews(cfg, n_per_class=2)

    # Test on first sample only
    result = explain_sample(
        text=samples[0]["text"],
        true_label=samples[0]["true_label"],
        sample_id=0,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        out_dir=out_dir
    )
    print("\n✅ Single sample SHAP working")