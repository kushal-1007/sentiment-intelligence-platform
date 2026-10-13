import json
import yaml
import sqlite3
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_corrections(db_path: str):
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM corrections WHERE used_in_training=0",
        con
    )
    con.close()
    logger.info(f"Loaded {len(df)} unused corrections from DB")
    return df


def load_replay_sample(splits_dir: Path, n: int = 2000):
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    sample   = train_df.sample(min(n, len(train_df)), random_state=42)
    logger.info(f"Loaded {len(sample)} replay samples from original training set")
    return sample


def build_finetune_df(corrections_df, replay_df):
    corrections_clean = pd.DataFrame({
        "review_text": corrections_df["text"],
        "sentiment":   corrections_df["correct_class"].astype(int)
    })
    replay_clean = pd.DataFrame({
        "review_text": replay_df["review_text"],
        "sentiment":   replay_df["sentiment"].astype(int)
    })
    combined = pd.concat([corrections_clean, replay_clean], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(
        f"Fine-tune set: {len(corrections_clean)} corrections "
        f"+ {len(replay_clean)} replay = {len(combined)} total"
    )
    return combined


class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts      = df["review_text"].tolist()
        self.labels     = df["sentiment"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }


def retrain(cfg, finetune_df):
    model_path = Path(cfg["training"]["output_dir"]) / "best"
    device     = torch.device(cfg["training"]["device"])
    logger.info(f"Loading model from {model_path} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model     = model.to(device)

    dataset = SentimentDataset(finetune_df, tokenizer, cfg["model"]["max_length"])
    loader  = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["training"]["learning_rate"]) * 0.5,
        weight_decay=cfg["training"]["weight_decay"]
    )
    total_steps  = len(loader) * 2
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    best_loss = float("inf")

    with mlflow.start_run(run_name=f"retrain-{datetime.now().strftime('%Y%m%d-%H%M')}"):
        mlflow.log_params({
            "retrain_samples":     len(finetune_df),
            "retrain_epochs":      2,
            "retrain_lr":          float(cfg["training"]["learning_rate"]) * 0.5,
            "retrain_batch_size":  16,
        })

        for epoch in range(1, 3):
            model.train()
            total_loss = 0
            all_preds, all_labels = [], []

            for batch in loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = loss_fn(outputs.logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                all_preds.extend(
                    outputs.logits.argmax(dim=-1).cpu().numpy()
                )
                all_labels.extend(labels.cpu().numpy())

            avg_loss = total_loss / len(loader)
            macro_f1 = f1_score(all_labels, all_preds, average="macro")

            logger.info(
                f"Epoch {epoch}/2 → Loss: {avg_loss:.4f} | F1: {macro_f1:.4f}"
            )
            mlflow.log_metrics(
                {"retrain_loss": avg_loss, "retrain_f1": macro_f1},
                step=epoch
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
                logger.info(f"Model saved to {model_path}")

        mlflow.log_metric("retrain_best_loss", best_loss)

    return best_loss


def mark_corrections_used(db_path: str, correction_ids: list):
    con = sqlite3.connect(db_path)
    ids = ",".join(str(i) for i in correction_ids)
    con.execute(
        f"UPDATE corrections SET used_in_training=1 WHERE id IN ({ids})"
    )
    con.commit()
    con.close()
    logger.info(f"Marked {len(correction_ids)} corrections as used")


def run_retraining():
    cfg       = load_config()
    db_path   = cfg["monitoring"]["db_path"]
    threshold = 20

    corrections_df = load_corrections(db_path)

    if len(corrections_df) < threshold:
        logger.warning(
            f"Only {len(corrections_df)} corrections — "
            f"need {threshold} to retrain. Exiting."
        )
        print(f"\n{'='*50}")
        print(f"  ⚠️  Not enough corrections to retrain")
        print(f"  Have : {len(corrections_df)}")
        print(f"  Need : {threshold}")
        print(f"{'='*50}\n")
        return

    replay_df    = load_replay_sample(
        Path(cfg["data"]["splits_dir"]), n=2000
    )
    finetune_df  = build_finetune_df(corrections_df, replay_df)
    best_loss    = retrain(cfg, finetune_df)

    mark_corrections_used(db_path, corrections_df["id"].tolist())

    summary = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "corrections_used":   len(corrections_df),
        "replay_samples":     min(2000, len(replay_df)),
        "total_finetune":     len(finetune_df),
        "best_retrain_loss":  round(float(best_loss), 4),
        "model_path":         str(
            Path(cfg["training"]["output_dir"]) / "best"
        )
    }

    out_path = Path("reports") / "retrain_summary.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  ✅ Retraining Complete")
    print(f"  Corrections used  : {summary['corrections_used']}")
    print(f"  Replay samples    : {summary['replay_samples']}")
    print(f"  Total fine-tune   : {summary['total_finetune']}")
    print(f"  Best loss         : {summary['best_retrain_loss']}")
    print(f"  Model saved       : {summary['model_path']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    run_retraining()