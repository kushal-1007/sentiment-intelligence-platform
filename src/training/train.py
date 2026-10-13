import json, yaml, logging
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
import mlflow
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

class YelpDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts  = df["review_text"].tolist()
        self.labels = df["sentiment"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_model(cfg):
    logger.info(f"Loading {cfg['model']['base_model']}...")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["base_model"],
        num_labels=cfg["model"]["num_labels"],
    )

    if cfg["model"]["freeze_base_layers"]:
        # freeze embeddings
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False

        # freeze transformer blocks 0-3, keep 4-5 trainable
        for i, layer in enumerate(model.distilbert.transformer.layer):
            if i < 4:
                for param in layer.parameters():
                    param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    print("\n" + "="*50)
    print("  MODEL SUMMARY")
    print("="*50)
    print(f"  Base model     : {cfg['model']['base_model']}")
    print(f"  Num labels     : {cfg['model']['num_labels']}")
    print(f"  Total params   : {total:,}")
    print(f"  Trainable      : {trainable:,}")
    print(f"  Frozen         : {total - trainable:,}")
    print(f"  Trainable %    : {trainable/total*100:.1f}%")
    print("="*50 + "\n")

    return model

def train_epoch(model, loader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, macro_f1, accuracy

@torch.no_grad()
def eval_epoch(model, loader, device, loss_fn):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels)

        total_loss += loss.item()
        all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, macro_f1, accuracy

def run_training():
    cfg    = load_config()
    device = torch.device(cfg["training"]["device"])
    logger.info(f"Using device: {device}")

    # ── Load data ──
    splits_dir = Path(cfg["data"]["splits_dir"])
    train_df   = pd.read_parquet(splits_dir / "train.parquet")
    val_df     = pd.read_parquet(splits_dir / "val.parquet")

    with open(splits_dir / "metadata.json") as f:
        metadata = json.load(f)

    # ── Class weights ──
    weights = metadata["class_weights"]
    class_weights = torch.tensor(
        [weights[str(i)] for i in range(cfg["model"]["num_labels"])],
        dtype=torch.float
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # ── Tokenizer + Datasets ──
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    train_ds  = YelpDataset(train_df, tokenizer, cfg["model"]["max_length"])
    val_ds    = YelpDataset(val_df,   tokenizer, cfg["model"]["max_length"])

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── Model ──
    model = build_model(cfg).to(device)

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=float(cfg["training"]["learning_rate"]),
    weight_decay=float(cfg["training"]["weight_decay"]),
    )
    total_steps   = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps  = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── MLflow ──
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    output_dir = Path(cfg["training"]["output_dir"]) / "best"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1     = 0.0
    patience_counter = 0
    patience         = cfg["training"]["early_stopping_patience"]

    with mlflow.start_run(run_name="distilbert-finetune"):
        mlflow.log_params({
            "model":      cfg["model"]["base_model"],
            "epochs":     cfg["training"]["epochs"],
            "batch_size": cfg["training"]["batch_size"],
            "lr":         cfg["training"]["learning_rate"],
            "max_length": cfg["model"]["max_length"],
            "device":     cfg["training"]["device"],
        })

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            print(f"\n{'─'*50}")
            print(f"  Epoch {epoch}/{cfg['training']['epochs']}")
            print(f"{'─'*50}")

            train_loss, train_f1, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, device, loss_fn)
            val_loss, val_f1, val_acc = eval_epoch(
                model, val_loader, device, loss_fn)

            print(f"  Train → Loss: {train_loss:.4f} | F1: {train_f1:.4f} | Acc: {train_acc:.4f}")
            print(f"  Val   → Loss: {val_loss:.4f} | F1: {val_f1:.4f} | Acc: {val_acc:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss, "train_f1": train_f1, "train_acc": train_acc,
                "val_loss":   val_loss,   "val_f1":   val_f1,   "val_acc":   val_acc,
            }, step=epoch)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print(f"  ✅ Best model saved (val_f1={val_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("  Early stopping triggered.")
                    break

        mlflow.log_metric("best_val_f1", best_val_f1)

    print(f"\n{'='*50}")
    print(f"  ✅ Process 3 Complete")
    print(f"  Best Val F1 : {best_val_f1:.4f}")
    print(f"  Model saved : {output_dir}")
    print(f"{'='*50}\n") 

    
if __name__ == "__main__":
    run_training()