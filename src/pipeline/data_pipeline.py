import os
import json
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_dataset(cfg):
    logger.info("Downloading Yelp Review Full from HuggingFace...")
    dataset = load_dataset(cfg["data"]["dataset_name"])

    dfs = []
    for split in ["train", "test"]:
        df = dataset[split].to_pandas()
        df["source_split"] = split
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.rename(columns={"text": "review_text", "label": "sentiment"})

    logger.info(f"Total records: {len(full_df):,}")
    logger.info(f"Label distribution:\n{full_df['sentiment'].value_counts().sort_index()}")
    return full_df
def sample_dataset(df, cfg):
    max_samples = cfg["data"]["max_samples"]
    seed = cfg["data"]["random_seed"]
    logger.info(f"Sampling {max_samples:,} records (stratified)...")
    _, df_sampled = train_test_split(
        df,
        test_size=max_samples,
        stratify=df["sentiment"],
        random_state=seed
    )
    return df_sampled.reset_index(drop=True)


def clean_text(df):
    logger.info("Cleaning text...")
    df = df.copy()
    df["review_text"] = (
        df["review_text"]
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[^\x00-\x7F]+", " ", regex=True)
    )
    before = len(df)
    df = df[df["review_text"].str.len() > 10].reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} empty/short reviews")
    return df


def split_dataset(df, cfg):
    seed = cfg["data"]["random_seed"]
    test_ratio = cfg["data"]["test_size"]
    val_ratio = cfg["data"]["val_size"]

    train_val, test = train_test_split(
        df, test_size=test_ratio,
        stratify=df["sentiment"], random_state=seed
    )
    val_ratio_adjusted = val_ratio / (1 - test_ratio)
    train, val = train_test_split(
        train_val, test_size=val_ratio_adjusted,
        stratify=train_val["sentiment"], random_state=seed
    )
    logger.info(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def save_splits(train, val, test, cfg):
    splits_dir = Path(cfg["data"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(splits_dir / "train.parquet", index=False)
    val.to_parquet(splits_dir / "val.parquet", index=False)
    test.to_parquet(splits_dir / "test.parquet", index=False)

    weights = compute_class_weight(
        "balanced",
        classes=np.unique(train["sentiment"]),
        y=train["sentiment"].values
    )
    class_weights = {int(c): round(float(w), 6) for c, w in enumerate(weights)}

    metadata = {
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "class_weights": class_weights,
        "label_distribution": {
            "train": train["sentiment"].value_counts().sort_index().to_dict(),
        }
    }

    with open(splits_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Splits saved to {splits_dir}/")
    logger.info(f"Class weights: {class_weights}")
    return metadata


def run_pipeline():
    cfg = load_config()
    df = download_dataset(cfg)
    df = sample_dataset(df, cfg)
    df = clean_text(df)
    train, val, test = split_dataset(df, cfg)
    metadata = save_splits(train, val, test, cfg)
    logger.info("✅ Process 1 Complete")
    return metadata


if __name__ == "__main__":
    run_pipeline()