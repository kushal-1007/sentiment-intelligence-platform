import json
import yaml
import sqlite3
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_predictions(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df  = pd.read_sql("SELECT * FROM predictions ORDER BY timestamp", con)
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(f"Loaded {len(df):,} prediction records from DB")
    return df

def split_reference_current(df: pd.DataFrame):
    mid = len(df) // 2
    reference = df.iloc[:mid].copy()
    current   = df.iloc[mid:].copy()
    logger.info(f"Reference: {len(reference)} records | Current: {len(current)} records")
    return reference, current


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "char_len":        df["char_len"].values,
        "confidence":      df["confidence"].values,
        "predicted_class": df["predicted_class"].values,
    })


def generate_drift_report(reference_df, current_df, out_dir: Path):
    ref = prepare_features(reference_df)
    cur = prepare_features(current_df)

    features = ["confidence", "char_len", "predicted_class"]
    drift_results = {}

    for feat in features:
        ref_mean = ref[feat].mean()
        cur_mean = cur[feat].mean()
        ref_std  = ref[feat].std()
        shift    = abs(cur_mean - ref_mean)
        drifted  = shift > 0.5 * ref_std
        drift_results[feat] = {
        "ref_mean": round(float(ref_mean), 4),
        "cur_mean": round(float(cur_mean), 4),
        "shift":    round(float(shift), 4),
        "drifted":  bool(drifted)
      }
        logger.info(f"  {feat}: ref={ref_mean:.3f} cur={cur_mean:.3f} "
                    f"shift={shift:.3f} {'⚠️ DRIFT' if drifted else '✅ OK'}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Drift Monitoring Report", fontsize=13, fontweight="bold")

    colors = {"reference": "#378ADD", "current": "#E24B4A"}

    for ax, feat in zip(axes, features):
        ax.hist(ref[feat], bins=10, alpha=0.6, label="Reference",
                color=colors["reference"], edgecolor="white")
        ax.hist(cur[feat], bins=10, alpha=0.6, label="Current",
                color=colors["current"], edgecolor="white")
        d = drift_results[feat]
        status = "DRIFT" if d["drifted"] else "OK"
        ax.set_title(f"{feat}\n{status}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / f"drift_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {plot_path}")

    return plot_path, drift_results


def check_confidence_alert(current_df, threshold: float) -> bool:
    avg = current_df["confidence"].mean()
    if avg < threshold:
        logger.warning(f"⚠️  Confidence alert: avg={avg:.3f} < threshold={threshold}")
        return True
    logger.info(f"Confidence OK: avg={avg:.3f}")
    return False


def save_summary(df, reference_df, current_df, plot_path, alert, out_dir):
    summary = {
       "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_predictions": len(df),
        "reference_size":    len(reference_df),
        "current_size":      len(current_df),
        "confidence_alert":  alert,
        "avg_confidence":    round(float(current_df["confidence"].mean()), 4),
        "prediction_dist":   current_df["predicted_label"].value_counts().to_dict(),
        "plot_path":         str(plot_path),
    }
    return summary


def run_monitoring():
    cfg       = load_config()
    db_path   = cfg["monitoring"]["db_path"]
    out_dir   = Path(cfg["monitoring"]["drift_report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = cfg["monitoring"]["confidence_threshold"]

    df = load_predictions(db_path)

    if len(df) < 10:
        logger.warning("Not enough predictions (need >= 10)")
        return

    reference_df, current_df = split_reference_current(df)
    plot_path, drift_results = generate_drift_report(reference_df, current_df, out_dir)
    alert                    = check_confidence_alert(current_df, threshold)

    any_drift = any(v["drifted"] for v in drift_results.values())
    summary   = save_summary(df, reference_df, current_df, plot_path, alert, out_dir)
    summary["drift_results"] = drift_results
    summary["any_drift"]     = any_drift

    with open(out_dir / "monitoring_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  ✅ Process 8 Complete")
    print(f"  Total predictions : {len(df)}")
    print(f"  Avg confidence    : {round(float(current_df['confidence'].mean()), 4)}")
    print(f"  Drift detected    : {any_drift}")
    print(f"  Confidence alert  : {alert}")
    print(f"  Plot saved to     : {plot_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    run_monitoring()