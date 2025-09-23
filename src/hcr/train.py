from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from .config import load_config
from .utils import ensure_dir, save_json
from .trainer import train_and_eval
from .data.tabular import arrays_from_dataframe

def chrono_split(df: pd.DataFrame, date_col: str = 'arrival_date', test_size=0.25):
    """Split hotel bookings chronologically by arrival date."""
    if date_col not in df.columns:
        # Fallback to random split if no date column
        return train_test_split(df, test_size=test_size, random_state=42, stratify=df.get('is_canceled'))

    df = df.sort_values(date_col)
    n = len(df)
    n_test = max(1, int(round(n*test_size)))
    return df.iloc[:-n_test], df.iloc[-n_test:]

def main(config_path: str = "configs/exp_baseline.yaml"):
    cfg = load_config(config_path)

    # Load processed hotel booking features
    feats_path = Path(cfg.paths['artifacts']) / "features.csv"
    print(f"Loading features from {feats_path}")
    feats = pd.read_csv(feats_path)

    # Ensure target column exists
    target_col = cfg.train.get('target', 'is_canceled')
    if target_col not in feats.columns:
        raise ValueError(f"Target column '{target_col}' not found in features")

    # Remove rows with missing target
    feats = feats.dropna(subset=[target_col])
    print(f"Dataset shape: {feats.shape}, Cancellation rate: {feats[target_col].mean():.3f}")

    # Get feature columns (exclude target)
    if 'features' in cfg.train:
        Xcols = cfg.train['features']
    else:
        # Use all columns except target as features
        Xcols = [col for col in feats.columns if col != target_col]

    print(f"Using {len(Xcols)} features for training")

    # Split data chronologically if possible, otherwise randomly
    test_size = cfg.train.get('test_size', 0.25)
    train, test = chrono_split(feats, cfg.train.get('date_col'), test_size)

    print(f"Train size: {len(train)}, Test size: {len(test)}")

    # Convert to arrays
    Xtr, ytr = arrays_from_dataframe(train, Xcols, target_col)
    Xte, yte = arrays_from_dataframe(test, Xcols, target_col)

    # Train and evaluate model
    result = train_and_eval(Xtr, ytr, Xte, yte, cfg)
    model = result["model"]
    prob = result["prob"]
    pred = result["pred"]

    # Calculate comprehensive metrics for hotel cancellation
    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, prob)) if len(set(yte)) > 1 else None,
        "n_test": int(len(yte)),
        "n_train": int(len(ytr)),
        "features": Xcols,
        "cancellation_rate_test": float(yte.mean()),
        "cancellation_rate_train": float(ytr.mean()),
    }

    # Add PR-AUC (important for imbalanced data)
    if len(set(yte)) > 1:
        precision, recall, _ = precision_recall_curve(yte, prob)
        metrics["pr_auc"] = float(auc(recall, precision))

    # Save metrics
    out = Path(cfg.paths['artifacts']) / "metrics.json"
    ensure_dir(out.parent)
    save_json(metrics, out)

    # Save model checkpoint
    ckpt_dir = Path(cfg.paths['artifacts']) / "checkpoints"
    ensure_dir(ckpt_dir)
    ckpt_path = ckpt_dir / "model"
    try:
        if hasattr(model, "save"):
            try:
                model.save(str(ckpt_path))
            except Exception:
                model.save(str(ckpt_path.with_suffix('.bin')))
        else:
            # Fallback: pickle
            import pickle
            with open(ckpt_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(model, f)
    except Exception as e:
        print("Warning: failed to save model checkpoint:", e)

    print(f"Saved metrics to {out}")
    print(f"Model performance: ROC-AUC={metrics.get('roc_auc', 'N/A'):.3f}, PR-AUC={metrics.get('pr_auc', 'N/A'):.3f}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/exp_baseline.yaml")
