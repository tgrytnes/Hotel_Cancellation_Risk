from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import yaml
import pandas as pd

plt.style.use('default')
sns.set_palette("husl")

def load_config(path: str):
    """Simple YAML config loader."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_experiment_results(artifacts_dir: Path):
    """Load predictions, metrics, and feature importance from experiment."""
    results = {}

    # Load predictions
    pred_path = artifacts_dir / "predictions.json"
    if pred_path.exists():
        with open(pred_path) as f:
            results['predictions'] = json.load(f)

    # Load metrics
    metrics_path = artifacts_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            results['metrics'] = json.load(f)

    # Load feature importance
    importance_path = artifacts_dir / "feature_importance.json"
    if importance_path.exists():
        with open(importance_path) as f:
            results['importance'] = json.load(f)

    return results

def load_all_experiments(base_dir: Path = None):
    """Load results from all available experiment artifacts directories."""
    if base_dir is None:
        base_dir = Path("..")

    experiments = {}

    artifact_dirs = {
        'rf_300': base_dir / 'artifacts_rf_300',
        'l2_moderate': base_dir / 'artifacts_l2_moderate',
        'xgb_lr1_d5': base_dir / 'artifacts_xgb_lr1_d5'
    }

    for exp_name, artifacts_dir in artifact_dirs.items():
        if artifacts_dir.exists():
            experiments[exp_name] = load_experiment_results(artifacts_dir)
            print(f"✓ Loaded {exp_name} model results")

    print(f"\nReady to generate visualizations for {len(experiments)} models...")
    return experiments

def plot_confusion_matrices(experiments: dict, save_path: Path):
    """Plot confusion matrices for all models side by side."""
    n_models = len(experiments)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axes = [axes]

    for idx, (exp_name, exp_data) in enumerate(experiments.items()):
        if 'predictions' not in exp_data:
            continue

        preds = exp_data['predictions']
        y_true = np.array(preds['y_true'])
        y_pred = np.array(preds['y_pred'])

        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'{exp_name}\nConfusion Matrix', fontsize=12)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Labels
        classes = ['Not Canceled', 'Canceled']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

        # Add text annotations with counts and percentages
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            text_color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f'{cm[i, j]:,}\n({cm_percent[i, j]:.1f}%)',
                   horizontalalignment="center", verticalalignment="center",
                   color=text_color, fontsize=10)

        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_comparison(experiments: dict, save_path: Path):
    """Plot ROC curves for all models on the same plot."""
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for idx, (exp_name, exp_data) in enumerate(experiments.items()):
        if 'predictions' not in exp_data:
            continue

        preds = exp_data['predictions']
        y_true = np.array(preds['y_true'])
        y_prob = np.array(preds['y_prob'])

        if len(set(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            color = colors[idx % len(colors)]
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{exp_name} (AUC = {roc_auc:.3f})')

    # Plot random classifier line
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--',
             label='Random Classifier (AUC = 0.500)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Hotel Cancellation Prediction\nModel Comparison', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_comparison(experiments: dict, save_path: Path):
    """Plot Precision-Recall curves for all models with baseline."""
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    baseline_precision = None

    for idx, (exp_name, exp_data) in enumerate(experiments.items()):
        if 'predictions' not in exp_data:
            continue

        preds = exp_data['predictions']
        y_true = np.array(preds['y_true'])
        y_prob = np.array(preds['y_prob'])

        if baseline_precision is None:
            baseline_precision = y_true.mean()  # Positive class rate

        if len(set(y_true)) > 1:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)

            color = colors[idx % len(colors)]
            plt.plot(recall, precision, color=color, lw=2,
                    label=f'{exp_name} (AUC = {pr_auc:.3f})')

    # Plot baseline (random classifier for imbalanced data)
    plt.axhline(y=baseline_precision, color='gray', linestyle='--', lw=2,
                label=f'Baseline (Random = {baseline_precision:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves - Hotel Cancellation Prediction\nModel Comparison (Class Imbalance: {baseline_precision:.1%} Cancellations)', fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(experiments: dict, save_path: Path, top_k: int = 20):
    """Plot feature importance from Random Forest or tree-based models."""
    # Find experiment with feature importance (typically Random Forest)
    importance_data = None
    model_name = None

    for exp_name, exp_data in experiments.items():
        if 'importance' in exp_data:
            importance_data = exp_data['importance']
            model_name = exp_name
            break

    if importance_data is None:
        print("No feature importance data found. Train a Random Forest or XGBoost model first.")
        return

    features = importance_data['feature_names']
    importances = np.array(importance_data['importance_values'])

    # Sort features by importance
    indices = np.argsort(importances)[::-1]

    # Take top K features
    top_indices = indices[:top_k]
    top_features = [features[i] for i in top_indices]
    top_importances = importances[top_indices]

    # Create horizontal bar plot
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_features))

    bars = plt.barh(y_pos, top_importances, color='skyblue', edgecolor='navy', alpha=0.7)

    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_importances)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', ha='left', va='center', fontsize=9)

    plt.yticks(y_pos, top_features)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_k} Feature Importance - {model_name}\nHotel Cancellation Prediction', fontsize=14)
    plt.gca().invert_yaxis()  # Top features at top
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_model_comparison_report(experiments: dict, save_dir: Path):
    """Create a comprehensive comparison report of all models."""
    report_path = save_dir / "model_comparison_report.txt"

    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("HOTEL CANCELLATION PREDICTION - MODEL COMPARISON REPORT\n")
        f.write("="*60 + "\n\n")

        for exp_name, exp_data in experiments.items():
            f.write(f"MODEL: {exp_name.upper()}\n")
            f.write("-" * 40 + "\n")

            if 'metrics' in exp_data:
                metrics = exp_data['metrics']
                f.write(f"Accuracy:     {metrics.get('accuracy', 'N/A'):.3f}\n")
                f.write(f"ROC-AUC:      {metrics.get('roc_auc', 'N/A'):.3f}\n")
                f.write(f"PR-AUC:       {metrics.get('pr_auc', 'N/A'):.3f}\n")
                f.write(f"Train Size:   {metrics.get('n_train', 'N/A'):,}\n")
                f.write(f"Test Size:    {metrics.get('n_test', 'N/A'):,}\n")
                f.write(f"Features:     {len(metrics.get('features', []))}\n")
                f.write(f"Cancel Rate:  {metrics.get('cancellation_rate_test', 'N/A'):.3f}\n")

            f.write("\n")

        f.write("="*60 + "\n")
        f.write("VISUALIZATION FILES GENERATED:\n")
        f.write("- confusion_matrices.pdf\n")
        f.write("- roc_comparison.pdf\n")
        f.write("- precision_recall_comparison.pdf\n")
        f.write("- feature_importance.pdf\n")
        f.write("="*60 + "\n")

def main(config_paths: list = None):
    """Generate comprehensive model visualizations."""
    if config_paths is None:
        # Default to baseline config
        config_paths = ["configs/exp_baseline.yaml"]

    experiments = {}

    # Load results from each experiment
    for config_path in config_paths:
        cfg = load_config(config_path)
        artifacts_dir = Path(cfg['paths']['artifacts'])

        exp_name = Path(config_path).stem.replace('exp_', '')
        experiments[exp_name] = load_experiment_results(artifacts_dir)

    if not experiments:
        print("No experiment results found. Run training first.")
        return

    # Create figures directory
    first_cfg = load_config(config_paths[0])
    fig_dir = Path(first_cfg['paths']['artifacts']) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating visualizations for {len(experiments)} models...")

    # Generate all visualizations
    plot_confusion_matrices(experiments, fig_dir / "confusion_matrices.pdf")
    print("✓ Confusion matrices saved")

    plot_roc_comparison(experiments, fig_dir / "roc_comparison.pdf")
    print("✓ ROC comparison saved")

    plot_precision_recall_comparison(experiments, fig_dir / "precision_recall_comparison.pdf")
    print("✓ Precision-Recall comparison saved")

    plot_feature_importance(experiments, fig_dir / "feature_importance.pdf")
    print("✓ Feature importance saved")

    create_model_comparison_report(experiments, fig_dir)
    print("✓ Model comparison report saved")

    print(f"\nAll visualizations saved to: {fig_dir}")
    print("Files generated:")
    for file in fig_dir.glob("*.pdf"):
        print(f"  - {file.name}")
    print(f"  - model_comparison_report.txt")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Multiple config files provided
        config_paths = sys.argv[1:]
    else:
        # Look for multiple experiment configs
        config_dir = Path("configs")
        if config_dir.exists():
            config_paths = list(config_dir.glob("exp_*.yaml"))
            config_paths = [str(p) for p in config_paths]
        else:
            config_paths = ["configs/exp_baseline.yaml"]

    main(config_paths)