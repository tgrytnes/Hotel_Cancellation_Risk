from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from .config import load_config

def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Hotel Cancellation Prediction')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(precision, recall, pr_auc, save_path):
    """Plot Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Hotel Cancellation Prediction')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Hotel Cancellation Prediction')
    plt.colorbar()

    classes = ['Not Canceled', 'Canceled']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(config_path: str = "configs/exp_baseline.yaml"):
    cfg = load_config(config_path)

    # Load metrics
    metrics_path = Path(cfg.paths['artifacts']) / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    print("=== Hotel Cancellation Prediction Results ===")
    print(f"Dataset size: {metrics.get('n_train', 'N/A')} train, {metrics.get('n_test', 'N/A')} test")
    print(f"Cancellation rate: Train={metrics.get('cancellation_rate_train', 'N/A'):.3f}, Test={metrics.get('cancellation_rate_test', 'N/A'):.3f}")
    print(f"Number of features: {len(metrics.get('features', []))}")
    print()
    print("Performance Metrics:")
    print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
    print(f"  ROC-AUC:  {metrics.get('roc_auc', 'N/A'):.3f}")
    print(f"  PR-AUC:   {metrics.get('pr_auc', 'N/A'):.3f}")

    # Create figures directory
    fig_dir = Path(cfg.paths['artifacts']) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Plot basic metrics bar chart
    plot_metrics = {k: v for k, v in metrics.items()
                   if isinstance(v, (int, float)) and k in ["accuracy", "roc_auc", "pr_auc"] and v is not None}

    if plot_metrics:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(list(plot_metrics.keys()), list(plot_metrics.values()),
                      color=['steelblue', 'darkorange', 'green'])
        plt.title("Hotel Cancellation Prediction - Model Performance")
        plt.ylabel("Score")
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, plot_metrics.values()):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        plt.grid(True, alpha=0.3)
        metrics_plot_path = fig_dir / "model_metrics.pdf"
        plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved metrics plot to {metrics_plot_path}")

    # Load predictions if available (for advanced plots)
    try:
        # This would require modifying train.py to save predictions
        # For now, just indicate where advanced plots would go
        print(f"Note: For ROC/PR curves and confusion matrix, predictions need to be saved during training")
        print(f"Advanced plots would be saved to: {fig_dir}")
    except FileNotFoundError:
        print("Predictions file not found. Only basic metrics plotted.")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/exp_baseline.yaml")