from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np

from .models import build_model


def train_and_eval(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    cfg: Any,
) -> Dict[str, Any]:
    model_name = cfg.train.get("model", "logreg_sklearn")
    model_args = dict(cfg.train.get("model_args", {}))

    # Inject common args
    input_dim = Xtr.shape[1]
    # For future DL models, trainer configuration would go here
    # Currently only sklearn models are supported

    model = build_model(model_name, input_dim=input_dim, **model_args)
    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)

    return {
        "model": model,
        "pred": pred,
        "prob": prob,
    }

