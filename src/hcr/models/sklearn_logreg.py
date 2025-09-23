from __future__ import annotations
from typing import Any
import pickle
import numpy as np
from .registry import register


class SKLogReg:
    def __init__(
        self,
        max_iter: int = 1000,
        random_state: int | None = 42,
        penalty: str = 'l2',
        C: float = 1.0,
        class_weight: str | dict | None = 'balanced',
        solver: str = 'liblinear'
    ):
        """
        Enhanced Logistic Regression baseline for hotel cancellation prediction.

        Args:
            max_iter: Maximum iterations for convergence
            random_state: Random seed for reproducibility
            penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
            C: Inverse regularization strength (smaller = stronger regularization)
            class_weight: Handle class imbalance ('balanced', dict, or None)
            solver: Algorithm for optimization ('liblinear', 'lbfgs', 'saga')
        """
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            penalty=penalty,
            C=C,
            class_weight=class_weight,
            solver=solver
        )
        self.preprocessor = None

    def fit(self, X: np.ndarray, y: np.ndarray, use_preprocessing: bool = True) -> "SKLogReg":
        """
        Fit the logistic regression model.

        Args:
            X: Feature matrix
            y: Target labels
            use_preprocessing: Whether to apply standard scaling
        """
        if use_preprocessing:
            from sklearn.preprocessing import StandardScaler
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder

            # Detect numeric and categorical columns
            if hasattr(X, 'dtypes'):  # pandas DataFrame
                numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            else:  # numpy array - assume all numeric
                numeric_cols = list(range(X.shape[1]))
                categorical_cols = []

            transformers = []
            if numeric_cols:
                transformers.append(('num', StandardScaler(), numeric_cols))
            if categorical_cols:
                transformers.append(('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols))

            if transformers:
                self.preprocessor = ColumnTransformer(transformers, remainder='passthrough')
                X_processed = self.preprocessor.fit_transform(X)
            else:
                X_processed = X
        else:
            X_processed = X

        self.model.fit(X_processed, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels."""
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score on test data."""
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        return self.model.score(X, y)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature coefficients as importance scores."""
        return np.abs(self.model.coef_[0]) if hasattr(self.model, 'coef_') else None

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str) -> "SKLogReg":
        with open(path, "rb") as f:
            model = pickle.load(f)
        obj = cls()
        obj.model = model
        return obj


@register("logreg_sklearn")
def _build_logreg_sklearn(**kwargs) -> Any:
    """Standard logistic regression baseline."""
    return SKLogReg(**kwargs)


@register("logreg_baseline")
def _build_logreg_baseline(**kwargs) -> Any:
    """Baseline logistic regression optimized for hotel cancellation prediction."""
    # Remove input_dim as sklearn models don't use it
    kwargs.pop('input_dim', None)
    defaults = {
        'penalty': 'l2',
        'C': 1.0,
        'class_weight': 'balanced',
        'solver': 'liblinear',
        'random_state': 42,
        'max_iter': 1000
    }
    defaults.update(kwargs)
    return SKLogReg(**defaults)


@register("logreg_l1")
def _build_logreg_l1(**kwargs) -> Any:
    """L1 regularized logistic regression for feature selection."""
    defaults = {
        'penalty': 'l1',
        'C': 0.1,
        'class_weight': 'balanced',
        'solver': 'liblinear',
        'random_state': 42,
        'max_iter': 1000
    }
    defaults.update(kwargs)
    return SKLogReg(**defaults)

