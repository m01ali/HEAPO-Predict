"""ANN training using sklearn MLPRegressor — no PyTorch dependency.

PyTorch 2.x on Python 3.14 + Apple M1 hangs on first CPU kernel execution
due to internal thread-pool or MPS shader compilation issues. sklearn's
MLPRegressor is pure numpy, always fast, and has zero GPU/MPS complexity.

Architecture: Input(n) → Dense(128) → ReLU → Dense(64) → ReLU → Dense(1)
Note: sklearn has no Dropout layer; L2 regularisation (alpha) serves a similar
      regularisation role.

Target convention: log1p(kWh) — same as PyTorch MLP. Back-transform via
    np.expm1(pred).clip(0).
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wrapper — keeps the same calling interface as the original PyTorch MLP
# ---------------------------------------------------------------------------

class ANN:
    """Thin wrapper around a fitted sklearn MLPRegressor.

    Exposes eval() / train() / __call__() so the training script can
    call it identically to the old PyTorch MLP wrapper.
    """

    def __init__(self, sklearn_mlp: MLPRegressor) -> None:
        self._model = sklearn_mlp

    # ---- sklearn has no train/eval mode distinction ----
    def eval(self) -> None:   # noqa: D401
        pass

    def train(self) -> None:  # noqa: D401
        pass

    def __call__(self, X: Any) -> np.ndarray:
        """Forward pass.  X may be a numpy array or a torch Tensor."""
        if hasattr(X, "numpy"):          # torch.Tensor
            X = X.detach().cpu().numpy()
        elif hasattr(X, "values"):       # pandas DataFrame
            X = X.values
        return self._model.predict(np.asarray(X, dtype=np.float32))

    def predict(self, X: Any) -> np.ndarray:
        """sklearn-compatible predict — delegates to __call__."""
        return self(X)

    # ---- save / load (joblib handles sklearn; these are for compatibility) ----
    def state_dict(self) -> dict:
        return {
            "coefs_":       [c.copy() for c in self._model.coefs_],
            "intercepts_":  [i.copy() for i in self._model.intercepts_],
        }

    def load_state_dict(self, sd: dict) -> None:
        self._model.coefs_      = [c.copy() for c in sd["coefs_"]]
        self._model.intercepts_ = [i.copy() for i in sd["intercepts_"]]


# ---------------------------------------------------------------------------
# fit_ann — same signature as the removed PyTorch version
# ---------------------------------------------------------------------------

def fit_ann(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    patience: int = 15,
    random_state: int = 42,
    hidden_layers: list[int] | None = None,
    dropouts: list[float] | None = None,   # accepted but ignored (sklearn has no dropout)
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 200,
) -> tuple[ANN, list[float], list[float], dict]:
    """Train a 2-layer MLP on log1p(kWh) using sklearn MLPRegressor.

    Args:
        X_train / X_val : Scaled numpy arrays (StandardScaler already applied).
        y_train / y_val : Log1p-transformed target.
        feature_names   : Column names (stored in metadata).
        patience        : Early-stopping patience in epochs.
        random_state    : RNG seed.
        hidden_layers   : Hidden layer sizes. Default [128, 64].
        dropouts        : Ignored (sklearn has no Dropout; kept for API compat).
        learning_rate   : Adam initial LR. Default 1e-3.
        batch_size      : Mini-batch size. Default 256.
        max_epochs      : Hard cap on training epochs. Default 200.

    Returns:
        (fitted_ann, train_losses, val_losses, meta_dict)
    """
    if hidden_layers is None:
        hidden_layers = [128, 64]

    np.random.seed(random_state)

    logger.info(
        f"ANN (sklearn MLPRegressor, no PyTorch)  "
        f"train={X_train.shape}  val={X_val.shape}"
    )
    logger.info(
        f"Architecture: {hidden_layers}  lr={learning_rate}  "
        f"batch={batch_size}  max_epochs={max_epochs}  patience={patience}"
    )

    # Use a random subsample of train to compute train-MSE each epoch cheaply.
    # (646k predict each epoch would add ~0.5 s; subsample adds ~0.05 s)
    TRAIN_SAMPLE = 20_000
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(X_train), size=min(TRAIN_SAMPLE, len(X_train)), replace=False)
    X_tr_sample = X_train[sample_idx]
    y_tr_sample = y_train[sample_idx]

    # Build model — max_iter=1 + warm_start=True means 1 epoch per fit() call.
    # tol and n_iter_no_change are set to disable sklearn's own early stopping;
    # we handle it manually using our held-out val set.
    mlp_sk = MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layers),
        activation="relu",
        solver="adam",
        alpha=1e-5,            # L2 weight decay (no dropout in sklearn)
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        learning_rate="constant",
        max_iter=1,            # 1 epoch per fit() call
        warm_start=True,       # continue from previous weights
        tol=1e-10,             # effectively disable sklearn's convergence check
        n_iter_no_change=max_epochs + 10,  # disable sklearn's internal patience
        early_stopping=False,
        verbose=False,
        random_state=random_state,
    )

    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_sd: dict | None = None
    stopped_epoch = max_epochs

    t0 = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress ConvergenceWarning (expected with max_iter=1)

        for epoch in range(1, max_epochs + 1):
            mlp_sk.fit(X_train, y_train)

            # --- losses ---
            train_pred = mlp_sk.predict(X_tr_sample)
            val_pred   = mlp_sk.predict(X_val)
            train_loss = float(mean_squared_error(y_tr_sample, train_pred))
            val_loss   = float(mean_squared_error(y_val,       val_pred))

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            elapsed = time.perf_counter() - t0
            logger.info(
                f"  ANN Epoch {epoch:3d}/{max_epochs}: "
                f"train_MSE={train_loss:.4f}  val_MSE={val_loss:.4f}  "
                f"patience={patience_counter}/{patience}  elapsed={elapsed:.0f}s"
            )

            # --- early stopping ---
            if val_loss < best_val_loss - 1e-6:
                best_val_loss    = val_loss
                patience_counter = 0
                best_sd = {
                    "coefs_":      [c.copy() for c in mlp_sk.coefs_],
                    "intercepts_": [i.copy() for i in mlp_sk.intercepts_],
                }
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(
                    f"ANN early stopping at epoch {epoch} — "
                    f"best val MSE(log)={best_val_loss:.5f}"
                )
                stopped_epoch = epoch
                break

        else:
            stopped_epoch = max_epochs
            logger.info(
                f"ANN reached max_epochs={max_epochs} — "
                f"best val MSE(log)={best_val_loss:.5f}"
            )

    # Restore best weights
    if best_sd is not None:
        mlp_sk.coefs_      = best_sd["coefs_"]
        mlp_sk.intercepts_ = best_sd["intercepts_"]

    elapsed_total = time.perf_counter() - t0
    logger.info(f"ANN training complete in {elapsed_total:.1f}s  (stopped epoch {stopped_epoch})")

    ann = ANN(mlp_sk)

    meta = {
        "backend":               "sklearn_mlp",
        "n_features":            int(X_train.shape[1]),
        "hidden_layers":         hidden_layers,
        "feature_names":         feature_names,
        "target":                "kWh_log1p",
        "epochs_trained":        stopped_epoch,
        "best_val_loss_log_mse": float(best_val_loss),
        "scaler_path":           "outputs/models/scaler_linear_A.pkl",
        "learning_rate":         learning_rate,
        "batch_size":            batch_size,
        "alpha_l2":              1e-5,
    }

    return ann, train_losses, val_losses, meta
