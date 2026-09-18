# Early stopping utility for GolfPose training.
#
# Monitors a validation metric (lower is better by default) and triggers
# a stop after ``patience`` epochs without improvement.

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Track a validation metric and signal when to stop.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after the last improvement.
    min_delta : float
        Minimum absolute change to qualify as an improvement.
    mode : str
        ``'min'`` (default) treats lower values as better;
        ``'max'`` treats higher values as better.
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value = None
        self.best_epoch = -1
        self.counter = 0
        self._stopped = False

    # ------------------------------------------------------------------
    def _is_better(self, current, best):
        if self.mode == "min":
            return current < best - self.min_delta
        return current > best + self.min_delta

    # ------------------------------------------------------------------
    def step(self, metric_value: float, epoch: int) -> bool:
        """Update state with the current epoch's metric.

        Returns
        -------
        should_stop : bool
            ``True`` when patience has been exhausted.
        """
        if self.best_value is None or self._is_better(metric_value, self.best_value):
            self.best_value = metric_value
            self.best_epoch = epoch
            self.counter = 0
            logger.info(
                "EarlyStopping: new best %.4f at epoch %d", metric_value, epoch
            )
        else:
            self.counter += 1
            logger.info(
                "EarlyStopping: no improvement for %d/%d epochs (best=%.4f @ epoch %d)",
                self.counter, self.patience, self.best_value, self.best_epoch,
            )

        if self.counter >= self.patience:
            self._stopped = True
            logger.warning(
                "EarlyStopping: triggered after %d epochs without improvement.",
                self.patience,
            )
            return True
        return False

    # ------------------------------------------------------------------
    @property
    def stopped(self) -> bool:
        return self._stopped

    def state_dict(self):
        return {
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
        }

    def load_state_dict(self, state: dict):
        self.best_value = state["best_value"]
        self.best_epoch = state["best_epoch"]
        self.counter = state["counter"]
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.mode = state.get("mode", self.mode)
