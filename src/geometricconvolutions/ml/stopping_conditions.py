from typing_extensions import Optional, Self
import numpy as np

import jax.numpy as jnp
import equinox as eqx


class StopCondition:
    def __init__(self: Self, verbose: int = 0) -> Self:
        assert verbose in {0, 1, 2}
        self.best_model = None
        self.verbose = verbose

    def stop(
        self: Self,
        model: eqx.Module,
        current_epoch: int,
        train_loss: float,
        val_loss: float,
        epoch_time: int,
    ) -> None:
        pass

    def log_status(
        self: Self, epoch: int, train_loss: float, val_loss: float, epoch_time: int
    ) -> None:
        if train_loss is not None:
            if val_loss is not None:
                print(
                    f"Epoch {epoch} Train: {train_loss:.7f} Val: {val_loss:.7f} Epoch time: {epoch_time:.5f}",
                )
            else:
                print(f"Epoch {epoch} Train: {train_loss:.7f} Epoch time: {epoch_time:.5f}")


class EpochStop(StopCondition):
    # Stop when enough epochs have passed.

    def __init__(self: Self, epochs: int, verbose: int = 0) -> Self:
        super(EpochStop, self).__init__(verbose=verbose)
        self.epochs = epochs

    def stop(
        self: Self,
        model: eqx.Module,
        current_epoch: int,
        train_loss: float,
        val_loss: float,
        epoch_time: int,
    ) -> bool:
        self.best_model = model

        if self.verbose == 2 or (
            self.verbose == 1 and (current_epoch % (self.epochs // np.min([10, self.epochs])) == 0)
        ):
            self.log_status(current_epoch, train_loss, val_loss, epoch_time)

        return current_epoch >= self.epochs


class TrainLoss(StopCondition):
    # Stop when the training error stops improving after patience number of epochs.

    def __init__(self: Self, patience: int = 0, min_delta: float = 0, verbose: int = 0) -> Self:
        super(TrainLoss, self).__init__(verbose=verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_train_loss = jnp.inf
        self.epochs_since_best = 0

    def stop(
        self: Self,
        model: eqx.Module,
        current_epoch: int,
        train_loss: Optional[float],
        val_loss: Optional[float],
        epoch_time: int,
    ) -> bool:
        if train_loss is None:
            return False

        if train_loss < (self.best_train_loss - self.min_delta):
            self.best_train_loss = train_loss
            self.best_model = model
            self.epochs_since_best = 0

            if self.verbose >= 1:
                self.log_status(current_epoch, train_loss, val_loss, epoch_time)
        else:
            self.epochs_since_best += 1

        return self.epochs_since_best > self.patience


class ValLoss(StopCondition):
    # Stop when the validation error stops improving after patience number of epochs.

    def __init__(self: Self, patience: int = 0, min_delta: float = 0, verbose: int = 0) -> Self:
        super(ValLoss, self).__init__(verbose=verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = jnp.inf
        self.epochs_since_best = 0

    def stop(
        self: Self,
        model: eqx.Module,
        current_epoch: int,
        train_loss: Optional[float],
        val_loss: Optional[float],
        epoch_time: int,
    ) -> bool:
        if val_loss is None:
            return False

        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.best_model = model
            self.epochs_since_best = 0

            if self.verbose >= 1:
                self.log_status(current_epoch, train_loss, val_loss, epoch_time)
        else:
            self.epochs_since_best += 1

        return self.epochs_since_best > self.patience
