from typing_extensions import Optional, Self
import numpy as np

import jax.numpy as jnp
from jaxtyping import ArrayLike
import equinox as eqx


class StopCondition:
    """
    Base StopCondition.
    """

    best_model: Optional[eqx.Module]
    verbose: int

    def __init__(self: Self, verbose: int = 0) -> None:
        """
        StopCondition constructor.

        args:
            verbose: verbose level, one of 0,1,2. 0 prints nothing, 1 prints every 10% of total
                epochs, and 2 prints every epoch.
        """
        assert verbose in {0, 1, 2}
        self.best_model = None
        self.verbose = verbose

    def stop(
        self: Self,
        model: eqx.Module,
        current_epoch: int,
        train_loss: Optional[ArrayLike],
        val_loss: Optional[ArrayLike],
        epoch_time: float,
    ) -> bool:
        return True

    def log_status(
        self: Self,
        epoch: int,
        train_loss: Optional[ArrayLike],
        val_loss: Optional[ArrayLike],
        epoch_time: float,
    ) -> None:
        if train_loss is not None:
            if val_loss is not None:
                print(
                    f"Epoch {epoch} Train: {train_loss:.7f} Val: {val_loss:.7f} Epoch time: {epoch_time:.5f}",
                )
            else:
                print(f"Epoch {epoch} Train: {train_loss:.7f} Epoch time: {epoch_time:.5f}")


class EpochStop(StopCondition):
    """
    Stop when enough epochs have passed.
    """

    def __init__(self: Self, epochs: int, verbose: int = 0) -> None:
        """
        EpochStop constructor.

        args:
            epochs: epoch limit
            verbose: verbose level, one of 0,1,2. 0 prints nothing, 1 prints every 10% of total
                epochs, and 2 prints every epoch.
        """
        super(EpochStop, self).__init__(verbose=verbose)
        self.epochs = epochs

    def stop(
        self: Self,
        model: eqx.Module,
        current_epoch: int,
        train_loss: Optional[ArrayLike],
        val_loss: Optional[ArrayLike],
        epoch_time: float,
    ) -> bool:
        """
        Stops if current_epoch is greater than or equal to the specified stop epoch, and log_status
        depending on the level of verbose.

        args:
            model: the current model, saved every epoch
            current_epoch: current epoch
            train_loss: current training loss
            val_loss: current valdiation loss
            epoch_time: how long the epoch took

        returns:
            whether to stop
        """
        self.best_model = model

        if self.verbose == 2 or (
            self.verbose == 1 and (current_epoch % (self.epochs // np.min([10, self.epochs])) == 0)
        ):
            self.log_status(current_epoch, train_loss, val_loss, epoch_time)

        return current_epoch >= self.epochs


class TrainLoss(StopCondition):
    """
    Stop when the training error stops improving after patience number of epochs.
    """

    def __init__(self: Self, patience: int = 0, min_delta: float = 0, verbose: int = 0) -> None:
        """
        TrainLoss constructor.

        args:
            patience: how many epochs of non-improvement to wait before stopping
            min_delta: the minimum decrease to count as an improvement
            verbose: the verbose level, one of 0,1. 0 don't log, 1 log on improvement.
        """
        super(TrainLoss, self).__init__(verbose=verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_train_loss = jnp.inf
        self.epochs_since_best = 0

    def stop(
        self: Self,
        model: eqx.Module,
        current_epoch: int,
        train_loss: Optional[ArrayLike],
        val_loss: Optional[ArrayLike],
        epoch_time: float,
    ) -> bool:
        """
        Stops if the training loss has not improved for a number of epochs equal to patience, and log_status
        depending on the level of verbose.

        args:
            model: the current model, saved every epoch
            current_epoch: current epoch
            train_loss: current training loss
            val_loss: current valdiation loss
            epoch_time: how long the epoch took

        returns:
            whether to stop
        """
        if train_loss is None or not isinstance(train_loss, float):
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
    """
    Stop when the validation error stops improving after patience number of epochs.
    """

    def __init__(self: Self, patience: int = 0, min_delta: float = 0, verbose: int = 0) -> None:
        """
        ValLoss constructor.

        args:
            patience: how many epochs of non-improvement to wait before stopping
            min_delta: the minimum decrease to count as an improvement
            verbose: the verbose level, one of 0,1. 0 don't log, 1 log on improvement.
        """
        super(ValLoss, self).__init__(verbose=verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = jnp.inf
        self.epochs_since_best = 0

    def stop(
        self: Self,
        model: eqx.Module,
        current_epoch: int,
        train_loss: Optional[ArrayLike],
        val_loss: Optional[ArrayLike],
        epoch_time: float,
    ) -> bool:
        """
        Stops if the val loss has not improved for a number of epochs equal to patience, and log_status
        depending on the level of verbose.

        args:
            model: the current model, saved every epoch
            current_epoch: current epoch
            train_loss: current training loss
            val_loss: current valdiation loss
            epoch_time: how long the epoch took

        returns:
            whether to stop
        """
        if val_loss is None or not isinstance(val_loss, float):
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
