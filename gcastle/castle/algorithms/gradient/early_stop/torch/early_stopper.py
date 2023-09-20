import numpy as np
import torch
from trustworthyAI.gcastle.castle.common.base import Tensor
# inspired by answers on : https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

EARLYSTOPPING = False

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, window=100, max_amplitude=1e-3, tolerance=2.5e-2):
        if not EARLYSTOPPING:
            self._allowed = False
        self.reset(patience, min_delta, window, max_amplitude, tolerance)

    def reset(self, patience=1, min_delta=0, window=10, max_amplitude=1e-3, tolerance=2.5e-2):
        self.patience : int = patience
        self.min_delta : float = min_delta
        self.tolerance : float = tolerance
        self.counter : int = 0
        self.min_validation_loss : float = np.inf
        self._early_stop : bool = False
        self.window : int = window
        self._loss_history : list[float] = []
        self.max_amplitude : float = np.max([max_amplitude, 0.0])
        self.counter : int = 0

    def __str__(self) -> str:
        return f"EarlyStopper(patience={self.patience}, min_delta={self.min_delta}, window={self.window}, max_amplitude={self.max_amplitude})"
    
    def __call__(self, validation_loss):
        if not EARLYSTOPPING:
            return None
        if not isinstance(validation_loss, float):
            if isinstance(validation_loss, (np.ndarray, list)):
                raise NotImplementedError("EarlyStopper does not yet support np.ndarray or list")
            elif isinstance(validation_loss, (torch.Tensor, Tensor)):
                validation_loss = validation_loss.item()
            else:
                try:
                    validation_loss = float(validation_loss)
                except:
                    raise TypeError(f"Expected float, got {type(validation_loss)}")
        if validation_loss is not None:
            self.update(validation_loss)

    @property
    def early_stop(self):
        if not EARLYSTOPPING:
            return False
        return self._early_stop

    @property
    def loss_history(self):
        return self._loss_history

    def loss_history_append(self, loss):
        self._loss_history.append(loss)

    @loss_history.setter
    def loss_history(self, loss_history):
        self._loss_history = loss_history

    @loss_history.deleter
    def loss_history(self):
        del self._loss_history

    def __earlyStopCriterionMin(self) -> bool:
        """
        Early stop criterion for convergence test.
        """
        history = self._loss_history
        window = self.window
        tolerance = self.tolerance
        if len(history) >= window:
            return (
                np.array([abs(history[-1] - el) for el in history[-window:]])
                < np.array([tolerance * history[-1]] * window)
            ).all()
            # return np.allclose(
            #     self._loss_history[-self.window :],
            #     self._loss_history[-1],
            #     atol=self._loss_history[-1] / 10,
            # )
        else:
            return False

    def __earlyStopCriterionSat(self) -> bool:
        """
        Early stop criterion for convergence test.
        """
        history = self._loss_history
        window = self.window
        tolerance = self.tolerance
        if len(self._loss_history) >= 2 * self.window:
            # if np.ptp(self._loss_history[-self.window :]) > (
            #     1.0 + self.max_amplitude
            # ) * np.max(self._loss_history[-2 * self.window : -self.window]):
            #     return False
            # else:
                return abs(
                    np.mean(history[-window:]) - np.mean(history[-2 * window : -window])
                ) < tolerance * np.mean(history[-window:])
        else:
            return False

    def __check_early_stop(self):
        if self.__earlyStopCriterionMin() or self.__earlyStopCriterionSat():
            if not self._early_stop:
                print("Early stopping due to absolute convergence.") if self.__earlyStopCriterionMin() else print(
                    "Early stopping due to saturation."
                )
            return True
        else:
            return False

    def update(self, validation_loss):
        if not self._early_stop:
            self._early_stop = self.step(validation_loss)
            print("Early stopping activated.") if self._early_stop else None
        
    def step(self, validation_loss):
        self.loss_history_append(validation_loss)
        flag = False
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            if not self.__check_early_stop():
                self.reset_counter()
        elif self.__check_early_stop():
            flag = self.increase_counter()
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            flag = self.increase_counter()
        return flag

    def reset_counter(self):
        print("Early stopping counter reset.") if self.counter > 0 else None
        self.counter = 0
        
    def increase_counter(self):
        self.counter += 1
        print("Early stopping counter: ", self.counter)
        if self.counter >= self.patience:
            print()
            return True
        return False