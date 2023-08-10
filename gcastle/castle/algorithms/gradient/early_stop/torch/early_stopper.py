import numpy as np

# inspired by answers on : https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, window=10, max_amplitude=1e-3):
        self.reset(patience, min_delta, window, max_amplitude)

    def reset(self, patience=1, min_delta=0, window=10, max_amplitude=1e-3):
        self.patience : int = patience
        self.min_delta : float = min_delta
        self.counter : int = 0
        self.min_validation_loss : float = np.inf
        self._early_stop : bool = False
        self.window : int = window
        self._loss_history : list[float] = []
        self.max_amplitude : float = np.max([max_amplitude, 0.0])

    def __str__(self) -> str:
        return f"EarlyStopper(patience={self.patience}, min_delta={self.min_delta}, window={self.window}, max_amplitude={self.max_amplitude})"
    
    def __call__(self, validation_loss):
        if not isinstance(validation_loss, float):
            if isinstance(validation_loss, [np.ndarray, list]):
                raise NotImplementedError("EarlyStopper does not yet support np.ndarray or list")
            else:
                raise TypeError(f"Expected float, got {type(validation_loss)}")
        if validation_loss is not None:
            self._early_stop = self.update(validation_loss)

    @property
    def early_stop(self):
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
        if len(self._loss_history) >= self.window:
            return np.allclose(
                self._loss_history[-self.window :],
                self._loss_history[-1],
                atol=self._loss_history[-1] / 20,
            )
        else:
            return False

    def __earlyStopCriterionSat(self) -> bool:
        """
        Early stop criterion for convergence test.
        """
        if len(self._loss_history) >= 2 * self.window:
            if np.ptp(self._loss_history[-self.window :]) > (
                1.0 + self.max_amplitude
            ) * np.max(self._loss_history[-2 * self.window : -self.window]):
                return False
            else:
                return np.isclose(
                    np.mean(self._loss_history[-self.window :]),
                    np.mean(self._loss_history[-2 * self.window : -self.window]),
                    atol=self._loss_history[-1] / 20,
                )
        else:
            return False

    def __check_early_stop(self):
        if self.__earlyStopCriterionMin() or self.__earlyStopCriterionSat():
            return True
        else:
            return False

    def update(self, validation_loss):
        if self._early_stop:
            return
        self.step(validation_loss)
        
    def step(self, validation_loss):
        self.loss_history_append(validation_loss)
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            if self.check_early_stop():
                pass
            else:
                self.counter = 0
        elif self.__check_early_stop():
            self.counter += 1
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
