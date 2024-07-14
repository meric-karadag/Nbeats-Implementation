import numpy as np
import torch
from torch import Tensor, tensor

from torchmetrics.functional.regression.r2 import _r2_score_compute, _r2_score_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from typing import Any, Optional, Sequence, Union


class my_R2Score(Metric):

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    sum_squared_error: Tensor
    sum_error: Tensor
    residual: Tensor
    total: Tensor

    def __init__(
        self,
        num_outputs: int = 1,
        adjusted: int = 0,
        multioutput: str = "uniform_average",
        position: int = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.num_outputs = num_outputs
        self.position = position

        if adjusted < 0 or not isinstance(adjusted, int):
            raise ValueError(
                "`adjusted` parameter should be an integer larger or equal to 0."
            )
        self.adjusted = adjusted

        allowed_multioutput = ("raw_values", "uniform_average", "variance_weighted")
        if multioutput not in allowed_multioutput:
            raise ValueError(
                f"Invalid input to argument `multioutput`. Choose one of the following: {allowed_multioutput}"
            )
        self.multioutput = multioutput

        self.add_state(
            "sum_squared_error",
            default=torch.zeros(self.num_outputs),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_error", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum"
        )
        self.add_state(
            "residual", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum"
        )
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        if self.position is not None:
            sum_squared_error, sum_error, residual, total = _r2_score_update(
                preds[:, self.position],
                target[:, self.position],
                
            )
        else:
            sum_squared_error, sum_error, residual, total = _r2_score_update(
                preds, target
            )

        self.sum_squared_error += sum_squared_error
        self.sum_error += sum_error
        self.residual += residual
        self.total += total

    def compute(self) -> Tensor:
        """Compute r2 score over the metric states."""
        return _r2_score_compute(
            self.sum_squared_error,
            self.sum_error,
            self.residual,
            self.total,
            self.adjusted,
            self.multioutput,
        )

    def plot(
        self,
        val: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        ax: Optional[_AX_TYPE] = None,
    ) -> _PLOT_OUT_TYPE:

        return self._plot(val, ax)


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01 * (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr


