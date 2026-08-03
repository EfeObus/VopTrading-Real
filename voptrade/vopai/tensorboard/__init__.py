# ensure users can still use a non-torch vopai version
try:
    from voptrade.vopai.tensorboard.lightgbm_callback import LightGBMTensorboardCallback
    from voptrade.vopai.tensorboard.tensorboard import TensorBoardCallback, TensorboardLogger

    TBLogger = TensorboardLogger
    TBCallback = TensorBoardCallback
    LightGBMCallback = LightGBMTensorboardCallback
except ModuleNotFoundError:
    from voptrade.vopai.tensorboard.base_tensorboard import (
        BaseTensorBoardCallback,
        BaseTensorboardLogger,
    )

    TBLogger = BaseTensorboardLogger  # type: ignore
    TBCallback = BaseTensorBoardCallback  # type: ignore
    LightGBMCallback = None  # type: ignore

__all__ = ("TBLogger", "TBCallback", "LightGBMCallback")
