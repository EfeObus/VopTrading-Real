# pragma pylint: disable=attribute-defined-outside-init

"""
This module load a custom model for vopai
"""

import logging
from pathlib import Path

from voptrade.constants import USERPATH_VOPAIMODELS, Config
from voptrade.exceptions import OperationalException
from voptrade.vopai.vopai_interface import IVopaiModel
from voptrade.resolvers import IResolver


logger = logging.getLogger(__name__)


class VopaiModelResolver(IResolver):
    """
    This class contains all the logic to load custom hyperopt loss class
    """

    object_type = IVopaiModel
    object_type_str = "VopaiModel"
    user_subdir = USERPATH_VOPAIMODELS
    initial_search_path = (
        Path(__file__).parent.parent.joinpath("vopai/prediction_models").resolve()
    )
    extra_path = "vopaimodel_path"

    @staticmethod
    def load_vopaimodel(config: Config) -> IVopaiModel:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary
        """
        disallowed_models = ["BaseRegressionModel"]

        vopaimodel_name = config.get("vopaimodel")
        if not vopaimodel_name:
            raise OperationalException(
                "No vopaimodel set. Please use `--vopaimodel` to "
                "specify the VopaiModel class to use.\n"
            )
        if vopaimodel_name in disallowed_models:
            raise OperationalException(
                f"{vopaimodel_name} is a baseclass and cannot be used directly. Please choose "
                "an existing child class or inherit from this baseclass.\n"
            )
        vopaimodel = VopaiModelResolver.load_object(
            vopaimodel_name,
            config,
            kwargs={"config": config},
        )

        return vopaimodel
