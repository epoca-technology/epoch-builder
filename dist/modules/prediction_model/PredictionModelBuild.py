from typing import List
from modules._types import IPredictionModelConfig
from modules.utils.Utils import Utils
from modules.regression.Regression import Regression
from modules.prediction_model.PredictionModelBuildAssets import PredictionModelBuildAssets




class PredictionModelBuild:
    """PredictionModelBuild Class

    This class handles the generation of the prediction model build. The output is ready
    to be evaluated and exported.

    Instance Properties:
        id: str
            The identifier of the build. This is not the id of the prediction model.
        regressions: List[Regression]
            The list of regressions that will be used to generate features.
        assets: PredictionModelBuildAssets
            The instance of the assets manager.
    """

    PRICE_CHANGE_REQUIREMENTS: List[float] = [2, 2.5, 3, 3.5]





    def __init__(self, regression_ids: List[str]):
        """Initializes the PredictionModelBuild Instance.
        
        Args:
            regression_ids: List[str]
                The list of regressions

        Raises:
            ValueError:
                If no regression ids are provided.
                If any of the regressions cannot be initialized.
        """
        # Make sure the regression ids were provided
        if not isinstance(regression_ids, list) or len(regression_ids) == 0:
            raise ValueError("A minimum of 1 regression must be provided in order to generate the prediction model build.")

        # Init the id
        self.id: str = Utils.generate_uuid4()

        # Initialize the regression instances
        self.regressions: List[Regression] = [Regression(reg_id) for reg_id in regression_ids]

        # Initialize the instance of the assets
        self.assets: PredictionModelBuildAssets = PredictionModelBuildAssets(
            build_id=self.id, 
            regressions=self.regressions,
            price_change_requirements=PredictionModelBuild.PRICE_CHANGE_REQUIREMENTS
        )







    def build(self) -> None:
        """Generates the Prediction Model Build.
        """
        pass











    def _generate_model_id(self) -> str:
        """Generates a random ID that will be assigned to a model variation
        within the build.

        Returns:
            str
        """
        return f"PM_{Utils.generate_uuid4()}"