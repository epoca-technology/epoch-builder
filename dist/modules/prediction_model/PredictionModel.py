from typing import List
from modules._types import IPredictionModelConfig
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.prediction_model.PredictionModelAssets import PredictionModelAssets




class PredictionModel:
    """PredictionModel Class

    This class handles the generation of the prediction model build. The output is ready
    to be evaluated and exported.

    Class Properties:
        ...

    Instance Properties:
        assets: PredictionModelBuildAssets
            The instance of the assets manager.
    """





    def __init__(self):
        """Initializes the PredictionModel Instance.
        
        Args:
            ...

        Raises:
            ValueError:
                If no regression ids are provided.
                If any of the regressions cannot be initialized.
        """
        # Initialize the instance of the assets
        self.assets: PredictionModelAssets = PredictionModelAssets()





    ###########
    ## Build ##
    ###########




    def build(self, limit: int) -> None:
        """
        """
        pass













    ###############################
    ## Profitable Configurations ##
    ###############################






    def find_profitable_configs(self, batch_file_name: str) -> None:
        """
        """
        pass



















    ##################
    ## Misc Helpers ##
    ##################



    def _generate_model_id(self) -> str:
        """Generates a random ID that will be assigned to a model variation
        within the build.

        Returns:
            str
        """
        return f"{Epoch.ID}_{Utils.generate_uuid4()}"