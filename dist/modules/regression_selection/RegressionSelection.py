from typing import List, Dict
from numpy import median, mean, arange
from modules._types import IKerasRegressionTrainingCertificate, IXGBRegressionTrainingCertificate
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch



class RegressionSelection:
    """RegressionSelection Class

    This class takes any number of KerasRegression|XGBRegression Certificates and calculates
    the optimal parameters that should be used to build Classifications.

    Instance Properties
        model_ids: List[str]
            The list of regression ids that have been selected.
        id: str
            Universally Unique Identifier (uuid4)
    """



    ## Initialization ##

    def __init__(self, model_ids: List[str]):
        """Initializes the RegressionSelection Instance and prepares it
        to be executed.

        Args:
            model_ids: List[str]
                The list of regression ids that have been selected.
        
        Raises:
            ValueError:
                If the model_ids has less than 5 regressions.
        """
        # Make sure that the limit is at least 5
        if len(model_ids) < 5:
            raise ValueError(f"A minimum of 5 regression ids are required. Received: {len(model_ids)}")

        # Initialize the model ids
        self.model_ids: List[str] = model_ids

        # Generate the ID
        self.id: str = Utils.generate_uuid4()


        






    ## Execution ## 





    def run(self) -> None:
        """Executes the RegressionSelection and stores the results once it
        completes.

        Raises:
            RuntimeError:
                If any of the certificates can't be loaded for any reason.
        """
        print(f"\n{self.id}:")
        # Extract all the backtest results
        print("    1/3) Extracting Certificates...")
        #@TODO

        # Build the Regression Selection
        print("    2/3) Building Selection...")
        #@TODO

        # Save the Regression Selection
        print("    3/3) Saving Selection...")
        #@TODO