from typing import List, Union
from numpy import median, mean, arange
from modules._types import IKerasRegressionTrainingCertificate, IXGBRegressionTrainingCertificate,\
    ISelectedRegression, IModel, IModelType, IPercentChangeInterpreterConfig, IKerasRegressionConfig, \
        IXGBRegressionConfig
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.model.ModelType import get_model_type, get_trainable_model_type


# Certificate Type Helper
ITrainingCertificate = Union[IKerasRegressionTrainingCertificate, IXGBRegressionTrainingCertificate]




# Class
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
                If the model_ids has less than 1 regressions.
                If any of the models is not a RegressionModel
        """
        # Make sure that the limit is at least 1
        if len(model_ids) < 1:
            raise ValueError(f"A minimum of 1 regression ids are required. Received: {len(model_ids)}")

        # Make sure all provided models are regressions
        non_regressions: filter = filter(
            lambda x: get_model_type(x) != "KerasRegressionModel" and get_model_type(x) != "XGBRegressionModel",
            model_ids
        )
        if next(non_regressions, None) != None:
            raise ValueError("Only RegressionModels can be passed through the RegressionSelection process.")

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
        # Extract all the training certificates
        print("    1/4) Extracting Certificates...")
        certificates: List[ITrainingCertificate] = [ 
            Epoch.FILE.get_banked_model_certificate(id, get_trainable_model_type(id)) for id in self.model_ids 
        ]

        # Build the Regression Selection
        print("    2/4) Building Selection...")
        selection: List[ISelectedRegression] = [self._build_selection(c) for c in certificates]

        # Calculate the price change mean
        print("    3/4) Calculating Price Change Mean...")
        price_change_mean: float = round(mean([s["discovery"]["successful_mean"] for s in selection]), 2)

        # Save the Regression Selection
        print("    4/4) Saving Selection...")
        Epoch.FILE.save_regression_selection({
            "id": self.id,
            "creation": Utils.get_time(),
            "price_change_mean": price_change_mean,
            "selection": selection
        })





    
    def _build_selection(self, certificate: ITrainingCertificate) -> ISelectedRegression:
        """Builds a selection based on the given Regression Certificate.

        Args:
            certificate: ITrainingCertificate
                The certificate that will be used to populate the selecton.

        Returns:
            ISelectedRegression

        Raises:
            ValueError:
                If the model config cannot be built for any reason
        """
        # Build the points median history
        points_hist: List[float] = [p["pts"] for p in certificate["regression_evaluation"]["positions"]]
        median_hist: List[float] = []
        for i in arange(0.1, 1.1, 0.1):
            median_hist.append(round(median(points_hist[:int(len(points_hist) * i)]), 2))

        # Finally, return the selection
        return {
            "id": certificate["id"],
            "model": self._build_model_config(certificate),
            "discovery": certificate["discovery"],
            "evaluation": certificate["regression_evaluation"],
            "points_median_hist": median_hist
        }






    def _build_model_config(self, cert: ITrainingCertificate) -> IModel:
        """Builds the model configuration based on its type.

        Args:
            cert: ITrainingCertificate
                The training certificate of the model

        Returns:
            IModel

        Raises:
            ValueError:
                If the model config cannot be built for any reason
        """
        # Retrieve the type
        model_type: IModelType = get_model_type(cert["id"])

        # Init the interpreter
        interpreter: IPercentChangeInterpreterConfig = {
            "min_increase_change": cert["discovery"]["increase_successful_mean"],
            "min_decrease_change": cert["discovery"]["decrease_successful_mean"],
        }

        # Init the regression
        regression: Union[IKerasRegressionConfig, IXGBRegressionConfig] = {
            "regression_id": cert["id"],
            "interpreter": interpreter,
            "regression": cert["regression_config"]
        }

        # Build a KerasRegressionModel
        if model_type == "KerasRegressionModel":
            return {"id": cert["id"], "keras_regressions": [ regression ] }

        # Build a XGBRegressionModel
        elif model_type == "XGBRegressionModel":
            return {"id": cert["id"], "xgb_regressions": [ regression ] }

        # Otherwise, raise an error
        else:
            raise ValueError(f"The RegressionModel ID {cert['id']} could not be built.")

