from typing import List
from json import loads
from numpy import ndarray
from h5py import File as h5pyFile
from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5
from keras import Sequential
from modules._types import IRegressionConfig, IDiscovery
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.keras_utils.KerasModelSummary import get_summary



class Regression:
    """Regression Class

    This class handles the initialization and management of a Keras Regression Model.

    Instance Properties:
        id: str
            The ID of the model that was set when training.
        description: str
            Important information regarding the trained model.
        lookback: int
            The number of prediction candlesticks that will be used to generate predictions.
        predictions: int
            The number of predictions to be generated.
        discovery: IDiscovery
            The model's discovery information. If the model has not yet been saved, this
            value will be an empty dict.
        model: Sequential
            The instance of the trained model.
    """




    ####################
    ## Initialization ##
    ####################



    def __init__(self, id: str):
        """Initializes the Regression Instance.

        Args:
            id: str
                The ID of the model that will be initialized.

        Raises:
            ValueError:
                If there is an issue loading the model.
                If the ID stored in the model's file is different to the one provided.
                If any of the other metadata is invalid.
        """
        with h5pyFile(Epoch.PATH.regression_model(id), mode="r") as model_file:
            self.id: str = model_file.attrs["id"]
            self.description: str = model_file.attrs["description"]
            self.lookback: int = int(model_file.attrs["lookback"])          # Downcast to int
            self.predictions: int = int(model_file.attrs["predictions"])    # Downcast to int
            self.discovery: IDiscovery = loads(model_file.attrs["discovery"])
            self.model: Sequential = load_model_from_hdf5(model_file)

        # Make sure the IDs are identical
        if self.id != id:
            raise ValueError(f"Regression ID Missmatch: {self.id} != {id}")

        # Make sure the description was extracted
        if not isinstance(self.description, str):
            raise ValueError(f"Regression Description is invalid: {str(self.description)}")
        
        # Make sure the lookback was extracted
        if not isinstance(self.lookback, int):
            raise ValueError(f"Regression Lookback is invalid: {str(self.lookback)}")

        # Make sure the predictions were extracted
        if not isinstance(self.predictions, int):
            raise ValueError(f"Regression Predictions is invalid: {str(self.predictions)}")

        # Make sure the discovery was extracted
        if not isinstance(self.discovery, dict):
            raise ValueError(f"Regression Discovery is invalid: {str(self.discovery)}")













    #################
    ## Prediction  ##
    #################





    def predict(self, features: ndarray) -> List[List[float]]:
        """Generates predictions based on the provided features.

        Args:
            features: ndarray
                The dataset that will be used as input in order to generate
                predictions. Keep in mind that it is best to generate all 
                the predictions in one go.

        Returns:
            List[float]
        """
        return self.model.predict(features).tolist()


















    ###################
    ## Configuration ## 
    ###################




    def get_config(self) -> IRegressionConfig:
        """Returns the configuration of the Keras Regression Model.

        Returns:
            IRegressionConfig
        """
        return {
            "id": self.id,
            "description": self.description,
            "lookback": self.lookback,
            "predictions": self.predictions,
            "discovery": self.discovery,
            "summary": get_summary(self.model),
        }