from typing import List
from numpy import ndarray, append
from pandas import Series
from h5py import File as h5pyFile
from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5
from keras import Sequential
from modules._types import IKerasRegressionConfig, KerasModelInterface
from modules.epoch.Epoch import Epoch
from modules.keras_models.KerasModelSummary import get_summary



class KerasRegression(KerasModelInterface):
    """KerasRegression Class

    This class handles the initialization of a Keras Regression.

    Instance Properties:
        id: str
            The ID of the model that was set when training.
        description: str
            Important information regarding the trained model.
        autoregressive: bool
            The type of regression.
        lookback: int
            The number of prediction candlesticks that will be used to generate predictions.
        predictions: int
            The number of predictions to be generated.
        model: Sequential
            The instance of the trained model.
    """





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
        # Load the model
        model_path: str = Epoch.FILE.get_active_model_path(id, "keras_regression")
        with h5pyFile(model_path, mode="r") as model_file:
            self.id: str = model_file.attrs["id"]
            self.description: str = model_file.attrs["description"]
            self.autoregressive: bool = bool(model_file.attrs["autoregressive"]) # Downcast to bool
            self.lookback: int = int(model_file.attrs["lookback"])          # Downcast to int
            self.predictions: int = int(model_file.attrs["predictions"])    # Downcast to int
            self.model: Sequential = load_model_from_hdf5(model_file)

        # Make sure the IDs are identical
        if self.id != id:
            raise ValueError(f"KerasRegressionModel ID Missmatch: {self.id} != {id}")

        # Make sure the description was extracted
        if not isinstance(self.description, str):
            raise ValueError(f"KerasRegressionModel Description is invalid: {str(self.description)}")
        
        # Make sure the type of regression was extracted
        if not isinstance(self.autoregressive, bool):
            raise ValueError(f"KerasRegressionModel Autoregressive Arg is invalid: {str(self.autoregressive)}-{type(self.autoregressive)}")
        
        # Make sure the lookback was extracted
        if not isinstance(self.lookback, int):
            raise ValueError(f"KerasRegressionModel Lookback is invalid: {str(self.lookback)}")

        # Make sure the predictions were extracted
        if not isinstance(self.predictions, int):
            raise ValueError(f"KerasRegressionModel Predictions is invalid: {str(self.predictions)}")











    def predict(self, close_prices: Series) -> List[float]:
        """Generates predictions based on a close price series.

        Args:
            close_prices: Series
                Lookback normalized close prices.

        Returns:
            List[float]
        """
        # Create a ndarray of the close prices
        close: ndarray = close_prices.to_numpy()

        # If the regression is autoregressive, generate 1 prediction at a time and re-use it for the next
        if self.autoregressive:
            # Iterate over the predictions range
            for _ in range(self.predictions):
                pred: float = self.model.predict(close[-self.lookback:].reshape((1, self.lookback)))[0]
                close = append(close, pred)

            # Finally, return the predictions
            return close[-self.predictions:].tolist()

        # If it is not autoregressive, generate all the predictions in one go
        else:
            return self.model.predict(close.reshape((1, self.lookback)))[0].tolist()












    def get_config(self) -> IKerasRegressionConfig:
        """Returns the configuration of the Keras Regression Model.

        Returns:
            IKerasRegressionConfig
        """
        return {
            "id": self.id,
            "description": self.description,
            "autoregressive": self.autoregressive,
            "lookback": self.lookback,
            "predictions": self.predictions,
            "summary": get_summary(self.model),
        }