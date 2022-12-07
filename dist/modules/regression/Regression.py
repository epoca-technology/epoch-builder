from typing import List
from numpy import ndarray
from h5py import File as h5pyFile
from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5
from keras import Sequential
from modules._types import IRegressionConfig
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.keras_utils.KerasModelSummary import get_summary



class Regression:
    """Regression Class

    This class handles the initialization and management of a Keras Regression Model.

    Class Properties:
        MIN_FEATURE_VALUE: float
        MAX_FEATURE_VALUE: float
            The minimum and maximum values features are allowed to have.

    Instance Properties:
        id: str
            The ID of the model that was set when training.
        description: str
            Important information regarding the trained model.
        lookback: int
            The number of prediction candlesticks that will be used to generate predictions.
        predictions: int
            The number of predictions to be generated.
        model: Sequential
            The instance of the trained model.
    """
    # Min and max feature values
    MIN_FEATURE_VALUE: float = 0.01
    MAX_FEATURE_VALUE: float = 5






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
            List[List[float]]
        """
        return self.model.predict(features, verbose=0).tolist()












    #############
    ## Feature ##
    #############





    def predict_feature(self, features: ndarray) -> List[float]:
        """Generates predictions for the entire features dataset. Then, it converts
        them into features.

        Args:
            features: ndarray
                The input dataset that will be used to generate predictions.

        Returns:
            List[float]
        """
        # Firstly, predict the entire dataset
        preds: List[List[float]] = self.predict(features)

        # Finally, return the predicted features
        return [
            self._normalize_feature(Utils.get_percentage_change(features[i, -1], p[-1])) for i, p in enumerate(preds)
        ]







    def _normalize_feature(self, predicted_change: float) -> float:
        """Given a predicted change, it will scale it to a range between
        -1 and 1 accordingly.

        Args:
            predicted_change: float
                The percentage change between the current price and the last
                predicted price.

        Returns:
            float
        """
        # Retrieve the adjusted change
        adjusted_change: float = self._calculate_adjusted_change(predicted_change)

        # Scale the increase change
        if adjusted_change > 0:
            return self._scale_feature(adjusted_change)
        
        # Scale the decrease change, keep in mind that the decrease data is in negative numbers.
        elif adjusted_change < 0:
            return -(self._scale_feature(-(adjusted_change)))
        
        # Otherwise, return 0 as a sign of neutrality
        else:
            return 0







    def _calculate_adjusted_change(self, change: float) -> float:
        """Adjusts the provided change to the min and max values in the
        regression discovery.

        Args:
            change: float
                The percentage change from the current price to the last 
                prediction.

        Returns:
            float
        """
        if change >= Regression.MIN_FEATURE_VALUE and change <= Regression.MAX_FEATURE_VALUE:
            return change
        elif change > Regression.MAX_FEATURE_VALUE:
            return Regression.MAX_FEATURE_VALUE
        elif change >= -(Regression.MAX_FEATURE_VALUE) and change <= -(Regression.MIN_FEATURE_VALUE):
            return change
        elif change < -(Regression.MAX_FEATURE_VALUE):
            return -(Regression.MAX_FEATURE_VALUE)
        else:
            return 0





    def _scale_feature(self, value: float) -> float:
        """Scales a prediction change based on the regression's min and max
        feature values

        Args:
            value: float
                The predicted price change that needs to be scaled.

        Returns: 
            float
        """
        return round(
            (value - Regression.MIN_FEATURE_VALUE) / (Regression.MAX_FEATURE_VALUE - Regression.MIN_FEATURE_VALUE), 
            6
        )










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
            "summary": get_summary(self.model),
        }