from typing import List
from numpy import ndarray, array, float32
from pandas import DataFrame
from h5py import File as h5pyFile
from tensorflow import data as tfdata
from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5
from keras import Sequential
from keras.preprocessing.timeseries import timeseries_dataset_from_array
from modules.keras_models import KerasModelInterface, KERAS_PATH, get_summary
from modules.model import IRegressionConfig



class Regression(KerasModelInterface):
    """Regression Class

    This class handles the initialization of a Keras Regression Model.

    Class Properties:
        ...

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



    ## Initialization ##



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
        with h5pyFile(f"{KERAS_PATH['models']}/{id}/model.h5", mode='r') as model_file:
            self.id: str = model_file.attrs['id']
            self.description: str = model_file.attrs['description']
            self.lookback: int = int(model_file.attrs['lookback'])          # Downcast to int
            self.predictions: int = int(model_file.attrs['predictions'])    # Downcast to int
            self.model: Sequential = load_model_from_hdf5(model_file)

        # Make sure the IDs are identical
        if self.id != id:
            raise ValueError(f"RegressionModel ID Missmatch: {self.id} != {id}")

        # Make sure the description was extracted
        if not isinstance(self.description, str):
            raise ValueError(f"RegressionModel Description is invalid: {str(self.description)}")
        
        # Make sure the lookback was extracted
        if not isinstance(self.lookback, int):
            raise ValueError(f"RegressionModel Lookback is invalid: {str(self.lookback)}")

        # Make sure the predictions were extracted
        if not isinstance(self.predictions, int):
            raise ValueError(f"RegressionModel Predictions is invalid: {str(self.predictions)}")






    ## Predictions ##




    def predict(self, normalized_lookback_df: DataFrame) -> List[float]:
        """Given a Lookback DataFrame, it will turn it into a Dataset and predict 
        future values.

        Args:
            normalized_lookback_df: DataFrame
                The input df that will be used to generate predictions.

        Returns:
            List[float]
        """
        # Make the input ds
        input_ds: tfdata.Dataset = self._make_input_dataset(normalized_lookback_df)

        # Return the predictions
        return self.model.predict(input_ds)[0]







    def _make_input_dataset(self, normalized_lookback_df: DataFrame) -> tfdata.Dataset:
        """Converts an Input DataFrame into a Dataset that is ready to be used to predict.

        Args:
            data: DataFrame
                The data to be converted into a TF Dataset
        
        Returns:
            tfdata.Dataset
        """
        # Convert the DataFrame into a numpy array
        data: ndarray = array(normalized_lookback_df, dtype=float32)

        # Initialize the Dataset
        ds: tfdata.Dataset = timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.lookback,
            sequence_stride=1,
            shuffle=False,
            batch_size=1
        )

        # Finally, return the features dataset
        return ds










    ## Regression Configuration ##




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