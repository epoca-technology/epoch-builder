from typing import Union, Any, Tuple
from pandas import DataFrame
from modules.candlestick import Candlestick
from modules.forecasting import IForecastingTrainingConfig, ForecastingTrainingWindowGenerator


class ForecastingTraining:
    """ForecastingTraining Class

    This class handles the training of a Forecasting Model.

    Class Properties:
        OUTPUT_PATH: str
            The directory in which the models will be stored.

    Instance Properties:
        test_mode: bool
            If test_mode is enabled, it won't initialize the candlesticks.
        id: str
            The ID of the model about to be trained.
        lookback: int
            The number of prediction candlesticks that will be used to generate predictions.
        predictions: int
            The number of predictions to be generated.
        window: ForecastingTrainingWindowGenerator
            The instance of the Window Generator
    """

    # Directory where the model and the training certificate will be stored
    OUTPUT_PATH: str = './forecasting_models'




    ## Initialization ##




    def __init__(self, config: IForecastingTrainingConfig, test_mode: bool = False):
        """Initializes the ForecastingTraining Instance.

        Args:
            config: IForecastingTrainingConfig
                The configuration that will be used to train the model.
            test_mode: bool
                Indicates if the execution is running from unit tests.
        """
        # Initialize the type of execution
        self.test_mode: bool = test_mode

        # Initialize the Identifier
        self.id: str = config['id']

        # Initialize the lookback
        self.lookback: int = config['lookback']

        # Initialize the predictions output
        self.predictions: int = config['predictions']

        # Initialize the candlesticks if not unit testing
        if not self.test_mode:
            Candlestick.init(self.lookback)

        # Split the candlesticks into train, val and test
        train_df, val_df, test_df = self._get_train_data()

        # Initialize the Window Instance
        self.window: ForecastingTrainingWindowGenerator = ForecastingTrainingWindowGenerator({
            "input_width": 0,
            "label_width": 0,
            "shift": 0,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "label_columns": ["c"]
        })





    

    def _get_train_data(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Splits the prediction candlesticks into train, val and test and normalizes them.

        Returns:
            Tuple[DataFrame, DataFrame, DataFrame] 
            (train_df, val_df, test_df)
        """
        # Create a copy of the DF
        df: DataFrame = Candlestick.PREDICTION_DF.copy()

        # Remove open and close times as they won't be needed
        df.drop(['ot', 'ct'], axis = 1, inplace=True)

        # Return the normalized DataFrames
        return self._normalize_data(
            df[0:int(df.shape[0]*0.7)],                     # Train
            df[int(df.shape[0]*0.7):int(df.shape[0]*0.9)],  # Val
            df[int(df.shape[0]*0.9):]                       # Test
        ) 






    def _normalize_data(self, train: DataFrame, val: DataFrame, test: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Normalizes the split DataFrames.

        Args:
            train: DataFrame
            val: DataFrame
            test: DataFrame

        Returns:
            Tuple[DataFrame, DataFrame, DataFrame]
            (train_df, val_df, test_df)
        """
        # Calculate the Train Mean & the Standard Deviation
        train_mean = train.mean()
        train_std = train.std()

        # Normalize the Dataframes
        train_df = (train - train_mean) / train_std
        val_df = (val - train_mean) / train_std
        test_df = (test - train_mean) / train_std

        # Return the packed tuple
        return train_df, val_df, test_df