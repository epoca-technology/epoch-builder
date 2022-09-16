from typing import List, Union, Tuple
from random import seed
from numpy.random import seed as npseed
from tensorflow import random as tf_random
from math import ceil
from pandas import DataFrame
from modules._types import IEpochConfig, IEpochDefaults
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.candlestick.Candlestick import Candlestick
from modules.epoch.EpochPath import EpochPath



# Class
class Epoch:
    """Epoch Class

    This singleton manages the creation, initialization and exporting of epochs.

    Class Properties:
        INITIALIZED: bool
            Indicates if the Epoch has been initialized.
        DEFAULTS: IEpochDefaults
            The default values that will be set if no data is provided through the CLI.
        SEED: int
            The set that will be used to set randomness in all required libs
        ID: str
            The ID of the Epoch.
        SMA_WINDOW_SIZE: int
            The window size that will be used to calculate the simple moving averages on
            the prediction candlesticks dataframe.
        TRAIN_SPLIT: float
            The split that will be applied to the epoch_width in order to train models.
        VALIDATION_SPLIT: float
            The split that will be applied to the train data in order to be able to evaluate
            the model's training performance.
        START: int
        END: int
            The date range of the Epoch.
        TEST_DS_START: int
        TEST_DS_END: int
            The date range of the test dataset.
        HIGHEST_PRICE: float
        LOWEST_PRICE: float
            Highest and lowest price within the Epoch. These values are stored as 
            they are used to scale the prediction candlesticks for trainable models.
            If the price was to go above the highest or below the lowest price, trading should be
            stopped and a new epoch should be published once the market is "stable"
        REGRESSION_LOOKBACK: int
        REGRESSION_PREDICTIONS: int
            The values that represent the input and the ouput of a regression.
            The lookback stands for the number of candlesticks from the past it needs to look at
            in order to generate a prediction.
            The predictions stand for the number of predictions the regressions will generate.
        POSITION_SIZE: float
        LEVERAGE: int
        IDLE_MINUTES_ON_POSITION_CLOSE: int
            The configuration values that are used to evaluate prediction models in trading
            simulations.
        PATH: EpochPath
            The instance of the Epoch's Path.
    """
    # Initialization State
    INITIALIZED: bool = False

    # Epoch Defaults
    DEFAULTS: IEpochDefaults = {
        "epoch_width": 24,
        "sma_window_size": 100,
        "train_split": 0.75,
        "validation_split": 0.2,
        "seed": 60184,
        "regression_lookback": 128,
        "regression_predictions": 32,
        "position_size": 10000,
        "leverage": 3,
        "idle_minutes_on_position_close": 30
    }

    # The set that will be used to set randomness in all required libs
    SEED: int

    # Identifier, must be preffixed with "_EPOCHNAME"
    ID: str

    # Simple Moving Average Window Size
    SMA_WINDOW_SIZE: int

    # Split used to train models
    TRAIN_SPLIT: float

    # Split used to evaluation regression's training performance
    VALIDATION_SPLIT: float

    # The date range of the Epoch
    START: int
    END: int

    # The date range of the test dataset
    TEST_DS_START: int
    TEST_DS_END: int

    # Normalization Price Range
    HIGHEST_PRICE: float
    LOWEST_PRICE: float

    # Regression Parameters
    REGRESSION_LOOKBACK: int
    REGRESSION_PREDICTIONS: int

    # Prediction Model Evaluation
    POSITION_SIZE: float
    LEVERAGE: int
    IDLE_MINUTES_ON_POSITION_CLOSE: int

    # EpochPath Instance
    PATH: EpochPath








    ## Creation ##



    @staticmethod
    def create(
        id: str, 
        epoch_width: int,
        sma_window_size: int,
        seed: int,
        train_split: float,
        validation_split: float,
        regression_lookback: int,
        regression_predictions: int,
        position_size: float,
        leverage: int,
        idle_minutes_on_position_close: int
    ) -> None:
        """Creates all the neccessary directories and files for the epoch
        to get started.
        
        Args:
            id: str
                The identifier of the epoch. Make sure to never reuse these names
                as the core infrastructure will have validations to prohibit this.
            epoch_width: int
                The number of months that comprise the epoch. This value is used
                to calculate the start and end timestamps of the epoch. This value
                is also used to calculate the Backtests' Date Range.
            sma_window_size: int
                The window size to be applied on the prediction candlesticks.
            seed: int
                The random seed to be set on all required libs and machines.
            train_split: float
                The split that will be applied to the epoch_width to train models.
            validation_split: float
                The split that will be applied to the train dataset in order to generate
                the validation dataset.
            regression_lookback: int
                The number of candlesticks from the past regressions will look at in order
                to generate predictions.
            regression_predictions: int
                The number of predictions regressions will generate.
            position_size: float
                The amount of USD that will be used to open positions.
            leverage: int
                The leverage that will be used to open positions
            idle_minutes_on_position_close: int
                The number of minutes a model must remain idle after closing a position.
                
        Raises:
            RuntimeError:
                If the Epoch's Singleton has been initialized
                If the previous epoch is still in the root directory.
                If the epoch directory already exists.
                If the candlestick bundle is not in place
            ValueError:
                If any of the provided values is invalid
        """
        # Make sure the epoch can be created
        Epoch._can_epoch_be_created(
            seed=seed,
            id=id,
            epoch_width=epoch_width,
            sma_window_size=sma_window_size,
            train_split=train_split,
            validation_split=validation_split,
            regression_lookback=regression_lookback,
            regression_predictions=regression_predictions,
            position_size=position_size,
            leverage=leverage,
            idle_minutes_on_position_close=idle_minutes_on_position_close
        )

        # Calculate the number of days in the epoch's width
        epoch_width_days: int = ceil(epoch_width * 30)

        # Extract the predictions candlestick df
        print("1/10) Extracting the Prediction Candlesticks DF...")
        prediction_df: DataFrame = Epoch._extract_epoch_prediction_candlesticks_df(epoch_width_days)

        # Calculate the epoch's date ranges
        print("2/10) Calculating the Epoch Range...")
        start: int = int(prediction_df.iloc[0]["ot"])
        end: int = int(prediction_df.iloc[-1]["ct"])
        test_ds_start, test_ds_end = Epoch._calculate_date_range(
            prediction_df, ceil(epoch_width_days * (1 - train_split))
        )

        # Initialize the Candlesticks
        print("5/10) Initializing Candlesticks...")
        highest_price, lowest_price = (0, 0)#TODO

        # Initialize the Epoch's directories
        print("5/10) Creating Directories...")
        EpochPath.init_directories(id)

        # Save the epoch's config file
        print("7/10) Saving Epoch Configuration...")
        epoch_config: IEpochConfig = {
            "seed": seed,
            "id": id,
            "sma_window_size": sma_window_size,
            "train_split": train_split,
            "validation_split": validation_split,
            "start": start,
            "end": end,
            "test_ds_start": test_ds_start,
            "test_ds_end": test_ds_end,
            "highest_price": highest_price,
            "lowest_price": lowest_price,
            "regression_lookback": regression_lookback,
            "regression_predictions": regression_predictions,
            "position_size": position_size,
            "leverage": leverage,
            "idle_minutes_on_position_close": idle_minutes_on_position_close
        }
        Configuration.update_epoch_config(epoch_config)

        # Add the Epoch's Directory to the gitignore file
        print("8/10) Adding Epoch to .gitignore file...")
        Epoch.add_epoch_to_gitignore_file(id)

        # Create the Epoch's Receipt
        print("9/10) Creating receipt...")
        Epoch._create_epoch_receipt(epoch_config)







    @staticmethod
    def _can_epoch_be_created(
        seed: int,
        id: str, 
        epoch_width: int,
        sma_window_size: int,
        train_split: float,
        validation_split: float,
        regression_lookback: int,
        regression_predictions: int,
        position_size: float,
        leverage: int,
        idle_minutes_on_position_close: int
    ) -> None:
        """Verifies if an Epoch can be created. Raises an error if any of the
        conditions is not met.

        Args:
            seed: int
            id: str
            epoch_width: int
            sma_window_size: int
            train_split: float
            validation_split: float
            regression_lookback: int
            regression_predictions: int
            position_size: float
            leverage: int
            idle_minutes_on_position_close: int
        Raises:
            RuntimeError:
                If the Epoch's Singleton has been initialized
                If the previous epoch is still in the root directory.
                If the epoch directory already exists.
                If the candlestick bundle is not in place
            ValueError:
                If any of the provided values is invalid
        """
        # Make sure the epoch has not been initialized
        if Epoch.INITIALIZED:
            raise RuntimeError("A new Epoch cannot be created if the singleton has been initialized.")

        # Validate the provided seed
        if not isinstance(seed, int) or seed < 1 or seed > 100000000:
            raise ValueError(f"The provided seed is invalid {seed}. It must be an int ranging 1-100000000")

        # Validate the provided id
        if not isinstance(id, str) or id[0] != "_" or len(id) < 4:
            raise ValueError(f"The provided Epoch ID {id} is invalid. It must contain at least 4 characters and be prefixed with _")

        # Validate the provided epoch width
        if not isinstance(epoch_width, int) or epoch_width < 6 or epoch_width > 48:
            raise ValueError(f"The provided epoch_width is invalid {epoch_width}. It must be an int ranging 6-48")

        # Validate the provided sma_window_size
        if not isinstance(sma_window_size, int) or sma_window_size < 10 or sma_window_size > 300:
            raise ValueError(f"The provided sma_window_size is invalid {sma_window_size}. It must be an int ranging 10-300")

        # Validate the provided train_split
        if not isinstance(train_split, float) or train_split < 0.6 or train_split > 0.95:
            raise ValueError(f"The provided train_split is invalid {train_split}. It must be an float ranging 0.6-0.95")

        # Validate the provided validation_split
        if not isinstance(validation_split, float) or validation_split < 0.15 or validation_split > 0.4:
            raise ValueError(f"The provided validation_split is invalid {validation_split}. It must be an float ranging 0.15-0.4")

        # Validate the provided regression_lookback
        if not isinstance(regression_lookback, int) or regression_lookback < 32 or regression_lookback > 512:
            raise ValueError(f"The provided regression_lookback is invalid {regression_lookback}. It must be an int ranging 32-512")

        # Validate the provided regression_predictions
        if not isinstance(regression_predictions, int) or regression_predictions < 32 or regression_predictions > 256:
            raise ValueError(f"The provided regression_predictions is invalid {regression_predictions}. It must be an int ranging 32-256")

        # Validate the provided position_size
        if not isinstance(position_size, (int, float)) or position_size < 100 or position_size > 100000000:
            raise ValueError(f"The provided position_size is invalid {position_size}. It must be an float ranging 100-100000000")

        # Validate the provided leverage
        if not isinstance(leverage, int) or leverage < 1 or leverage > 5:
            raise ValueError(f"The provided leverage is invalid {leverage}. It must be an int ranging 1-5")

        # Validate the provided idle_minutes_on_position_close
        if not isinstance(idle_minutes_on_position_close, int) or idle_minutes_on_position_close < 0 or idle_minutes_on_position_close > 1000:
            raise ValueError(f"The provided idle_minutes_on_position_close is invalid {idle_minutes_on_position_close}. It must be an int ranging 0-1000")

        # Retrieve the current epoch (if any)
        current_config: Union[IEpochConfig, None] = Configuration.get_epoch_config(allow_empty=True)

        # Make sure the previous epoch is no longer in the project
        if current_config is not None and Utils.directory_exists(current_config["id"]):
            raise RuntimeError(f"Cannot create a new epoch because the previous one is still in the root directory: {current_config['id']}")

        # Make sure the new epoch doesn't already exist
        if Utils.directory_exists(id):
            raise RuntimeError(f"The epoch directory {id} already exists.")

        # Make sure the candlestick bundle exists
        if not Utils.file_exists(Candlestick.DEFAULT_CANDLESTICK_CONFIG["csv_file"]) or\
            not Utils.file_exists(Candlestick.PREDICTION_CANDLESTICK_CONFIG["csv_file"]):
            raise RuntimeError(f"The candlestick bundle must exist for an Epoch to be created.")






    @staticmethod
    def _extract_epoch_prediction_candlesticks_df(epoch_width_days: int) -> DataFrame:
        """Extracts the prediction candlesticks only covering the epoch. This df is 
        used to calculate date ranges as well as generating the 
        normalized prediction candlesticks df.

        Args:
            epoch_width_days: int
                The number of days that comprise the epoch.

        Returns:
            DataFrame
        """
        # Extract the entire csv
        raw_df: DataFrame = Candlestick._get_df(Candlestick.PREDICTION_CANDLESTICK_CONFIG)

        # Subset the rows that are part of the epoch
        df: DataFrame = raw_df.iloc[-(Epoch._calculate_candlesticks_in_range(epoch_width_days)):]
        df.reset_index(drop=True, inplace=True)

        # Finally, return the DF
        return df







    @staticmethod
    def _calculate_date_range(df: DataFrame, days: int) -> Tuple[int, int]:
        """Calculates the date range for a given number of days.

        Args:
            df: DataFrame
                The prediction candlesticks dataframe.
            days: int
                The number of days that comprise the date range that will be calculated

        Returns:
            Tuple[int, int]
            (start, end)
        """
        # Subset the last items based on the range
        range_df: DataFrame = df.iloc[-(Epoch._calculate_candlesticks_in_range(days)):]

        # Finally, return the first ot and the last ct
        return int(range_df.iloc[0]["ot"]), int(range_df.iloc[-1]["ct"])






    @staticmethod
    def _calculate_candlesticks_in_range(days: int) -> int:
        """Calculates the number of prediction candlesticks that fit within a given
        number of days

        Args:
            days: int
                The number of days in which the candlesticks will be fit.

        Returns:
            int
        """
        mins_in_a_day: int = 24 * 60
        candles_in_a_day: float = mins_in_a_day / Candlestick.PREDICTION_CANDLESTICK_CONFIG["interval_minutes"]
        return ceil(candles_in_a_day * days)






    @staticmethod
    def add_epoch_to_gitignore_file(epoch_id: str) -> None:
        """Loads the entire .gitignore file and appends the epoch's
        id at the end of it.

        Args:
            epoch_id: str
                The ID of the epoch to be added to the gitignore file.
        """
        # Init the path of the file
        path: str = "./.gitignore"

        # Init the file
        gitignore: str = Utils.read(path)

        # Append the new Epoch
        gitignore += f"\n{epoch_id}"

        # Save the file
        Utils.write(path, gitignore)




    @staticmethod
    def _create_epoch_receipt(config: IEpochConfig) -> None:
        """Creates the Epoch's receipt and stores it in the root directory.

        Args:
            config: IEpochConfig
                The configuration that was used to create the epoch.
        """
        # Init values
        receipt: str = f"{config['id']}\n\n"

        # General
        receipt += f"Creation: {Utils.from_milliseconds_to_date_string(Utils.get_time())}\n"
        receipt += f"Seed: {config['seed']}\n"
        receipt += f"Train Split: {config['train_split']}\n"
        receipt += f"Validation Split: {config['validation_split']}\n"
        receipt += f"Regression Lookback: {config['regression_lookback']}\n"
        receipt += f"Regression Predictions: {config['regression_predictions']}\n"
        receipt += f"Position Size: ${config['position_size']}\n"
        receipt += f"Leverage: x{config['leverage']}\n"
        receipt += f"Idle Minutes On Position Close: {config['idle_minutes_on_position_close']}\n"

        # Price Range
        receipt += "\n\nPrice Range:\n"
        receipt += f"Highest: {config['highest_price']}\n"
        receipt += f"Lowest: {config['lowest_price']}\n"

        # Epoch Date Range
        receipt += "\n\nEpoch Range:\n"
        receipt += f"Start: {Utils.from_milliseconds_to_date_string(config['start'])}\n"
        receipt += f"End: {Utils.from_milliseconds_to_date_string(config['end'])}\n"

        # Test Dataset Date Range
        receipt += "\nTest Dataset Range:\n"
        receipt += f"Start: {Utils.from_milliseconds_to_date_string(config['test_ds_start'])}\n"
        receipt += f"End: {Utils.from_milliseconds_to_date_string(config['test_ds_end'])}\n"

        # Finally, save the receipt
        Utils.write(f"{config['id']}/{config['id']}_receipt.txt", receipt)








    ## Initialization ##


    @staticmethod
    def init() -> None:
        """Initializes an already created Epoch.

        Raises:
            RuntimeError: 
                if the epoch has already been initialized.
        """
        # Make sure the Epoch has not already been initialized
        if Epoch.INITIALIZED:
            raise RuntimeError("The Epoch can only be initialized once.")

        # Load the configuration file
        config: IEpochConfig = Configuration.get_epoch_config()

        # Make sure the Epoch's directory also exists
        if not Utils.directory_exists(config["id"]):
            raise RuntimeError(f"The Epochs directory does not exist {config['id']}.")

        # Populate epoch properties
        Epoch.SEED = config["seed"]
        Epoch.ID = config["id"]
        Epoch.SMA_WINDOW_SIZE = config["sma_window_size"]
        Epoch.TRAIN_SPLIT = config["train_split"]
        Epoch.VALIDATION_SPLIT = config["validation_split"]
        Epoch.START = config["start"]
        Epoch.END = config["end"]
        Epoch.TEST_DS_START = config["test_ds_start"]
        Epoch.TEST_DS_END = config["test_ds_end"]
        Epoch.HIGHEST_PRICE = config["highest_price"]
        Epoch.LOWEST_PRICE = config["lowest_price"]
        Epoch.REGRESSION_LOOKBACK = config["regression_lookback"]
        Epoch.REGRESSION_PREDICTIONS = config["regression_predictions"]
        Epoch.POSITION_SIZE = config["position_size"]
        Epoch.LEVERAGE = config["leverage"]
        Epoch.IDLE_MINUTES_ON_POSITION_CLOSE = config["idle_minutes_on_position_close"]

        # Set a static seed on all required libraries
        seed(Epoch.SEED)
        npseed(Epoch.SEED)
        tf_random.set_seed(Epoch.SEED)

        # Initialize the File Instance
        Epoch.PATH = EpochPath(Epoch.ID)
        
        # Set the state of the Epoch as Initialized
        Epoch.INITIALIZED = True












    ## Export ## 




    @staticmethod
    def export() -> None:
        """Builds the Epoch's Manifest, as well as all the required assets.
        The Epoch's File is saved in zip format and placed in the root directory
        of the Epoch. If any changes need to be made, the original archive will be
        replaced.

        Raises:
            RuntimeError:
                If the class training data for unit tests has not been set
                If the take profit or the stop loss has not been set
                If the epoch's directory does not exist.
                ...
        """
        # Retrieve the Epoch
        config: IEpochConfig = Configuration.get_epoch_config()

        # Make sure the epoch can be exported
        Epoch._can_epoch_be_exported(config)

        # @TODO






    @staticmethod
    def _can_epoch_be_exported(config: IEpochConfig) -> None:
        """Verifies if an Epoch can be exported. It raises an error if any
        if the requirements is not met.

        Args:
            config: IEpochConfig
                The configuration of the epoch that will be exported

        Raises:
            RuntimeError:
                If the epoch's directory does not exist.
                ...
        """
        # Make sure the epoch's directory exists
        if not Utils.directory_exists(config["id"]):
            raise RuntimeError(f"The Epoch directory {config['id']} does not exist.")

        # @TODO