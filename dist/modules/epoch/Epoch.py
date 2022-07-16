from typing import List, Union, Tuple
from random import seed
from numpy.random import seed as npseed
from tensorflow import random as tf_random
from math import ceil
from pandas import DataFrame
from modules.types import IEpochConfig, IEpochDefaults
from modules.candlestick.Candlestick import Candlestick
from modules.epoch.EpochFile import EpochFile
from modules.epoch.PredictionRangeIndexer import create_indexer



# Class
class Epoch:
    """Epoch Class

    This singleton manages the initialization, creation and exporting of epochs.

    Class Properties:
        INITIALIZED: bool
            Indicates if the Epoch has been initialized.
        DEFAULTS: IEpochDefaults
            The default values that will be set if no data is provided through the CLI.
        ID: str
            The ID of the Epoch.
        START: int
        END: int
            The range of the Epoch and is used to calculate the backtest and training ranges
        BACKTEST_START: int
        BACKTEST_END: int
            The range that will be used to backtest all the models (epoch_width * 0.5)
        TRAINING_START: int
        TRAINING_END: int
            The range that will be used to train the Keras & XGBoost Models (epoch_width * 1.5)
        IDLE_MINUTES_ON_POSITION_CLOSE: int
            The number of minutes a model must remain idle once a position is closed.
        UT_CLASS_TRAINING_DATA_ID: Union[str, None]
            The ID of the training data that will be used in the unit tests.
        TAKE_PROFIT: Union[float, None]
        STOP_LOSS: Union[float, None]
            The Exit Combination that came victorious in the Regression Selection Process.
            This value is set once the RegressionSelection has concluded.
    """
    # Initialization State
    INITIALIZED: bool = False

    # Epoch Defaults
    DEFAULTS: IEpochDefaults = {
        "epoch_width": 365,
        "seed": 60184,
        "price_change_requirement": 3,
        "idle_minutes_on_position_close": 30
    }

    # Identifier, must be preffixed with "_EPOCHNAME"
    ID: str

    # The date range of the Epoch
    START: int
    END: int

    # The range that will be used to backtest all the models (epoch_width * 0.5)
    BACKTEST_START: int
    BACKTEST_END: int

    # The range that will be used to train the Keras & XGBoost Models (epoch_width * 1.5)
    TRAINING_START: int
    TRAINING_END: int

    # Price Change Requirement
    PRICE_CHANGE_REQUIREMENT: float

    # Idle minutes on position close
    IDLE_MINUTES_ON_POSITION_CLOSE: int

    # The identifier of the classification training data for unit tests
    UT_CLASS_TRAINING_DATA_ID: Union[str, None] = None

    # Best Exit Combinations - Populated based on the Regression Selection Results
    TAKE_PROFIT: Union[float, None] = None
    STOP_LOSS: Union[float, None] = None

    # EpochFile Instance
    FILE: EpochFile





    ## Creation ##


    @staticmethod
    def create(
        id: str, 
        epoch_width: int,
        seed: int,
        price_change_requirement: float,
        idle_minutes_on_position_close: int
    ) -> None:
        """Creates all the neccessary directories and files for the epoch
        to get started.
        
        Args:
            id: str
                The identifier of the epoch. Make sure to never reuse these names
                as the core infrastructure will have validations to prohibit this.
            epoch_width: int
                The number of days that comprise the epoch. This value is used
                to calculate the start and end timestamps of the epoch. Notice that the
                training timestamps will be calculated as follows: epoch_width * 1.5.
            seed: int
                The random seed to be set on all required libs and machines.
            price_change_requirement: float
                The best position exit combination known so far.
            idle_minutes_on_position_close: int
                The number of minutes a model must remain idle after closing a position.
        """
        # Make sure the epoch can be created
        Epoch._can_epoch_be_created(id)

        # Initialize the candlesticks based on the prediction range lookbacks
        lookbacks: List[int] = [ 100, 300 ]
        Candlestick.init(max(lookbacks))

        # Check if the candlesticks' prediction range need to be indexed
        if EpochFile.file_exists(Candlestick.PREDICTION_RANGE_INDEXER_PATH):
            print("1/6) Creating Prediction Range Indexer: Skipped")
        else:
            create_indexer(lookbacks, "1/6) Creating Prediction Range Indexer")

        # Calculate the epoch's date ranges
        print("2/6) Calculating the Epoch Range...")
        start, end = Epoch._calculate_date_range(epoch_width)
        backtest_start, backtest_end = Epoch._calculate_date_range(ceil(epoch_width * 0.5))
        training_start, training_end = Epoch._calculate_date_range(ceil(epoch_width * 1.5))

        # Initialize the Epoch's directories
        print("3/6) Creating Directories...")
        EpochFile.create_epoch_directories(id)

        # Generate Arima's Backtest Configurations
        print("4/6) Generating Arima Backtest Configuration Files...")


        # Create the Epoch's Default Files
        print("5/6) Creating Default Files...")


        # Save the epoch's config file
        print("6/6) Saving Epoch Configuration...")
        EpochFile.update_epoch_config({
            "seed": seed,
            "id": id,
            "start": start,
            "end": end,
            "backtest_start": backtest_start,
            "backtest_end": backtest_end,
            "training_start": training_start,
            "training_end": training_end,
            "price_change_requirement": price_change_requirement,
            "idle_minutes_on_position_close": idle_minutes_on_position_close
        })





    @staticmethod
    def _can_epoch_be_created(
        id: str, 
        epoch_width: int,
        seed: int,
        price_change_requirement: float,
        idle_minutes_on_position_close: int
    ) -> None:
        """Verifies if an Epoch can be created. Raises an error if any of the
        conditions is not met.

        Args:
            id: str
            epoch_width: int
            seed: int
            price_change_requirement: float
            idle_minutes_on_position_close: int
        Raises:
            RuntimeError:
                If the Epoch's Singleton has been initialized
                If the previous epoch is still in the root directory.
                If the epoch directory already exists.
                If the candlestick bundle is not in place
            ValueError:
                If the id is invalid.
                If the epoch_width is invalid.
                If the price_change_requirement is invalid.
                If the idle_minutes_on_position_close is invalid.
        """
        # Make sure the epoch has not been initialized
        if Epoch.INITIALIZED:
            raise RuntimeError("A new Epoch cannot be created if the singleton has been initialized.")

        # Validate the provided id
        if not isinstance(id, str) or id[0] != "_" or len(id) < 4:
            raise ValueError(f"The provided Epoch ID {id} is invalid. It must contain at least 4 characters and be prefixed with _")

        # Validate the provided epoch width
        if not isinstance(epoch_width, int) or epoch_width < 100 or epoch_width > 1000:
            raise ValueError(f"The provided epoch_width is invalid {epoch_width}. It must be an int ranging 100-1000")

        # Validate the provided seed
        if not isinstance(seed, int) or seed < 1 or seed > 100000000:
            raise ValueError(f"The provided seed is invalid {seed}. It must be an int ranging 1-100000000")

        # Validate the provided price_change_requirement
        if not isinstance(price_change_requirement, (int, float)) or price_change_requirement < 1 or price_change_requirement > 10:
            raise ValueError(f"The provided price_change_requirement is invalid {price_change_requirement}. It must be a float ranging 1-10")

        # Validate the provided idle_minutes_on_position_close
        if not isinstance(idle_minutes_on_position_close, int) or idle_minutes_on_position_close < 0 or idle_minutes_on_position_close > 1000:
            raise ValueError(f"The provided idle_minutes_on_position_close is invalid {idle_minutes_on_position_close}. It must be an int ranging 0-1000")

        # Retrieve the current epoch (if any)
        current_config: Union[IEpochConfig, None] = EpochFile.get_epoch_config(allow_empty=True)

        # Make sure the previous epoch is no longer in the project
        if current_config is not None and EpochFile.directory_exists(current_config["id"]):
            raise RuntimeError(f"Cannot create a new epoch because the previous one is still in the root directory: {current_config['id']}")

        # Make sure the new epoch doesn't already exist
        if EpochFile.directory_exists(id):
            raise RuntimeError(f"The epoch directory {id} already exists.")

        # Make sure the candlestick bundle exists
        if not EpochFile.file_exists(Candlestick.DEFAULT_CANDLESTICK_CONFIG["csv_file"]) or\
            not EpochFile.file_exists(Candlestick.PREDICTION_CANDLESTICK_CONFIG["csv_file"]):
            raise RuntimeError(f"The candlestick bundle must exist for an Epoch to be created.")






    @staticmethod
    def _calculate_date_range(epoch_width: int) -> Tuple[int, int]:
        """Based on the provided epoch_width (Number of days), it will calculate
        the start and end timestamps.

        Args:
            epoch_width: int
                The number of days that comprise the epoch.

        Returns:
            Tuple[int, int]
            (start, end)
        """
        # Calculate the number of candlesticks that will be in the range
        mins_in_a_day: int = 24 * 60
        candles_in_a_day: float = mins_in_a_day / Candlestick.PREDICTION_CANDLESTICK_CONFIG["interval_minutes"]
        candles_in_range: int = ceil(candles_in_a_day * epoch_width)

        # Subset the last items based on the range
        df: DataFrame = Candlestick.PREDICTION_DF.iloc[-candles_in_range:]

        # Finally, return the first ot and the last ct
        return int(df.iloc[0]["ot"]), int(df.iloc[-1]["ct"])














    ## Init ##


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
        #config: IEpochConfig = EpochFile.get_epoch_config()

        # Make sure the Epoch's directory also exists
        #if not EpochFile.directory_exists(config["id"]):
        #    raise RuntimeError(f"The Epochs directory does not exist {config['id']}.")

        # Populate epoch properties
        #Epoch.ID = config["id"]
        #Epoch.START = config["start"]
        #Epoch.END = config["end"]
        #Epoch.BACKTEST_START = config["backtest_start"]
        #Epoch.BACKTEST_END = config["backtest_end"]
        #Epoch.TRAINING_START = config["training_start"]
        #Epoch.TRAINING_END = config["training_end"]
        #Epoch.PRICE_CHANGE_REQUIREMENT = config["price_change_requirement"]
        #Epoch.IDLE_MINUTES_ON_POSITION_CLOSE = config["idle_minutes_on_position_close"]
        #Epoch.UT_CLASS_TRAINING_DATA_ID = config.get("ut_class_training_data_id")

        # Set a static seed on all required libraries
        seed(Epoch.DEFAULTS["seed"])
        npseed(Epoch.DEFAULTS["seed"])
        tf_random.set_seed(Epoch.DEFAULTS["seed"])

        # Initialize the File Instance
        #Epoch.FILE = EpochFile(Epoch.ID)
        
        # Set the state of the Epoch as Initialized
        Epoch.INITIALIZED = True










    ## Classification Training Data Unit Test ##



    @staticmethod
    def set_ut_class_training_data_id(id: str) -> None:
        """Updates the Epoch's Configuration File and sets the provided
        classification training data id to be used in the unit tests.

        Args:
            id: str
                The identifier of the training data.
        """
        # Retrieve the current configuration
        config: IEpochConfig = EpochFile.get_epoch_config()

        # Set the id
        config["ut_class_training_data_id"] = id

        # Update the file
        EpochFile.update_epoch_config(config)









    ## Position Exit Combination ##


    @staticmethod
    def set_position_exit_combination(take_profit: float, stop_loss: float) -> None:
        """Updates the Epoch's Configuration File and sets the provided
        position exit combination. This value will be used when generating the 
        training data and evaluating classification models.

        Args:
            take_profit: float
            stop_loss: float
                The position exit combination to be set on the epoch based on the 
                Regression Selection's results.
        """
        # Retrieve the current configuration
        config: IEpochConfig = EpochFile.get_epoch_config()

        # Set the combination
        config["take_profit"] = take_profit
        config["stop_loss"] = stop_loss

        # Update the file
        EpochFile.update_epoch_config(config)










    ## Export ## 


    @staticmethod
    def export() -> None:
        """
        """
        pass
