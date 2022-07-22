from typing import List, Union, Tuple
from random import seed
from numpy.random import seed as npseed
from tensorflow import random as tf_random
from math import ceil
from pandas import DataFrame
from modules.types import IEpochConfig, IEpochDefaults
from modules.utils.Utils import Utils
from modules.candlestick.Candlestick import Candlestick
from modules.epoch.EpochFile import EpochFile
from modules.epoch.CandlestickNormalization import normalize_prediction_candlesticks
from modules.epoch.PredictionRangeIndexer import create_indexer
from modules.epoch.BacktestConfigFactory import BacktestConfigFactory
from modules.epoch.EpochDefaultFiles import create_default_files



# Class
class Epoch:
    """Epoch Class

    This singleton manages the initialization, creation and exporting of epochs.

    Class Properties:
        INITIALIZED: bool
            Indicates if the Epoch has been initialized.
        DEFAULTS: IEpochDefaults
            The default values that will be set if no data is provided through the CLI.
        SEED: int
            The set that will be used to set randomness in all required libs
        ID: str
            The ID of the Epoch.
        START: int
        END: int
            The range of the Epoch. These values are used for:
            1) Calculate the training evaluation range (epoch_width * 0.1)
            2) Calculate the backtest range (epoch_width * 0.2)
        TRAINING_EVALUATION_START: int
        TRAINING_EVALUATION_END: int
            The training evaluation range is used for the following:
            1) Backtest ArimaModels in all position exit combinations
            2) Evaluate freshly trained Regression Models
            3) Backtest shortlisted RegressionModels in all position exit combinations
            4) Evaluate freshly trained Classification Models
            training_evaluation_range = epoch_width * 0.1
        BACKTEST_START: int
        BACKTEST_END: int
            The backtest range is used for the following:
            1) Backtest shortlisted ClassificationModels
            2) Backtest generated ConsensusModels
            backtest_range = epoch_width * 0.2
        HIGHEST_PRICE: float
        LOWEST_PRICE: float
            Highest and lowest price within the Epoch. These values are stored as 
            they are used to scale the prediction candlesticks for trainable models.
            If the price was to go above the highest or below the lowest price, trading should be
            stopped and a new epoch should be published once the market is "stable"
        REGRESSION_PRICE_CHANGE_REQUIREMENT: float
            This value is used to evaluate Keras & XGB Regression Models.
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
        "epoch_width": 36,
        "seed": 60184,
        "regression_price_change_requirement": 3,
        "idle_minutes_on_position_close": 30
    }

    # The set that will be used to set randomness in all required libs
    SEED: int

    # Identifier, must be preffixed with "_EPOCHNAME"
    ID: str

    # The date range of the Epoch
    START: int
    END: int

    # Training Evaluation Range
    TRAINING_EVALUATION_START: int
    TRAINING_EVALUATION_END: int

    # Backtest Range
    BACKTEST_START: int
    BACKTEST_END: int

    # Normalization Price Range
    HIGHEST_PRICE: float
    LOWEST_PRICE: float

    # Regression Price Change Requirement
    REGRESSION_PRICE_CHANGE_REQUIREMENT: float

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
        regression_price_change_requirement: float,
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
            seed: int
                The random seed to be set on all required libs and machines.
            regression_price_change_requirement: float
                The best position exit combination known so far.
            idle_minutes_on_position_close: int
                The number of minutes a model must remain idle after closing a position.
                
        Raises:
            RuntimeError:
                If the Epoch's Singleton has been initialized
                If the previous epoch is still in the root directory.
                If the epoch directory already exists.
                If the candlestick bundle is not in place
            ValueError:
                If the id is invalid.
                If the epoch_width is invalid.
                If the regression_price_change_requirement is invalid.
                If the idle_minutes_on_position_close is invalid.
        """
        # Make sure the epoch can be created
        Epoch._can_epoch_be_created(
            id=id,
            epoch_width=epoch_width,
            seed=seed,
            regression_price_change_requirement=regression_price_change_requirement,
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
        training_evaluation_start, training_evaluation_end = Epoch._calculate_date_range(prediction_df, ceil(epoch_width_days * 0.1))
        backtest_start, backtest_end = Epoch._calculate_date_range(prediction_df, ceil(epoch_width_days * 0.2))

        # Check if the normalized prediction candlesticks csv needs to be created
        print("3/10) Creating the Normalized Prediction Candlesticks CSV...")
        highest_price, lowest_price = normalize_prediction_candlesticks(prediction_df)

        # Initialize the candlesticks based on the prediction range lookbacks
        lookbacks: List[int] = [ 100, 300 ]
        Candlestick.init(max(lookbacks), start=start, end=end)

        # Check if the candlesticks' prediction range need to be indexed
        if EpochFile.file_exists(Candlestick.PREDICTION_RANGE_INDEXER_PATH):
            print("4/10) Creating Prediction Range Indexer: Skipped")
        else:
            create_indexer(lookbacks, "4/10) Creating Prediction Range Indexer")

        # Initialize the Epoch's directories
        print("5/10) Creating Directories...")
        EpochFile.create_epoch_directories(id)

        # Generate Arima's Backtest Configurations
        print("6/10) Generating Arima Backtest Configuration Files...")
        BacktestConfigFactory.build_arima_backtest_configs(id, idle_minutes_on_position_close)

        # Create the Epoch's Default Files
        print("7/10) Creating Default Files...")
        create_default_files(id)

        # Save the epoch's config file
        print("8/10) Saving Epoch Configuration...")
        epoch_config: IEpochConfig = {
            "seed": seed,
            "id": id,
            "start": start,
            "end": end,
            "training_evaluation_start": training_evaluation_start,
            "training_evaluation_end": training_evaluation_end,
            "backtest_start": backtest_start,
            "backtest_end": backtest_end,
            "highest_price": highest_price,
            "lowest_price": lowest_price,
            "regression_price_change_requirement": regression_price_change_requirement,
            "idle_minutes_on_position_close": idle_minutes_on_position_close
        }
        EpochFile.update_epoch_config(epoch_config)

        # Add the Epoch's Directory to the gitignore file
        print("9/10) Adding Epoch to .gitignore file...")
        Epoch.add_epoch_to_gitignore_file(id)

        # Create the Epoch's Receipt
        print("10/10) Creating receipt...")
        Epoch._create_epoch_receipt(epoch_config)







    @staticmethod
    def _can_epoch_be_created(
        id: str, 
        epoch_width: int,
        seed: int,
        regression_price_change_requirement: float,
        idle_minutes_on_position_close: int
    ) -> None:
        """Verifies if an Epoch can be created. Raises an error if any of the
        conditions is not met.

        Args:
            id: str
            epoch_width: int
            seed: int
            regression_price_change_requirement: float
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
                If the regression_price_change_requirement is invalid.
                If the idle_minutes_on_position_close is invalid.
        """
        # Make sure the epoch has not been initialized
        if Epoch.INITIALIZED:
            raise RuntimeError("A new Epoch cannot be created if the singleton has been initialized.")

        # Validate the provided id
        if not isinstance(id, str) or id[0] != "_" or len(id) < 4:
            raise ValueError(f"The provided Epoch ID {id} is invalid. It must contain at least 4 characters and be prefixed with _")

        # Validate the provided epoch width
        if not isinstance(epoch_width, int) or epoch_width < 6 or epoch_width > 48:
            raise ValueError(f"The provided epoch_width is invalid {epoch_width}. It must be an int ranging 6-48")

        # Validate the provided seed
        if not isinstance(seed, int) or seed < 1 or seed > 100000000:
            raise ValueError(f"The provided seed is invalid {seed}. It must be an int ranging 1-100000000")

        # Validate the provided regression_price_change_requirement
        if not isinstance(regression_price_change_requirement, (int, float)) or regression_price_change_requirement < 1 or regression_price_change_requirement > 5:
            raise ValueError(f"The provided regression_price_change_requirement is invalid {regression_price_change_requirement}. It must be a float ranging 1-5")

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
        df: DataFrame = Candlestick._get_df(Candlestick.PREDICTION_CANDLESTICK_CONFIG)

        # Subset the rows that are part of the epoch
        return df.iloc[-(Epoch._calculate_candlesticks_in_range(epoch_width_days)):]







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
        gitignore: str = EpochFile.read(path)

        # Append the new Epoch
        gitignore += f"\n{epoch_id}"

        # Save the file
        EpochFile.write(path, gitignore)




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
        receipt += f"Regression Price Change Requirement: {config['regression_price_change_requirement']}%\n"
        receipt += f"Idle Minutes On Position Close: {config['idle_minutes_on_position_close']}\n"

        # Price Range
        receipt += "\n\nPrice Range:\n"
        receipt += f"Highest: {config['highest_price']}\n"
        receipt += f"Lowest: {config['lowest_price']}\n"

        # Epoch Date Range
        receipt += "\n\nEpoch Range:\n"
        receipt += f"Start: {Utils.from_milliseconds_to_date_string(config['start'])}\n"
        receipt += f"End: {Utils.from_milliseconds_to_date_string(config['end'])}\n"

        # Training Evaluation Date Range
        receipt += "\nTraining Evaluation Range:\n"
        receipt += f"Start: {Utils.from_milliseconds_to_date_string(config['training_evaluation_start'])}\n"
        receipt += f"End: {Utils.from_milliseconds_to_date_string(config['training_evaluation_end'])}\n"

        # Backtest Date Range
        receipt += "\nBacktest Range:\n"
        receipt += f"Start: {Utils.from_milliseconds_to_date_string(config['backtest_start'])}\n"
        receipt += f"End: {Utils.from_milliseconds_to_date_string(config['backtest_end'])}"

        # Finally, save the receipt
        EpochFile.write(f"{config['id']}/{config['id']}_receipt.txt", receipt)








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
        config: IEpochConfig = EpochFile.get_epoch_config()

        # Make sure the Epoch's directory also exists
        if not EpochFile.directory_exists(config["id"]):
            raise RuntimeError(f"The Epochs directory does not exist {config['id']}.")

        # Populate epoch properties
        Epoch.SEED = config["seed"]
        Epoch.ID = config["id"]
        Epoch.START = config["start"]
        Epoch.END = config["end"]
        Epoch.TRAINING_EVALUATION_START = config["training_evaluation_start"]
        Epoch.TRAINING_EVALUATION_END = config["training_evaluation_end"]
        Epoch.BACKTEST_START = config["backtest_start"]
        Epoch.BACKTEST_END = config["backtest_end"]
        Epoch.HIGHEST_PRICE = config["highest_price"]
        Epoch.LOWEST_PRICE = config["lowest_price"]
        Epoch.REGRESSION_PRICE_CHANGE_REQUIREMENT = config["regression_price_change_requirement"]
        Epoch.IDLE_MINUTES_ON_POSITION_CLOSE = config["idle_minutes_on_position_close"]
        Epoch.UT_CLASS_TRAINING_DATA_ID = config.get("ut_class_training_data_id")
        Epoch.TAKE_PROFIT = config.get("take_profit")
        Epoch.STOP_LOSS = config.get("stop_loss")

        # Set a static seed on all required libraries
        seed(Epoch.SEED)
        npseed(Epoch.SEED)
        tf_random.set_seed(Epoch.SEED)

        # Initialize the File Instance
        Epoch.FILE = EpochFile(Epoch.ID)
        
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

        Raises:
            ValueError:
                If the id is invalid.
                If the training data file is not in the correct directory.
        """
        # Make sure the provided id is valid
        if not Utils.is_uuid4(id):
            raise ValueError(f"The provided id {id} does not meet the uuid4 requirements.")

        # Retrieve the current configuration
        config: IEpochConfig = EpochFile.get_epoch_config()

        # Make sure the training data file is in the right directory
        td_path: str = f"{config['id']}/{EpochFile.MODEL_PATH['classification_training_data']}/{id}.json"
        if not EpochFile.file_exists(td_path):
            raise ValueError(f"The training data file was not found in the path: {td_path}")

        # Set the id
        config["ut_class_training_data_id"] = id

        # Update the file
        EpochFile.update_epoch_config(config)









    ## Position Exit Combination ##


    @staticmethod
    def set_position_exit_combination(take_profit: float, stop_loss: float) -> None:
        """Updates the Epoch's Configuration File and sets the provided
        position exit combination. These values will be used when generating the 
        training data and evaluating classification models.

        Args:
            take_profit: float
            stop_loss: float
                The position exit combination to be set on the epoch based on the 
                Regression Selection's results.

        Raises:
            ValueError:
                If the take_profit or stop_loss is invalid.
            RuntimeError:
                If the Classification Training Data for Unit Tests has not been set.
        """
        # Validate the provided take profit and stop loss
        if not isinstance(take_profit, (int, float)) or take_profit < 1 or take_profit > 5:
            raise ValueError(f"The take profit must be a valid float ranging 1-5. Instead, received: {take_profit}")
        if not isinstance(stop_loss, (int, float)) or stop_loss < 1 or stop_loss > 5:
            raise ValueError(f"The stop loss must be a valid float ranging 1-5. Instead, received: {stop_loss}")

        # Retrieve the current configuration
        config: IEpochConfig = EpochFile.get_epoch_config()

        # Make sure the classification training data for unit tests has been set
        td_id: Union[str, None] = config.get("ut_class_training_data_id")
        if not isinstance(td_id, str) or not Utils.is_uuid4(td_id):
            raise RuntimeError("The Classification Training Data for Unit Tests must be set prior to the position exit combination.")

        # Set the combination
        config["take_profit"] = take_profit
        config["stop_loss"] = stop_loss

        # Update the file
        EpochFile.update_epoch_config(config)










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
        config: IEpochConfig = EpochFile.get_epoch_config()

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
                If the class training data for unit tests has not been set
                If the take profit or the stop loss has not been set
                If the epoch's directory does not exist.
                ...
        """
        # Make sure the values that are meant to be set overtime have all been populated
        ut_class_training_data_id: int = config.get("ut_class_training_data_id")
        take_profit: float = config.get("take_profit")
        stop_loss: float = config.get("stop_loss")
        if not isinstance(ut_class_training_data_id, str):
            raise RuntimeError("The Classification Training Data for Unit Tests has not been set.")
        if not isinstance(take_profit, (int, float)) or not isinstance(stop_loss, (int, float)):
            raise RuntimeError("The Take Profit or the Stop Loss has not been set.")

        # Make sure the epoch's directory exists
        if not EpochFile.directory_exists(config["id"]):
            raise RuntimeError("The Epoch directory does not exist.")

        # @TODO