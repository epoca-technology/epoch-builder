from typing import List, Union, Tuple
from random import seed
from numpy.random import seed as npseed
from tensorflow import random as tf_random
from math import ceil
from pandas import DataFrame
from modules._types import IEpochConfig, IEpochDefaults
from modules.utils.Utils import Utils
from modules.database.Database import Database
from modules.candlestick.Candlestick import Candlestick
from modules.epoch.EpochFile import EpochFile
from modules.epoch.CandlestickNormalization import normalize_prediction_candlesticks
from modules.epoch.PredictionRangeIndexer import create_indexer
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
        TRAIN_SPLIT: float
            The split that will be applied to the epoch_width in order to train models.
        START: int
        END: int
            The range of the Epoch. These values are used for:
            1) Calculate the training evaluation range (1 - train_split)
            2) Calculate the backtest range (epoch_width * backtest_split)
        TRAINING_EVALUATION_START: int
        TRAINING_EVALUATION_END: int
            The training evaluation range is used for the following:
            1) Evaluate freshly trained Regression Models
            2) Evaluate freshly trained Classification Models
            training_evaluation_range = 1 - train_split
        BACKTEST_START: int
        BACKTEST_END: int
            The backtest range is used for the following:
            1) Discover Regressions & Classifications
            2) Backtest shortlisted ClassificationModels
            3) Backtest generated ConsensusModels
            backtest_range = epoch_width * backtest_split
        HIGHEST_PRICE: float
        LOWEST_PRICE: float
            Highest and lowest price within the Epoch. These values are stored as 
            they are used to scale the prediction candlesticks for trainable models.
            If the price was to go above the highest or below the lowest price, trading should be
            stopped and a new epoch should be published once the market is "stable"
        MODEL_DISCOVERY_STEPS: int
            This value is used to discover Regressions and Classifications
        IDLE_MINUTES_ON_POSITION_CLOSE: int
            The number of minutes a model must remain idle once a position is closed.
        CLASSIFICATION_TRAINING_DATA_ID_UT: Union[str, None]
            The ID of the training data that will be used in the unit tests.
    """
    # Initialization State
    INITIALIZED: bool = False

    # Epoch Defaults
    DEFAULTS: IEpochDefaults = {
        "epoch_width": 24,
        "train_split": 0.85,
        "backtest_split": 0.3,
        "seed": 60184,
        "model_discovery_steps": 7,
        "idle_minutes_on_position_close": 30
    }

    # The set that will be used to set randomness in all required libs
    SEED: int

    # Identifier, must be preffixed with "_EPOCHNAME"
    ID: str

    # Split used to train models
    TRAIN_SPLIT: float

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

    # Steps used to discover models
    MODEL_DISCOVERY_STEPS: int

    # Idle minutes on position close
    IDLE_MINUTES_ON_POSITION_CLOSE: int

    # The identifier of the classification training data for unit tests
    CLASSIFICATION_TRAINING_DATA_ID_UT: Union[str, None] = None

    # EpochFile Instance
    FILE: EpochFile








    ## Creation ##


    @staticmethod
    def create(
        id: str, 
        epoch_width: int,
        seed: int,
        train_split: float,
        backtest_split: float,
        model_discovery_steps: int,
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
            train_split: float
                The split that will be applied to the epoch_width to train models.
            backtest_split: float
                The split that will be applied to the epoch_width to backtest models. 
            model_discovery_steps: int
                The steps that will be used during the model discovery process.
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
            id=id,
            epoch_width=epoch_width,
            seed=seed,
            train_split=train_split,
            backtest_split=backtest_split,
            model_discovery_steps=model_discovery_steps,
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
        training_evaluation_start, training_evaluation_end = Epoch._calculate_date_range(prediction_df, ceil(epoch_width_days * (1 - train_split)))
        backtest_start, backtest_end = Epoch._calculate_date_range(prediction_df, ceil(epoch_width_days * backtest_split))

        # Check if the normalized prediction candlesticks csv needs to be created
        print("3/10) Creating the Normalized Prediction Candlesticks CSV...")
        highest_price, lowest_price = normalize_prediction_candlesticks(prediction_df)

        # Initialize the candlesticks based on the prediction range lookbacks
        lookbacks: List[int] = [ 100 ]
        Candlestick.init(max(lookbacks), start=start, end=end)

        # Check if the candlesticks' prediction range need to be indexed
        if EpochFile.file_exists(Candlestick.PREDICTION_RANGE_INDEXER_PATH):
            print("4/10) Creating Prediction Range Indexer: Skipped")
        else:
            create_indexer(lookbacks, "4/10) Creating Prediction Range Indexer")

        # Initialize the Epoch's directories
        print("5/10) Creating Directories...")
        EpochFile.create_epoch_directories(id)

        # Create the Epoch's Default Files
        print("6/10) Creating Default Files...")
        create_default_files(id)

        # Save the epoch's config file
        print("7/10) Saving Epoch Configuration...")
        epoch_config: IEpochConfig = {
            "seed": seed,
            "id": id,
            "train_split": train_split,
            "start": start,
            "end": end,
            "training_evaluation_start": training_evaluation_start,
            "training_evaluation_end": training_evaluation_end,
            "backtest_start": backtest_start,
            "backtest_end": backtest_end,
            "highest_price": highest_price,
            "lowest_price": lowest_price,
            "model_discovery_steps": model_discovery_steps,
            "idle_minutes_on_position_close": idle_minutes_on_position_close
        }
        EpochFile.update_epoch_config(epoch_config)

        # Add the Epoch's Directory to the gitignore file
        print("8/10) Adding Epoch to .gitignore file...")
        Epoch.add_epoch_to_gitignore_file(id)

        # Create the Epoch's Receipt
        print("9/10) Creating receipt...")
        Epoch._create_epoch_receipt(epoch_config)

        # Initialize the Database
        print("10/10) Initializing Database...")
        Database.delete_tables()
        Database.initialize_tables()







    @staticmethod
    def _can_epoch_be_created(
        id: str, 
        epoch_width: int,
        seed: int,
        train_split: float,
        backtest_split: float,
        model_discovery_steps: int,
        idle_minutes_on_position_close: int
    ) -> None:
        """Verifies if an Epoch can be created. Raises an error if any of the
        conditions is not met.

        Args:
            id: str
            epoch_width: int
            seed: int
            train_split: float
            backtest_split: float
            model_discovery_steps: int
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

        # Validate the provided id
        if not isinstance(id, str) or id[0] != "_" or len(id) < 4:
            raise ValueError(f"The provided Epoch ID {id} is invalid. It must contain at least 4 characters and be prefixed with _")

        # Validate the provided epoch width
        if not isinstance(epoch_width, int) or epoch_width < 6 or epoch_width > 48:
            raise ValueError(f"The provided epoch_width is invalid {epoch_width}. It must be an int ranging 6-48")

        # Validate the provided seed
        if not isinstance(seed, int) or seed < 1 or seed > 100000000:
            raise ValueError(f"The provided seed is invalid {seed}. It must be an int ranging 1-100000000")

        # Validate the provided train_split
        if not isinstance(train_split, float) or train_split < 0.6 or train_split > 0.95:
            raise ValueError(f"The provided train_split is invalid {train_split}. It must be an float ranging 0.6-0.95")

        # Validate the provided backtest_split
        if not isinstance(backtest_split, float) or backtest_split < 0.2 or backtest_split > 0.6:
            raise ValueError(f"The provided backtest_split is invalid {backtest_split}. It must be an float ranging 0.2-0.6")

        # Validate the provided model_discovery_steps
        if not isinstance(model_discovery_steps, int) or model_discovery_steps < 1 or model_discovery_steps > 20:
            raise ValueError(f"The provided model_discovery_steps is invalid {model_discovery_steps}. It must be an int ranging 1-20")

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
        receipt += f"Train Split: {config['train_split']}\n"
        receipt += f"Model Discovery Steps: {config['model_discovery_steps']}\n"
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
        Epoch.TRAIN_SPLIT = config["train_split"]
        Epoch.START = config["start"]
        Epoch.END = config["end"]
        Epoch.TRAINING_EVALUATION_START = config["training_evaluation_start"]
        Epoch.TRAINING_EVALUATION_END = config["training_evaluation_end"]
        Epoch.BACKTEST_START = config["backtest_start"]
        Epoch.BACKTEST_END = config["backtest_end"]
        Epoch.HIGHEST_PRICE = config["highest_price"]
        Epoch.LOWEST_PRICE = config["lowest_price"]
        Epoch.MODEL_DISCOVERY_STEPS = config["model_discovery_steps"]
        Epoch.IDLE_MINUTES_ON_POSITION_CLOSE = config["idle_minutes_on_position_close"]
        Epoch.CLASSIFICATION_TRAINING_DATA_ID_UT = config.get("classification_training_data_id_ut")

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
    def set_classification_training_data_id_ut(id: str) -> None:
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
        config["classification_training_data_id_ut"] = id

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
        if not isinstance(ut_class_training_data_id, str):
            raise RuntimeError("The Classification Training Data for Unit Tests has not been set.")

        # Make sure the epoch's directory exists
        if not EpochFile.directory_exists(config["id"]):
            raise RuntimeError("The Epoch directory does not exist.")

        # @TODO