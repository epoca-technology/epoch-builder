from typing import List, Union, Any
from os import makedirs
from os.path import exists, isfile, dirname, splitext
from shutil import rmtree
from json import load, dumps
from dist.modules.types.backtest_types import IBacktestID
from modules.types import IConfigPath, IBacktestAssetsPath, IModelAssetsPath, IEpochConfig, \
    IBacktestConfig, IRegressionTrainingBatch, ITrainingDataConfig, IClassificationTrainingBatch,\
        IBacktestResult, ITrainingDataFile
from modules.model.ModelType import TRAINABLE_MODEL_TYPES
from modules.utils.Utils import Utils
from modules.epoch.PositionExitCombination import PositionExitCombination




# Class
class EpochFile:
    """EpochFile Class

    This class handles all the interactions with the file system. Some methods are static
    as some processes cannot initialize the Epoch Module.

    Class Properties:
        CONFIG_PATH: IConfigPath
            The paths to the configuration files that govern the software.
        BACKTEST_PATH: IBacktestAssetsPath
            The paths for all the directories within the backtest_assets directory.
        MODEL_PATH: IModelAssetsPath
            The paths for all the directories within the model_assets directory.

    Instance Properties:
        epoch_id: str
            The identifier of the epoch and root directory for all the assets.
    """
    # Configuration Files' Paths
    CONFIG_PATH: IConfigPath = {
        "epoch":                                    "config/Epoch.json",
        "backtest":                                 "config/Backtest.json",
        "classification_training_data":             "config/ClassificationTrainingData.json",
        "keras_classification_training":            "config/KerasClassificationTraining.json",
        "keras_regression_training":                "config/KerasRegressionTraining.json",
        "xgb_classification_training":              "config/XGBClassificationTraining.json",
        "xgb_regression_training":                  "config/XGBRegressionTraining.json",
    }

    # Backtest Files' Paths
    BACKTEST_PATH: IBacktestAssetsPath = {
        "assets":                                   "backtest_assets",
        "configurations":                           "backtest_assets/configurations",
        "results":                                  "backtest_assets/results",
        "regression_selection":                     "backtest_assets/regression_selection",
    }

    # Model Files' Paths
    MODEL_PATH: IModelAssetsPath = {
        "assets":                                   "model_assets",
        "batched_training_certificates":            "model_assets/batched_training_certificates",
        "classification_training_data":             "model_assets/classification_training_data",
        "classification_training_data_configs":     "model_assets/classification_training_data_configs",
        "models":                                   "model_assets/models",
        "models_bank":                              "model_assets/models_bank",
        "keras_classification_training_configs":    "model_assets/keras_classification_training_configs",
        "keras_regression_training_configs":        "model_assets/keras_regression_training_configs",
        "xgb_classification_training_configs":      "model_assets/xgb_classification_training_configs",
        "xgb_regression_training_configs":          "model_assets/xgb_regression_training_configs",
    }



    def __init__(self, epoch_id: str) -> None:
        """Initializes the EpochFile instance.

        Args:
            epoch_id: str
                The ID of the current epoch.
        """
        # Init the identifier
        self.epoch_id: str = epoch_id









    ## Epoch Files Management ##







    # Classification Training Data
    # The classification training data is saved once the process is completed and then
    # read in order to validate the integrity of the file.


    def get_classification_training_data(self, id: str) -> ITrainingDataFile:
        """Retrieves a Classification Training Data File.

        Args:
            id: str
                The ID of the training data.

        Returns:
            ITrainingDataFile

        Raises:
            RuntimeError:
                If the training data file does not exist.
        """
        # Initialize the path
        path: str = self._p(f"{EpochFile.MODEL_PATH['classification_training_data']}/{id}.json")

        # Return the results
        return EpochFile.read(path)




    def save_classification_training_data(self, training_data: ITrainingDataFile) -> None:
        """Saves the training data file in the appropiate directory.

        Args:
            training_data: ITrainingDataFile
                The training data file data.
        """
        # Init the path
        path: str = self._p(f"{EpochFile.MODEL_PATH['classification_training_data']}/{training_data['id']}.json")

        # Save the file
        EpochFile.write(path, training_data)









    # Backtest Results Management
    # The backtest results are saved once all the models within have completed. These results
    # can also be read by modules such as RegressionSelection.




    def get_backtest_results(self, backtest_id: IBacktestID, take_profit: float, stop_loss: float) -> List[IBacktestResult]:
        """Retrieves all the results for a given backtest.

        Args:
            backtest_id: IBacktestID
                The Id of the backtest.
            take_profit: float
            stop_loss: float
                The position exit combination of the backtest.

        Returns:
            List[IBacktestResult]

        Raises:
            RuntimeError:
                If the backtest result file does not exist.
        """
        # Initialize the path
        path: str = self._p(
            f"{EpochFile.BACKTEST_PATH['results']}/{PositionExitCombination.get_path(take_profit, stop_loss)}/{backtest_id}.json"
        )

        # Return the results
        return EpochFile.read(path)






    def save_backtest_results(self, results: List[IBacktestResult]) -> None:
        """Saves a series of backtest results in the corresponding
        directories. If the backtest is the unit test, it will save 
        it in the root of the backtest results ignoring the position
        exit combination.

        Args:
            results: List[IBacktestResult]
                The list of backtest results to be stored.

        Raises:
            ValueError:
                If the results list is empty.
        """
        # Make sure that at least 1 result has been provided
        if len(results) == 0:
            raise ValueError("Cannot save the backtest results because the provided list is empty.")

        # Init values
        path: str = self._p(EpochFile.BACKTEST_PATH['results'])
        id: IBacktestID = results[0]["backtest"]["id"]
        tp: float = results[0]["backtest"]["take_profit"]
        sl: float = results[0]["backtest"]["stop_loss"]

        # Check if it is the unit test
        if len(results) == 0 and "unit_test" in id:
            path = f"{path}/{id}.json"

        # Otherwise, save the result based on the position exit combination
        else:
            path = f"{path}/{PositionExitCombination.get_path(tp, sl)}/{id}.json"

        # Save the results
        EpochFile.write(path, results)





    # Misc Helpers


    def _p(self, path: str) -> str:
        """Adds the Epoch's name to the beggining of a given path.

        Args:
            path: str
                The path that will be completed with the epoch id.

        Returns: 
            str
        """
        return f"{self.epoch_id}/{path}"











    ## Epoch Directories Creation ##




    # Epoch Directories
    # For the Epoch to be able to operate in a scalable way, it needs to follow
    # strict guidelines when storing configurations, results, models, etc.
    # This function creates the entire skeleton for both, backtest and model 
    # management.


    @staticmethod
    def create_epoch_directories(epoch_id: str) -> None:
        """Creates all the directories required for the epoch to function.

        Args:
            epoch_id: str
                The identifier of the epoch.
        """
        # Create all the backtest asset directories
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.BACKTEST_PATH['assets']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.BACKTEST_PATH['configurations']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.BACKTEST_PATH['results']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.BACKTEST_PATH['regression_selection']}")
        for exit_combination in PositionExitCombination.get_records():
            EpochFile.make_directory(f"{epoch_id}/{EpochFile.BACKTEST_PATH['configurations']}/{exit_combination['path']}")
            EpochFile.make_directory(f"{epoch_id}/{EpochFile.BACKTEST_PATH['results']}/{exit_combination['path']}")

        # Create all the model asset directories
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['assets']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['batched_training_certificates']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['batched_training_certificates']}/unit_tests")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['classification_training_data']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['classification_training_data_configs']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['models']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['models_bank']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['models_bank']}/unit_tests")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['keras_classification_training_configs']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['keras_regression_training_configs']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['xgb_classification_training_configs']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['xgb_regression_training_configs']}")
        for trainable_model in TRAINABLE_MODEL_TYPES:
            EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['batched_training_certificates']}/{trainable_model}")
            EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['models_bank']}/{trainable_model}")















    ## Configuration Files Management ##







    # Epoch Configuration
    # The Epoch Configuration File holds global configuration variables that are
    # used by several modules.

    @staticmethod
    def get_epoch_config(allow_empty: bool = False) -> Union[IEpochConfig, None]:
        """Retrieves the Epoch Configuration.

        Args:
            allow_empty: bool
                If enabled, an error won't be raised in case the file doesn't exist
                and instead returns None.

        Returns:
            Union[IEpochConfig, None]
        """
        return EpochFile.read(EpochFile.CONFIG_PATH["epoch"], allow_empty=allow_empty)


    @staticmethod
    def update_epoch_config(new_config: IEpochConfig) -> None:
        """Updates the current Epoch Configuration.

        Args:
            new_config: IEpochConfig
                The new config to be set on the file
        """
        return EpochFile.write(EpochFile.CONFIG_PATH["epoch"], data=new_config, indent=4)




    # Backtest Configuration
    # The Backtest Configuration File holds the configuration that will be used
    # to run the Backtest Process.

    def get_backtest_config(self) -> IBacktestConfig:
        """Retrieves the Backtest Configuration.

        Returns:
            IBacktestConfig
        """
        return EpochFile.read(EpochFile.CONFIG_PATH["backtest"])




    # Classification Training Data Configuration
    # The configuration file holds the data that will be used to generate the Classification
    # Training Data


    
    def get_classification_training_data_config(self) -> ITrainingDataConfig:
        """Retrieves the ClassificationTrainingData Configuration.

        Returns:
            ITrainingDataConfig
        """
        return EpochFile.read(EpochFile.CONFIG_PATH["classification_training_data"])



    # Keras Regression Training Configuration
    # The configuration file holds the data that will be used to train Keras Regression Models.
    
    def get_keras_regression_training_config(self) -> IRegressionTrainingBatch:
        """Retrieves the configuration for training Keras Regression Models.

        Returns:
            IRegressionTrainingBatch
        """
        return EpochFile.read(EpochFile.CONFIG_PATH["keras_regression_training"])

    
    def update_keras_regression_training_config(self, new_config: IRegressionTrainingBatch) -> None:
        """Updates the Keras Regression Training configuration.

        Args:
            new_config: IRegressionTrainingBatch
                The new config to be set on the file
        """
        return EpochFile.write(EpochFile.CONFIG_PATH["keras_regression_training"], data=new_config, indent=4)




    # Keras Classification Training Configuration
    # The configuration file holds the data used to train Keras Classification Models.

    def get_keras_classification_training_config(self) -> IClassificationTrainingBatch:
        """Retrieves the KerasClassificationTraining Configuration.

        Returns:
            IClassificationTrainingBatch
        """
        return EpochFile.read(EpochFile.CONFIG_PATH["keras_classification_training"])

   
    def update_keras_classification_training_config(self, new_config: IClassificationTrainingBatch) -> None:
        """Updates the current KerasClassificationTraining Configuration.

        Args:
            new_config: IClassificationTrainingBatch
                The new config to be set on the file
        """
        return EpochFile.write(EpochFile.CONFIG_PATH["keras_classification_training"], data=new_config, indent=4)




    # XGBoost Regression Training Configuration
    # The configuration file holds the data that will be used to train XGB Regression Models.
    
    def get_xgb_regression_training_config(self) -> Any:
        """Retrieves the configuration for training XGB Regression Models.

        Returns:
            ...
        """
        raise NotImplemented("XGBoost based models have not yet been implemented.")

    
    def update_xgb_regression_training_config(self, new_config: Any) -> None:
        """Updates the XGB Regression Training configuration.

        Args:
            new_config: ...
                ...
        """
        raise NotImplemented("XGBoost based models have not yet been implemented.")




    # XGBoost Classification Training Configuration
    # The configuration file holds the data used to train XGB Classification Models.

    def get_xgb_classification_training_config(self) -> Any:
        """Retrieves the XGBClassificationTraining Configuration.

        Returns:
            ...
        """
        raise NotImplemented("XGBoost based models have not yet been implemented.")

   
    def update_xgb_classification_training_config(self, new_config: Any) -> None:
        """Updates the current XGBClassificationTraining Configuration.

        Args:
            new_config: ...
                ...
        """
        raise NotImplemented("XGBoost based models have not yet been implemented.")












    ## File System Management ##






    # Path Existance


    @staticmethod
    def directory_exists(path: str) -> bool:
        """Checks if a given directory path exists.

        Args:
            path: str
                The path to be checked for existance.

        Returns:
            bool
        """
        return exists(path)





    @staticmethod
    def file_exists(path: str) -> bool:
        """Checks if a given file path exists.

        Args:
            path: str
                The path to be checked for existance.

        Returns:
            bool
        """
        return isfile(path)








    # Directory Management



    @staticmethod
    def make_directory(path: str) -> None:
        """Creates a directory at a given path if it doesnt already exist.

        Args:
            path: str
                The path in which the directory should be created.
        """
        if not EpochFile.directory_exists(path):
            makedirs(path)





    @staticmethod
    def remove_directory(path: str) -> None:
        """Removes a directory and its contents.

        Args:
            path: str
                The path of the directory that will be removed.

        Raises:
            RuntimeError:
                If the directory does not exist
        """
        # Make sure the directory exists
        if not EpochFile.directory_exists(path):
            raise RuntimeError(f"The directory {path} does not exist.")

        # Remove the directory
        rmtree(path)








    # File Reading / Writting




    @staticmethod
    def read(path: str, allow_empty: bool = False) -> Any:
        """Reads a file located at a given path and returns
        its contents.

        Args:
            path: str
                The path in which the file is located.
            allow_empty: bool
                If enabled, the function won't raise an error if the file
                does not exist.
        
        Returns:
            Any

        Raises:
            RuntimeError:
                If the file does not exist and allow_empty is set to False.
        """
        # Check if the file exists
        if EpochFile.file_exists(path):
            # Split the path into path name and extension
            _, extension = splitext(path)

            # Read the file according to its format
            if extension == ".json":
                return load(open(path))
            else:
                return open(path).read()

        # Otherwise, check if an error needs to raised
        else:
            if allow_empty:
                return None
            else:
                raise RuntimeError(f"The file {path} does not exist.")






    @staticmethod
    def write(path: str, data: Any, timestamp_file_name: bool = False, indent: Union[int, None] = None) -> None:
        """Writes a file on given path.

        Args:
            path: str
                The path of the file that will be written. Note that if the file
                exists, it will overwrite it.
            data: Any
                The data to be stored in the file. If it is a JSON file, the data must
                be compatible.
            timestamp_file_name: bool
                If enabled, it will append the current timestamp to the file name in the
                following way: my_file.json -> my_file_1657887972213.json
            indent: Union[int, None]
                The indenting to be applied on the JSON File. Defaults to no indenting.
        """
        # Make sure the directory exists
        dir_name: str = dirname(path)
        if not EpochFile.directory_exists(dir_name):
            EpochFile.make_directory(dir_name)

        # Split the path into path name and extension
        path_name, extension = splitext(path)

        # Check if the file name needs to be timestamped
        if timestamp_file_name:
            path_name += f"_{Utils.get_time()}"

        # Write the File based on its format
        with open(path_name + extension, "w") as file_wrapper:
            if extension == ".json":
                file_wrapper.write(dumps(data, indent=indent))
            else:
                file_wrapper.write(data)