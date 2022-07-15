from typing import List, Union, Any
from os import makedirs
from os.path import exists, isfile, dirname, splitext
from shutil import rmtree
from json import load, dumps
from modules.types import IConfigPath, IBacktestAssetsPath, IModelAssetsPath, IEpochConfig, \
    IBacktestConfig, IRegressionTrainingBatch, ITrainingDataConfig, IClassificationTrainingBatch
from modules.utils.Utils import Utils




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
        "epoch":                                "config/Epoch.json",
        "backtest":                             "config/Backtest.json",
        "regression_training":                  "config/RegressionTraining.json",
        "classification_training_data":         "config/ClassificationTrainingData.json",
        "classification_training":              "config/ClassificationTraining.json"
    }

    # Backtest Files' Paths
    BACKTEST_PATH: IBacktestAssetsPath = {
        "assets":                               "backtest_assets",
        "configurations":                       "backtest_assets/configurations",
        "results":                              "backtest_assets/results",
        "regression_selection":                 "backtest_assets/regression_selection",
    }

    # Model Files' Paths
    MODEL_PATH: IModelAssetsPath = {
        "assets":                               "model_assets",
        "batched_training_certificates":        "model_assets/batched_training_certificates",
        "classification_training_configs":      "model_assets/classification_training_configs",
        "classification_training_data":         "model_assets/classification_training_data",
        "classification_training_data_configs": "model_assets/classification_training_data_configs",
        "models":                               "model_assets/models",
        "models_bank":                          "model_assets/models_bank",
        "regression_training_configs":          "model_assets/regression_training_configs",
    }



    def __init__(self, epoch_id: str) -> None:
        """
        """
        # Init the identifier
        self.epoch_id: str = epoch_id






    ## Epoch Paths ##









    def _epoch_path(self, path: str) -> str:
        """Adds the Epoch's name to the beggining of a given path.

        Args:
            path: str
                The path that will be completed with the epoch id.

        Returns: 
            str
        """
        return f"{self.epoch_id}/{path}"











    ## Epoch Directories Init ##



















    ## Configuration Files Management ##





    # Epoch Configuration



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



    def get_backtest_config(self) -> IBacktestConfig:
        """Retrieves the Backtest Configuration.

        Returns:
            IBacktestConfig
        """
        return EpochFile.read(EpochFile.CONFIG_PATH["backtest"])







    # Regression Training Configuration



    
    def get_regression_training_config(self) -> IRegressionTrainingBatch:
        """Retrieves the RegressionTraining Configuration.

        Returns:
            IRegressionTrainingBatch
        """
        return EpochFile.read(EpochFile.CONFIG_PATH["regression_training"])




    
    def update_regression_training_config(self, new_config: IRegressionTrainingBatch) -> None:
        """Updates the current Epoch Configuration.

        Args:
            new_config: IRegressionTrainingBatch
                The new config to be set on the file
        """
        return EpochFile.write(EpochFile.CONFIG_PATH["regression_training"], data=new_config, indent=4)






    # Classification Training Data Configuration




    
    def get_classification_training_data_config(self) -> ITrainingDataConfig:
        """Retrieves the ClassificationTrainingData Configuration.

        Returns:
            ITrainingDataConfig
        """
        return EpochFile.read(EpochFile.CONFIG_PATH["classification_training_data"])







    # Classification Training Configuration



    
    def get_classification_training_config(self) -> IClassificationTrainingBatch:
        """Retrieves the ClassificationTraining Configuration.

        Returns:
            IClassificationTrainingBatch
        """
        return EpochFile.read(EpochFile.CONFIG_PATH["classification_training"])




   
    def update_classification_training_config(self, new_config: IClassificationTrainingBatch) -> None:
        """Updates the current ClassificationTraining Configuration.

        Args:
            new_config: IClassificationTrainingBatch
                The new config to be set on the file
        """
        return EpochFile.write(EpochFile.CONFIG_PATH["classification_training"], data=new_config, indent=4)















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
        """Creates a directory at a given path.

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








    # JSON File Reading / Writting




    @staticmethod
    def read(path: str, allow_empty: bool = False) -> Any:
        """Reads a JSON file located at a given path and returns
        its contents.

        Args:
            path: str
                The path in which the JSON file is located.
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
            return load(open(path))

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