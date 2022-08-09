from typing import List, Union, Any, Tuple
from os import makedirs, listdir
from os.path import exists, isfile, dirname, splitext
from shutil import rmtree, move
from json import load, dumps
from modules._types import IConfigPath, IBacktestAssetsPath, IModelAssetsPath, IEpochConfig, \
    IBacktestConfig, IKerasRegressionTrainingBatch, ITrainingDataConfig, IKerasClassificationTrainingBatch,\
        IBacktestResult, ITrainingDataFile, IKerasClassificationTrainingCertificate, IKerasRegressionTrainingCertificate,\
            ITrainableModelType, ITrainableModelExtension, IBacktestID, IRegressionSelectionFile, \
                IXGBClassificationTrainingBatch, IXGBRegressionTrainingBatch, IXGBClassificationTrainingCertificate,\
                    IXGBRegressionTrainingCertificate
from modules.model.ModelType import TRAINABLE_MODEL_TYPES
from modules.utils.Utils import Utils
from modules.model.ModelType import get_trainable_model_type







## Trainable Model Type Helpers ##

# Training Batch
ITrainingBatch = Union[
    IKerasRegressionTrainingBatch, IKerasClassificationTrainingBatch, 
    IXGBRegressionTrainingBatch, IXGBClassificationTrainingBatch
]

# Training Certificate
ITrainingCertificate = Union[
    IKerasRegressionTrainingCertificate, IKerasClassificationTrainingCertificate,
    IXGBRegressionTrainingCertificate, IXGBClassificationTrainingCertificate
]

# Training Certificate Lists
ITrainingCertificateList = Union[
    List[IKerasRegressionTrainingCertificate], List[IKerasClassificationTrainingCertificate],
    List[IXGBRegressionTrainingCertificate], List[IXGBClassificationTrainingCertificate]
]







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
        "keras_classification_training":            "config/KerasClassificationTraining.json",
        "keras_regression_training":                "config/KerasRegressionTraining.json",
        "xgb_classification_training":              "config/XGBClassificationTraining.json",
        "xgb_regression_training":                  "config/XGBRegressionTraining.json",
    }

    # Backtest Files' Paths
    BACKTEST_PATH: IBacktestAssetsPath = {
        "assets":                                   "backtest_assets",
        "configurations":                           "backtest_assets/configurations",
        "regression_selection":                     "backtest_assets/regression_selection",
        "results":                                  "backtest_assets/results"
    }

    # Model Files' Paths
    MODEL_PATH: IModelAssetsPath = {
        "assets":                                   "model_assets",
        "batched_training_certificates":            "model_assets/batched_training_certificates",
        "classification_training_data":             "model_assets/classification_training_data",
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






    # Models Management
    # When models are trained, a model file and a certificate file are created and 
    # placed inside of the model's directory.
    # Active models live in the models directory and the entire lib should be placed 
    # in the models_bank directory. The Model Activation functionality will handle
    # the management of these dirs.





    def save_training_certificate(self, certificate: ITrainingCertificate) -> None:
        """Saves a training certificate batch into the proper directory.

        Args:
            certificate: ITrainingCertificate
                The training certificate to be stored.
        """
        EpochFile.write(self.get_active_model_certificate_path(certificate["id"]), certificate)






    def save_training_certificate_batch(
        self, 
        trainable_model_type: ITrainableModelType, 
        batch_name: str, 
        certificates: ITrainingCertificateList
    ) -> None:
        """Saves a training certificate batch into the proper directory.

        Args:
            trainable_model_type: ITrainableModelType
                The type of models that are being trained.
            batch_name: str
                The name of the training batch.
            certificates: ITrainingCertificateList
                The certificates built on training completion.
        """
        # Init the path
        path: str = self._p(EpochFile.MODEL_PATH["batched_training_certificates"])

        # If it is a unit test, save the file in the unit test directory
        if "UNIT_TEST" in batch_name:
            path = f"{path}/unit_tests/{batch_name}.json"

        # Otherwise, save it according to the model type
        else:
            path = f"{path}/{trainable_model_type}/{batch_name}.json"

        # Finally, save the batch
        EpochFile.write(path, certificates, timestamp_file_name=True)





    def move_trained_models_to_bank(self, model_type: ITrainableModelType, certificates: ITrainingCertificateList) -> None:
        """Given a list of training certificates, it will move all the models from
        the active directory into the bank.

        Args:
            model_type: ITrainableModelType, 
                The type of the trainable model.
            certificates: ITrainingCertificateList
                The list of certificates issued when training the models.
        """
        for cert in certificates:
            EpochFile.move_file_or_dir(
                source=self.get_active_model_dir_path(cert["id"]),
                destination=self.get_banked_model_dir_path(cert["id"], model_type),
            )











    # Model Management Misc Helpers
    # All models must be kept in the models_bank directory. Only active models should
    # be placed in the models directory.


    # General


    def model_exists(self, model_id: str, model_type: ITrainableModelType) -> bool:
        """Checks if a model exists in the active and bank directories.

        Args:
            model_id: str
                The identifier of the model.
            model_type: ITrainableModelType
                The type of model.

        Returns:
            bool
        """
        return EpochFile.file_exists(self.get_active_model_path(model_id, model_type)) or\
            EpochFile.file_exists(self.get_banked_model_path(model_id, model_type))








    # Active Model


    def get_active_model_dir_path(self, model_id: str) -> str:
        """Retrieves the path of a directory that holds the model and certificate files

        Args:
            model_id: str
                The identifier of the model.

        Returns:
            str
        """
        return self._p(f"{EpochFile.MODEL_PATH['models']}/{model_id}")






    def make_active_model_dir(self, model_id: str) -> None:
        """Keras Models are stored using the h5 format and therefore, it does not
        make use of the EpochFile.write functionality. For this reason,
        the directory of the model should be created prior to saving it.

        Args:
            model_id: str
                The ID of the model
        """
        EpochFile.make_directory(self.get_active_model_dir_path(model_id))




    def get_active_model_path(self, model_id: str, model_type: ITrainableModelType) -> str:
        """Retrieves the path of a model based on its type in the active directory (models).

        Args:
            model_id: str
                The identifier of the model.
            model_type: ITrainableModelType
                The type of model.

        Returns:
            str
        """
        return f"{self.get_active_model_dir_path(model_id)}/model.{self.get_model_extension(model_type)}"





    def get_active_model_certificate_path(self, model_id: str) -> str:
        """Retrieves the path of a model training certificate based on its type in the active directory (models).

        Args:
            model_id: str
                The identifier of the model.

        Returns:
            str
        """
        return f"{self.get_active_model_dir_path(model_id)}/certificate.json"





    def get_active_model_certificate(self, model_id: str) -> ITrainingCertificate:
        """Retrieves a model training certificate based on its type in the active directory (models).

        Args:
            model_id: str
                The identifier of the model.

        Returns:
            ITrainingCertificate
        
        Raises:
            RuntimeError:
                If the certificate does not exist.
        """
        return EpochFile.read(self.get_active_model_certificate_path(model_id))





    def get_active_model_ids(self, model_type: ITrainableModelType, exclude_unit_test: bool = False) -> List[str]:
        """Extracts the ids of all the active models within the models directory.

        Args:
            model_type: ITrainableModelType
                The type of model.
            exclude_unit_test: bool
                If enabled, it won't include the UNIT_TEST model in the list.

        Returns:
            List[str]
        """
        # Retrieve the directory contents
        directories, _ = EpochFile.get_directory_content(self._p(EpochFile.MODEL_PATH["models"]))

        # Filter the directories and add only the ones related to the provided model type.
        model_ids: List[str] = list(filter(lambda x: get_trainable_model_type(x) == model_type, directories))

        # Exclude the unit test model if applies
        if exclude_unit_test:
            model_ids = list(filter(lambda x: "UNIT_TEST" not in x, model_ids))

        # Finally, return the list of ids
        return model_ids




    def active_model_has_certificate(self, model_id: str) -> bool:
        """Verifies if an active model has a training certificate.

        Args:
            model_id: str
                The identifier of the model.

        Returns:
            bool
        """
        return EpochFile.file_exists(self.get_active_model_certificate_path(model_id))





    def remove_active_model(self, model_id: str) -> None:
        """Deletes an active model from the models directory.

        Args:
            model_id: str
                The identifier of the model.
        """
        EpochFile.remove_directory(self.get_active_model_dir_path(model_id))







    # Model Bank



    def get_banked_model_certificate(self, model_id: str) -> ITrainingCertificate:
        """Retrieves a model training certificate based on its type in the banked directory (models_bank).

        Args:
            model_id: str
                The identifier of the model.

        Returns:
            ITrainingCertificate
        
        Raises:
            RuntimeError:
                If the certificate does not exist.
        """
        return EpochFile.read(f"{self.get_banked_model_dir_path(model_id)}/certificate.json")
        





    def get_banked_model_dir_path(self, model_id: str, model_type: ITrainableModelType) -> str:
        """Retrieves the path of a banked model that holds the model and certificate files.

        Args:
            model_id: str
                The identifier of the model.
            model_type: ITrainableModelType
                The type of model.

        Returns:
            str
        """
        return self._p(f"{EpochFile.MODEL_PATH['models_bank']}/{model_type}/{model_id}")





    def get_banked_model_path(self, model_id: str, model_type: ITrainableModelType) -> str:
        """Retrieves the path of a model based on its type in the bank directory (models_bank).

        Args:
            model_id: str
                The identifier of the model.
            model_type: ITrainableModelType
                The type of model.

        Returns:
            str
        """
        return f"{self.get_banked_model_dir_path(model_id, model_type)}/model.{self.get_model_extension(model_type)}"









    # Model File Extension


    def get_model_extension(self, model_type: ITrainableModelType) -> ITrainableModelExtension:
        """Retrieves the extension of a model based on its type.

        Args:
            model_type: ITrainableModelType
                The trainable type of the model.

        Returns:
            ITrainableModelExtension
        """
        if model_type == "keras_regression" or model_type == "keras_classification":
            return "h5"
        elif model_type == "xgb_regression" or model_type == "xgb_classification":
            return "json"
        else:
            raise ValueError(f"Could not find the model extension for: {model_type}.")









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
        return EpochFile.read(self._p(f"{EpochFile.MODEL_PATH['classification_training_data']}/{id}.json"))




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









    # Hyperparams
    # The Hyperparams Module builds many different models and saves the configurations in
    # batches so the models can be trained by multiple machines within the cluster or in 
    # any external environment.
    # In order to find the best possible models, it is recommended to make use of several
    # technologies. 




    def save_hyperparams_batch(
        self, 
        model_type: ITrainableModelType,
        batch_type: str,
        batch: ITrainingBatch
    ) -> None:
        """Saves a Keras Hyperparams batch for a given network.

        Args:
            model_type: ITrainableModelType
                The trainable type of the model.
            batch_type: str
                The type of batch. This value will be used to create the directory
                in which the batches will be placed.
            batch: ITrainingBatch
                The configs batch to be saved.
        """
        # Init the path
        path: str = f"{self.get_hyperparams_dir_path(model_type)}/{batch_type}/{batch['name']}.json"

        # Save the file
        EpochFile.write(path, batch, indent=4)





    def save_hyperparams_receipt(self, model_type: ITrainableModelType, receipt: str) -> None:
        """Saves a Hyperparams receipt that covers the recently generated 
        configurations.

        Args:
            model_type: ITrainableModelType
                The trainable type of the model.
            receipt: str
                The receipt to be stored.
        """
        # Init the path
        path: str = f"{self.get_hyperparams_dir_path(model_type)}/receipt.txt"

        # Save the file
        EpochFile.write(path, receipt)





    def get_hyperparams_dir_path(self, model_type: ITrainableModelType) -> str:
        """Retrieves the path of the directory that holds hyperparam configurations
        by trainable model type.

        Args:
            model_type: ITrainableModelType
                The type of models hyperparams will be generated for.

        Returns:
            str
        """
        # Check if it is a Keras Regression
        if model_type == "keras_regression":
            return self._p(EpochFile.MODEL_PATH["keras_regression_training_configs"])

        # Check if it is a Keras Classification
        elif model_type == "keras_classification":
            return self._p(EpochFile.MODEL_PATH["keras_classification_training_configs"])

        # Check if it is an XGB Regression
        elif model_type == "xgb_regression":
            return self._p(EpochFile.MODEL_PATH["xgb_regression_training_configs"])

        # Check if it is an XGB Classification
        elif model_type == "xgb_classification":
            return self._p(EpochFile.MODEL_PATH["xgb_classification_training_configs"])
        
        # Otherwise, raise an error
        else:
            raise ValueError(f"The provided model_type {model_type} is invalid.")












    # Backtest Results Management
    # The backtest results are saved once all the models within have completed. These results
    # can also be read by modules such as RegressionSelection.




    def get_backtest_results(self, backtest_id: IBacktestID) -> List[IBacktestResult]:
        """Retrieves all the results for a given backtest.

        Args:
            backtest_id: IBacktestID
                The Id of the backtest.

        Returns:
            List[IBacktestResult]

        Raises:
            RuntimeError:
                If the backtest result file does not exist.
        """
        return EpochFile.read(f"{EpochFile.BACKTEST_PATH['results']}/{backtest_id}.json")







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
        path: str = self._p(f"{EpochFile.BACKTEST_PATH['results']}/{results[0]['backtest']['id']}.json")

        # Save the results
        EpochFile.write(path, results)










    # Regression Selection
    # This process is performed in order to find out what regression models and position
    # exit combinations perform best. 



    def save_regression_selection(self, file: IRegressionSelectionFile) -> None:
        """Saves a Regression Selection Result into the proper directory

        Args:
            file: IRegressionSelectionFile
                The selection to be stored
        """
        # Init the path
        path: str = self._p(f"{EpochFile.BACKTEST_PATH['regression_selection']}/{file['id']}.json")

        # Save the file
        EpochFile.write(path, file)
    




    def list_regression_selections(self) -> List[str]:
        """Retrieves the list of regression selection ids in the assets
        directory.

        Returns:
            List[str]
        """
        # Retrieve the directory contents
        _, files = EpochFile.get_directory_content(self._p(EpochFile.BACKTEST_PATH["regression_selection"]))

        # Filter the directories and add only the ones related to the provided model type.
        return list(filter(lambda x: ".json" in x, files))






    def get_regression_selection(self, id: str) -> IRegressionSelectionFile:
        """Retrieves a regression selection file.

        Args:
            id: str
                The identifier of the regression selection.

        Returns:
            IRegressionSelectionFile

        Raises:
            RuntimeError:
                If the regression selection file does not exist.
        """
        return EpochFile.read(self._p(f"{EpochFile.BACKTEST_PATH['regression_selection']}/{id}.json"))










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
        EpochFile.write(EpochFile.CONFIG_PATH["epoch"], data=new_config, indent=4)







    # Backtest Configuration
    # The Backtest Configuration File holds the configuration that will be used
    # to run the Backtest Process.

    def get_backtest_config(self) -> IBacktestConfig:
        """Retrieves the Backtest Configuration.

        Returns:
            IBacktestConfig
        """
        return EpochFile.read(EpochFile.CONFIG_PATH["backtest"])









    # Keras Regression Training Configuration
    # The configuration file holds the data that will be used to train Keras Regression Models.
    

    def get_keras_regression_training_config(self) -> IKerasRegressionTrainingBatch:
        """Retrieves the configuration for training Keras Regression Models.

        Returns:
            IKerasRegressionTrainingBatch
        """
        return EpochFile.read(EpochFile.CONFIG_PATH["keras_regression_training"])

    


    def update_keras_regression_training_config(self, new_config: IKerasRegressionTrainingBatch) -> None:
        """Updates the Keras Regression Training configuration.

        Args:
            new_config: IKerasRegressionTrainingBatch
                The new config to be set on the file
        """
        EpochFile.write(EpochFile.CONFIG_PATH["keras_regression_training"], data=new_config, indent=4)






    # Keras Classification Training Configuration
    # The configuration file holds the data used to train Keras Classification Models.



    def get_keras_classification_training_config(self) -> IKerasClassificationTrainingBatch:
        """Retrieves the KerasClassificationTraining Configuration.

        Returns:
            IKerasClassificationTrainingBatch
        """
        return EpochFile.read(EpochFile.CONFIG_PATH["keras_classification_training"])

   


    def update_keras_classification_training_config(self, new_config: IKerasClassificationTrainingBatch) -> None:
        """Updates the current KerasClassificationTraining Configuration.

        Args:
            new_config: IKerasClassificationTrainingBatch
                The new config to be set on the file
        """
        EpochFile.write(EpochFile.CONFIG_PATH["keras_classification_training"], data=new_config, indent=4)






    # XGBoost Regression Training Configuration
    # The configuration file holds the data that will be used to train XGB Regression Models.
    


    def get_xgb_regression_training_config(self) -> IXGBRegressionTrainingBatch:
        """Retrieves the configuration for training XGB Regression Models.

        Returns:
            IXGBRegressionTrainingBatch
        """
        return EpochFile.read(EpochFile.CONFIG_PATH["xgb_regression_training"])




    
    def update_xgb_regression_training_config(self, new_config: IXGBRegressionTrainingBatch) -> None:
        """Updates the XGB Regression Training configuration.

        Args:
            new_config: IXGBRegressionTrainingBatch
                The configuration to be set on the file.
        """
        EpochFile.write(EpochFile.CONFIG_PATH["xgb_regression_training"], data=new_config, indent=4)







    # XGBoost Classification Training Configuration
    # The configuration file holds the data used to train XGB Classification Models.


    def get_xgb_classification_training_config(self) -> IXGBClassificationTrainingBatch:
        """Retrieves the XGBClassificationTraining Configuration.

        Returns:
            IXGBClassificationTrainingBatch
        """
        return EpochFile.read(EpochFile.CONFIG_PATH["xgb_classification_training"])





    def update_xgb_classification_training_config(self, new_config: IXGBClassificationTrainingBatch) -> None:
        """Updates the current XGBClassificationTraining Configuration.

        Args:
            new_config: IXGBClassificationTrainingBatch
                The configuration to be set on the file.
        """
        EpochFile.write(EpochFile.CONFIG_PATH["xgb_classification_training"], data=new_config, indent=4)















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
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.BACKTEST_PATH['regression_selection']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.BACKTEST_PATH['results']}")

        # Create all the model asset directories
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['assets']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['batched_training_certificates']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['batched_training_certificates']}/unit_tests")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['classification_training_data']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['models']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['models_bank']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['keras_classification_training_configs']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['keras_regression_training_configs']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['xgb_classification_training_configs']}")
        EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['xgb_regression_training_configs']}")
        for trainable_model in TRAINABLE_MODEL_TYPES:
            EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['batched_training_certificates']}/{trainable_model}")
            EpochFile.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['models_bank']}/{trainable_model}")











    ## File System Management ##




    # General


    @staticmethod
    def move_file_or_dir(source: str, destination: str) -> None:
        """Moves a directory or file from source to destination

        Args:
            source: str
                The path that will be moved to the destination.
            destination: str
                The path in which the source will be moved to.
        """
        # Firstly make sure the source exists
        if not EpochFile.file_exists(source) and not EpochFile.directory_exists(source):
            raise RuntimeError(f"The file/dir cannot be moved because the source does not exist: {source}")

        # Finally, move the file/dir
        move(source, destination)







    # Directory/File Existance


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







    @staticmethod
    def get_directory_content(path: str) -> Tuple[List[str], List[str]]:
        """Retrieves all the directories and files located in the
        provided path.

        Args:
            path: str
                The path of the directory

        Returns:
            Tuple[List[str], List[str]]
            (directories, files)
        
        Raises:
            RuntimeError:
                If the directory does not exist.
        """
        # Init values
        directories: List[str] = []
        files: List[str] = []

        # Make sure the directory exists
        if not EpochFile.directory_exists(path):
            raise RuntimeError(f"The directory {path} does not exist.")

        # Iterate over each item in the directory
        for item in listdir(path):
            # Check if it is a file
            if isfile(f"{path}/{item}"):
                files.append(item)
            
            # Otherwise, it is a directory
            else:
                directories.append(item)

        # Finally, return the contents
        return directories, files







    # Read / Write File Actions




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



