from typing import List, Union
from modules._types import IBacktestAssetsPath, IModelAssetsPath, IBacktestConfig, IKerasRegressionTrainingBatch, \
    IKerasClassificationTrainingBatch, IBacktestResult, ITrainingDataFile, IKerasClassificationTrainingCertificate, \
        IKerasRegressionTrainingCertificate, ITrainableModelType, ITrainableModelExtension, IBacktestID, \
            IRegressionSelectionFile, IXGBClassificationTrainingBatch, IXGBRegressionTrainingBatch, \
                IXGBClassificationTrainingCertificate, IXGBRegressionTrainingCertificate, IHyperparamsCategory
from modules.utils.Utils import Utils
from modules.model.ModelType import TRAINABLE_MODEL_TYPES, get_trainable_model_type







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
        BACKTEST_PATH: IBacktestAssetsPath
            The paths for all the directories within the backtest_assets directory.
        MODEL_PATH: IModelAssetsPath
            The paths for all the directories within the model_assets directory.

    Instance Properties:
        epoch_id: str
            The identifier of the epoch and root directory for all the assets.

    Hyperparams:
        save_hyperparams_batch(model_type: ITrainableModelType, batch_type: str, batch: ITrainingBatch) -> None
        save_hyperparams_receipt(model_type: ITrainableModelType, receipt: str) -> None
        list_training_config_categories(self, model_type: ITrainableModelType) -> List[IHyperparamsCategory]
        list_training_config_names(self, model_type: ITrainableModelType, category: IHyperparamsCategory) -> List[str]
        get_training_config_path(self, model_type: ITrainableModelType, category: IHyperparamsCategory, batch_file_name: str) -> str
        get_hyperparams_dir_path(model_type: ITrainableModelType) -> str

    Models:
        save_training_certificate(certificate: ITrainingCertificate) -> None
        save_training_certificate_batch(trainable_model_type: ITrainableModelType, batch_name: str, certificates: ITrainingCertificateList) -> None
        move_trained_models_to_bank(model_type: ITrainableModelType, certificates: ITrainingCertificateList) -> None
        activate_model(model_id: str) -> None
        model_exists(model_id: str, model_type: ITrainableModelType) -> bool
        get_active_model_dir_path(model_id: str) -> str
        make_active_model_dir(model_id: str) -> None
        get_active_model_path(model_id: str, model_type: ITrainableModelType) -> str
        get_active_model_certificate_path(model_id: str) -> str
        get_active_model_certificate(model_id: str) -> ITrainingCertificate
        get_active_model_ids(model_type: ITrainableModelType, exclude_unit_test: bool = False) -> List[str]
        active_model_has_certificate(model_id: str) -> bool
        remove_active_model(model_id: str) -> None
        is_model_active(model_id: str) -> bool
        get_banked_model_certificate(model_id: str, model_type: ITrainableModelType) -> ITrainingCertificate
        get_banked_model_dir_path(model_id: str, model_type: ITrainableModelType) -> str
        get_banked_model_path(model_id: str, model_type: ITrainableModelType) -> str
        get_model_extension(model_type: ITrainableModelType) -> ITrainableModelExtension

    Classification Training Data:
        get_classification_training_data(id: str) -> ITrainingDataFile
        list_classification_training_data_ids() -> List[str]
        save_classification_training_data(training_data: ITrainingDataFile) -> None

    Backtests:
        list_backtest_configs() -> List[str]
        get_backtest_config(file_name: str) -> IBacktestConfig
        save_backtest_config(config: IBacktestConfig) -> None
        get_backtest_results(backtest_id: IBacktestID) -> List[IBacktestResult]
        save_backtest_results(results: List[IBacktestResult]) -> None

    Regression Selection:
        save_regression_selection(file: IRegressionSelectionFile) -> None
        list_regression_selection_ids() -> List[str]
        get_regression_selection(id: str) -> IRegressionSelectionFile

    Epoch Path:
        p(path: str) -> str

    Epoch Directories:
        create_epoch_directories(epoch_id: str) -> None
    """
    # Backtest Files' Paths
    BACKTEST_PATH: IBacktestAssetsPath = {
        "assets":                                   "backtest_assets",
        "configurations":                           "backtest_assets/configurations",
        "results":                                  "backtest_assets/results"
    }

    # Model Files' Paths
    MODEL_PATH: IModelAssetsPath = {
        "assets":                                   "model_assets",
        "batched_training_certificates":            "model_assets/batched_training_certificates",
        "classification_training_data":             "model_assets/classification_training_data",
        "models":                                   "model_assets/models",
        "models_bank":                              "model_assets/models_bank",
        "regression_selection":                     "model_assets/regression_selection",
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









    #########################################################################################
    ## HYPERPARAMS                                                                         ##
    ## The Hyperparams Module builds many different models and saves the configurations in ##
    ## batches so the models can be trained by multiple machines within the cluster or in  ##
    ## any external environment.                                                           ##
    ## In order to find the best possible models, it is recommended to make use of several ##
    ## technologies.                                                                       ##
    ##                                                                                     ##
    ## TRAINING                                                                            ##
    ## Since models can take a very long time to train and the process can be stopped by   ##
    ## things like power outages, the flow needs to be a bit different. Instead of just    ##
    ## reading the configuration file, it should create a temp copy and place it in the    ##
    ## root config directory.                                                              ##
    ## When a batch completes the training, the temp configuration file is deleted from    ##
    ## the root config but maintained in the epoch's directory.                            ##
    ## If a batch is to be interrupted for any reason, the merge_training_certificates     ##
    ## functionality will save whatever progress there was and update the temp config file ##
    ## so the batch can be resumed.                                                        ##
    #########################################################################################



    # Hyperparams Creation


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
        Utils.write(path, batch, indent=4)





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
        Utils.write(path, receipt)






    # Model Training Configuration Management




    def list_training_config_categories(self, model_type: ITrainableModelType) -> List[IHyperparamsCategory]:
        """Retrieves the list of directories within the training configurations. Moreover,
        it appends the unit test config at the end of the list.

        Args:
            model_type: ITrainableModelType
                The type of model being trained.

        Returns:
            List[IHyperparamsCategory]
        """
        # Extract the directories within the hyperparams dir path
        directories, _ = Utils.get_directory_content(self.get_hyperparams_dir_path(model_type))

        # Return the directories within as well as the unit test category
        return directories + [ "UNIT_TEST" ]





    def list_training_config_names(self, model_type: ITrainableModelType, category: IHyperparamsCategory) -> List[str]:
        """Lists the file names of the configuration files within a category.

        Args:
            model_type: ITrainableModelType
                The type of model being trained.
            category: IHyperparamsCategory 
                The category of the batch config.

        Returns:
            List[str]
        """
        # Retrieve all the config files within the category
        _, files = Utils.get_directory_content(f"{self.get_hyperparams_dir_path(model_type)}/{category}", only_file_ext=".json")

        # Create the files with order list
        def _get_order(file_name: str) -> int:
            try:
                return int(file_name.split("_")[-2])
            except Exception as e:
                print(e)
                return 0
        files_with_order: List[str] = [{ "name": f, "order": _get_order(f)} for f in files]
        files_with_order = sorted(files_with_order, key=lambda d: d["order"]) 

        # Finally, return the file names
        return [file["name"] for file in files_with_order]







    def get_training_config_path(self, model_type: ITrainableModelType, category: IHyperparamsCategory, batch_file_name: str) -> str:
        """Retrieves the path for a specific training batch configuration file.

        Args:
            model_type: ITrainableModelType
                The type of model being trained.
            category: IHyperparamsCategory 
                The category of the batch config.
            batch_file_name: str
                The name of the batch config.

        Returns:
            str
        """
        # Check if it is a unit test
        if category == "UNIT_TEST":
            return f"{self.get_hyperparams_dir_path(model_type)}/{batch_file_name}"
        else:
            return f"{self.get_hyperparams_dir_path(model_type)}/{category}/{batch_file_name}"








    # Hyperparams Directory Path
    # Each technology and each type of model has its own dedicated directory in which
    # the configuration files are stored, grouped by the trainable model type.

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
            return self.p(EpochFile.MODEL_PATH["keras_regression_training_configs"])

        # Check if it is a Keras Classification
        elif model_type == "keras_classification":
            return self.p(EpochFile.MODEL_PATH["keras_classification_training_configs"])

        # Check if it is an XGB Regression
        elif model_type == "xgb_regression":
            return self.p(EpochFile.MODEL_PATH["xgb_regression_training_configs"])

        # Check if it is an XGB Classification
        elif model_type == "xgb_classification":
            return self.p(EpochFile.MODEL_PATH["xgb_classification_training_configs"])
        
        # Otherwise, raise an error
        else:
            raise ValueError(f"The provided model_type {model_type} is invalid.")














    #####################################################################################
    ## MODELS                                                                          ##
    ## When models are trained, a model file and a certificate file are created and    ##
    ## placed inside of the model's directory within the active models directory       ##
    ## so it can be evaluated (if applies). Once the evaluation completes, the model's ##
    ## directory is moved into the bank.                                               ##
    ## Later on, a model can be activated if needed.                                   ##
    #####################################################################################





    def save_training_certificate(self, certificate: ITrainingCertificate) -> None:
        """Saves a training certificate batch into the proper directory.

        Args:
            certificate: ITrainingCertificate
                The training certificate to be stored.
        """
        Utils.write(self.get_active_model_certificate_path(certificate["id"]), certificate)






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
        path: str = self.p(EpochFile.MODEL_PATH["batched_training_certificates"])

        # If it is a unit test, save the file in the unit test directory
        if "UNIT_TEST" in batch_name:
            path = f"{path}/unit_tests/{batch_name}.json"

        # Otherwise, save it according to the model type
        else:
            path = f"{path}/{trainable_model_type}/{batch_name}.json"

        # Finally, save the batch
        Utils.write(path, certificates, timestamp_file_name=True)





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
            Utils.move_file_or_dir(
                source=self.get_active_model_dir_path(cert["id"]),
                destination=self.get_banked_model_dir_path(cert["id"], model_type),
            )







    def activate_model(self, model_id: str) -> None:
        """Verifies if a model exists in the active directory. If not, it copies it
        from the bank.

        Args:
            model_id: str
                The ID of the model to be activated.
        """
        # Firstly, make sure the model is not already active
        if not self.is_model_active(model_id):
            # Retrieve the trainable type
            trainable_type: ITrainableModelType = get_trainable_model_type(model_id)

            # Finally, Copy the file to the active directory from the bank
            Utils.copy_file_or_dir(
                source=self.get_banked_model_dir_path(model_id, trainable_type), 
                destination=self.get_active_model_dir_path(model_id)
            )










    # General Model Helpers



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
        return Utils.file_exists(self.get_active_model_path(model_id, model_type)) or\
            Utils.file_exists(self.get_banked_model_path(model_id, model_type))








    # Active Model Misc Helpers
    # An active model is a model that is placed in the models directory and 
    # is ready to be loaded in order to generate predictions.




    def get_active_model_dir_path(self, model_id: str) -> str:
        """Retrieves the path of a directory that holds the model and certificate files

        Args:
            model_id: str
                The identifier of the model.

        Returns:
            str
        """
        return self.p(f"{EpochFile.MODEL_PATH['models']}/{model_id}")







    def make_active_model_dir(self, model_id: str) -> None:
        """Keras Models are stored using the h5 format and therefore, it does not
        make use of the EpochFile.write functionality. For this reason,
        the directory of the model should be created prior to saving it.

        Args:
            model_id: str
                The ID of the model
        """
        Utils.make_directory(self.get_active_model_dir_path(model_id))





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
        return Utils.read(self.get_active_model_certificate_path(model_id))





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
        directories, _ = Utils.get_directory_content(self.p(EpochFile.MODEL_PATH["models"]))

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
        return Utils.file_exists(self.get_active_model_certificate_path(model_id))






    def remove_active_model(self, model_id: str) -> None:
        """Deletes an active model from the models directory.

        Args:
            model_id: str
                The identifier of the model.
        """
        Utils.remove_directory(self.get_active_model_dir_path(model_id))





    def is_model_active(self, model_id: str) -> bool:
        """Verifies if a model is in the active directory.

        Args:
            model_id: str
                The identifier of the model.

        Returns: bool
        """
        return Utils.directory_exists(self.get_active_model_dir_path(model_id))













    # Model Bank
    # Once a model is trained, it is moved from the active directory 
    # into the bank. The active directory should only be left with
    # models that are being used.




    def get_banked_model_certificate(self, model_id: str, model_type: ITrainableModelType) -> ITrainingCertificate:
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
        return Utils.read(f"{self.get_banked_model_dir_path(model_id, model_type)}/certificate.json")
        





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
        return self.p(f"{EpochFile.MODEL_PATH['models_bank']}/{model_type}/{model_id}")






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
    # The extension of the model's file depends on the technology behind it.



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
















    #########################################################################################
    ## CLASSIFICATION TRAINING DATA                                                        ##
    ## The classification training data is saved once the process is completed. This value ##
    ## can be read by other modules and is exported with the Epoch. 
    #########################################################################################





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
        return Utils.read(self.p(f"{EpochFile.MODEL_PATH['classification_training_data']}/{id}.json"))






    def list_classification_training_data_ids(self) -> List[str]:
        """Retrieves the list of classification training data ids in the assets
        directory.

        Returns:
            List[str]
        """
        # Retrieve the directory contents
        _, files = Utils.get_directory_content(
            path=self.p(EpochFile.MODEL_PATH["classification_training_data"]), 
            only_file_ext=".json"
        )

        # Init the ids
        ids: List[str] = list(filter(lambda x: ".json" in x, files))

        # Remove the extension from the id
        return [id.replace(".json", "") for id in ids]






    def save_classification_training_data(self, training_data: ITrainingDataFile) -> None:
        """Saves the training data file in the appropiate directory.

        Args:
            training_data: ITrainingDataFile
                The training data file data.
        """
        # Init the path
        path: str = self.p(f"{EpochFile.MODEL_PATH['classification_training_data']}/{training_data['id']}.json")

        # Save the file
        Utils.write(path, training_data)


















    #####################################################################################
    ## BACKTESTS                                                                       ##
    ## The backtest process reads the configuration files and saves the results in     ##
    ## the corresponding paths so they can be visualized by the user or other modules. ##
    #####################################################################################






    def list_backtest_configs(self) -> List[str]:
        """Retrieves the list of backtest configuration files within the directory.

        Returns:
            List[str]
        """
        # Retrieve the directory contents
        _, files = Utils.get_directory_content(
            path=self.p(EpochFile.BACKTEST_PATH["configurations"]), 
            only_file_ext=".json"
        )

        # Return the files
        return files





    def get_backtest_config(self, file_name: str) -> IBacktestConfig:
        """Retrieves all the results for a given backtest.

        Args:
            file_name: str
                The name of the configuration file.

        Returns:
            IBacktestConfig

        Raises:
            RuntimeError:
                If the backtest config file does not exist.
        """
        return Utils.read(f"{EpochFile.BACKTEST_PATH['configurations']}/{file_name}")





    def save_backtest_config(self, config: IBacktestConfig) -> None:
        """Saves a backtest configuration file in the corresponding path.

        Args:
            config: IBacktestConfig
                The configuration to save.
        """
        # Init values
        path: str = self.p(f"{EpochFile.BACKTEST_PATH['configurations']}/{config['id']}.json")

        # Save the results
        Utils.write(path, config)





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
        return Utils.read(f"{EpochFile.BACKTEST_PATH['results']}/{backtest_id}.json")







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
        path: str = self.p(f"{EpochFile.BACKTEST_PATH['results']}/{results[0]['backtest']['id']}.json")

        # Save the results
        Utils.write(path, results)
















    ###############################################################################################
    ## REGRESSION SELECTION                                                                      ##
    ## This process analyzes a given list of regressions and groups all the neccessary data      ##
    ## in order to validate the selection prior to generating the Classification Training Data.  ##
    ###############################################################################################





    def save_regression_selection(self, file: IRegressionSelectionFile) -> None:
        """Saves a Regression Selection Result into the proper directory

        Args:
            file: IRegressionSelectionFile
                The selection to be stored
        """
        # Init the path
        path: str = self.p(f"{EpochFile.MODEL_PATH['regression_selection']}/{file['id']}.json")

        # Save the file
        Utils.write(path, file)
    




    def list_regression_selection_ids(self) -> List[str]:
        """Retrieves the list of regression selection ids in the assets
        directory.

        Returns:
            List[str]
        """
        # Retrieve the directory contents
        _, files = Utils.get_directory_content(
            path=self.p(EpochFile.MODEL_PATH["regression_selection"]), 
            only_file_ext=".json"
        )

        # Remove the extension from the id
        return [id.replace(".json", "") for id in files]





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
        return Utils.read(self.p(f"{EpochFile.MODEL_PATH['regression_selection']}/{id}.json"))















    ###############################################################################
    ## EPOCH PATH                                                                ##
    ## All the assets generated during the creation of an epoch, are stored in a ##
    ## directory named after the Epoch's ID. For this reason, when interacting   ##
    ## with Epoch Files, it is important to call the p method.                  ##
    ###############################################################################





    def p(self, path: str) -> str:
        """Adds the Epoch's name to the beggining of a given path.

        Args:
            path: str
                The path that will be completed with the epoch id.

        Returns: 
            str
        """
        return f"{self.epoch_id}/{path}"



















    ###############################################################################
    ## EPOCH DIRECTORIES                                                         ##
    ## For the Epoch to be able to operate in a scalable way, it needs to follow ##
    ## strict guidelines when storing configurations, results, models, etc.      ##
    ## This function creates the entire skeleton for both, backtest and model    ##
    ## management.                                                               ##
    ###############################################################################



    @staticmethod
    def create_epoch_directories(epoch_id: str) -> None:
        """Creates all the directories required for the epoch to function.

        Args:
            epoch_id: str
                The identifier of the epoch.
        """
        # Create all the backtest asset directories
        Utils.make_directory(f"{epoch_id}/{EpochFile.BACKTEST_PATH['assets']}")
        Utils.make_directory(f"{epoch_id}/{EpochFile.BACKTEST_PATH['configurations']}")
        Utils.make_directory(f"{epoch_id}/{EpochFile.BACKTEST_PATH['regression_selection']}")
        Utils.make_directory(f"{epoch_id}/{EpochFile.BACKTEST_PATH['results']}")

        # Create all the model asset directories
        Utils.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['assets']}")
        Utils.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['batched_training_certificates']}")
        Utils.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['batched_training_certificates']}/unit_tests")
        Utils.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['classification_training_data']}")
        Utils.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['models']}")
        Utils.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['models_bank']}")
        Utils.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['keras_classification_training_configs']}")
        Utils.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['keras_regression_training_configs']}")
        Utils.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['xgb_classification_training_configs']}")
        Utils.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['xgb_regression_training_configs']}")
        for trainable_model in TRAINABLE_MODEL_TYPES:
            Utils.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['batched_training_certificates']}/{trainable_model}")
            Utils.make_directory(f"{epoch_id}/{EpochFile.MODEL_PATH['models_bank']}/{trainable_model}")