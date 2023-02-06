from typing import Union
from modules._types import IRegressionTrainingConfigCategory
from modules.utils.Utils import Utils









# Class
class EpochPath:
    """EpochPath Class

    This class handles the management of all the paths used by the Epoch.

    Class Properties:
        ...

    Instance Properties:
        epoch_id: str
            The identifier of the epoch and root directory for all the assets.
    """



    def __init__(self, epoch_id: str) -> None:
        """Initializes the EpochFile instance.

        Args:
            epoch_id: str
                The ID of the current epoch.
        """
        # Init the identifier
        self.epoch_id: str = epoch_id





    # Regressions


    def regression_training_configs(
        self, 
        category: Union[IRegressionTrainingConfigCategory, None] = None, 
        batch_file_name: Union[str, None] = None
    ) -> str:
        """Builds the path for the training configs based on the provided
        args.

        Args:
            category: Union[IRegressionTrainingConfigCategory, None]
                The category of the training config.
            batch_file_name: Union[str, None]
                The file name of the batch.

        Returns:
            str
        """
        # Check if the category was provided
        if isinstance(category, str):
            # Check if the file name was provided
            if isinstance(batch_file_name, str):
                return self.p(f"regression_training_configs/{category}/{batch_file_name}")
            
            # Otherwise, return the category's root path
            else:
                return self.p(f"regression_training_configs/{category}")

        # Otherwise, return the root path
        else:
            return self.p("regression_training_configs")






    def regression_training_checkpoints(self, id: str) -> str:
        """Builds the path for the regression training checkpoints. This directory
        should be removed once the model finishes training.

        Args:
            id: str
                The identifier of the regression.

        Returns:
            str
        """
        return self.p(f"regression_training_checkpoints/{id}")




    def regression_training_active_epoch(self, id: str) -> str:
        """As epochs are completed, their number is stored in a file 
        in case the training was interrupted for any reason.

        Args:
            id: str
                The identifier of the regression.

        Returns:
            str
        """
        return self.p(f"regression_training_checkpoints/{id}/active_epoch.txt")






    def regression_batched_certificates(self) -> str:
        """Builds the path for the regression batched training certificates.

        Returns:
            str
        """
        return self.p("regression_batched_certificates")







    def regressions(self, id: Union[str, None]) -> str:
        """Builds the path for the regressions root directory. If the model 
        id is passed, it will return the root path for the model instead.

        Args:
            id: Union[str, None]
                The identifier of the model.

        Returns:
            str
        """
        # Check if the model id was provided
        if isinstance(id, str):
            return self.p(f"regressions/{id}")

        # Otherwise, return the root directory
        else:
            return self.p("regressions")







    def regression_certificate(self, id: str) -> str:
        """Builds the path for a specific regression's certificate file.

        Args:
            id: str
                The identifier of the regression.

        Returns:
            str
        """
        return self.regressions(f"/{id}/certificate.json")







    def regression_model(self, id: str) -> str:
        """Builds the path for a specific regression's model file.

        Args:
            id: str
                The identifier of the regression.

        Returns:
            str
        """
        return self.regressions(f"/{id}/model.h5")








    # Prediction Models



    def prediction_models(self) -> str:
        """Builds the path for the prediction models root directory.

        Returns:
            str
        """
        return self.p("prediction_models")







    def prediction_models_assets(self) -> str:
        """Builds the path for the prediction models assets root directory.

        Returns:
            str
        """
        return f"{self.prediction_models()}/assets"







    def prediction_models_features(self) -> str:
        """Builds the path for the prediction models features file.

        Returns:
            str
        """
        return f"{self.prediction_models_assets()}/features.json"







    def prediction_models_labels(self) -> str:
        """Builds the path for the prediction models labels file.

        Returns:
            str
        """
        return f"{self.prediction_models_assets()}/labels.json"







    def prediction_models_lookback_indexer(self) -> str:
        """Builds the path for the prediction models lookback indexer file.

        Returns:
            str
        """
        return f"{self.prediction_models_assets()}/lookback_indexer.json"







    def profitable_configs_journal(self) -> str:
        """Builds the path for the profitable prediction models journal file.

        Returns:
            str
        """
        return f"{self.prediction_models()}/journal.json"






    def prediction_models_configs_receipt(self) -> str:
        """Builds the path for the configurations receipt.

        Returns:
            str
        """
        return f"{self.prediction_models()}/configs_receipt.txt"






    def prediction_models_configs(self, batch_file_name: Union[str, None] = None) -> str:
        """Retrieves the path for the prediction models configurations directory. If 
        a batch file name is provided, it will return the path for that file instead.

        Args:
            batch_file_name: Union[str, None]
                The name of the config file. If provided, it must contain the ext.

        Returns:
            str
        """
        if isinstance(batch_file_name, str):
            return f"{self.prediction_models()}/configs/{batch_file_name}"
        else:
            return f"{self.prediction_models()}/configs"







    def prediction_models_profitable_configs(self, batch_file_name: Union[str, None] = None) -> str:
        """Retrieves the path for the prediction models profitable configurations directory. If 
        a batch file name is provided, it will return the path for that file instead.

        Args:
            batch_file_name: Union[str, None]
                The name of the config file. If provided, it must contain the ext.

        Returns:
            str
        """
        if isinstance(batch_file_name, str):
            return f"{self.prediction_models()}/profitable_configs/{batch_file_name}"
        else:
            return f"{self.prediction_models()}/profitable_configs"







    def prediction_models_build(self) -> str:
        """Retrieves the path for the prediction models build file.

        Returns:
            str
        """
        return f"{self.prediction_models()}/build.json"













    # Epoch Export




    def export(self) -> str:
        """Builds the path for the export root directory.

        Returns:
            str
        """
        return self.p("export")



    


    def export_prediction_model_certificate(self) -> str:
        """Builds the path for the prediction model certificate that
        will be placed in the export build.

        Returns:
            str
        """
        return f"{self.export()}/prediction_model_certificate.json"





    def export_regression_certificates(self) -> str:
        """Builds the path for the regression training certificates that
        will be placed in the export build.

        Returns:
            str
        """
        return f"{self.export()}/regression_certificates.json"





    def export_regression_model(self, regression_id: str) -> str:
        """Builds the path for the regression model file that
        will be placed in the export build.

        Args:
            regression_id: str
                The identifier of the regression that will be exported.

        Returns:
            str
        """
        return f"{self.export()}/{regression_id}.h5"





    def export_epoch_config(self) -> str:
        """Builds the path for the epoch's configuration that
        will be placed in the export build.

        Returns:
            str
        """
        return f"{self.export()}/epoch.json"





    def epoch_file(self) -> str:
        """Builds the path for the Epoch File.

        Returns:
            str
        """
        return self.p(f"{self.epoch_id}")









    # Epoch Path


    def p(self, path: str) -> str:
        """All the assets generated during the creation of an epoch, are stored in a
        directory named after the Epoch's ID. Therefore, this function needs to
        be invoked when interacting with epoch paths.

        Args:
            path: str
                The path that will be completed with the epoch id.

        Returns: 
            str
        """
        return f"{self.epoch_id}/{path}"












    # Epoch Directories Initialization


    @staticmethod
    def init_directories(epoch_id: str) -> None:
        """For the Epoch to be able to operate in a scalable way, it needs to follow
        strict guidelines when storing configurations, results, models, etc.
        This function creates the entire directory skeleton when the epoch is 
        created in order to simplify the usage.

        Args:
            epoch_id: str
                The identifier of the epoch.
        """
        Utils.make_directory(f"{epoch_id}/regression_training_configs")
        Utils.make_directory(f"{epoch_id}/regression_training_checkpoints")
        Utils.make_directory(f"{epoch_id}/regression_batched_certificates")
        Utils.make_directory(f"{epoch_id}/regressions")
        Utils.make_directory(f"{epoch_id}/prediction_models")
        Utils.make_directory(f"{epoch_id}/prediction_models/assets")
        Utils.make_directory(f"{epoch_id}/prediction_models/configs")
        Utils.make_directory(f"{epoch_id}/prediction_models/profitable_configs")