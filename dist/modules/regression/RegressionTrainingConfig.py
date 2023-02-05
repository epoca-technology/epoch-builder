from typing import Union, List, Tuple
from functools import reduce
from copy import deepcopy
from math import ceil
from random import shuffle
from modules._types import IKerasModelConfig, IKerasOptimizer, IKerasActivation, IRegressionTrainingConfigLoss,\
    IKerasActivation, IRegressionTrainingConfig, IRegressionTrainingConfigBatch, IRegressionTrainingConfigCategory,\
        IKerasModelTemplateName, IKerasUnit, IKerasFilter, IKerasKernelSize, IKerasPoolSize, IRegressionBatchSizes,\
            IRegressionTrainingConfigNetworkReceipt, IRegressionHyperparams, IRegressionCategoryHyperparams
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.keras_utils.KerasModel import KerasModel
from modules.regression.network_architectures import NETWORKS_BY_CATEGORY






class RegressionTrainingConfig:
    """RegressionTrainingConfig Class

    This singleton handles the generation and management of the regression training configurations.

    Class Properties:
        BATCH_SIZE: IRegressionBatchSizes
            The number of model configs that will be included per batch
        HYPERPARAMS: IRegressionHyperparams
            A dict containing all the hyperparameters by category.
    """
    # The number of model configs that will be included per batch
    BATCH_SIZE: IRegressionBatchSizes = {
        "CNN": 15,
        "DNN": 100,
        "CDNN": 30,
        "LSTM": 3,
        "BDLSTM": 3,
        "CLSTM": 3,
        "GRU": 3,
        "UNIT_TEST": 1
    }

    # Hyperparameters that will be used to build the training configurations
    HYPERPARAMS: IRegressionHyperparams = {
        "CNN": {
            "learning_rates": [ -1, 0.001, 0.0001 ],
            "optimizers": [ "adam" ],
            "loss_functions": [
                { "name": "mean_absolute_error", "metric": "mean_squared_error" },
                { "name": "mean_squared_error", "metric": "mean_absolute_error" }
            ],
            "activations": [ "relu" ]
        },
        "DNN": {
            "learning_rates": [ -1, 0.001, 0.0001 ],
            "optimizers": [ "adam" ],
            "loss_functions": [ 
                { "name": "mean_absolute_error", "metric": "mean_squared_error" },
                { "name": "mean_squared_error", "metric": "mean_absolute_error" }
            ],
            "activations": [ "relu" ]
        },
        "CDNN": {
            "learning_rates": [ -1, 0.001, 0.0001 ],
            "optimizers": [ "adam" ],
            "loss_functions": [
                { "name": "mean_absolute_error", "metric": "mean_squared_error" },
                { "name": "mean_squared_error", "metric": "mean_absolute_error" }
            ],
            "activations": [ "relu" ]
        },
        "LSTM": {
            "learning_rates": [ -1 ],
            "optimizers": [ "adam" ],
            "loss_functions": [
                { "name": "mean_absolute_error", "metric": "mean_squared_error" },
                { "name": "mean_squared_error", "metric": "mean_absolute_error" }
            ],
            "activations": [ None ]
        },
        "BDLSTM": {
            "learning_rates": [ -1 ],
            "optimizers": [ "adam" ],
            "loss_functions": [
                { "name": "mean_absolute_error", "metric": "mean_squared_error" },
                { "name": "mean_squared_error", "metric": "mean_absolute_error" }
            ],
            "activations": [ None ]
        },
        "CLSTM": {
            "learning_rates": [ -1 ],
            "optimizers": [ "adam" ],
            "loss_functions": [ 
                { "name": "mean_absolute_error", "metric": "mean_squared_error" },
                { "name": "mean_squared_error", "metric": "mean_absolute_error" }
            ],
            "activations": [ "relu" ]
        },
        "GRU": {
            "learning_rates": [ -1 ],
            "optimizers": [ "adam" ],
            "loss_functions": [
                { "name": "mean_absolute_error", "metric": "mean_squared_error" },
                { "name": "mean_squared_error", "metric": "mean_absolute_error" }
            ],
            "activations": [ None ]
        },
    }









    #######################################
    ## TRAINING CONFIGURATIONS GENERATOR ##
    #######################################



    @staticmethod
    def generate() -> None:
        """Generates all the hyperparams models and saves them as well as
        the receipt.

        Raises:
            ValueError:
                If a KerasModel cannot be initialized for any reason.
        """
        # Init the network receipts
        network_receipts: List[IRegressionTrainingConfigNetworkReceipt] = []

        # Iterate over the networks
        for category, network_variations in NETWORKS_BY_CATEGORY.items():
            # Init the network's list of configs
            print(f"\n\nGenerating {category} training configuration batches...")
            configs: List[IRegressionTrainingConfig] = []

            # Init the category's hyperparams
            category_hyperparams: IRegressionCategoryHyperparams = RegressionTrainingConfig.HYPERPARAMS[category]

            # Iterate over the variations
            for variation_name, variations in network_variations.items():

                # Iterate over the learning rates
                for learning_rate in category_hyperparams["learning_rates"]:

                    # Iterate over the optimizers
                    for optimizer in category_hyperparams["optimizers"]:

                        # Iterate over the loss functions
                        for loss in category_hyperparams["loss_functions"]:

                            # Generate all the combinations for the variation and concatenate them
                            configs = configs + RegressionTrainingConfig._get_variation_configs(
                                category_hyperparams=category_hyperparams,
                                keras_model_name= variation_name,
                                keras_model_variations=variations,
                                learning_rate=learning_rate,
                                optimizer=optimizer,
                                loss=loss
                            )

            # Build batches and save the network configurations
            models, batches = RegressionTrainingConfig._build_and_save_batches(category, configs)
            network_receipts.append({ "name": category, "models": models, "batches": batches})

        # Save the unit test config batch
        RegressionTrainingConfig._save_unit_test_training_config_batch()

        # Finally, save the receipt
        RegressionTrainingConfig._save_training_configs_receipt(network_receipts)





    def _build_and_save_batches(
        category: IRegressionTrainingConfigCategory, 
        configs: List[IRegressionTrainingConfig]
    ) -> Tuple[int, int]:
        """Given a list of model configs, it will split them and save them based on the 
        provided batch size.

        Args:
            category: IRegressionTrainingConfigCategory
                The category of the configurations. F.e. DNN, CDNN...
            configs: List[IRegressionTrainingConfig]
                The list of configurations that were built for all variations.

        Returns:
            Tuple[int, int]
            (models, batches)
        """
        # Shuffle the configurations in order to ensure any architectures can be found
        # at any batch and therefore remove the need to train all the batches.
        shuffle(configs)

        # Init counts
        models: int = len(configs)
        batches: int = ceil(models / RegressionTrainingConfig.BATCH_SIZE[category])

        # Init the batched training config file
        training_config: IRegressionTrainingConfigBatch = {
            "name": "", # Placeholder
            "configs": [] # Placeholder
        }

        # Save the configurations in batches
        slice_start: int = 0
        for batch_number in range(1, batches+1):
            # Include the name
            training_config["name"] = f"KR_{category}_{batch_number}_{batches}"

            # Include the sliced configs
            slice_end: int = slice_start + RegressionTrainingConfig.BATCH_SIZE[category]
            training_config["configs"] = configs[slice_start:slice_end]

            # Save the batch
            batch_path: str = Epoch.PATH.regression_training_configs(
                category, 
                f"{training_config['name']}.json"
            )
            Utils.write(batch_path, training_config, indent=4)

            # Set the end of the slice as the new start
            slice_start = slice_end

        # Finally, pack and return the network counts
        return models, batches







    @staticmethod
    def _get_variation_configs(
        category_hyperparams: IRegressionCategoryHyperparams,
        keras_model_name: IKerasModelTemplateName,
        keras_model_variations: List[IKerasModelConfig],
        learning_rate: float,
        optimizer: IKerasOptimizer,
        loss: IRegressionTrainingConfigLoss
    ) -> List[IRegressionTrainingConfig]:
        """Iterates over all the variations and adds all the possible combinations
        by network.

        Args:
            category_hyperparams: IRegressionCategoryHyperparams
            keras_model_name: IKerasModelTemplateName
            keras_model_variations: List[IKerasModelConfig]
            learning_rate: float
            optimizer: IKerasOptimizer
            loss: IKerasHyperparamsLoss
        Returns:
            List[IRegressionTrainingConfig]
        """
        # Init the Keras Models Configs
        keras_model_configs: List[IKerasModelConfig] = []

        # Iterate over each activation function
        for activation in category_hyperparams["activations"]:
            
            # Iterate over each variation and add it to the list
            for variation in keras_model_variations:
                keras_model_configs.append({
                    "activations": [activation]*len(variation["activations"]) if activation is not None else None,
                    "units": variation.get("units"),
                    "filters": variation.get("filters"),
                    "kernel_sizes": variation.get("kernel_sizes"),
                    "pool_sizes": variation.get("pool_sizes")
                })

        # Finally, return the list of configs
        return [RegressionTrainingConfig._generate_model_config(
            keras_model_name=keras_model_name,
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss=loss,
            activations=c.get("activations"),
            units=c.get("units"),
            filters=c.get("filters"),
            kernel_sizes=c.get("kernel_sizes"),
            pool_sizes=c.get("pool_sizes")
        ) for c in keras_model_configs]







    @staticmethod
    def _generate_model_config(
        keras_model_name: IKerasModelTemplateName,
        learning_rate: float,
        optimizer: IKerasOptimizer,
        loss: IRegressionTrainingConfigLoss,
        activations: Union[List[IKerasActivation], None]=None,
        units: Union[List[IKerasUnit], None]=None,
        filters: Union[List[IKerasFilter], None]=None,
        kernel_sizes: Union[List[IKerasKernelSize], None]=None,
        pool_sizes: Union[List[IKerasPoolSize], None]=None
    ) -> IRegressionTrainingConfig:
        """Builds the configuration for a model ready to be trained and evaluated.

        Args:
            keras_model_name: IKerasModelTemplateName
            learning_rate: float
            optimizer: IKerasOptimizer
            loss: IRegressionTrainingConfigLoss
            activations: Union[List[IKerasActivation], None]
            units: Union[List[IKerasUnit], None]
            filters: Union[List[IKerasFilter], None]
            kernel_sizes: Union[List[IKerasKernelSize], None]
            pool_sizes: Union[List[IKerasPoolSize], None]

        Returns:
            IRegressionTrainingConfig

        Raises:
            ValueError:
                If the KerasModel cannot be initialized for any reason.
        """
        # Init values
        id: str = f"{keras_model_name}_{Utils.generate_uuid4()}"
        description: str = f"Generated by the {Epoch.ID} RegressionTrainingConfig Module on {Utils.from_milliseconds_to_date_string(Utils.get_time())}."
        keras_model: IKerasModelConfig = { "name": keras_model_name }

        # Populate the units if any
        if isinstance(units, list):
            keras_model["units"] = units

        # Populate the activations if any
        if isinstance(activations, list):
            keras_model["activations"] = activations

        # Populate the filters if any
        if isinstance(filters, list):
            keras_model["filters"] = filters

        # Populate the kernel_sizes if any
        if isinstance(kernel_sizes, list):
            keras_model["kernel_sizes"] = kernel_sizes

        # Populate the pool_sizes if any
        if isinstance(pool_sizes, list):
            keras_model["pool_sizes"] = pool_sizes

        # Validate the integrity of the model
        RegressionTrainingConfig._validate_model_integrity(keras_model)
        
        # Finally, return the configuration
        return {
            "id": id,
            "description": description,
            "lookback": Epoch.REGRESSION_LOOKBACK,
            "predictions": Epoch.REGRESSION_PREDICTIONS,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "loss": loss["name"],
            "metric": loss["metric"],
            "keras_model": keras_model
        }







    @staticmethod
    def _validate_model_integrity(model: IKerasModelConfig) -> None:
        """Validates a Keras Model by building an actual instance.

        Args:
            model: IKerasModelConfig
                The model to be validated.

        Raises:
            ValueError:
                If the KerasModel cannot be initialized for any reason.
        """
        # Create a copy of the model config
        keras_model_val = deepcopy(model)

        # Add regression model properties
        keras_model_val["lookback"] = Epoch.REGRESSION_LOOKBACK
        keras_model_val["predictions"] = Epoch.REGRESSION_PREDICTIONS

        # Initialize the KerasModel. If any value is invalid, this function will raise an error and
        # stop the execution.
        KerasModel(keras_model_val)






    # Misc Helpers




    @staticmethod
    def _save_training_configs_receipt(network_receipts: List[IRegressionTrainingConfigNetworkReceipt]) -> None:
        """Once the execution has saved all the configurations, a receipt is saved
        at the base of the path providing general information regarding the generated
        configurations as well as keeping track of the training progress.

        Args:
            network_receipts: List[IRegressionTrainingConfigNetworkReceipt]
                The list of network receipts generated.
        """
        # Calculate the total number of models
        total_models_result: int = reduce(lambda x, y: { "models": x["models"] + y["models"] }, network_receipts)
        total_models: int = total_models_result["models"]

        # Init the receipt
        receipt: str = f"{Epoch.ID}: Regression Training Configs\n\n"

        # Configuration
        receipt += f"Creation: {Utils.from_milliseconds_to_date_string(Utils.get_time())}\n"
        receipt += f"Total Models: {total_models}\n"

        # Networks
        for net in network_receipts:
            # General Info
            receipt += f"\n\n{net['name']}\n"
            receipt += f"Models: {net['models']}\n"
            receipt += f"Batches: {net['batches']}\n"

            # Fillable Batches
            for batch_number in range(1, net["batches"] + 1, 1):
                receipt += f"{net['name']}_{batch_number}: \n"

        # Finally, write the receipt in a text file
        Utils.write(f"{Epoch.PATH.regression_training_configs()}/receipt.txt", receipt)





    @staticmethod
    def _save_unit_test_training_config_batch() -> None:
        """Builds and saves the unit test training config batch.
        """
        # Init the config
        config: IRegressionTrainingConfigBatch = {
            "name": "KR_UNIT_TEST",
            "configs": [
                {
                    "id": "KR_UNIT_TEST",
                    "description": "This is the official Regression for Unit Tests.",
                    "lookback": Epoch.REGRESSION_LOOKBACK,
                    "predictions": Epoch.REGRESSION_PREDICTIONS,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                    "loss": "mean_absolute_error",
                    "metric": "mean_squared_error",
                    "keras_model": {
                        "name": "KR_DNN_S2",
                        "units": [32, 32],
                        "activations": ["relu", "relu"]
                    }
                }
            ]
        }

        # Save the batch
        batch_path: str = Epoch.PATH.regression_training_configs("UNIT_TEST", "UNIT_TEST.json")
        Utils.write(batch_path, config, indent=4)










    


    ################################
    ## Training Config Retrievers ##
    ################################





    @staticmethod
    def get_batch(category: IRegressionTrainingConfigCategory, batch_file_name: str) -> IRegressionTrainingConfigBatch:
        """Retrieves a training config batch based on provided args.

        Args:
            category: IRegressionTrainingConfigCategory
                The category in which the batch is located.
            batch_file_name: str
                The batch file to be loaded.

        Returns:
            IRegressionTrainingConfigBatch

        Raises:
            RuntimeError:
                If the batch file does not exist.
        """
        return Utils.read(Epoch.PATH.regression_training_configs(category, batch_file_name))