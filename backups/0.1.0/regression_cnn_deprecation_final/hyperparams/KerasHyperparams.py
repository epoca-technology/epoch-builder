from typing import Union, List, Tuple
from os import makedirs
from os.path import exists
from json import dumps
from functools import reduce
from copy import deepcopy
from math import ceil
from modules.types import IKerasModelConfig, IKerasHyperparamsLoss, IKerasHyperparamsNetworkReceipt, \
    IRegressionTrainingConfig, IRegressionTrainingBatch, IClassificationTrainingConfig, \
    IClassificationTrainingBatch, IKerasOptimizer, IKerasActivation, ITrainableModelType
from modules.utils.Utils import Utils
from modules.keras_models.KerasPath import KERAS_PATH
from modules.keras_models.KerasModel import KerasModel
from modules.model.RegressionModel import RegressionModel
from modules.hyperparams.RegressionNeuralNetworks import REGRESSION_NEURAL_NETWORKS
from modules.hyperparams.ClassificationNeuralNetworks import CLASSIFICATION_NEURAL_NETWORKS



class KerasHyperparams:
    """KerasHyperparams Class

    This class handles the generation of n number of Keras Model Configurations
    ready to be trained in hyperparams mode.

    Class Properties:
        REGRESSION_BATCH_SIZE: int
        CLASSIFICATION_BATCH_SIZE: int
            The maximum number of models per batch that will be used if none was provided.
        OPTIMIZERS: List[IKerasOptimizer]
            The list of optimizers that will be used when compiling models.
        REGRESSION_LOSS_FUNCTIONS: List[IKerasHyperparamsLoss]
        CLASSIFICATION_LOSS_FUNCTIONS: List[IKerasHyperparamsLoss]
            The list of loss and metric combinations that will be used when compiling models. 
            Keep in mind that regressions don't use a metric function.
        ACTIVATIONS: List[IKerasActivation]
            The list of different activations that will be used to structure models.
        DROPOUT_RATES: List[float]
            The list of different dropout rates that will be used to structure models.

    Instance Properties:
        model_type: ITrainableModelType
            The type of model to generate hyperparameters for.
        networks: INeuralNetworks
            The list of networks supported by the model type.
        prefix: str
            The prefix of the model based on its type. It can be "C_" or "R_".
        batch_size: int
            The max number of models that will go on each batch.
        start: Union[str, None]
        end: Union[str, None]
            The date range that will be used to train the model. Notice that this value is 
            only used in keras_regression
        training_data_id: Union[str, None]
            The identifier of the classification training data.
        epoch_name: str
            The name of the output.
        base_path: str
            The base training configuration directory based on the model type.
        output_path: str
            The output path where the configuration files will be stored based on the
            provided out_dir.

    """
    # Default Batch Size
    REGRESSION_BATCH_SIZE: int = 40
    CLASSIFICATION_BATCH_SIZE: int = 60

    # Optimizers
    OPTIMIZERS: List[IKerasOptimizer] = [ "adam", "rmsprop" ]

    # Regression Loss Functions
    REGRESSION_LOSS_FUNCTIONS: List[IKerasHyperparamsLoss] = [ 
        { "name": "mean_absolute_error", "metric": None },
        #{ "name": "mean_squared_error", "metric": None }
    ]

    # Classification Loss Functions
    CLASSIFICATION_LOSS_FUNCTIONS: List[IKerasHyperparamsLoss] = [ 
        { "name": "categorical_crossentropy", "metric": "categorical_accuracy" },
        { "name": "binary_crossentropy", "metric": "binary_accuracy" }
    ]

    # Activations
    ACTIVATIONS: List[IKerasActivation] = [ "relu" ]   # Reduced from ["relu", "tanh"]

    # Dropout Rates
    DROPOUT_RATES: List[float] = [ 0.25 ]   # Reduced from [0.25, 0.5]







    ## Initialization ##



    def __init__(
        self, 
        epoch_name: str, 
        model_type: ITrainableModelType, 
        batch_size: int, 
        start: Union[str, None] = None, 
        end: Union[str, None] = None, 
        training_data_id: Union[str, None] = None
    ):
        """Initializes the KerasHyperparams instance based on the provided configuration.

        Args:
            epoch_name: str
                The name of the directory that in which the configuration files will be 
                included.
            model_type: ITrainableModelType "keras_regression"|"keras_classification"
                The type of model which hyperparams will be generated for.
            batch_size: int
                The number of models that will go on each batch.
            start: Union[str, None]
            end: Union[str, None]
                The date range that will be used to train the model. Notice that this value is 
                only used in keras_regression
            training_data_id: Union[str, None]
                The training data id that will be included in the model configurations. Notice
                that this value is only used in keras_classification.
        """
        # Initialize the type of model
        self.model_type: str = model_type

        # Init the networks
        self.networks = REGRESSION_NEURAL_NETWORKS if self.model_type == "keras_regression" \
            else CLASSIFICATION_NEURAL_NETWORKS

        # Initialize the prefix
        self.prefix: str = "C_" if self.model_type == "keras_classification" else "R_"

        # Initialize the batch size
        self.batch_size: int = batch_size

        # Initialize the date range
        self.start: Union[str, None] = start
        self.end: Union[str, None] = end

        # Initialize the training data id
        self.training_data_id: Union[str, None] = training_data_id

        # Initialize the base of the output path
        self.base_path: str = KERAS_PATH["classification_training_configs"] if self.model_type == "keras_classification" \
            else KERAS_PATH["regression_training_configs"]

        # Initialize the output name
        self.epoch_name: str = epoch_name

        # Initialize the out path
        self.output_path: str = f"{self.base_path}/{self.epoch_name}"

        # If the base output directory doesn't exist, create it
        if not exists(self.base_path):
            makedirs(self.base_path)

        # Make sure the final output path does not exist
        if exists(self.output_path):
            raise ValueError(f"The path {self.output_path} already exists.")
        else:
            makedirs(self.output_path)












    ## Execution ##




    def generate(self) -> None:
        """Generates all the hyperparams models and saves them as well as
        the receipt.

        Raises:
            ValueError:
                If a KerasModel cannot be initialized for any reason.
        """
        # Init the network receipts
        network_receipts: List[IKerasHyperparamsNetworkReceipt] = []

        # Iterate over the networks
        for network_type, network_variations in self.networks.items():
            # Init the network's list of configs
            configs: Union[List[IRegressionTrainingConfig], List[IClassificationTrainingConfig]] = []

            # Iterate over the variations
            for variation_name, variations in network_variations.items():

                # Iterate over the optimizers
                for optimizer in KerasHyperparams.OPTIMIZERS:

                    # Iterate over the loss functions
                    for loss in KerasHyperparams.REGRESSION_LOSS_FUNCTIONS if self.model_type == "keras_regression" \
                        else KerasHyperparams.CLASSIFICATION_LOSS_FUNCTIONS:

                        # Generate all the combinations for the variation and concatenate them
                        configs = configs + self._get_variation_configs(
                            network_type=network_type,
                            keras_model_name= variation_name,
                            keras_model_variations=variations,
                            optimizer=optimizer,
                            loss=loss
                        )

            # Build batches and save the network configurations
            models, batches = self._build_and_save_batches(network_type, configs)
            network_receipts.append({ "name": network_type, "models": models, "batches": batches})

        # Finally, save the receipt
        self._save_receipt(network_receipts)





    def _build_and_save_batches(
        self, 
        network_type: str, 
        configs: Union[List[IRegressionTrainingConfig], List[IClassificationTrainingConfig]]
    ) -> Tuple[int, int]:
        """Given a list of model configs, it will split them and save them based on the 
        provided batch size.

        Args:
            network_type: str
                The type of network the files belong to. F.e. DNN, CNN...
            configs: Union[List[IRegressionTrainingConfig], List[IClassificationTrainingConfig]]
                The list of configurations that were built for all variations.

        Returns:
            Tuple[int, int]
            (models, batches)
        """
        # Init counts
        models: int = len(configs)
        batches: int = ceil(models / self.batch_size)

        # Init the batched training config file
        training_config: Union[IRegressionTrainingBatch, IClassificationTrainingBatch] = {
            "name": "", # Placeholder
            "models": [] # Placeholder
        }

        # Add the regression specific values
        if self.model_type == "keras_regression":
            training_config["start"] = self.start
            training_config["end"] = self.end
        
        # Add the classification specific values
        else:
            training_config["training_data_id"] = self.training_data_id

        # Save the configurations in batches
        slice_start: int = 0
        for batch_number in range(1, batches+1):
            # Include the name
            training_config["name"] = f"{self.prefix}{network_type}_{batch_number}_{batches}"

            # Include the sliced configs
            slice_end: int = slice_start + self.batch_size
            training_config["models"] = configs[slice_start:slice_end]

            # Save the batch
            self._save_batch(network_type, batch_number, training_config)

            # Set the end of the slice as the new start
            slice_start = slice_end

        # Finally, pack and return the network counts
        return models, batches








    def _get_variation_configs(
        self,
        network_type: str,
        keras_model_name: str,
        keras_model_variations: List[IKerasModelConfig],
        optimizer: str,
        loss: IKerasHyperparamsLoss
    ) -> Union[List[IRegressionTrainingConfig], List[IClassificationTrainingConfig]]:
        """Iterates over all the variations and adds all the possible combinations
        by network.

        Args:
            network_type: str
            keras_model_name: str
            keras_model_variations: List[IKerasModelConfig]
            optimizer: str
            loss: IKerasHyperparamsLoss
        Returns:
            Union[List[IRegressionTrainingConfig], List[IClassificationTrainingConfig]]
        """
        # Init the Keras Models Configs
        keras_model_configs: List[IKerasModelConfig] = []

        # Init the activations
        activations: List[Union[str, None]] = KerasHyperparams.ACTIVATIONS if network_type != "LSTM" else [None]

        # Iterate over each activation function
        for activation in activations:
            
            # Iterate over each variation
            for variation in keras_model_variations:

                # Dropout Variations
                # Add a configuration per dropout variation specified in the Class Properties
                if variation.get("dropout_rates") != None:
                    for dropout in KerasHyperparams.DROPOUT_RATES:
                        keras_model_configs.append({
                            "activations": [activation]*len(variation["activations"]) if activation is not None else None,
                            "units": variation.get("units"),
                            "filters": variation.get("filters"),
                            "kernel_sizes": variation.get("kernel_sizes"),
                            "pool_sizes": variation.get("pool_sizes"),
                            "dropout_rates": [dropout]*len(variation["dropout_rates"])
                        })

                # Otherwise, just add the traditional model
                else:
                    keras_model_configs.append({
                        "activations": [activation]*len(variation["activations"]) if activation is not None else None,
                        "units": variation.get("units"),
                        "filters": variation.get("filters"),
                        "kernel_sizes": variation.get("kernel_sizes"),
                        "pool_sizes": variation.get("pool_sizes")
                    })

        # Build the initial list of configs
        configs: Union[List[IRegressionTrainingConfig], List[IClassificationTrainingConfig]] = [self._generate_model_config(
            keras_model_name=keras_model_name,
            optimizer=optimizer,
            loss=loss,
            autoregressive=True,
            activations=c.get("activations"),
            units=c.get("units"),
            filters=c.get("filters"),
            kernel_sizes=c.get("kernel_sizes"),
            pool_sizes=c.get("pool_sizes"),
            dropout_rates=c.get("dropout_rates"),
        ) for c in keras_model_configs]

        # If it is a Classification, just return them as they are
        if self.model_type == "keras_classification":
            return configs

        # Otherwise, add the autoregressive variation
        else:
            #ar_configs: Union[List[IRegressionTrainingConfig], List[IClassificationTrainingConfig]] = [self._generate_model_config(
            #    keras_model_name=keras_model_name,
            #    optimizer=optimizer,
            #    loss=loss,
            #    autoregressive=True,
            #    activations=c.get("activations"),
            #    units=c.get("units"),
            #    filters=c.get("filters"),
            #    kernel_sizes=c.get("kernel_sizes"),
            #    pool_sizes=c.get("pool_sizes"),
            #    dropout_rates=c.get("dropout_rates"),
            #) for c in keras_model_configs]

            # Finally, return the concatenated list
            #return configs + ar_configs

            # Single-Shot Regressions have been deprecated because of the high
            # resource consumption and poor performance
            return configs








    def _generate_model_config(
        self,
        keras_model_name: str,
        optimizer: str,
        loss: IKerasHyperparamsLoss,
        autoregressive: Union[bool, None]=None,
        activations: Union[List[str], None]=None,
        units: Union[List[int], None]=None,
        filters: Union[List[int], None]=None,
        kernel_sizes: Union[List[int], None]=None,
        pool_sizes: Union[List[int], None]=None,
        dropout_rates: Union[List[float], None]=None
    ) -> Union[IRegressionTrainingConfig, IClassificationTrainingConfig]:
        """Builds the configuration for a model ready to be trained and evaluated.

        Args:
            keras_model_name: str
            optimizer: str
            loss: IKerasHyperparamsLoss
            autoregressive: Union[bool, None]
            units: Union[List[int], None]
            activations: Union[List[str], None]
            filters: Union[List[int], None]
            kernel_sizes: Union[List[int], None]
            pool_sizes: Union[List[int], None]
            dropout_rates: Union[List[float], None]

        Returns:
            Union[IRegressionTrainingConfig, IClassificationTrainingConfig]

        Raises:
            ValueError:
                If the KerasModel cannot be initialized for any reason.
        """
        # Init values
        id: str = self._generate_model_id(keras_model_name)
        description: str = f"Generated by Hyperparams ({self.epoch_name})."
        keras_model: IKerasModelConfig = { "name": keras_model_name }

        # Populate the units if any
        if isinstance(units, list):
            keras_model["units"] = units

        # Populate the activations if any
        if isinstance(activations, list):
            keras_model["activations"] = activations

        # Populate the dropout_rates if any
        if isinstance(dropout_rates, list):
            keras_model["dropout_rates"] = dropout_rates

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
        self._validate_model_integrity(keras_model, autoregressive=autoregressive)
        
        # Finally, return the configuration based on the type of model
        if self.model_type == "keras_classification":
            return {
                "id": id,
                "description": description,
                "optimizer": optimizer,
                "loss": loss["name"],
                "metric": loss["metric"],
                "keras_model": keras_model
            }
        else:
            return {
                "id": id,
                "description": description,
                "autoregressive": autoregressive if isinstance(autoregressive, bool) else False,
                "lookback": RegressionModel.DEFAULT_LOOKBACK,
                "predictions": RegressionModel.DEFAULT_PREDICTIONS,
                "optimizer": optimizer,
                "loss": loss["name"],
                "keras_model": keras_model
            }






    def _validate_model_integrity(self, model: IKerasModelConfig, autoregressive: Union[bool, None]) -> None:
        """Validates a Keras Model by building an actual instance.

        Args:
            model: IKerasModelConfig
                The model to be validated.
            autoregressive: Union[bool, None]
                The type of regression.

        Raises:
            ValueError:
                If the KerasModel cannot be initialized for any reason.
        """
        # Create a copy of the model config
        keras_model_val = deepcopy(model)

        # Add model type specific properties
        if self.model_type == "keras_regression":
            keras_model_val["autoregressive"] = autoregressive if isinstance(autoregressive, bool) else False
            keras_model_val["lookback"] = RegressionModel.DEFAULT_LOOKBACK
            keras_model_val["predictions"] = RegressionModel.DEFAULT_PREDICTIONS
        else:
            keras_model_val["features_num"] = 5 # Minimum number of features allowed

        # Initialize the KerasModel. If any value is invalid, this function will raise an error and
        # stop the execution.
        KerasModel(config=keras_model_val)








    def _generate_model_id(self, model_name: str) -> str:
        """Generates a random string that is appended to the model's name in order to 
        prevent duplicate ID issues.

        Returns:
            str
        """
        return f"{model_name}_{Utils.generate_uuid4()}"












    ## Saving ##





    def _save_batch(
        self, 
        network: str, 
        batch_num: int, 
        network_file: Union[IRegressionTrainingBatch, IClassificationTrainingBatch]
    ) -> None:
        """Saves a Keras Training Batch File into the configs directory.

        Args:
            network: str
                The type of network that is going to be saved.
            batch_num: int
                The batch number for the given network.
            network_file: 
                The configuration file for given network.
        """
        # If the network's directory doesn't exist, create it
        if not exists(f"{self.output_path}/{network}"):
            makedirs(f"{self.output_path}/{network}")

        # Write the results on a JSON File
        with open(f"{self.output_path}/{network}/{self.prefix}{network}_{batch_num}.json", "w") as outfile:
            outfile.write(dumps(network_file, indent=4))








    def _save_receipt(self, network_receipts: List[IKerasHyperparamsNetworkReceipt]) -> None:
        """Once the execution has saved all the configurations, a receipt is saved
        at the base of the path providing general information regarding the contents
        of the Hyperparams configs.

        Args:
            network_receipts: List[IKerasHyperparamsNetworkReceipt]
                The list of network receipts generated.
        """
        # Calculate the total number of models
        total_models_result: int = reduce(lambda x, y: { "models": x["models"] + y["models"] }, network_receipts)
        total_models: int = total_models_result["models"]

        # Init the receipt
        receipt: str = f"{self.epoch_name}: {self.model_type} hyperparams\n\n"

        # Configuration
        receipt += f"Creation: {Utils.from_milliseconds_to_date_string(Utils.get_time())}\n"
        receipt += f"Total Models: {total_models}\n"
        receipt += f"Batch Size: {self.batch_size}\n"
        if self.model_type == "keras_regression":
            receipt += f"Start: {self.start}\n"
            if isinstance(self.end, str):
                receipt += f"End: {self.end}\n"
        if self.model_type == "keras_classification":
            receipt += f"Training Data ID: {self.training_data_id}\n"

        # Networks
        for net in network_receipts:
            # General Info
            receipt += f"\n\n{net['name']}\n"
            receipt += f"Models: {net['models']}\n"
            receipt += f"Batches: {net['batches']}\n"

            # Fillable Batches
            receipt += "\n"
            for batch_number in range(1, net["batches"] + 1, 1):
                receipt += f"{net['name']}_{batch_number}: \n"

        # Finally, write the receipt in a text file
        with open(f"{self.output_path}/receipt.txt", "w") as outfile:
            outfile.write(receipt)







