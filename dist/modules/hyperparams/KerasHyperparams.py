from typing import Union, List, Tuple
from functools import reduce
from copy import deepcopy
from math import ceil
from modules._types import IKerasModelConfig, IKerasHyperparamsLoss, IKerasHyperparamsNetworkReceipt, \
    IKerasRegressionTrainingConfig, IKerasRegressionTrainingBatch, IKerasClassificationTrainingConfig, \
    IKerasClassificationTrainingBatch, IKerasParams, ITrainableModelType, IModelIDPrefix, IKerasOptimizer,\
        IKerasActivation, IKerasHyperparamsNetworks
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.keras_models.KerasModel import KerasModel
from modules.model.ModelType import get_prefix_by_trainable_model_type
from modules.hyperparams.KerasNetworks import REGRESSION_NETWORKS, CLASSIFICATION_NETWORKS






class KerasHyperparams:
    """KerasHyperparams Class

    This class handles the generation of n number of Keras Model Configurations
    ready to be trained in hyperparams mode.

    Class Properties:
        REGRESSION_PARAMS: IKerasParams
        CLASSIFICATION_PARAMS: IKerasParams
            The parameters that will be used to generate and build the model 
            variations.

    Instance Properties:
        model_type: ITrainableModelType
            The type of model to generate hyperparameters for.
        networks: INeuralNetworks
            The list of networks supported by the model type.
        params: IKerasParams
            The parameters that will be used to generate variations based 
            on the model type.
        batch_size: int
            The max number of models that will go on each batch.
        lookback: int
            The number of candlesticks the regression needs to look into the past in order to generate a prediction
        predictions: int
            The number of predictions that will be generated (Only used by Regressions)
        training_data_id: Union[str, None]
            The identifier of the classification training data.
    """
    # Regression Model Params
    REGRESSION_PARAMS: IKerasParams = {
        "batch_size": 150,
        "learning_rates": [-1, 0.01, 0.001, 0.0001],
        "optimizers": [ "adam", "rmsprop" ],
        "loss_functions": [ 
            { "name": "mean_absolute_error", "metric": "mean_squared_error" },
            { "name": "mean_squared_error", "metric": "mean_absolute_error" }
        ],
        "activations": [ "relu", "tanh" ],
        "dropout_rates": [ 0.25, 0.5, 0.75 ]
    }

    # Classification Model Params
    CLASSIFICATION_PARAMS: IKerasParams = {
        "batch_size": 300,
        "learning_rates": [-1, 0.01, 0.001, 0.0001, 0.00005 ],
        "optimizers": [ "adam", "rmsprop" ],
        "loss_functions": [ 
            { "name": "categorical_crossentropy", "metric": "categorical_accuracy" },
            { "name": "binary_crossentropy", "metric": "binary_accuracy" }
        ],
        "activations": [ "relu", "tanh" ],
        "dropout_rates": [ 0.25, 0.5, 0.75 ]
    }







    ## Initialization ##



    def __init__(
        self, 
        model_type: ITrainableModelType, 
        batch_size: int, 
        training_data_id: Union[str, None] = None
    ):
        """Initializes the KerasHyperparams instance based on the provided configuration.

        Args:
            model_type: ITrainableModelType "keras_regression"|"keras_classification"
                The type of model which hyperparams will be generated for.
            batch_size: int
                The number of models that will go on each batch.
            training_data_id: Union[str, None]
                The training data id that will be included in the model configurations. Notice
                that this value is only used in classification models.
        """
        # Initialize the type of model
        self.model_type: ITrainableModelType = model_type

        # Init the networks
        self.networks: IKerasHyperparamsNetworks = REGRESSION_NETWORKS if self.model_type == "keras_regression" \
            else CLASSIFICATION_NETWORKS

        # Init the params
        self.params = KerasHyperparams.REGRESSION_PARAMS if self.model_type == "keras_regression" \
            else KerasHyperparams.CLASSIFICATION_PARAMS

        # Initialize the batch size based on the input
        self.batch_size: int = batch_size

        # Initialize the lookback
        self.lookback: int = Epoch.REGRESSION_LOOKBACK

        # Initialize the predictions
        self.predictions: int = Epoch.REGRESSION_PREDICTIONS

        # Initialize the training data id
        self.training_data_id: Union[str, None] = training_data_id












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
            print(f"Generating {network_type} Neural Networks...")
            configs: Union[List[IKerasRegressionTrainingConfig], List[IKerasClassificationTrainingConfig]] = []

            # Iterate over the variations
            for variation_name, variations in network_variations.items():

                # Iterate over the learning rates
                for learning_rate in self.params["learning_rates"]:

                    # Iterate over the optimizers
                    for optimizer in self.params["optimizers"]:

                        # Iterate over the loss functions
                        for loss in self.params["loss_functions"]:

                            # Generate all the combinations for the variation and concatenate them
                            configs = configs + self._get_variation_configs(
                                network_type=network_type,
                                keras_model_name= variation_name,
                                keras_model_variations=variations,
                                learning_rate=learning_rate,
                                optimizer=optimizer,
                                loss=loss
                            )

            # Build batches and save the network configurations
            models, batches = self._build_and_save_batches(network_type, configs)
            network_receipts.append({ "name": network_type, "models": models, "batches": batches})

        # Finally, save the receipt
        print("Saving receipt...")
        self._save_receipt(network_receipts)





    def _build_and_save_batches(
        self, 
        network_type: str, 
        configs: Union[List[IKerasRegressionTrainingConfig], List[IKerasClassificationTrainingConfig]]
    ) -> Tuple[int, int]:
        """Given a list of model configs, it will split them and save them based on the 
        provided batch size.

        Args:
            network_type: str
                The type of network the files belong to. F.e. DNN, CNN...
            configs: Union[List[IKerasRegressionTrainingConfig], List[IKerasClassificationTrainingConfig]]
                The list of configurations that were built for all variations.

        Returns:
            Tuple[int, int]
            (models, batches)
        """
        # Init counts
        models: int = len(configs)
        batches: int = ceil(models / self.batch_size)

        # Init the batched training config file
        training_config: Union[IKerasRegressionTrainingBatch, IKerasClassificationTrainingBatch] = {
            "name": "", # Placeholder
            "models": [] # Placeholder
        }
        
        # Add the classification specific values
        if self.model_type == "keras_classification":
            training_config["training_data_id"] = self.training_data_id

        # Save the configurations in batches
        slice_start: int = 0
        for batch_number in range(1, batches+1):
            # Include the name
            prefix: IModelIDPrefix = get_prefix_by_trainable_model_type(self.model_type)
            training_config["name"] = f"{prefix}{network_type}_{batch_number}_{batches}"

            # Include the sliced configs
            slice_end: int = slice_start + self.batch_size
            training_config["models"] = configs[slice_start:slice_end]

            # Save the batch
            Epoch.FILE.save_hyperparams_batch(self.model_type, network_type, training_config)

            # Set the end of the slice as the new start
            slice_start = slice_end

        # Finally, pack and return the network counts
        return models, batches








    def _get_variation_configs(
        self,
        network_type: str,
        keras_model_name: str,
        keras_model_variations: List[IKerasModelConfig],
        learning_rate: float,
        optimizer: IKerasOptimizer,
        loss: IKerasHyperparamsLoss
    ) -> Union[List[IKerasRegressionTrainingConfig], List[IKerasClassificationTrainingConfig]]:
        """Iterates over all the variations and adds all the possible combinations
        by network.

        Args:
            network_type: str
            keras_model_name: str
            keras_model_variations: List[IKerasModelConfig]
            learning_rate: float
            optimizer: IKerasOptimizer
            loss: IKerasHyperparamsLoss
        Returns:
            Union[List[IKerasRegressionTrainingConfig], List[IKerasClassificationTrainingConfig]]
        """
        # Init the Keras Models Configs
        keras_model_configs: List[IKerasModelConfig] = []

        # Init the activations
        activations: List[Union[IKerasActivation, None]] = self.params["activations"] if network_type != "LSTM" else [None]

        # Iterate over each activation function
        for activation in activations:
            
            # Iterate over each variation
            for variation in keras_model_variations:

                # Dropout Variations
                # Add a configuration per dropout variation specified in the Class Properties
                if variation.get("dropout_rates") != None:
                    for dropout in self.params["dropout_rates"]:
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

        # Finally, return the list of configs
        return [self._generate_model_config(
            keras_model_name=keras_model_name,
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss=loss,
            activations=c.get("activations"),
            units=c.get("units"),
            filters=c.get("filters"),
            kernel_sizes=c.get("kernel_sizes"),
            pool_sizes=c.get("pool_sizes"),
            dropout_rates=c.get("dropout_rates"),
        ) for c in keras_model_configs]








    def _generate_model_config(
        self,
        keras_model_name: str,
        learning_rate: float,
        optimizer: IKerasOptimizer,
        loss: IKerasHyperparamsLoss,
        activations: Union[List[IKerasActivation], None]=None,
        units: Union[List[int], None]=None,
        filters: Union[List[int], None]=None,
        kernel_sizes: Union[List[int], None]=None,
        pool_sizes: Union[List[int], None]=None,
        dropout_rates: Union[List[float], None]=None
    ) -> Union[IKerasRegressionTrainingConfig, IKerasClassificationTrainingConfig]:
        """Builds the configuration for a model ready to be trained and evaluated.

        Args:
            keras_model_name: str
            learning_rate: str
            optimizer: IKerasOptimizer
            loss: IKerasHyperparamsLoss
            units: Union[List[int], None]
            activations: Union[List[IKerasActivation], None]
            filters: Union[List[int], None]
            kernel_sizes: Union[List[int], None]
            pool_sizes: Union[List[int], None]
            dropout_rates: Union[List[float], None]

        Returns:
            Union[IKerasRegressionTrainingConfig, IKerasClassificationTrainingConfig]

        Raises:
            ValueError:
                If the KerasModel cannot be initialized for any reason.
        """
        # Init values
        id: str = self._generate_model_id(keras_model_name)
        description: str = f"Generated by Hyperparams ({Epoch.ID})."
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
        self._validate_model_integrity(keras_model)
        
        # Finally, return the configuration based on the type of model
        if self.model_type == "keras_classification":
            return {
                "id": id,
                "description": description,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "loss": loss["name"],
                "metric": loss["metric"],
                "keras_model": keras_model
            }
        else:
            return {
                "id": id,
                "description": description,
                "lookback": self.lookback,
                "predictions": self.predictions,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "loss": loss["name"],
                "metric": loss["metric"],
                "keras_model": keras_model
            }






    def _validate_model_integrity(self, model: IKerasModelConfig) -> None:
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

        # Add model type specific properties
        if self.model_type == "keras_regression":
            keras_model_val["lookback"] = self.lookback
            keras_model_val["predictions"] = self.predictions
        else:
            keras_model_val["features_num"] = 3 # Minimum number of features allowed

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












    ## Hyperparams Receipt ##





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
        receipt: str = f"{Epoch.ID}: {self.model_type} hyperparams\n\n"

        # Configuration
        receipt += f"Creation: {Utils.from_milliseconds_to_date_string(Utils.get_time())}\n"
        receipt += f"Total Models: {total_models}\n"
        receipt += f"Batch Size: {self.batch_size}\n"
        if self.model_type == "keras_classification":
            receipt += f"Training Data ID: {self.training_data_id}\n"

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
        Epoch.FILE.save_hyperparams_receipt(self.model_type, receipt)