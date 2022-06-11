from typing import Union, List, Dict
from random import choice
from string import ascii_uppercase, digits
from os import makedirs
from os.path import exists
from json import dumps
from modules.utils import Utils
from modules.keras_models import KERAS_PATH, IKerasModelConfig
from modules.classification import IClassificationTrainingConfig, IClassificationTrainingBatch
from modules.hyperparams import IKerasLoss
import modules.hyperparams.src.KerasClassification as source



class KerasClassificationHyperparams:
    """KerasClassificationHyperparams Class

    This class handles the generation of n number of Keras Classification Model Configurations
    ready to be trained.

    Class Properties:
        LEARNING_RATES: List[float]
            The list of learning rates that will be used when compiling models.
        OPTIMIZERS: List[str]
            The list of optimizers that will be used when compiling models.
        LOSS: List[IKerasLoss]
            The list of loss and metric combinations that will be used when compiling models.
        ACTIVATIONS: List[str]
            The list of different activations that will be used to structure models.
        DROPOUT_RATES: List[float]
            The list of different dropout rates that will be used to structure models.
        POOL_SIZES: List[int]
            The list of different pool sizes that will be used to structure CNN models.
    """


    # Learning Rates
    LEARNING_RATES: List[float] = [ 0.01, 0.001, 0.0001 ]


    # Optimizers
    OPTIMIZERS: List[str] = [ "adam", "rmsprop" ]


    # Loss
    LOSS: List[IKerasLoss] = [ 
        { "func_name": "categorical_crossentropy", "metric": "categorical_accuracy" },
        { "func_name": "binary_crossentropy", "metric": "binary_accuracy" }
    ]


    # Activations
    ACTIVATIONS: List[str] = [ "relu", "tanh" ]


    # Dropout Rates
    DROPOUT_RATES: List[float] = [ 0.25 ] # Changed from [0.25, 0.5] in order to lower the number of tests


    # Pool Sizes
    POOL_SIZES: List[int] = [ 2 ] # Changed from [2, 4] in order to lower the number of tests




    @staticmethod
    def generate(network_type: str, training_data_id: str) -> None:
        """Generates a series of Keras Classification Model Configurations for the given
        network type and outputs it to the classification_training_configs directory.

        Args:
            network_type: str ("DNN", "CNN", "LSTM", "CLSTM")
                The type of network that will be focused.
            training_data_id: str
                The identifier of the training data that will be used for the models.
        """
        # Init the model Variations
        network_models: Dict[str, List[IKerasModelConfig]] = KerasClassificationHyperparams._get_network_models(network_type)

        # Init the list of configs
        configs: List[IClassificationTrainingConfig] = []

        # Iterate over each variation
        for variation_key, variations in network_models.items():

            # Iterate over the learning rates
            for learning_rate in KerasClassificationHyperparams.LEARNING_RATES:

                # Iterate over the optimizers
                for optimizer in KerasClassificationHyperparams.OPTIMIZERS:

                    # Iterate over the loss functions
                    for loss in KerasClassificationHyperparams.LOSS:
                        
                        # Generate all the combinations for the variation and concatenate them
                        configs = configs + KerasClassificationHyperparams._get_variation_configs(
                            network_type=network_type,
                            keras_model_name= variation_key,
                            keras_model_variations=variations,
                            learning_rate=learning_rate,
                            optimizer=optimizer,
                            loss=loss
                        )
                
        # Finally, save the file
        KerasClassificationHyperparams._save(network_type, {
            "name": f"{network_type}_VARIATIONS",
            "training_data_id": training_data_id,
            "hyperparams_mode": True,
            "models": configs
        })





    @staticmethod
    def _get_variation_configs(
        network_type: str,
        keras_model_name: str,
        keras_model_variations: List[IKerasModelConfig],
        learning_rate: float,
        optimizer: str,
        loss: IKerasLoss
    ) -> List[IClassificationTrainingConfig]:
        """Iterates over all the variations and adds all the possible combinations.

        Args:
            network_type: str
            keras_model_name: str
            keras_model_variations: List[IKerasModelConfig]
            learning_rate: float
            optimizer: str
            loss: IKerasLoss
        Returns:
            List[IClassificationTrainingConfig]
        """
        # Init the Keras Models Configs
        keras_model_configs: List[IKerasModelConfig] = []

        # Init the activations
        activations: List[Union[str, None]] = KerasClassificationHyperparams.ACTIVATIONS if network_type != "LSTM" else [None]

        # Iterate over each activation function
        for activation in activations:
            
            # Iterate over each variation
            for variation in keras_model_variations:

                # Check if the variation only has dropout rates
                if variation.get("dropout_rates") != None and variation.get("pool_sizes") == None:
                    # Iterate over the dropout variations
                    for dropout in KerasClassificationHyperparams.DROPOUT_RATES:
                        keras_model_configs.append({
                            "activations": [activation]*len(variation["activations"]) if activation is not None else None,
                            "units": variation.get("units"),
                            "filters": variation.get("filters"),
                            "dropout_rates": [dropout]*len(variation["dropout_rates"])
                        })

                # Check if the variation only has pool sizes
                elif variation.get("dropout_rates") == None and variation.get("pool_sizes") != None:
                    # Iterate over the pool size variations
                    for pool_size in KerasClassificationHyperparams.POOL_SIZES:
                        keras_model_configs.append({
                            "activations": [activation]*len(variation["activations"]) if activation is not None else None,
                            "units": variation.get("units"),
                            "filters": variation.get("filters"),
                            "pool_sizes": [pool_size]*len(variation["pool_sizes"])
                        })


                # Check if the variation only both, dropouts and pool sizes
                elif variation.get("dropout_rates") != None and variation.get("pool_sizes") != None:
                    # Iterate over the dropout variations
                    for dropout in KerasClassificationHyperparams.DROPOUT_RATES:
                        # Iterate over the pool size variations
                        for pool_size in KerasClassificationHyperparams.POOL_SIZES:
                            keras_model_configs.append({
                                "activations": [activation]*len(variation["activations"]) if activation is not None else None,
                                "units": variation.get("units"),
                                "filters": variation.get("filters"),
                                "dropout_rates": [dropout]*len(variation["dropout_rates"]),
                                "pool_sizes": [pool_size]*len(variation["pool_sizes"])
                            })

                # Otherwise, add the traditional model
                else:
                    keras_model_configs.append({
                        "activations": [activation]*len(variation["activations"]) if activation is not None else None,
                        "units": variation.get("units"),
                        "filters": variation.get("filters")
                    })

        # Finally, return the configs
        return [KerasClassificationHyperparams._generate_model_config(
            keras_model_name=keras_model_name,
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss=loss,
            activations=c.get("activations"),
            units=c.get("units"),
            filters=c.get("filters"),
            dropout_rates=c.get("dropout_rates"),
            pool_sizes=c.get("pool_sizes"),
        ) for c in keras_model_configs]







    @staticmethod
    def _generate_model_config(
        keras_model_name: str,
        learning_rate: float,
        optimizer: str,
        loss: IKerasLoss,
        activations: Union[List[str], None]=None,
        units: Union[List[int], None]=None,
        filters: Union[List[int], None]=None,
        dropout_rates: Union[List[float], None]=None,
        pool_sizes: Union[List[int], None]=None,
    ) -> IClassificationTrainingConfig:
        """Builds the configuration for a model ready to be trained and evaluated.

        Args:
            keras_model_name: str
            learning_rate: float
            optimizer: str
            loss: IKerasLoss
            units: Union[List[int], None]
            activations: Union[List[str], None]
            dropout_rates: Union[List[float], None]
            filters: Union[List[int], None]
            pool_sizes: Union[List[int], None]

        Returns:
            IClassificationTrainingConfig
        """
        # Init the keras model configuration
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

        # Populate the pool_sizes if any
        if isinstance(pool_sizes, list):
            keras_model["pool_sizes"] = pool_sizes
        
        # Finally, return the configuration
        return {
            "id": f"{keras_model_name}_{KerasClassificationHyperparams._generate_id_suffix()}",
            "description": "Generated by Plutus Hyperparams.",
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "loss": loss["func_name"],
            "metric": loss["metric"],
            "keras_model": keras_model
        }







    @staticmethod
    def _save(network: str, network_file: IClassificationTrainingBatch) -> None:
        """Saves the Classification Training Batch File into the configs directory.

        Args:
            network: str
                The type of network that is going to be trained.
            network_file: 
        """
        # If the results directory doesn't exist, create it
        if not exists(KERAS_PATH["classification_training_configs"]):
            makedirs(KERAS_PATH["classification_training_configs"])

        # Write the results on a JSON File
        with open(f"{KERAS_PATH['classification_training_configs']}/C_{network}_{str(Utils.get_time())}.json", "w") as outfile:
            outfile.write(dumps(network_file, indent=4))















    ## Misc Helpers ##




    @staticmethod
    def _get_network_models(network_type: str) -> Union[source.IDNN, source.ICNN, source.ILSTM, source.ICLSTM]:
        """Returns the appropiate network models

        Args:
            network_type: str
                The network to retrieve the models and variations for.

        Returns:
            Union[source.IDNN, source.ICNN, source.ILSTM, source.ICLSTM]
        """
        if network_type == "DNN":
            return source.DNN
        elif network_type == "CNN":
            return source.CNN
        elif network_type == "LSTM":
            return source.LSTM
        elif network_type == "CLSTM":
            return source.CLSTM
        else:
            raise ValueError(f"The provided network type ({str(network_type)}) is invalid.")






    @staticmethod
    def _generate_id_suffix() -> str:
        """Generates a random string that will be added to the end of the
        model's id in order to prevent duplicate ID issues.

        Returns:
            str
        """
        return ''.join(choice(ascii_uppercase + digits) for _ in range(10))