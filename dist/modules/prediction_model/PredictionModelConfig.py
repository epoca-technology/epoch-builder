from typing import List, Tuple
from itertools import combinations
from math import ceil
from random import shuffle
from modules._types import IMinSumFunction, IRegressionsPerModel, IPredictionModelMinifiedConfig
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch




class PredictionModelConfig:
    """PredictionModelConfig Class

    This singleton handles the initialization of the prediction model assets, as well
    as the prediction model configurations.

    Class Properties:
        BATCH_SIZE: int
            The number of minified configs that can be placed in a single batch.
        PRICE_CHANGE_REQUIREMENTS: List[float]
        MIN_SUM_FUNCTIONS: List[IMinSumFunction]
        MIN_SUM_ADJUSTMENT_FACTORS: List[float]
        REGRESSIONS_PER_MODEL: List[IRegressionsPerModel]
            The lists of hyperparameters that will be used to generate configurations.

    Instance Properties:
        ...
    """
    # Batch Size
    BATCH_SIZE: int = 100000

    # The list of price change requirements that will be used to build configs
    PRICE_CHANGE_REQUIREMENTS: List[float] = [ 3 ]

    # Min Sum Functions
    MIN_SUM_FUNCTIONS: List[IMinSumFunction] = [ "mean", "median" ]

    # Min Sum Adjustment Factors
    MIN_SUM_ADJUSTMENT_FACTORS: List[float] = [ 1.5, 2 ]

    # Regressions per model
    REGRESSIONS_PER_MODEL: List[IRegressionsPerModel] = [ 8 ]








    @staticmethod
    def create(regression_ids: List[str]) -> None:
        """Creates and saves all the possible combinations in config
        batches.

        Args:
            regression_ids: List[str]
                The list of selected regression ids.
        """
        # Generate the list of model configs with all the possible combinations
        print("\n\nGenerating model configurations...")
        models: List[IPredictionModelMinifiedConfig] = []

        # Generate the combinations by regressions per model
        combs_by_rpm: List[List[Tuple[str]]] = [
            list(combinations(regression_ids, rpm)) for rpm in PredictionModelConfig.REGRESSIONS_PER_MODEL
        ]

        # Flatten the combinations
        combs: List[List[str]] = [item for sublist in combs_by_rpm for item in sublist]

        # Iterate over each price change requirement
        for pcr in PredictionModelConfig.PRICE_CHANGE_REQUIREMENTS:
            # Iterate over each sum function
            for min_sum_func in PredictionModelConfig.MIN_SUM_FUNCTIONS:
                # Iterate over each min sum adjustment factor
                for adj_factor in PredictionModelConfig.MIN_SUM_ADJUSTMENT_FACTORS:
                    # Iterate over each combination
                    for comb in combs:
                        # Append the model to the list
                        models.append({ 
                            "pcr": pcr, 
                            "msf": min_sum_func, 
                            "msaf": adj_factor, 
                            "ri": list(comb)
                        })

        # Shuffle the configurations in order to make sure that all (or most) batches contain profitable
        # configurations and therefore, keep track of the progress. 
        shuffle(models)

        # Calculate the number of batches that will be stored
        batches: int = ceil(len(models) / PredictionModelConfig.BATCH_SIZE)
        
        # Save the configurations in batches
        slice_start: int = 0
        for batch_number in range(1, batches+1):
            # Init the name of the file
            file_name = f"{Epoch.ID}_{batch_number}_{batches}.json"

            # Calculate the end of the slice
            slice_end: int = slice_start + PredictionModelConfig.BATCH_SIZE

            # Save the batch including only the sliced configs
            Utils.write(Epoch.PATH.prediction_models_configs(file_name), models[slice_start:slice_end])

            # Set the end of the slice as the new start
            slice_start = slice_end

        # Build and save the receipt
        receipt: str = f"{Epoch.ID}: Prediction Models\n\n"
        receipt += f"Creation: {Utils.from_milliseconds_to_date_string(Utils.get_time())}\n"
        receipt += f"Batch Size: {PredictionModelConfig.BATCH_SIZE}\n\n"
        receipt += f"\nRegression Combinations:\n"
        for i, comb in enumerate(combs_by_rpm):
            receipt += f"R{PredictionModelConfig.REGRESSIONS_PER_MODEL[i]}: {len(comb)}\n"
        receipt += f"Total Combinations: {len(combs)}\n\n"
        receipt += f"Total Models: {len(models)}\n\n"
        receipt += f"Configuration Batches ({batches}):\n"
        for batch_number in range(1, batches + 1, 1):
            receipt += f"{Epoch.ID}_{batch_number}: \n"
        Utils.write(Epoch.PATH.prediction_models_configs_receipt(), receipt)

        






    @staticmethod
    def get_batch(batch_file_name: str) -> List[IPredictionModelMinifiedConfig]:
        """Retrieves a configuration batch.

        Args:
            batch_file_name: str
                The name of the batch to be retrieved. It must include the ext.

        Returns:
            List[IPredictionModelMinifiedConfig]
        """
        return Utils.read(Epoch.PATH.prediction_models_configs(batch_file_name))








    @staticmethod
    def save_profitable_configs(batch_file_name: str, configs: List[IPredictionModelMinifiedConfig]) -> None:
        """Retrieves a configuration batch.

        Args:
            batch_file_name: str
                The name of the batch where the profitable configs were found.
            configs: List[IPredictionModelMinifiedConfig]
                The list of all the profitable configurations.
        """
        return Utils.write(Epoch.PATH.prediction_models_profitable_configs(batch_file_name), configs)








    @staticmethod
    def get_profitable_configs() -> List[IPredictionModelMinifiedConfig]:
        """Retrieves a list with all of the profitable configurations found.

        Returns:
            List[IPredictionModelMinifiedConfig]
        
        Raises:
            RuntimeError:
                If no profitable configs are found
        """
        # Init the list
        profitable_configs: List[IPredictionModelMinifiedConfig] = []

        # Retrieve all the profitable config files
        _, config_files = Utils.get_directory_content(Epoch.PATH.prediction_models_profitable_configs(), only_file_ext=".json")
        
        # Load the files and append them to the list
        for file in config_files:
            # Retrieve the configs
            configs: List[IPredictionModelMinifiedConfig] = Utils.read(Epoch.PATH.prediction_models_profitable_configs(file))

            # Concatenate them to the list if there is at least 1
            if len(configs) > 0:
                profitable_configs = profitable_configs + configs
        
        # Make sure at least 1 config was found
        if len(profitable_configs) == 0:
            raise RuntimeError("No profitable prediction model configurations were found.")

        # Finally, return the list of configs
        return profitable_configs
