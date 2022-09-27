from typing import List, Tuple, Dict
from tqdm import tqdm
from modules._types import IPredictionModelMinifiedConfig, IDiscovery, IBacktestPerformance, IPredictionModelCertificate,\
    IRegressionConfig
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.regression.Regression import Regression
from modules.prediction_model.PredictionModelConfig import PredictionModelConfig
from modules.prediction_model.PredictionModelAssets import PredictionModelAssets
from modules.prediction_model.PredictionModelDiscovery import PredictionModelDiscovery
from modules.prediction_model.PredictionModelBacktest import PredictionModelBacktest




class PredictionModel:
    """PredictionModel Class

    This class handles the generation of the prediction model build. The output is ready
    to be evaluated and exported.

    Class Properties:
        ...

    Instance Properties:
        assets: PredictionModelAssets
            The instance of the assets manager.
        backtest: PredictionModelBacktest
            The instance of the backtester.
    """





    def __init__(self):
        """Initializes the PredictionModel Instance.
        
        Args:
            ...

        Raises:
            ValueError:
                If no regression ids are provided.
                If any of the regressions cannot be initialized.
        """
        # Initialize the instance of the assets
        self.assets: PredictionModelAssets = PredictionModelAssets()

        # Initialize the Backtest Instance
        self.backtest: PredictionModelBacktest = PredictionModelBacktest(self.assets.features_num, self.assets.lookback_indexer)






    ###############################
    ## Profitable Configurations ##
    ###############################






    def find_profitable_configs(self, batch_file_name: str) -> None:
        """Given a batch config file name, it will find and save all the 
        profitable model configurations.

        Args:
            batch_file_name: str
                The name of the configuration file that will be explored.
        """
        # Retrieve the configs
        configs: List[IPredictionModelMinifiedConfig] = PredictionModelConfig.get_batch(batch_file_name)

        # Init the profitable configs
        profitable_configs: List[IPredictionModelMinifiedConfig] = []

        # A model is considered to be profitable if it generates at least 50% of the 
        # value of the position_size property.
        min_profit: float = Epoch.POSITION_SIZE * 0.5

        # Init the progress bar
        print(f"\nLooking for profitable prediction models...")
        progress_bar = tqdm(bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=len(configs))

        # Iterate over each config
        for config in configs:
            # Build the features
            features, features_sum = self._build_features(config["ri"])

            # Discovery the model
            disc: IDiscovery = PredictionModelDiscovery().discover(features_sum, self.assets.labels[str(config["pcr"])])

            # Backtest the model
            performance: IBacktestPerformance = self.backtest.calculate_performance(
                price_change_requirement=config["pcr"],
                min_increase_sum=disc["increase_successful_mean"] if config["msf"] == "mean" else disc["increase_successful_median"],
                min_decrease_sum=disc["decrease_successful_mean"] if config["msf"] == "mean" else disc["decrease_successful_median"],
                features=features,
                features_sum=features_sum
            )

            # Shortlist the model if it is profitable
            if performance["profit"] >= min_profit:
                profitable_configs.append(config)

            # Update the progress
            progress_bar.update()

        # Finally, save the profitable models
        PredictionModelConfig.save_profitable_configs(batch_file_name, profitable_configs)









    ###########
    ## Build ##
    ###########




    def build(self, limit: int) -> None:
        """Loads all the profitable configurations and re-evaluates them. Once this
        part of the process is complete, it will order the models by profit and apply
        the slice based on the provided limit. Finally, builds the certificates for 
        the selected top and stores the build.

        Args:
            limit: int
                The maximum number of models that can be placed in the build.

        Raises:
            RuntimeError:
                If there are no profitable model configurations.
        """
        # Init the profitable configs
        profitable_configs: List[IPredictionModelMinifiedConfig] = PredictionModelConfig.get_profitable_configs()

        # Init the list of certificates
        certs: List[IPredictionModelCertificate] = []

        # Init constant values
        creation: int = Utils.get_time()
        regression_configs: Dict[str, IRegressionConfig] = { 
            reg_id: Regression(reg_id).get_config() for reg_id in self.assets.feature_ids
        }

        # Init the progress bar
        print(f"\nBuilding profitable prediction models...")
        progress_bar = tqdm(bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=len(profitable_configs))

        # Iterate over each config
        for config in profitable_configs:
            # Build the features
            features, features_sum = self._build_features(config["ri"])

            # Discovery the model
            disc: IDiscovery = PredictionModelDiscovery().discover(features_sum, self.assets.labels[str(config["pcr"])])

            # Backtest the model
            performance: IBacktestPerformance = self.backtest.calculate_performance(
                price_change_requirement=config["pcr"],
                min_increase_sum=disc["increase_successful_mean"] if config["msf"] == "mean" else disc["increase_successful_median"],
                min_decrease_sum=disc["decrease_successful_mean"] if config["msf"] == "mean" else disc["decrease_successful_median"],
                features=features,
                features_sum=features_sum
            )

            # Append the certificate to the list
            id: str = self._generate_model_id()
            certs.append({
                "id": id,
                "creation": creation,
                "test_ds_start": Epoch.TEST_DS_START,
                "test_ds_end": Epoch.TEST_DS_END,
                "model": {
                    "id": id,
                    "price_change_requirement": config["pcr"],
                    "min_sum_function": config["msf"],
                    "min_increase_sum": disc["increase_successful_mean"] if config["msf"] == "mean" else disc["increase_successful_median"],
                    "min_decrease_sum": disc["decrease_successful_mean"] if config["msf"] == "mean" else disc["decrease_successful_median"],
                    "regressions": [regression_configs[reg_id] for reg_id in config["ri"]]
                },
                "discovery": disc,
                "backtest": performance
            })

            # Update the progress
            progress_bar.update()

        # Make sure profitable certificates were built
        if len(certs) == 0:
            raise RuntimeError("No profitable certificates were built.")

        # Sort the models by profit from high to low
        certs = sorted(certs, key=lambda x: x["backtest"]["profit"], reverse=True)

        # Finally, apply a slice based on the provided limit and save the build
        Utils.write(Epoch.PATH.prediction_models_build(), certs[:limit])






















    ##################
    ## Misc Helpers ##
    ##################





    def _build_features(self, regression_ids: List[str]) -> Tuple[List[List[float]], List[float]]:
        """Builds the features lists and sums structured by index for a given
        list of regressions.

        Args:
            regression_ids: List[str]
                The list of regressions in the model.

        Returns:
            Tuple[List[List[float]], List[float]]
            (features, features_sum)
        """
        # Init values
        features: List[List[float]] = []
        features_sum: List[float] = []

        # Iterate over each item
        for index in range(self.assets.features_num):
            # Append the list of features for the index
            feature_list: List[float] = [self.assets.features[id][index] for id in regression_ids]
            features.append(feature_list)

            # Append the sum of the features
            features_sum.append(sum(feature_list))

        # Finally, return the packed feature lists
        return features, features_sum










    def _generate_model_id(self) -> str:
        """Generates a random ID that will be assigned to a model variation
        within the build.

        Returns:
            str
        """
        return f"{Epoch.ID}_{Utils.generate_uuid4()}"