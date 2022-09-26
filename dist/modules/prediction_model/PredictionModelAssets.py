from typing import List, Tuple, Dict, Union
from numpy import ndarray, array
from pandas import DataFrame
from tqdm import tqdm
from modules._types import ILookbackIndexer, ITestDatasetFeatures, ITestDatasetLabel, ITestDatasetLabels
from modules.utils.Utils import Utils
from modules.candlestick.Candlestick import Candlestick
from modules.epoch.Epoch import Epoch
from modules.regression.Regression import Regression




class PredictionModelAssets:
    """PredictionModelAssets Class

    This class handles the generation and management of all the assets needed by
    the PredictionModel.

    Class Properties:
        ...

    Instance Properties:
        features: ITestDatasetFeatures
            A dict containing the list of features within the test dataset. They are grouped
            by regression ID and follow the adjusted prediction indexing.
        labels: ITestDatasetLabels
            A dict containing the list of labels (outcomes) within the test dataset. They are
            grouped by price change requirement in string format.
            It is also important to mention that they follow the adjusted prediction indexing
            and there may be less labels than features in some cases.
        lookback_indexer: ILookbackIndexer
            The lookback indexer contains a dict with 1m candlestick open times as keys and 
            prediction candlestick indexes as values. 
    """



    def __init__(self):
        """Initializes the PredictionModelAssets Instance. When invoked,
        it will load all the assets.
        """
        # Init the features
        self.features: ITestDatasetFeatures = Utils.read(Epoch.PATH.prediction_models_features())
        self.feature_ids: List[str] = list(self.features.keys())
        self.features_num: int = len(self.features[self.feature_ids[0]])

        # Init the test ds labels
        self.labels: ITestDatasetLabels = Utils.read(Epoch.PATH.prediction_models_labels())

        # Init the lookback indexer
        self.lookback_indexer: ILookbackIndexer = Utils.read(Epoch.PATH.prediction_models_lookback_indexer())










    ####################
    ## Assets Builder ##
    ####################





    @staticmethod
    def build(regression_ids: List[str], price_change_requirements: List[float]) -> None:
        """Builds and stores the features, labels and the lookback indexer.

        Args:
            regression_ids: List[str]
                The list of regressions that will be used to generate prediction models.
            price_change_requirements: List[float]
                The list of price change requirements that will be used to generate
                prediction model variations.
        """
        # Make sure there at least 20 regressions in the list
        if len(regression_ids) < 20:
            raise ValueError(f"A minimum of 20 regressions must be provided in order to build the prediction model's assets. Received: {len(regression_ids)}")

        # Generate the features
        PredictionModelAssets._generate_features(regression_ids)

        # Generate the labels
        PredictionModelAssets._generate_labels(price_change_requirements)

        # Generate the lookback indexer
        PredictionModelAssets._generate_lookback_indexer()












    ##############
    ## Features ##
    ##############




    @staticmethod
    def _generate_features(regression_ids: List[str]) -> None:
        """Builds and saves the test dataset features obtained from
        the build's regressions. The features are saved so they can analyzed
        externally.

        Args:
            regression_ids: List[str]
                The list of regressions that will be used to generate prediction models.
            price_change_requirements: List[float]

        Returns:
            ITestDatasetFeatures
        """
        print("Generating Features...\n")
        # Init the input ds
        input_ds: ndarray = PredictionModelAssets._build_input_ds()

        # Build the features dict
        features: ITestDatasetFeatures = {
            reg_id: Regression(reg_id).predict_feature(input_ds) for reg_id in regression_ids
        }
        
        # Finally, save the features
        Utils.write(Epoch.PATH.prediction_models_features(), features)






    @staticmethod
    def _build_input_ds() -> ndarray:
        """Builds the input dataset that will be used in order to predict all
        the features for each regression.

        Returns:
            ndarray
        """
        # Init the df, grabbing only the close prices
        df: DataFrame = Candlestick.NORMALIZED_PREDICTION_DF[["c"]].copy()

        # Init features list
        features_raw: List[List[float]] = []

        # Iterate over the normalized ds and build the features & labels
        for i in range(Epoch.REGRESSION_LOOKBACK, df.shape[0]):
            # Make sure there are enough items remaining
            if i < (df.shape[0]-Epoch.REGRESSION_PREDICTIONS):
                features_raw.append(df.iloc[i-Epoch.REGRESSION_LOOKBACK:i, 0])

        # Finally, return the features as a numpy array
        return array(features_raw)














    ############
    ## Labels ##
    ############




    @staticmethod
    def _generate_labels(price_change_requirements: List[float]) -> None:
        """Builds and saves the labels file for all price_change_requirements. In case it has 
        already been saved, it loads it into memory.

        Args:
            price_change_requirements: List[float]
                The list of price change requirements that will be used to generate the labels.
        """
        print("Generating Labels...")
        # Initialize the labels dict
        labels: ITestDatasetLabels = {}

        # Iterate over each pcr, generating the labels
        for pcr in price_change_requirements:
            labels[str(pcr)] = PredictionModelAssets._generate_labels_for_pcr(pcr)
        
        # Store the file for future uses
        Utils.write(Epoch.PATH.prediction_models_labels(), labels)






    @staticmethod
    def _generate_labels_for_pcr(price_change_requirement: float) -> List[ITestDatasetLabel]:
        """Generates the test dataset labels (outcomes) based on a provided price
        change requirement.

        Args:
            price_change_requirement: float
                The price percentage change required in order to determine the
                label.

        Returns:
            List[ITestDatasetLabel]
        """
        # Init the label list
        labels: List[ITestDatasetLabel] = []
        print(f"\nGenerating PCR {price_change_requirement}% Labels...")
        progress_bar = tqdm(bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=Candlestick.PREDICTION_DF.shape[0] - Epoch.REGRESSION_LOOKBACK)
        
        # Iterate over each prediction candlestick, after the initial lookback
        for prediction_candlestick in Candlestick.PREDICTION_DF.iloc[Epoch.REGRESSION_LOOKBACK:].to_records():
            # Retrieve the label
            label: Union[ITestDatasetLabel, None] = PredictionModelAssets._get_label(price_change_requirement, prediction_candlestick)

            # Add it if a label was determined
            if label is not None:
                labels.append(label)

            # Otherwise, stop looking for labels
            else:
                break

            # Update progress
            progress_bar.update()

        # Finally, return the labels
        return labels






    @staticmethod
    def _get_label(price_change_requirement: float, pred_candlestick: Dict[str, float]) -> ITestDatasetLabel:
        """Based on a provided prediction candlestick, it will determine what the outcome 
        will be in the future, based on the pcr.

        Args:
            price_change_requirement: float
                The percentage change required for an outcome to be determined.
            pred_candlestick: Dict[str, float]
                The current prediction candlestick.

        Returns:
            ITestDatasetLabel
        """
        # Calculate the exit prices
        increase_price, decrease_price = PredictionModelAssets._get_exit_prices(price_change_requirement, pred_candlestick["o"])

        # Iterate over 1m candlesticks until the label is determined
        for candlestick in Candlestick.DF[Candlestick.DF["ot"] > pred_candlestick["ot"]].to_records():
            if candlestick["h"] >= increase_price:
                return 1
            elif candlestick["l"] <= decrease_price:
                return 0
        
        # If no label is determined, return None
        return None






    @staticmethod
    def _get_exit_prices(price_change_requirement: float, current_price: float) -> Tuple[float, float]:
        """Calculates the prices to determine if the price had an increase or a decrease outcome.

        Args:
            price_change_requirement: float
                The percentage change the price needs to experience in order to determine
                an increase or decrease outcome.
            current_price: float
                The open price of the current candlestick
        """
        return Utils.alter_number_by_percentage(current_price, price_change_requirement), \
            Utils.alter_number_by_percentage(current_price, -price_change_requirement)




















    ######################
    ## Lookback Indexer ##
    ######################




    @staticmethod
    def _generate_lookback_indexer() -> None:
        """Builds and saves the lookback indexer file.
        """
        # Init the indexer
        indexer: ILookbackIndexer = {}
        print("\n\nGenerating Lookback Indexer")
        progress_bar: tqdm = tqdm(bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=Candlestick.DF.shape[0])

        # Iterate over each candlestick and populate the indexer
        for candlestick_1m in Candlestick.DF.to_records():
            indexer[int(candlestick_1m["ot"])] = PredictionModelAssets._get_prediction_candlestick_index(candlestick_1m["ot"])
            progress_bar.update()

        # Store the indexer for future use
        Utils.write(Epoch.PATH.prediction_models_lookback_indexer(), indexer)




    @staticmethod
    def _get_prediction_candlestick_index(current_time: int) -> int:
        """Based on the time of a 1 minute candlestick, it returns the index
        it belongs to in the prediction df.

        Args:
            current_time: int
                The open time of the current 1m candlestick.

        Returns:
            int
        """
        return int(Candlestick.PREDICTION_DF[Candlestick.PREDICTION_DF["ot"] <= current_time].iloc[-1:].index.values[0] - Epoch.REGRESSION_LOOKBACK)