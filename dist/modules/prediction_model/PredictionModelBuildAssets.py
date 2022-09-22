from typing import List, Tuple, Dict, Union
from numpy import ndarray, array
from tqdm import tqdm
from modules._types import ILookbackIndexer, ITestDatasetFeatures, ITestDatasetLabel, ITestDatasetLabels
from modules.candlestick.Candlestick import Candlestick
from modules.utils.Utils import Utils
from modules.epoch.Epoch import Epoch
from modules.regression.Regression import Regression




class PredictionModelBuildAssets:
    """PredictionModelBuildAssets Class

    This class handles the generation and management of all the assets needed by
    the builder.

    Instance Properties:
        build_id: str
            The identifier of the build.
        regressions: List[Regression]
            The list of regressions that will be used to generate features.
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



    def __init__(self, build_id: str, regressions: List[Regression], price_change_requirements: List[float]):
        """Initializes the PredictionModelBuildAssets Instance. When invoked,
        it will generate all the assets that do not exist.
        
        Args:
            build_id: str
                The identifier of the build.
            regression_ids: List[Regression]
                The list of regression instances.
            price_change_requirements: List[float]
                The list of price change requirements that will need labels.
        """
        # Init the id
        self.build_id: str = build_id

        # Init the regressions
        self.regressions: List[Regression] = regressions

        # Init the features
        self.features: ITestDatasetFeatures = self._get_features()

        # Init the test ds labels
        self.labels: ITestDatasetLabels = self._get_labels(price_change_requirements)

        # Init the lookback indexer
        self.lookback_indexer: ILookbackIndexer = self._get_lookback_indexer()










    ##############
    ## Features ##
    ##############





    def _get_features(self) -> ITestDatasetFeatures:
        """Builds, saves and returns the test dataset features obtained from
        the build's regressions. The features are saved so they can analyzed
        externally.

        Returns:
            ITestDatasetFeatures
        """
        # Init the input ds
        input_ds: ndarray = self._build_input_ds()

        # Build the features dict
        features: ITestDatasetFeatures = {r.id: r.predict_feature(input_ds) for r in self.regressions}
        
        # Save the features
        Utils.write(Epoch.PATH.prediction_models_features(self.build_id), features)

        # Finally, return them
        return features






    def _build_input_ds(self) -> ndarray:
        """Builds the input dataset that will be used in order to predict all
        the features for each regression.

        Returns:
            ndarray
        """
        # Init features list
        features_raw: List[List[float]] = []

        # Iterate over the normalized ds and build the features & labels
        for i in range(Epoch.REGRESSION_LOOKBACK, Candlestick.NORMALIZED_PREDICTION_DF.shape[0]):
            # Make sure there are enough items remaining
            if i < (Candlestick.NORMALIZED_PREDICTION_DF.shape[0]-Epoch.REGRESSION_PREDICTIONS):
                features_raw.append(Candlestick.NORMALIZED_PREDICTION_DF.iloc[i-Epoch.REGRESSION_LOOKBACK:i, 0])

        # Finally, return the features as a numpy array
        return array(features_raw)














    ############
    ## Labels ##
    ############





    def _get_labels(self, price_change_requirements: List[float]) -> ITestDatasetLabels:
        """Builds and saves the labels file for all price_change_requirements. In case it has 
        already been saved, it loads it into memory.
        """
        # Init the path
        path: str = Epoch.PATH.prediction_models_labels()

        # Check if the file exists
        if Utils.file_exists(path):
            return Utils.read(path)

        # Otherwise, generate it
        else:
            # Initialize the labels dict
            labels: ITestDatasetLabels = {}

            # Iterate over each pcr, generating the labels
            for pcr in price_change_requirements:
                labels[str(pcr)] = self._generate_labels(pcr)
            
            # Store the file for future uses
            Utils.write(path, labels)

            # Finally, return it
            return labels







    def _generate_labels(self, price_change_requirement: float) -> List[ITestDatasetLabel]:
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
        null_label_indexes: List[int] = []
        progress_bar: tqdm = tqdm(bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=Candlestick.PREDICTION_DF.shape[0] - Epoch.REGRESSION_LOOKBACK)
        progress_bar.set_description(f"Generating PCR {price_change_requirement}%")
        
        # Iterate over each prediction candlestick, after the initial lookback
        i: int = 0
        for prediction_candlestick in Candlestick.PREDICTION_DF.iloc[Epoch.REGRESSION_LOOKBACK:].to_records():
            # Retrieve the label
            label: Union[ITestDatasetLabel, None] = self._get_label(prediction_candlestick)

            # Add it if a label was determined
            if label is not None:
                labels.append(label)

            # Otherwise, add it to the null labels
            else:
                null_label_indexes.append(i)

            # Update progress
            i += 1
            progress_bar.update()

        # Report null labels
        print(f"\nThe following indexes returned null labels ({len(null_label_indexes)}): {str(null_label_indexes)}\n")

        # Finally, return the labels
        return labels







    def _get_label(self, price_change_requirement: float, pred_candlestick: Dict[str, float]) -> ITestDatasetLabel:
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
        increase_price, decrease_price = self._get_exit_prices(price_change_requirement, pred_candlestick["o"])

        # Iterate over 1m candlesticks until the label is determined
        for candlestick in Candlestick.DF[Candlestick.DF["ot"] > pred_candlestick["ot"]].to_records():
            if candlestick["h"] >= increase_price:
                return 1
            elif candlestick["l"] <= decrease_price:
                return 0
        
        # If no label is determined, return None
        return None







    def _get_exit_prices(self, price_change_requirement: float, current_price: float) -> Tuple[float, float]:
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





    def _get_lookback_indexer(self) -> ILookbackIndexer:
        """Builds and saves the lookback indexer file. In case it has 
        already been saved, it loads it into memory.

        Returns:
            ILookbackIndexer
        """
        # Init the path
        path: str = Epoch.PATH.prediction_models_lookback_indexer()

        # Check if the file exists
        if Utils.file_exists(path):
            return Utils.read(path)

        # Otherwise, generate it
        else:
            # Init the indexer
            indexer: ILookbackIndexer = {}
            progress_bar: tqdm = tqdm(bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=Candlestick.DF.shape[0])
            progress_bar.set_description("Generating Lookback Indexer")

            # Iterate over each candlestick and populate the indexer
            for candlestick_1m in Candlestick.DF.to_records():
                indexer[int(candlestick_1m["ot"])] = self.get_prediction_candlestick_index(candlestick_1m["ot"])
                progress_bar.update()

            # Store the indexer for future use
            Utils.write(path, indexer)

            # Finally, return it
            return indexer





    def get_prediction_candlestick_index(self, current_time: int) -> int:
        """Based on the time of a 1 minute candlestick, it returns the index
        it belongs to in the prediction df.

        Args:
            current_time: int
                The open time of the current 1m candlestick.

        Returns:
            int
        """
        return Candlestick.PREDICTION_DF[Candlestick.PREDICTION_DF["ot"] <= current_time].iloc[-1:].index.values[0] - Epoch.REGRESSION_LOOKBACK