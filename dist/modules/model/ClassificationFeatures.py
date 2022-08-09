from typing import List, Union
from pandas import DataFrame
from modules._types import ITechnicalAnalysis
from modules.candlestick.Candlestick import Candlestick
from modules.model.RegressionModelFactory import RegressionModel
from modules.technical_analysis.TechnicalAnalysis import TechnicalAnalysis




def build_features(
    current_timestamp: int, 
    regressions: List[RegressionModel],
    max_lookback: int,
    include_rsi: bool,
    include_aroon: bool,
    lookback_df: Union[DataFrame, None]
) -> List[float]:
    """Builds the list of features that will be used by the Classification to predict.
    As well as dealing with Regression Predictions it will also build the TA values
    if enabled.

    Args:
        current_timestamp: int
            The open time of the current 1 minute candlestick.
        lookback_df: Union[DataFrame, None]
            ConsensusModels pass the lookback df for optimization reasons.

    Returns:
        List[float]
    """
    # Init the lookback_df
    lookback: DataFrame = Candlestick.get_lookback_df(max_lookback, current_timestamp) \
        if lookback_df is None else lookback_df

    # Generate predictions with all the regression models within the classification
    features: List[float] = [ r.feature(current_timestamp, lookback_df=lookback) for r in regressions ]

    # Check if any Technical Anlysis feature needs to be added
    if include_rsi or include_aroon:
        # Retrieve the technical analysis
        ta: ITechnicalAnalysis = TechnicalAnalysis.get_technical_analysis(
            lookback,
            include_rsi=include_rsi,
            include_aroon=include_aroon
        )

        # Populate the RSI feature if enabled
        if include_rsi:
            features.append(ta["rsi"])

        # Populate the Aroon feature if enabled
        if include_aroon:
            features.append(ta["aroon"])

    # Finally, return all the features
    return features