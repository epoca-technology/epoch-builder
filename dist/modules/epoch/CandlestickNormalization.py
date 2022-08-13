from typing import Tuple
from pandas import DataFrame
from modules.candlestick.Candlestick import Candlestick





def normalize_prediction_candlesticks(prediction_df: DataFrame) -> Tuple[float, float]:
    """Normalizes the Prediction Candlesticks DataFrame and saves it in the 
    appropiate directory. 

    Args:
        prediction_df: DataFrame
            DataFrame containing all the prediction candlesticks within the Epoch.

    Returns:
        Tuple[float, float]
        (highest_price, lowest_price)
    """
    # Create a copy of the prediction df
    df: DataFrame = prediction_df[["ot", "ct", "c"]].copy()

    # Populate the min and max
    max: float = df["c"].max()
    min: float = df["c"].min()

    # Normalize the close prices
    df["c"] = df[["c"]].apply(lambda x: (x - min) / (max - min))

    # When the dataset normalization takes place, the lowest price is converted to 0.
    # This value can bring negative impacts to the model's training process and therefore,
    # it should be replaced with the second lowest price recorded.
    df.loc[df["c"].nsmallest(1).index, "c"] = df["c"].nsmallest(2).iloc[-1]

    # Save the normalized df
    df.to_csv(Candlestick.NORMALIZED_PREDICTION_CANDLESTICK_CONFIG["csv_file"], index=False)

    # Finally, return the highest and lowest prices
    return max, min