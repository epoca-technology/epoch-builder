from typing import Union
from math import ceil
from pandas import DataFrame, read_csv
from modules._types import ICandlestickConfig, ICandlestickBuildPayload








# Class
class Candlestick:
    """Candlestick Class

    This singleton initializes the candlestick data and provides efficient ways of retrieving it.


    Class Properties:
        Paths:
            ASSETS_PATH: str
                Candlesticks Assets Path.
        
        Configs:
            DEFAULT_CANDLESTICK_CONFIG: ICandlestickConfig
                The settings to be used for managing the default candlesticks.
            PREDICTION_CANDLESTICK_CONFIG: ICandlestickConfig
                The settings to be used for managing the prediction candlesticks.
            NORMALIZED_PREDICTION_CANDLESTICK_CONFIG: ICandlestickConfig
                The settings to be used for managing the normalized prediction candlesticks.
        
        DataFrames:
            DF: DataFrame
                One Minute Candlesticks DataFrame with the following columns: ot, ct, o, h, l, c
            PREDICTION_DF: DataFrame 
                Prediction Candlesticks DataFrame with the following columns: ot, ct, o, h, l, c, v
            NORMALIZED_PREDICTION_DF: DataFrame
                Normalized Prediction Candlesticks DataFrame with the following columns: ot, ct, c.
    """
    # Assets' Paths
    ASSETS_PATH: str = "candlesticks"


    # Default Candlesticks Configuration
    DEFAULT_CANDLESTICK_CONFIG: ICandlestickConfig = {
        "columns": ("ot", "ct", "o", "h", "l", "c"),
        "csv_file": f"{ASSETS_PATH}/candlesticks.csv",
        "interval_minutes": 1
    }


    # Prediction Candlesticks Configuration
    PREDICTION_CANDLESTICK_CONFIG: ICandlestickConfig = {
        "columns": ("ot", "ct", "o", "h", "l", "c", "v"),
        "csv_file": f"{ASSETS_PATH}/prediction_candlesticks.csv",
        "interval_minutes": 30
    }


    # Normalized Prediction Candlesticks Configuration
    NORMALIZED_PREDICTION_CANDLESTICK_CONFIG: ICandlestickConfig = {
        "columns": ("ot", "ct", "c"),
        "csv_file": f"{ASSETS_PATH}/normalized_prediction_candlesticks.csv",
        "interval_minutes": PREDICTION_CANDLESTICK_CONFIG["interval_minutes"]
    }


    # DataFrames
    DF: DataFrame = DataFrame()
    PREDICTION_DF: DataFrame = DataFrame()
    NORMALIZED_PREDICTION_DF: DataFrame = DataFrame()







    ####################
    ## Initialization ##
    ####################


    @staticmethod
    def init(lookback: int, start: int, end: int) -> None:
        """Initializes the Candlestick Class based on the provided date range (If any).
        It also removes the 1m candlesticks that are within the lookback period.

        Args:
            lookback: int
                The lookback value used by regressions.
            start: int
                Start Date for the candlestick dataframes.
            end: int
                End Date for the candlestick dataframes.

        Raises:
            ValueError: 
                If it cannot load the DataFrames for any reason or the values are invalid.
        """
        # Init the Candlestick DataFrames
        Candlestick.DF: DataFrame = Candlestick.load_df(Candlestick.DEFAULT_CANDLESTICK_CONFIG, start, end)
        Candlestick.PREDICTION_DF: DataFrame = Candlestick.load_df(Candlestick.PREDICTION_CANDLESTICK_CONFIG, start, end)
        Candlestick.NORMALIZED_PREDICTION_DF: DataFrame = Candlestick.load_df(Candlestick.NORMALIZED_PREDICTION_CANDLESTICK_CONFIG, start, end)

        # The models need data prior to the current time to perform predictions. Since the default candlesticks
        # will be used for simulating, the df needs to start from a point in which there are enough prediction
        # candlesticks in order to make a prediction. Once the subsetting is done, reset the indexes.
        Candlestick.DF = Candlestick.DF[Candlestick.DF["ot"] >= Candlestick.PREDICTION_DF.iloc[lookback]["ot"]]
        Candlestick.DF.reset_index(drop=True, inplace=True)

        # The default df should start at the same time as the prediction df at the lookback index
        if Candlestick.DF.iloc[0]["ot"] != Candlestick.PREDICTION_DF.iloc[lookback]["ot"]:
            raise ValueError(f"The default and prediction candlestick dataframes dont start at the same time. \
                {Candlestick.DF.iloc[0]['ot']} != {Candlestick.PREDICTION_DF.iloc[lookback]['ot']}")

        # The prediction and normalized df should start at the same time
        if Candlestick.PREDICTION_DF.iloc[0]["ot"] != Candlestick.NORMALIZED_PREDICTION_DF.iloc[0]["ot"]:
            raise ValueError(f"The prediction and normalized candlestick dataframes dont start at the same time. \
                {Candlestick.PREDICTION_DF.iloc[0]['ot']} != {Candlestick.NORMALIZED_PREDICTION_DF.iloc[0]['ot']}")

        # The default df should end at the same time as the prediction and normalized dfs
        if Candlestick.DF.iloc[-1]["ct"] != Candlestick.PREDICTION_DF.iloc[-1]["ct"] or Candlestick.DF.iloc[-1]["ct"] != Candlestick.NORMALIZED_PREDICTION_DF.iloc[-1]["ct"]:
            raise ValueError(f"The default and prediction candlestick dataframes dont start at the same time. \
                {Candlestick.DF.iloc[-1]['ct']} != {Candlestick.PREDICTION_DF.iloc[-1]['ct']} != {Candlestick.NORMALIZED_PREDICTION_DF.iloc[-1]['ct']}")

        # The default dataset must have more rows than the prediction dataset
        if Candlestick.DF.shape[0] <= Candlestick.PREDICTION_DF.shape[0]:
            raise ValueError(f"The default candlesticks dataframe must contain more rows than the prediction dataframe. \
                {Candlestick.DF.shape[0]} <= {Candlestick.PREDICTION_DF.shape[0]}")

        # The prediction and its normalized variant should have the same number of rows
        if Candlestick.PREDICTION_DF.shape[0] != Candlestick.NORMALIZED_PREDICTION_DF.shape[0]:
            raise ValueError(f"The prediction and the normalized prediction candlestick dataframes have different number of rows. \
                {Candlestick.PREDICTION_DF.shape[0]} != {Candlestick.NORMALIZED_PREDICTION_DF.shape[0]}")











    ##########################
    ## Candlesticks Builder ##
    ##########################



    @staticmethod
    def build_candlesticks(
        epoch_width: int, 
        sma_window_size: int, 
        train_split: float
    ) -> ICandlestickBuildPayload:
        """Initializes the Candlestick Class based on the provided date range (If any).
        It also removes the 1m candlesticks that are within the lookback period.

        Args:
            epoch_width: int
                The number of months that comprise the Epoch.
            sma_window_size: int
                The simple moving average window size that will be used to build the 
                normalized df.
            train_split: float
                The split that will be applied on the data to build the train and test
                datasets.

        Returns:
            ICandlestickBuildPayload

        Raises:
            RuntimeError: 
                If it cannot load/save the DataFrames for any reason or the values are invalid.
                If there are not enough candlesticks in the prediction dataframe in order to build the epoch.
                If there are null values in the sma_df for the epoch width
        """
        # Calculate the number of candlesticks that form the epoch
        epoch_width_days: int = ceil(epoch_width * 30)
        mins_in_a_day: int = 24 * 60
        candles_in_a_day: float = mins_in_a_day / 30
        candles_in_range: int = ceil(candles_in_a_day * epoch_width_days)

        # Load the entire prediction candlesticks df
        pred_df: DataFrame = Candlestick.load_df(Candlestick.PREDICTION_CANDLESTICK_CONFIG)

        # Make sure there are enough items in the df
        if candles_in_range > pred_df.shape[0]:
            raise RuntimeError(f"There are not enough candlesticks in the prediction df as it contains {pred_df.shape[0]} but needs {candles_in_range}")

        # Initialize the sma_df
        sma_df: DataFrame = pred_df[["ot", "ct", "c"]].copy()
        sma_df["c"] = sma_df["c"].rolling(sma_window_size).mean()

        # Subset the sma_df to the epoch's width and make sure there are no null values
        sma_df = sma_df.iloc[-(candles_in_range):]
        if sma_df.isnull().values.any():
            raise RuntimeError(f"The SMA DataFrame has null values in the epoch width. This may be caused by not having enough candlesticks in the prediction df.")
       
        # Calculate the highest and lowest price sma
        highest_price_sma: float = sma_df["c"].max()
        lowest_price_sma: float = sma_df["c"].min()

        # Normalize the sma df
        sma_df["c"] = sma_df["c"].apply(lambda x: (x - lowest_price_sma) / (highest_price_sma - lowest_price_sma))

        # When the dataset normalization takes place, the lowest price is converted to 0.
        # This value can bring negative impacts to the model's training process and therefore,
        # it should be replaced with the second lowest price recorded.
        sma_df.loc[sma_df["c"].nsmallest(1).index, "c"] = sma_df["c"].nsmallest(2).iloc[-1]
        if sma_df["c"].min() <= 0 or sma_df["c"].max() > 1:
            raise RuntimeError(f"The sma_df values were not normalized correctly: {sma_df['c'].min()} | {sma_df['c'].max()}")

        # Calculate the epoch's date range
        start: int = int(sma_df.iloc[0]["ot"])
        end: int = int(sma_df.iloc[-1]["ct"])

        # Calculate the test ds date range
        sma_test_df: DataFrame = sma_df.iloc[int(sma_df.shape[0] * train_split):]
        test_ds_start: int = int(sma_test_df.iloc[0]["ot"])
        test_ds_end: int = int(sma_test_df.iloc[-1]["ct"])

        # Subset the pred_df to the epoch's width
        pred_df = pred_df.iloc[-(candles_in_range):]

        # Load the default candlesticks and subset them to the epoch's width
        default_df: DataFrame = Candlestick.load_df(Candlestick.DEFAULT_CANDLESTICK_CONFIG)
        default_df = default_df[(default_df["ot"] >= start) & (default_df["ct"] <= end)]

        # Make sure the beggining and the ends match perfectly
        if default_df.iloc[0]["ot"] != pred_df.iloc[0]["ot"] or default_df.iloc[0]["ot"] != sma_df.iloc[0]["ot"]:
            raise RuntimeError(f"Candlestick DFs OT Discrepancy: {default_df.iloc[0]['ot']} | {pred_df.iloc[0]['ot']} | {sma_df.iloc[0]['ot']}")
        if default_df.iloc[-1]["ct"] != pred_df.iloc[-1]["ct"] or default_df.iloc[-1]["ct"] != sma_df.iloc[-1]["ct"]:
            raise RuntimeError(f"Candlestick DFs CT Discrepancy: {default_df.iloc[-1]['ct']} | {pred_df.iloc[-1]['ct']} | {sma_df.iloc[-1]['ct']}")

        # The pred and sma dfs must have the same exact number of rows
        if pred_df.shape[0] != sma_df.shape[0]:
            raise RuntimeError(f"The prediction and sma dfs have different number of rows: {pred_df.shape[0]} != {sma_df.shape[0]}")

        # Save the DataFrames
        default_df.to_csv(Candlestick.DEFAULT_CANDLESTICK_CONFIG["csv_file"], index=False)
        pred_df.to_csv(Candlestick.PREDICTION_CANDLESTICK_CONFIG["csv_file"], index=False)
        sma_df.to_csv(Candlestick.NORMALIZED_PREDICTION_CANDLESTICK_CONFIG["csv_file"], index=False)

        # Finally, return the payload
        return {
            "start": start,
            "end": end,
            "test_ds_start": test_ds_start,
            "test_ds_end": test_ds_end,
            "highest_price_sma": highest_price_sma,
            "lowest_price_sma": lowest_price_sma
        }













    ######################
    ## DataFrame Loader ##
    ######################



    @staticmethod
    def load_df(config: ICandlestickConfig, start: Union[int, None]=None, end: Union[int, None]=None) -> DataFrame:
        """ Retrieves the DataFrame for the candlesticks based on the start-end range. If no start or end
        are provided, it will load all the candlesticks.

        Args:
            config: ICandlestickConfig
                The configuration used to extract the candlestick csv file and create the DataFrame
            start: Union[int, None]
                The start time of the candlestick's dataframe. Any rows before this time will be deleted.
            end: Union[int, None]
                The end time of the candlestick's dataframe. Any rows after this time will be deleted.
        
        Returns:
            DataFrame
        
        Raises:
            ValueError: 
                If the data frame does not contain at least 1 row.
                If the data frame does not contain the correct number of columns.
        """
        # Retrieve the CSV File
        df: DataFrame = read_csv(config["csv_file"], usecols=config["columns"])

        # Modify the DataFrame's date range if applies
        if isinstance(start, int) and isinstance(end, int):
            df = df[(df["ot"] >= start) & (df["ct"] <= end)]
            df.reset_index(drop=True, inplace=True)
        
        # Make sure it has the correct amount of rows & columns
        if df.shape[0] == 0:
            raise ValueError("The candlesticks dataframe does not have the correct amount of rows. Expected > 0 but got 0")
        elif df.shape[1] != len(config["columns"]):
            raise ValueError(f"The candlesticks dataframe does not have the correct amount of columns. Expected {len(config['columns'])} but got {df.shape[1]}")
        
        # Return the DataFrame
        return df