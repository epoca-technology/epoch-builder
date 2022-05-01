from typing import Tuple, Union, TypedDict
from pandas import DataFrame, Series, read_csv
from ta.momentum import rsi
from ta.trend import ema_indicator
from modules.utils import Utils





# Candlesticks Config Type
class ICandlestickConfig(TypedDict):
    columns: Tuple[str]
    csv_file_name: str






# Class
class Candlestick:
    """Candlestick Class

    This singleton initializes the candlestick data and provides efficient ways of retrieving it.


    Class Properties:
        DEFAULT_CANDLESTICK_CONFIG: ICandlestickConfig
            The settings to be used for managing the default candlesticks.
        PREDICTION_CANDLESTICK_CONFIG: ICandlestickConfig
            The settings to be used for managing the prediction candlesticks.
        DF: DataFrame
            One Minute Candlesticks DataFrame with the following columns: ot, ct, o, h, l, c
        PREDICTION_DF: DataFrame 
            Prediction Candlesticks DataFrame with the following columns: ot, ct, c

    Public Methods:
        
    """

    # Default Candlesticks Configuration
    DEFAULT_CANDLESTICK_CONFIG: ICandlestickConfig = {
        "columns": ("ot", "ct", "o", "h", "l", "c"),
        "csv_file": "candlesticks/candlesticks.csv"
    }

    # Prediction Candlesticks Configuration
    PREDICTION_CANDLESTICK_CONFIG: ICandlestickConfig = {
        "columns": ("ot", "ct", "c"),
        "csv_file": "candlesticks/prediction_candlesticks.csv"
    }


    # DataFrames
    DF: DataFrame = DataFrame()
    PREDICTION_DF: DataFrame = DataFrame()


    


    ## DataFrames Initializer ##



    @staticmethod
    def init(
        max_lookback: int, 
        start: Union[str, int, None] = None, 
        end: Union[str, int, None] = None
    ) -> None:
        """Initializes the Candlestick Class based on the provided date range (If any).
        It also removes the 1m candlesticks that are within the lookback period.

        Args:
            max_lookback: int
                The highest lookback value contained by all the models in the simulation.
            start: Union[str, int, None]
                Start Date for the candlestick dataframes.
            end: Union[str, int, None]
                End Date for the candlestick dataframes.

        Raises:
            ValueError: 
                If it cannot load the DataFrames for any reason or the values are invalid.
        """
        # Initialize the start and the end timestamps if provided
        start: Union[int, None] = Candlestick._get_date_timestamp(start)
        end: Union[int, None] = Candlestick._get_date_timestamp(end)

        # Init the Default & Forecast Candlestick DataFrames
        Candlestick.DF: DataFrame = Candlestick._get_df(Candlestick.DEFAULT_CANDLESTICK_CONFIG, start, end)
        Candlestick.PREDICTION_DF: DataFrame = Candlestick._get_df(Candlestick.PREDICTION_CANDLESTICK_CONFIG, start, end)

        # The models need data prior to the current time to perform predictions. Since the default candlesticks
        # will be used for simulating, the df needs to start from a point in which there are enough forecast
        # candlesticks in order to make a prediction. Once the subsetting is done, reset the indexes.
        Candlestick.DF = Candlestick.DF[Candlestick.DF['ot'] >= Candlestick.PREDICTION_DF.iloc[max_lookback]['ot']]
        Candlestick.DF.reset_index(drop=True, inplace=True)

        # Both datasets should start at the same time. The first forecast candlestick must be selected based on
        # the max_lookback
        if Candlestick.DF.iloc[0]['ot'] != Candlestick.PREDICTION_DF.iloc[max_lookback]['ot']:
            raise ValueError(f"The candlestick dataframes dont start at the same time. \
                {Candlestick.DF.iloc[0]['ot']} != {Candlestick.PREDICTION_DF.iloc[max_lookback]['ot']}")
        
        # The default dataset must have more rows than the forecast dataset
        if Candlestick.DF.shape[0] <= Candlestick.PREDICTION_DF.shape[0]:
            raise ValueError(f"The default candlesticks dataframe must contain more rows than the prediction dataframe. \
                {Candlestick.DF.shape[0]} <= {Candlestick.PREDICTION_DF.shape[0]}")





    @staticmethod
    def _get_date_timestamp(date_value: Union[str, int, None]) -> Union[int, None]:
        """Given a date_value, it will process it according to its format and return the 
        equivalent timestamp in milliseconds.

        Args:
            date_value: Union[str, int, None] 
                The date that needs to be converted.

        Returns:
            Union[int, None]
        """
        # Handle a string conversion
        if isinstance(date_value, str):
            return Utils.from_date_string_to_milliseconds(date_value)
        # If it already is a timestamp, return the provided value
        elif isinstance(date_value, int):
            return date_value
        # If none of the types are met, return None
        else:
            return None





    @staticmethod
    def _get_df(config: ICandlestickConfig, start: Union[int, None], end: Union[int, None]) -> DataFrame:
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

        ## Modify the DataFrame's date range if applies ##

        # Start and End Range have been provided
        if isinstance(start, int) and isinstance(end, int):
            df = df[(df['ot'] >= start) & (df['ct'] <= end)]
            df.reset_index(drop=True, inplace=True)

        # Only the Start was provided
        elif isinstance(start, int):
            df = df[df['ot'] >= start]
            df.reset_index(drop=True, inplace=True)

        # Only the End was provided
        elif isinstance(end, int):
            df = df[df['ct'] <= end]
            df.reset_index(drop=True, inplace=True)
        
        # Make sure it has the correct amount of rows & columns
        if df.shape[0] == 0:
            raise ValueError('The candlesticks dataframe does not have the correct amount of rows. Expected > 0 but got 0')
        elif df.shape[1] != len(config["columns"]):
            raise ValueError(f'The candlesticks dataframe does not have the correct amount of columns. Expected {len(config["columns"])} but got {df.shape[1]}')
        
        # Return the DataFrame
        return df














    ## Prediction Candlesticks Data ##


    @staticmethod
    def get_data_to_predict_on(
        current_timestamp: int,
        lookback: int, 
        include_rsi: bool,
        include_ema: bool
    ) -> Tuple[Series, Union[float, None], Union[float, None], Union[float, None]]:
        """Retrieves the prediction data based on provided params.

        Args:
            current_timestamp: int
                Current 1m candlestick's open timestamp in milliseconds
            lookback: int
                The lookback number set in the model.
            include_rsi: bool 
                Includes the RSI value in the packed tuple. If False, will
                populate the value with None instead.
            include_ema: bool
                Includes the short and long EMA values in the packed tuple. 
                If False, will populate the value with None instead.

        Returns:
            Tuple[Series, Union[float, None], Union[float, None], Union[float, None]]
                (Series, RSI, Short EMA, Long EMA)

        Raises:
            ValueError:
                If the subset forecast df has less or more rows than the provided lookback.
        """
        # Init TA values
        rsi: Union[bool, None] = None
        short_ema: Union[bool, None] = None
        long_ema: Union[bool, None] = None

        # Subset the Prediction DF to only include the rows that will be used and reset indexes
        df: DataFrame = Candlestick.PREDICTION_DF[Candlestick.PREDICTION_DF['ct'] <= current_timestamp].iloc[-lookback:]
        #df.reset_index(drop=True, inplace=True)

        # Make sure the number of rows in the df matches the lookback value
        if df.shape[0] != lookback:
            raise ValueError(f"The number of rows in the subset prediction df is different to the lookback provided. \
                DF Shape: {str(df.shape[0])}, Lookback: {lookback}")

        # Include the RSI if applies
        if include_rsi:
            rsi = Candlestick._get_rsi(df['c'])

        # Include the EMA if applies
        if include_ema:
            short_ema, long_ema = Candlestick._get_ema(df['c'])

        # Return the packed values
        return df['c'], rsi, short_ema, long_ema






    @staticmethod
    def _get_rsi(close_series: Series, window: int = 7) -> float:
        """Returns the last RSI value for a given series.

        Args:
            close_series: Series
                The close price series that will be used by the indicator.
            window: int 
                The number of periods to be used in the RSI calculation (Default: 7).
        
        Returns:
            float
        
        Raises:
            ValueError:
                If the number of rows in the series is equals or less to the RSI window size
        """
        # Make sure the number of rows is greater than the window
        if close_series.shape[0] <= window:
            raise ValueError(f"The number of rows in the series must be greater than the RSI Window. \
                Received: {close_series.shape[0]}")
                
        # Calculate and return the RSI
        rsi_result: Series = rsi(close_series, window=window)
        return round(rsi_result.iloc[-1], 2)





    @staticmethod
    def _get_ema(close_series: Series, short_window: int = 7, long_window = 20) -> Tuple[float, float]:
        """Returns the last EMA values for a given series.

        Args:
            close_series: Series
                The close price series that will be used by the indicator.
            short_window: int 
                The number of periods to be used by the Short EMA (Default: 7).
            long_window: int
                The number of periods to be used by the Long EMA (Default: 20).
        
        Returns:
            Tuple(float, float)

        Raises:
            ValueError:
                If the rows in the series are less than the short or long EMA window.
        """
        # Make sure the number of rows is greater than the short and long window
        if close_series.shape[0] <= short_window or close_series.shape[0] <= long_window:
            raise ValueError(f"The number of rows in the series must be greater than the short and long EMA windows. \
                Received: {close_series.shape[0]}")

        # Init the short and long EMAs
        short_ema: Series = ema_indicator(close_series, window=short_window)
        long_ema: Series = ema_indicator(close_series, window=long_window)
        
        # Pack the last values and return them
        return round(short_ema.iloc[-1], 2), round(long_ema.iloc[-1], 2)








    @staticmethod
    def get_current_prediction_range(lookback: int, current_time: int) -> Tuple[int, int]:
        """Based on the model's lookback and the current time, it will retrieve the open time
        and the close time of the first and the last candlestick used to generate the prediction.

        Args:
            lookback: int
                The number of candlesticks the model looks into the past to make a prediction.
            current_time: int
                The ot of the current 1m candlestick.

        Returns:
            Tuple[int, int] (first_ot, last_ct)
        """
        # Subset the DataFrame
        df: DataFrame = Candlestick.PREDICTION_DF[Candlestick.PREDICTION_DF['ct'] <= current_time].iloc[-lookback:]

        # Return the first ot and the last ct
        return int(df.iloc[0]['ot']), int(df.iloc[-1]['ct'])