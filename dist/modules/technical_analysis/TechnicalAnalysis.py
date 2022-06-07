from typing import Tuple, TypedDict, Union, List
from pandas import DataFrame, Series
from ta.momentum import rsi
from ta.trend import aroon_up, aroon_down
from modules.database import Database





# Type
class ITechnicalAnalysis(TypedDict):
    rsi: Union[float, None]
    aroon_up: Union[float, None]
    aroon_down: Union[float, None]





# Class
class TechnicalAnalysis:
    """TechnicalAnalysis Class

    This singleton manages the technical analysis indicators for Classifications.


    Class Properties:
        SHORT_WINDOW: int 
        MEDIUM_WINDOW: int 
        LONG_WINDOW: int 
            The windows that will be use for indicators.
        DECIMALS: int
            The number of decimals that will be used in the normalization.
        
    """
    # Windows
    SHORT_WINDOW: int = 7
    MEDIUM_WINDOW: int = 25
    LONG_WINDOW: int = 50

    # Normalization Decimals
    DECIMALS: int = 6






    @staticmethod
    def get_technical_analysis(
        lookback_df: DataFrame, 
        include_rsi: bool=False, 
        include_aroon: bool=False
    ) -> ITechnicalAnalysis:
        """Builds the Technical Analysis Dictionary. The process checks for the results
        in the Database first. If any of the indicators are not found, they will be 
        calculated and then saved.

        Args:
            lookback_df: DataFrame
                The candlestick lookback that will be used to calculate the indicators.
            include_rsi: bool
                If enabled, the RSI result will be included in the response.
            include_aroon: bool
                If enabled, The AROON_UP and AROON_DOWN results will be included in the response.

        Returns:
            ITechnicalAnalysis
        """
        # Initialize the ID
        id: str = TechnicalAnalysis._get_id(lookback_df.iloc[0]["ot"], lookback_df.iloc[-1]["ct"])

        # Init action values
        create: bool = False
        update: bool = False

        # Download the record from the Database if exists
        ta: Union[ITechnicalAnalysis, None] = TechnicalAnalysis._read(id)

        # If the record does not exist, replace it with an empty dict and set the create action
        if ta is None:
            ta = {}
            create = True

        # Handle the RSI Indicator if it has to be included
        if include_rsi and not isinstance(ta.get("rsi"), float):
            ta["rsi"] = TechnicalAnalysis._calculate_rsi(lookback_df["c"])
            update = True if not create else False

        # Handle the Aroon Indicator if it has to be included
        if include_aroon and (not isinstance(ta.get("aroon_up"), float) or not isinstance(ta.get("aroon_down"), float)):
            up, down = TechnicalAnalysis._calculate_aroon(lookback_df["c"]) 
            ta["aroon_up"] = up
            ta["aroon_down"] = down
            update = True if not create else False

        # Check if the TA needs to be saved
        if create or update:
            TechnicalAnalysis._save(id, ta, update=update)

        # Finally, return the analysis
        return ta






    ## Database Record Management ##




    @staticmethod
    def _read(id: str) -> Union[ITechnicalAnalysis, None]:
        """Downloads a technical analysis record from the Database.

        Args:
            id: str
                The identifier of the technical analysis.

        Returns:
            Union[ITechnicalAnalysis, None]
        """
        # Retrieve the technical analysis if any
        snap: List[ITechnicalAnalysis] = Database.read_query(
            f"SELECT ta FROM {Database.tn('technical_analysis')} WHERE id = %s LIMIT 1",
            (id,)
        )

        # Return the prediction if any
        return snap[0]['ta'] if len(snap) == 1 else None






    @staticmethod
    def _save(id: str, ta: ITechnicalAnalysis, update: bool) -> None:
        """Creates or Updates the technical analysis dict in the Database.

        Args:
            id: str
                The identifier of the technical analysis.
            ta: ITechnicalAnalysis
                The up to date technical analysis dict.
            update: bool
                If enabled, it will update the record instead of inserting it.
        """
        # Check if the record needs to be updated
        if update:
            Database.write_query(
                f"UPDATE {Database.tn('technical_analysis')} SET ta=%s WHERE id=%s",
                (ta, id)
            )

        # Otherwise, insert the record
        else:
            # Perform the insert safely as there could be a different machine writting 
            # at the exact same time and therefore, trigger a unique id violation error.
            try:
                Database.write_query(
                    f"INSERT INTO {Database.tn('technical_analysis')}(id, ta) VALUES (%s, %s)",
                    (id, ta)
                )
            except Exception as e:
                print(f"TA Insert Error: {str(e)}")








    @staticmethod
    def _delete(id: str) -> None:
        """Deletes a technical analysis record from the Database.

        Args:
            id: str
                The identifier of the technical analysis.
        """
        Database.write_query(
            f"DELETE FROM {Database.tn('technical_analysis')} WHERE id = %s",
            (id,)
        )












    ## Calculators ##




    @staticmethod
    def _calculate_rsi(close_prices: Series) -> float:
        """Returns the current RSI value for a given series.

        Args:
            close_prices: Series
                The close price series that will be used by the indicator.
        
        Returns:
            float
        """
        # Calculate the RSI Series
        rsi_result: Series = rsi(close_prices, window=TechnicalAnalysis.SHORT_WINDOW)

        # Normalizer
        def _normalize(val: float) -> float:
            return round(((val / 100) - 0.5)*2, 6)
        
        # Return the current item in a normalized format
        return _normalize(rsi_result.iloc[-1])






    @staticmethod
    def _calculate_aroon(close_prices: Series) -> Tuple[float, float]:
        """Returns the current Aroon up and down values for a given series.

        Args:
            close_prices: Series
                The close price series that will be used by the indicator.
        
        Returns:
            Tuple(float, float)
            (aroon_up, aroon_down)
        """
        # Calculate the Aroon Series
        up: Series = aroon_up(close_prices, window=TechnicalAnalysis.SHORT_WINDOW)
        down: Series = aroon_down(close_prices, window=TechnicalAnalysis.SHORT_WINDOW)

        # Normalizers
        def _normalize_up(val: float) -> float:
            return round((val / 100), 6)
        def _normalize_down(val: float) -> float:
            return round((val / 100)*-1, 6)

        # Pack the current values and return them
        return _normalize_up(up.iloc[-1]), _normalize_down(down.iloc[-1])











    ## Misc Helpers ##





    @staticmethod
    def _get_id(first_ot: int, last_ct: int) -> str:
        """Given a lookback range, it will turn it into an id that's compatible with the
        cache system.

        Args:
            first_ot: int
                The open time of the first candlestick of the lookback.
            last_ct: int
                The close time of the last candlestick of the lookback.

        Returns:
            str
        """
        return f"{str(int(first_ot))}_{str(int(last_ct))}"







