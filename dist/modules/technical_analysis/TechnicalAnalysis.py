from typing import Union, List
from pandas import DataFrame, Series
from ta.momentum import rsi
from ta.trend import AroonIndicator
from modules.types import ITechnicalAnalysis
from modules.database.Database import Database











# Class
class TechnicalAnalysis:
    """TechnicalAnalysis Class

    This singleton manages the technical analysis indicators for Classifications.

    The docs for the ta library can be found here:
    https://technical-analysis-library-in-python.readthedocs.io/
    """






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
                If enabled, the normalized 'rsi' result will be included in the response.
            include_aroon: bool
                If enabled, the normalized 'aroon' result will be included in the response.

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
        if include_aroon and not isinstance(ta.get("aroon"), float):
            ta["aroon"] = TechnicalAnalysis._calculate_aroon(lookback_df["c"])
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
        return snap[0]["ta"] if len(snap) > 0 else None






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
    def _calculate_rsi(close: Series) -> float:
        """Returns the current RSI value for a given series.

        Args:
            close: Series
                The close price series that will be used by the indicator.
        
        Returns:
            float
        """
        # Calculate the RSI Series
        result: Series = rsi(close, window=14)
        
        # Return the current item in a normalized format
        return TechnicalAnalysis._normalize(result.iloc[-1], 0, 100)







    @staticmethod
    def _calculate_aroon(close: Series) -> float:
        """Returns the current Aroon value for a given series.

        Args:
            close: Series
                The close price series that will be used by the indicator.
        
        Returns:
            float
        """
        # Calculate the Aroon Series
        result: Series = AroonIndicator(close, window=25).aroon_indicator()

        # Return the current item in a normalized format
        return TechnicalAnalysis._normalize(result.iloc[-1], -100, 100)











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






    @staticmethod
    def _normalize(value: float, minimum: float, maximum: float) -> float:
        """Normalizes a value to a -1 / 1 range based on its min and max
        possible values.

        Args:
            value: float
                The value to be normalized.
            minimum: float
                The minimum value that can be generated by the indicator.
            maximum: float
                The maximum value that can be generated by the indicator.
        """
        # Handle a case in which the value does not need to be normalized
        if minimum == -1 and maximum == 1:
            return round(value, 6)
        
        # Handle a case in which the value just needs to be divider by 100
        elif minimum == -100 and maximum == 100:
            return round((value / 100), 6)

        # Handle a case in which the value needs to be divider by 100 and scaled tu support negative values
        elif minimum == 0 and maximum == 100:
            return round(((value / 100) - 0.5)*2, 6)

        # Handle a case in which the value needs to be scaled tu support negative values
        elif minimum == 0 and maximum == 1:
            return round((value - 0.5)*2, 6)
        
        # Otherwise, throw an error
        else:
            raise ValueError("The technical analysis min and/or max args are invalid.")

