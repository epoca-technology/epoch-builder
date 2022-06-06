import unittest
from typing import Union
from pandas import Series, DataFrame
from modules.candlestick import Candlestick
from modules.technical_analysis import TechnicalAnalysis, ITechnicalAnalysis



# Start Candlestick
START_CANDLESTICK: Series = Candlestick.DF.iloc[879956].copy()

# Lookback DF
LOOKBACK_DF: DataFrame = Candlestick.get_lookback_df(300, START_CANDLESTICK["ot"])
FIRST_OT: int = LOOKBACK_DF.iloc[0]["ot"]
LAST_CT: int = LOOKBACK_DF.iloc[-1]["ct"]
ID: str = TechnicalAnalysis._get_id(FIRST_OT, LAST_CT)




## Test Class ##
class TechnicalAnalysisTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        TechnicalAnalysis._delete(ID)

    # After Tests
    def tearDown(self):
        TechnicalAnalysis._delete(ID)



    ## Technical Anlysis Flow ##



    # Can generate the RSI Indicator and verify it has been cached
    def testGenerateRSI(self):
        # Make sure the record does not exist yet
        ta: Union[ITechnicalAnalysis, None] = TechnicalAnalysis._read(ID)
        self.assertEqual(ta, None)

        # Generate the RSI
        ta = TechnicalAnalysis.get_technical_analysis(LOOKBACK_DF, include_rsi=True)

        # Make sure the dict has the correct RSI value
        rsi_calc: float = TechnicalAnalysis._calculate_rsi(LOOKBACK_DF["c"])
        self.assertDictEqual(ta, {"rsi": rsi_calc})

        # Make sure the TA has been stored in the db and that it is exactly the same
        # as the one retrieved previously
        cached_ta: Union[ITechnicalAnalysis, None] = TechnicalAnalysis._read(ID)
        self.assertDictEqual(ta, cached_ta)

        # If the RSI is generated again, it will come directly from the db
        ta = TechnicalAnalysis.get_technical_analysis(LOOKBACK_DF, include_rsi=True)
        self.assertDictEqual(ta, cached_ta)

        # Delete the TA Record and make sure it is gone
        TechnicalAnalysis._delete(ID)
        cached_ta = TechnicalAnalysis._read(ID)
        self.assertEqual(cached_ta, None)




    # Can generate the Aroon Indicator and verify it has been cached
    def testGenerateAroon(self):
        # Generate the Aroon
        ta = TechnicalAnalysis.get_technical_analysis(LOOKBACK_DF, include_aroon=True)

        # Make sure the dict has the correct Aroon value
        up, down = TechnicalAnalysis._calculate_aroon(LOOKBACK_DF["c"])
        self.assertDictEqual(ta, {"aroon_up": up, "aroon_down": down})

        # Make sure the TA has been stored in the db and that it is exactly the same
        # as the one retrieved previously
        cached_ta: Union[ITechnicalAnalysis, None] = TechnicalAnalysis._read(ID)
        self.assertDictEqual(ta, cached_ta)

        # If the TA is generated again, it will come directly from the db
        ta = TechnicalAnalysis.get_technical_analysis(LOOKBACK_DF, include_aroon=True)
        self.assertDictEqual(ta, cached_ta)




    # Can generate the RSI and then complete with the record with Aroon
    def testGenerateRSIAndCompleteWithAroon(self):
        # Generate the RSI
        ta = TechnicalAnalysis.get_technical_analysis(LOOKBACK_DF, include_rsi=True)
        rsi_calc: float = TechnicalAnalysis._calculate_rsi(LOOKBACK_DF["c"])
        self.assertDictEqual(ta, {"rsi": rsi_calc})
        ta_snap: Union[TechnicalAnalysis, None] = TechnicalAnalysis._read(ID)
        self.assertDictEqual(ta_snap, {"rsi": rsi_calc})

        # Now generate Aroon as well
        ta = TechnicalAnalysis.get_technical_analysis(LOOKBACK_DF, include_aroon=True)
        up, down = TechnicalAnalysis._calculate_aroon(LOOKBACK_DF["c"])
        self.assertDictEqual(ta, {"rsi": rsi_calc, "aroon_up": up, "aroon_down": down})
        ta_snap: Union[TechnicalAnalysis, None] = TechnicalAnalysis._read(ID)
        self.assertDictEqual(ta_snap, {"rsi": rsi_calc, "aroon_up": up, "aroon_down": down})

        # Can retrieve the full dict including both indicators
        ta = TechnicalAnalysis.get_technical_analysis(LOOKBACK_DF, include_rsi=True, include_aroon=True)
        self.assertDictEqual(ta, {"rsi": rsi_calc, "aroon_up": up, "aroon_down": down})



    # Can generate Aroon and then complete with the record with RSI
    def testGenerateAroonAndCompleteWithRSI(self):
        # Generate Aroon
        ta = TechnicalAnalysis.get_technical_analysis(LOOKBACK_DF, include_aroon=True)
        up, down = TechnicalAnalysis._calculate_aroon(LOOKBACK_DF["c"])
        self.assertDictEqual(ta, {"aroon_up": up, "aroon_down": down})
        ta_snap: Union[TechnicalAnalysis, None] = TechnicalAnalysis._read(ID)
        self.assertDictEqual(ta_snap, {"aroon_up": up, "aroon_down": down})

        # Generate the RSI
        ta = TechnicalAnalysis.get_technical_analysis(LOOKBACK_DF, include_rsi=True)
        rsi_calc: float = TechnicalAnalysis._calculate_rsi(LOOKBACK_DF["c"])
        self.assertDictEqual(ta, {"rsi": rsi_calc, "aroon_up": up, "aroon_down": down})
        ta_snap: Union[TechnicalAnalysis, None] = TechnicalAnalysis._read(ID)
        self.assertDictEqual(ta_snap, {"rsi": rsi_calc, "aroon_up": up, "aroon_down": down})

        # Can retrieve the full dict including both indicators
        ta = TechnicalAnalysis.get_technical_analysis(LOOKBACK_DF, include_rsi=True, include_aroon=True)
        self.assertDictEqual(ta, {"rsi": rsi_calc, "aroon_up": up, "aroon_down": down})




    # Can generate any number of indicators in one go
    def testGenerateSeveralInidicators(self):
        # Generate the indicators
        ta = TechnicalAnalysis.get_technical_analysis(LOOKBACK_DF, include_rsi=True, include_aroon=True)

        # Calculate the results
        rsi_calc: float = TechnicalAnalysis._calculate_rsi(LOOKBACK_DF["c"])
        up, down = TechnicalAnalysis._calculate_aroon(LOOKBACK_DF["c"])
        expected_ta: ITechnicalAnalysis = {"rsi": rsi_calc, "aroon_up": up, "aroon_down": down}

        # Validate the ta
        self.assertDictEqual(ta, expected_ta)

        # Pull the snap and make sure it is identical
        ta_snap: Union[TechnicalAnalysis, None] = TechnicalAnalysis._read(ID)
        self.assertDictEqual(expected_ta, ta_snap)










    ## Calculators ##



    # Can calculate the RSI Indicator for a given series
    def testCalculateRSI(self):
        # Calculate the RSI and make sure it is a valid float
        rsi: float = TechnicalAnalysis._calculate_rsi(LOOKBACK_DF["c"])
        self.assertIsInstance(rsi, float)
        self.assertLessEqual(rsi, 1)




    # Can calculate the Aroon Indicator for a given series
    def testCalculateAroon(self):
        # Calculate the Aroons and make sure it is a valid float
        aroon_up, aroon_down = TechnicalAnalysis._calculate_aroon(LOOKBACK_DF["c"])
        self.assertIsInstance(aroon_up, float)
        self.assertLessEqual(aroon_up, 1)
        self.assertIsInstance(aroon_down, float)
        self.assertLessEqual(aroon_down, 1)
        self.assertNotEqual(aroon_up, aroon_down)








    ## Misc Helpers ##




    # Can put together an ID from a prediction range
    def testGetID(self):
        self.assertEqual(
            TechnicalAnalysis._get_id(FIRST_OT, LAST_CT),
            f"{int(FIRST_OT)}_{int(LAST_CT)}"
        )
        self.assertEqual(
            TechnicalAnalysis._get_id(1502942400000, 1503007199999),
            "1502942400000_1503007199999"
        )



    # Can normalize a technical analysis value
    def testNormalizeValue(self):
        self.assertEqual(TechnicalAnalysis._normalize_value(14.285714), 0.142857)
        self.assertEqual(TechnicalAnalysis._normalize_value(42.857143), 0.428571)
        self.assertEqual(TechnicalAnalysis._normalize_value(71.428571), 0.714286)
        self.assertEqual(TechnicalAnalysis._normalize_value(100), 1)
        self.assertEqual(TechnicalAnalysis._normalize_value(0), 0)





# Test Execution
if __name__ == '__main__':
    unittest.main()
