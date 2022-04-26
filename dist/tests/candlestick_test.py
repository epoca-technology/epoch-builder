import unittest
from modules.utils import Utils
from modules.candlestick import Candlestick


# Initialize the candlesticks
LOOKBACK = 300



# Test Class
class CandlestickTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass





    # Can retrieve the Prediction Data for a series with and without TA
    def testGetPredictionData(self):
        # The first default candlestick must match the Prediction candlestick placed in the lookback index
        self.assertEqual(Candlestick.DF.iloc[0]['ot'], Candlestick.PREDICTION_DF.iloc[LOOKBACK]['ot'])

        # Initialize a default candlestick
        default_df = Candlestick.DF.iloc[288]

        # Retrieve the Prediction data starting at this point without TA
        series, rsi, short_ema, long_ema = Candlestick.get_data_to_predict_on(default_df['ot'], LOOKBACK, False, False)
        self.assertTrue(series.shape[0], LOOKBACK)
        self.assertEqual(rsi, None)
        self.assertEqual(short_ema, None)
        self.assertEqual(long_ema, None)

        # Retrieve the Prediction data with RSI
        series, rsi, short_ema, long_ema = Candlestick.get_data_to_predict_on(default_df['ot'], 270, True, False)
        self.assertTrue(series.shape[0], 270)
        self.assertIsInstance(rsi, float)
        self.assertEqual(short_ema, None)
        self.assertEqual(long_ema, None)

        # Retrieve the Prediction data with EMA
        series, rsi, short_ema, long_ema = Candlestick.get_data_to_predict_on(default_df['ot'], 200, False, True)
        self.assertTrue(series.shape[0], 200)
        self.assertEqual(rsi, None)
        self.assertIsInstance(short_ema, float)
        self.assertIsInstance(long_ema, float)

        # Retrieve the Prediction data with RSI & EMA
        series, rsi, short_ema, long_ema = Candlestick.get_data_to_predict_on(default_df['ot'], 50, True, True)
        self.assertTrue(series.shape[0], 50)
        self.assertIsInstance(rsi, float)
        self.assertIsInstance(short_ema, float)
        self.assertIsInstance(long_ema, float)




    # Can retrieve the range of a prediction (first_ot & last_ct)
    def testGetCurrentPredictionRange(self):
        # Initialize a default candlestick
        default_df = Candlestick.DF.iloc[2998]

        # Retrieve the current range
        first_ot, last_ct = Candlestick.get_current_prediction_range(LOOKBACK, default_df['ot'])

        # Make sure both values are lower than the current time
        self.assertLess(first_ot, default_df['ot'])
        self.assertLess(last_ct, default_df['ot'])
        #print("Current: ", Utils.from_milliseconds_to_date_string(default_df['ot']))
        #print("Last CT: ", Utils.from_milliseconds_to_date_string(last_ct))
        #print("First OT: ", Utils.from_milliseconds_to_date_string(first_ot))






    # Can retrieve the RSI for a series of close prices
    def testGetRSI(self):
        # Retrieve the RSI for the first 100 values
        first_rsi: float = Candlestick._get_rsi(Candlestick.PREDICTION_DF.iloc[0: 100]['c'])
        self.assertIsInstance(first_rsi, float)
        self.assertGreater(first_rsi, 0)

        # Retrieve the RSI for the last 200 values
        second_rsi: float = Candlestick._get_rsi(Candlestick.PREDICTION_DF.iloc[-200:]['c'])
        self.assertIsInstance(second_rsi, float)
        self.assertGreater(second_rsi, 0)

        # Both RSIs should be different numbers (There is the possibility that they could match though)
        self.assertFalse(first_rsi == second_rsi)






    # Can retrieve the EMAs for a series of close prices
    def testGetEMA(self):
        # Retrieve the EMAs for the first 50 values
        first_short_ema, first_long_ema = Candlestick._get_ema(Candlestick.PREDICTION_DF.iloc[0: 100]['c'])
        self.assertIsInstance(first_short_ema, float)
        self.assertIsInstance(first_long_ema, float)
        self.assertGreater(first_short_ema, 0)
        self.assertGreater(first_long_ema, 0)

        # Retrieve the EMAs for the first 300 values
        second_short_ema, second_long_ema = Candlestick._get_ema(Candlestick.PREDICTION_DF.iloc[0: 300]['c'])
        self.assertIsInstance(second_short_ema, float)
        self.assertIsInstance(second_long_ema, float)
        self.assertGreater(second_short_ema, 0)
        self.assertGreater(second_long_ema, 0)

        # Retrieve the EMAs for the last 150 values
        third_short_ema, third_long_ema = Candlestick._get_ema(Candlestick.PREDICTION_DF.iloc[-300:]['c'])
        self.assertIsInstance(third_short_ema, float)
        self.assertIsInstance(third_long_ema, float)
        self.assertGreater(third_short_ema, 0)
        self.assertGreater(third_long_ema, 0)





# Test Execution
if __name__ == '__main__':
    unittest.main()