import unittest
from pandas import Series
from modules.model import ArimaModel
from modules.candlestick import Candlestick
from modules.utils import Utils



# Initialize the candlesticks
LOOKBACK = ArimaModel.DEFAULT_LOOKBACK



# Test Class
class CandlestickTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass



    # Can retrieve a lookback df
    def testGetLookbackDF(self):
        # Initialize a default candlestick
        default_df = Candlestick.DF.iloc[185255]

        # Retrieve the lookback df
        df = Candlestick.get_lookback_df(300, default_df['ot'])
        self.assertEqual(df.shape[0], 300)
        self.assertEqual(df.shape[1], len(Candlestick.PREDICTION_CANDLESTICK_CONFIG['columns']))

        # The values should not be normalized
        norm_df = df[(df['o'] <= 1) | (df['h'] <= 1) | (df['l'] <= 1) | (df['c'] <= 1)]
        self.assertEqual(norm_df.shape[0], 0)




    # Can retrieve a normalized lookback df
    def testGetNormalizedLookbackDF(self):
        # Initialize a default candlestick
        default_df = Candlestick.DF.iloc[185255]

        # Retrieve the lookback df
        df = Candlestick.get_lookback_df(300, default_df['ot'], normalized=True)
        self.assertEqual(df.shape[0], 300)
        self.assertEqual(df.shape[1], 4) # o, h, l, c
        
        # The values should not be normalized
        norm_df = df[(df['o'] <= 1) | (df['h'] <= 1) | (df['l'] <= 1) | (df['c'] <= 1)]
        self.assertEqual(norm_df.shape[0], 300)




    # Can retrieve the close prices for any lookback
    def testGetLookbackClosePrices(self):
        # The first default candlestick must match the Prediction candlestick placed in the lookback index
        self.assertEqual(Candlestick.DF.iloc[0]['ot'], Candlestick.PREDICTION_DF.iloc[LOOKBACK]['ot'])

        # Initialize a default candlestick
        default_df = Candlestick.DF.iloc[288]

        # Retrieve the Prediction data starting at this point without TA
        series: Series = Candlestick.get_lookback_close_prices(LOOKBACK, default_df['ot'])
        self.assertEqual(series.shape[0], LOOKBACK)




    # Can retrieve the lookback prediction range (first_ot & last_ct)
    def testGetLookbackPredictionRange(self):
        # Initialize a default candlestick
        default_df = Candlestick.DF.iloc[2998]

        # Retrieve the current range
        first_ot, last_ct = Candlestick.get_lookback_prediction_range(LOOKBACK, default_df['ot'])

        # Make sure both values are lower than the current time
        self.assertLess(first_ot, default_df['ot'])
        self.assertLess(last_ct, default_df['ot'])
        #print("Current: ", Utils.from_milliseconds_to_date_string(default_df['ot']))
        #print("Last CT: ", Utils.from_milliseconds_to_date_string(last_ct))
        #print("First OT: ", Utils.from_milliseconds_to_date_string(first_ot))










# Test Execution
if __name__ == '__main__':
    unittest.main()