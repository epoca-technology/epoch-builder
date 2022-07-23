import unittest
from typing import List
from pandas import Series
from modules.types import IRegressionConfig, IRegressionTrainingCertificate
from modules.database.Database import Database
from modules.epoch.Epoch import Epoch
from modules.candlestick.Candlestick import Candlestick
from modules.regression.Regression import Regression



## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")





# TRAINING CERTIFICATE
MODEL_ID: str = "R_UNIT_TEST"
CERT: IRegressionTrainingCertificate = Epoch.FILE.get_active_model_certificate(MODEL_ID)











## Test Class ##
class RegressionTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass





    # Initialize an instance with valid data and validate the integrity
    def testInitialize(self):
        # Initialize the instance
        r: Regression = Regression(MODEL_ID)

        # Output the config and make sure everything matches
        config: IRegressionConfig = r.get_config()
        self.assertEqual(config["id"], MODEL_ID)
        self.assertEqual(config["description"], CERT["description"])
        self.assertEqual(config["autoregressive"], CERT["autoregressive"])
        self.assertEqual(config["lookback"], CERT["lookback"])
        self.assertEqual(config["predictions"], CERT["predictions"])
        self.assertIsInstance(config["summary"], dict)

        # Can generate a prediction for a random selected series
        close_prices: Series = Candlestick.get_lookback_df(r.lookback, Candlestick.DF.iloc[214554]["ot"], normalized=True)
        preds: List[float] = r.predict(close_prices["c"])

        # The predictions should match the model's properties
        self.assertEqual(len(preds), r.predictions)
        self.assertTrue(all(isinstance(x, float) for x in preds))
        self.assertTrue(all(list(map(lambda x: x >= 0 and x <= 1, preds))))








# Test Execution
if __name__ == '__main__':
    unittest.main()
