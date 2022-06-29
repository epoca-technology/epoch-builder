import unittest
from modules.types import IModel, IPrediction
from modules.database.Database import Database
from modules.candlestick.Candlestick import Candlestick
from modules.model.RegressionModel import RegressionModel



## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")





# TRAINING DATA FILE
CONFIG: IModel = {
    "id": "R_UNIT_TEST",
    "regression_models": [{"regression_id": "R_UNIT_TEST", "interpreter": { "long": 1, "short": 1 }}]
}







## Test Class ##
class RegressionModelTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass



    ## Initialization ##


    # Initialize an instance with valid data and validate the integrity
    def testInitialize(self):
        # Initialize the instance
        model: RegressionModel = RegressionModel(CONFIG)

        # Make sure the instance is recognized
        self.assertIsInstance(model, RegressionModel)
        self.assertEqual(type(model).__name__, "RegressionModel")

        # Init the test candlestick time
        time: int = Candlestick.DF.iloc[799551]["ot"]
        first_ot, last_ct = Candlestick.get_lookback_prediction_range(model.get_lookback(), time)

        # Make sure the prediction does not exist
        self.assertEqual(model.cache.get(first_ot, last_ct), None)

        # Perform a random prediction
        pred: IPrediction = model.predict(time, enable_cache=False)
        self.assertIsInstance(pred, dict)
        self.assertIsInstance(pred["r"], int)
        self.assertIsInstance(pred["t"], int)
        self.assertIsInstance(pred["md"], list)
        self.assertEqual(len(pred["md"]), 1)
        self.assertIsInstance(pred['md'][0]['npl'], list)
        self.assertEqual(len(pred['md'][0]['npl']), model.regression.predictions)
        self.assertTrue(all(isinstance(x, float) for x in pred['md'][0]['npl']))
        self.assertTrue(all(list(map(lambda x: x >= 0 and x <= 1, pred['md'][0]['npl']))))

        # Make sure the prediction has not been cached
        self.assertEqual(model.cache.get(first_ot, last_ct), None)

        # Perform a the same prediction with cache enabled
        second_pred = model.predict(time, enable_cache=True)
        self.assertIsInstance(second_pred, dict)
        self.assertIsInstance(second_pred["r"], int)
        self.assertIsInstance(second_pred["t"], int)
        self.assertIsInstance(second_pred["md"], list)
        self.assertEqual(len(second_pred["md"]), 1)
        self.assertDictEqual(second_pred["md"][0], {"d": pred['md'][0]["d"]})
        
        # Make sure the prediction has been cached
        self.assertDictEqual(model.cache.get(first_ot, last_ct), second_pred)

        # Clean up the prediction
        model.cache.delete(first_ot, last_ct)
        cached_pred = model.cache.get(first_ot, last_ct)
        self.assertTrue(cached_pred == None)

        # Retrieve the summary and make sure it is a dict
        summary: IModel = model.get_model()
        self.assertIsInstance(summary, dict)








# Test Execution
if __name__ == '__main__':
    unittest.main()
