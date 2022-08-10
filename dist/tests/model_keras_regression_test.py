import unittest
from modules._types import IModel, IPrediction
from modules.database.Database import Database
from modules.candlestick.Candlestick import Candlestick
from modules.model.KerasRegressionModel import KerasRegressionModel



## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")





# TEST DATA
CONFIG: IModel = {
    "id": "KR_UNIT_TEST",
    "keras_regressions": [ { "regression_id": "KR_UNIT_TEST" } ]
}







## Test Class ##
class KerasRegressionModelTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass






    # Can initialize a model instance and make use of its functionalities
    def testInitialize(self):
        # Initialize the instance
        model: KerasRegressionModel = KerasRegressionModel(CONFIG, enable_cache=False)

        # Make sure the instance is recognized
        self.assertIsInstance(model, KerasRegressionModel)
        self.assertEqual(type(model).__name__, "KerasRegressionModel")

        # Init the test candlestick time
        time: int = Candlestick.DF.iloc[99551]["ot"]
        first_ot, last_ct = Candlestick.get_lookback_prediction_range(model.get_lookback(), time)


        ## Predictions ## 

        # Make sure the prediction does not exist
        self.assertEqual(model.prediction_cache.get(first_ot, last_ct), None)

        # Perform a random prediction
        pred: IPrediction = model.predict(time)
        self.assertIsInstance(pred, dict)
        self.assertIsInstance(pred["r"], int)
        self.assertIsInstance(pred["t"], int)
        self.assertIsInstance(pred["md"], list)
        self.assertEqual(len(pred["md"]), 1)
        self.assertIsInstance(pred['md'][0]['pl'], list)
        self.assertEqual(len(pred['md'][0]['pl']), model.regression.predictions + 1)
        self.assertTrue(all(isinstance(x, float) for x in pred['md'][0]['pl']))
        self.assertTrue(all(list(map(lambda x: x >= 0 and x <= 1, pred['md'][0]['pl']))))

        # Make sure the prediction has not been cached
        self.assertEqual(model.prediction_cache.get(first_ot, last_ct), None)


        ## Features ##

        # Make sure the feature does not exist
        self.assertEqual(model.feature_cache.get(first_ot, last_ct), None)

        # Generate a random feature
        feature: float = model.feature(time)
        self.assertIsInstance(feature, (int, float))
        self.assertTrue(feature >= -1 and feature <= 1)
        
        # Make sure the feature has not been cached
        self.assertEqual(model.feature_cache.get(first_ot, last_ct), None)


        ## Model Config

        # Retrieve the summary and make sure it is a dict
        summary: IModel = model.get_model()
        self.assertIsInstance(summary, dict)





    # Can cache predictions and delete them afterwards
    def testPredictionCache(self):
        # Initialize the instance
        model: KerasRegressionModel = KerasRegressionModel(CONFIG, enable_cache=True)

        # Init the test candlestick time
        time: int = Candlestick.DF.iloc[61551]["ot"]
        first_ot, last_ct = Candlestick.get_lookback_prediction_range(model.get_lookback(), time)

        # Make sure the prediction does not exist
        self.assertEqual(model.prediction_cache.get(first_ot, last_ct), None)

        # Perform a random prediction
        pred: IPrediction = model.predict(time)
        self.assertIsInstance(pred, dict)
        self.assertIsInstance(pred["r"], int)
        self.assertIsInstance(pred["t"], int)
        self.assertIsInstance(pred["md"], list)
        self.assertEqual(len(pred["md"]), 1)
        
        # Make sure the prediction has been cached
        self.assertDictEqual(model.prediction_cache.get(first_ot, last_ct), pred)

        # Clean up the prediction
        model.prediction_cache.delete(first_ot, last_ct)
        cached_pred = model.prediction_cache.get(first_ot, last_ct)
        self.assertTrue(cached_pred == None)







    # Can cache features and delete them afterwards
    def testFeatureCache(self):
        # Initialize the instance
        model: KerasRegressionModel = KerasRegressionModel(CONFIG, enable_cache=True)

        # Init the test candlestick time
        time: int = Candlestick.DF.iloc[76500]["ot"]
        first_ot, last_ct = Candlestick.get_lookback_prediction_range(model.get_lookback(), time)

        # Make sure the feature does not exist
        self.assertEqual(model.feature_cache.get(first_ot, last_ct), None)

        # Generate a random feature
        feature: float = model.feature(time)
        self.assertIsInstance(feature, (int, float))
        self.assertTrue(feature >= -1 and feature <= 1)
        
        # Make sure the feature has been cached
        self.assertEqual(model.feature_cache.get(first_ot, last_ct), feature)

        # Clean up the prediction
        model.feature_cache.delete(first_ot, last_ct)
        cached_feature = model.feature_cache.get(first_ot, last_ct)
        self.assertTrue(cached_feature == None)







# Test Execution
if __name__ == '__main__':
    unittest.main()
