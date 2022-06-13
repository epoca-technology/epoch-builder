import unittest
from modules.candlestick import Candlestick
from modules.model import IModel, ClassificationModel, IPrediction




# TRAINING DATA FILE
CONFIG: IModel = {
    "id": "C_UNIT_TEST",
    "classification_models": [{"classification_id": "C_UNIT_TEST","interpreter": { "min_probability": 0.51 }}]
}







## Test Class ##
class ClassificationModelTestCase(unittest.TestCase):
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
        model: ClassificationModel = ClassificationModel(CONFIG)

        # Init the test candlestick time
        time: int = Candlestick.DF.iloc[655858]["ot"]
        first_ot, last_ct = Candlestick.get_lookback_prediction_range(model.get_lookback(), time)

        # Perform a random prediction
        pred: IPrediction = model.predict(time, enable_cache=False)
        self.assertIsInstance(pred, dict)
        self.assertIsInstance(pred["r"], int)
        self.assertIsInstance(pred["t"], int)
        self.assertIsInstance(pred["md"], list)
        self.assertEqual(len(pred["md"]), 1)
        self.assertAlmostEqual(pred["md"][0]["up"]+pred["md"][0]["dp"], 1, delta=0.00001)

        # Make sure the prediction has not been cached
        self.assertEqual(model.cache.get(first_ot, last_ct), None)

        # Perform a the same prediction with cache enabled
        pred = model.predict(time, enable_cache=True)
        self.assertIsInstance(pred, dict)
        self.assertIsInstance(pred["r"], int)
        self.assertIsInstance(pred["t"], int)
        self.assertIsInstance(pred["md"], list)
        
        # Make sure the prediction has been cached
        self.assertDictEqual(model.cache.get(first_ot, last_ct), pred)








# Test Execution
if __name__ == '__main__':
    unittest.main()
