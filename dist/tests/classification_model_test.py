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

        # Perform a random prediction
        pred: IPrediction = model.predict(Candlestick.DF.iloc[655858]["ot"], enable_cache=False)
        self.assertIsInstance(pred, dict)
        self.assertIsInstance(pred["r"], int)
        self.assertIsInstance(pred["t"], int)
        self.assertIsInstance(pred["md"], list)
        self.assertEqual(len(pred["md"]), 1)
        self.assertEqual(pred["md"][0]["up"]+pred["md"][0]["dp"], 1)
        




# Test Execution
if __name__ == '__main__':
    unittest.main()
