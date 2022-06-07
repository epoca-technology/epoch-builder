import unittest
from typing import List
from json import load
from modules.keras_models import KERAS_PATH
from modules.classification import ITrainingDataFile, Classification




# TRAINING DATA FILE
FILE_ID: str = "d5f0f65e-66e7-4f6e-ab9d-b6dd6ad31267"
TRAINING_DATA: ITrainingDataFile = load(open(f"{KERAS_PATH['classification_training_data']}/{FILE_ID}.json"))
MODEL_ID: str = "C_UNIT_TEST"











## Test Class ##
class ClassificationTestCase(unittest.TestCase):
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
        c: Classification = Classification(MODEL_ID)

        # Validate model properties
        self.assertEqual(c.id, MODEL_ID)
        self.assertEqual(c.training_data_id, TRAINING_DATA["id"])
        self.assertEqual(c.include_rsi, TRAINING_DATA["include_rsi"])
        self.assertEqual(c.include_aroon, TRAINING_DATA["include_aroon"])
        self.assertEqual(c.features_num, TRAINING_DATA["features_num"])
        for index, model in enumerate(c.regressions):
            self.assertDictEqual(model, TRAINING_DATA["models"][index])

        # Can generate a prediction with some dummy features
        pred: List[float] = c.predict(features=[0]*c.features_num)
        self.assertEqual(len(pred), 2)
        self.assertEqual(pred[0]+pred[1], 1)

        # Can build the configuration dict
        self.assertIsInstance(c.get_config(), dict)




# Test Execution
if __name__ == '__main__':
    unittest.main()
