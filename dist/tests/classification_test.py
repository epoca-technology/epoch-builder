import unittest
from typing import List
from modules.types import ITrainingDataFile
from modules.database.Database import Database
from modules.epoch.Epoch import Epoch
from modules.classification.Classification import Classification



## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")

    



# TRAINING DATA FILE
TRAINING_DATA: ITrainingDataFile = Epoch.FILE.get_classification_training_data(Epoch.UT_CLASS_TRAINING_DATA_ID)
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
        self.assertEqual(c.include_stoch, TRAINING_DATA["include_stoch"])
        self.assertEqual(c.include_aroon, TRAINING_DATA["include_aroon"])
        self.assertEqual(c.include_stc, TRAINING_DATA["include_stc"])
        self.assertEqual(c.include_mfi, TRAINING_DATA["include_mfi"])
        self.assertEqual(c.features_num, TRAINING_DATA["features_num"])
        for index, model in enumerate(c.regressions):
            self.assertDictEqual(model, TRAINING_DATA["models"][index])

        # Can generate a prediction with some dummy features
        pred: List[float] = c.predict(features=[0]*c.features_num)
        self.assertEqual(len(pred), 2)
        self.assertAlmostEqual(pred[0]+pred[1], 1, delta=0.00001)

        # Can build the configuration dict
        self.assertIsInstance(c.get_config(), dict)




# Test Execution
if __name__ == '__main__':
    unittest.main()
