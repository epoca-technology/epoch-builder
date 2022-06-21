import unittest
from typing import List
from copy import deepcopy
from numpy import ndarray
from modules.types import IKerasModelConfig, IRegressionTrainingConfig
from modules.database.Database import Database
from modules.keras_models.KerasPath import KERAS_PATH
from modules.regression.RegressionTraining import RegressionTraining



## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")





# Default Configuration
CONFIG: IRegressionTrainingConfig = {
    "id": "R_UNIT_TEST",
    "description": "Regression Training Unit Test.",
    "autoregressive": False,
    "lookback": 100,
    "predictions": 30,
    "optimizer": "adam",
    "loss": "mean_squared_error",
    "keras_model": {
        "name": "R_UNIT_TEST",
        "units": [64],
        "activations": ["relu"]
    }
}










## Test Class ##
class RegressionTrainingTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass



    ## Initialization ##


    # Initialize an instance with valid data and validate the integrity
    def testInitialize(self):
        # Init the config
        config: IRegressionTrainingConfig = deepcopy(CONFIG)

        # Initialize the instance
        training: RegressionTraining = RegressionTraining(
            config=config,
            hyperparams_mode=False,
            test_mode=True
        )

        # Validate properties
        self.assertEqual(training.test_mode, True)
        self.assertEqual(training.hyperparams_mode, False)
        self.assertDictEqual(training.training_config, RegressionTraining.SHORTLISTED_TRAINING_CONFIG)
        self.assertEqual(training.id, config["id"])
        self.assertEqual(training.model_path, f"{KERAS_PATH['models']}/{config['id']}")
        self.assertEqual(training.description, config["description"])
        self.assertEqual(training.autoregressive, config["autoregressive"])
        self.assertEqual(training.lookback, config["lookback"])
        self.assertEqual(training.predictions, config["predictions"])
        self.assertEqual(training.optimizer._name.lower(), config["optimizer"])
        self.assertEqual(training.loss.name, config["loss"])
        expected_keras_model: IKerasModelConfig = deepcopy(config["keras_model"])
        expected_keras_model["autoregressive"] = config["autoregressive"]
        expected_keras_model["lookback"] = config["lookback"]
        expected_keras_model["predictions"] = config["predictions"]
        self.assertDictEqual(training.keras_model, expected_keras_model)
        self.assertIsInstance(training.train_x, ndarray)
        self.assertIsInstance(training.train_y, ndarray)
        self.assertIsInstance(training.test_x, ndarray)
        self.assertIsInstance(training.test_y, ndarray)
        self.assertTrue(len(training.train_x) > 0)
        self.assertTrue(len(training.train_y) > 0)
        self.assertTrue(len(training.test_x) > 0)
        self.assertTrue(len(training.test_y) > 0)






# Test Execution
if __name__ == '__main__':
    unittest.main()