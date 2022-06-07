import unittest
from typing import List
from json import load
from copy import deepcopy
from modules.keras_models import KERAS_PATH, IKerasModelConfig
from modules.classification import IClassificationTrainingConfig, ITrainingDataFile, ClassificationTraining




# TRAINING DATA FILE
FILE_ID: str = "d5f0f65e-66e7-4f6e-ab9d-b6dd6ad31267"
TRAINING_DATA: ITrainingDataFile = load(open(f"{KERAS_PATH['classification_training_data']}/{FILE_ID}.json"))


# Default Configuration
CONFIG: IClassificationTrainingConfig = {
    "id": "C_UNIT_TEST",
    "description": "Executed from Unit Tests",
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "metric": "binary_accuracy",
    "keras_model": {
        "name": "C_UNIT_TEST",
        "units": [64],
        "activations": ["relu"]
    }
}









## Test Class ##
class ClassificationTrainingTestCase(unittest.TestCase):
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
        config: IClassificationTrainingConfig = deepcopy(CONFIG)

        # Initialize the instance
        training: ClassificationTraining = ClassificationTraining(
            training_data_file=TRAINING_DATA,
            config=config,
            max_evaluations=1000,
            test_mode=True
        )

        # Validate properties
        self.assertEqual(training.id, config["id"])
        self.assertEqual(training.description, config["description"])
        self.assertEqual(len(training.models), len(TRAINING_DATA["models"]))
        self.assertEqual(len(training.regressions), len(TRAINING_DATA["models"]))
        self.assertEqual(training.learning_rate, config["learning_rate"])
        self.assertEqual(training.optimizer._name.lower(), config["optimizer"])
        self.assertEqual(training.loss.name, config["loss"])
        self.assertEqual(training.metric.name, config["metric"])
        self.assertEqual(training.training_data_summary["regression_selection_id"], TRAINING_DATA["regression_selection_id"])
        self.assertEqual(training.training_data_summary["id"], TRAINING_DATA["id"])
        self.assertEqual(training.training_data_summary["description"], TRAINING_DATA["description"])
        self.assertEqual(training.training_data_summary["start"], TRAINING_DATA["start"])
        self.assertEqual(training.training_data_summary["end"], TRAINING_DATA["end"])
        self.assertEqual(training.training_data_summary["steps"], TRAINING_DATA["steps"])
        self.assertEqual(training.training_data_summary["up_percent_change"], TRAINING_DATA["up_percent_change"])
        self.assertEqual(training.training_data_summary["down_percent_change"], TRAINING_DATA["down_percent_change"])
        self.assertEqual(training.training_data_summary["include_rsi"], TRAINING_DATA["include_rsi"])
        self.assertEqual(training.training_data_summary["include_aroon"], TRAINING_DATA["include_aroon"])
        self.assertEqual(training.training_data_summary["features_num"], TRAINING_DATA["features_num"])
        expected_keras_model: IKerasModelConfig = deepcopy(config["keras_model"])
        expected_keras_model["features_num"] = TRAINING_DATA["features_num"]
        self.assertDictEqual(training.keras_model, expected_keras_model)






# Test Execution
if __name__ == '__main__':
    unittest.main()
