from unittest import TestCase, main
from copy import deepcopy
from modules._types import IKerasModelConfig, IKerasClassificationTrainingConfig, ITrainingDataFile,\
    IClassificationDatasets
from modules.database.Database import Database
from modules.epoch.Epoch import Epoch
from modules.classification_training_data.Datasets import make_datasets
from modules.keras_classification.KerasClassificationTraining import KerasClassificationTraining




## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")

    


# TRAINING DATA FILE
TRAINING_DATA_FILE: ITrainingDataFile = Epoch.FILE.get_classification_training_data(Epoch.CLASSIFICATION_TRAINING_DATA_ID_UT)


# Default Configuration
CONFIG: IKerasClassificationTrainingConfig = {
    "id": "KC_UNIT_TEST",
    "description": "This is the official KerasClassificationModel for Unit Tests.",
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "metric": "binary_accuracy",
    "keras_model": {
        "name": "KC_DNN_S3",
        "units": [256, 128, 64],
        "activations": ["relu", "relu", "relu"]
    }
}









## Test Class ##
class ClassificationTrainingTestCase(TestCase):
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
        config: IKerasClassificationTrainingConfig = deepcopy(CONFIG)

        # Build the datasets
        datasets: IClassificationDatasets = make_datasets(TRAINING_DATA_FILE["training_data"], Epoch.TRAIN_SPLIT)

        # Initialize the instance
        training: KerasClassificationTraining = KerasClassificationTraining(
            training_data_file=TRAINING_DATA_FILE,
            config=config,
            datasets=datasets,
            test_mode=True
        )

        # Validate properties
        self.assertEqual(training.id, config["id"])
        self.assertEqual(training.description, config["description"])
        self.assertEqual(len(training.regressions), len(TRAINING_DATA_FILE["regressions"]))
        self.assertEqual(training.learning_rate, config["learning_rate"])
        self.assertEqual(training.optimizer._name.lower(), config["optimizer"])
        self.assertEqual(training.loss.name, config["loss"])
        self.assertEqual(training.metric.name, config["metric"])
        self.assertEqual(training.training_data_summary["regression_selection_id"], TRAINING_DATA_FILE["regression_selection_id"])
        self.assertEqual(training.training_data_summary["id"], TRAINING_DATA_FILE["id"])
        self.assertEqual(training.training_data_summary["description"], TRAINING_DATA_FILE["description"])
        self.assertEqual(training.training_data_summary["start"], TRAINING_DATA_FILE["start"])
        self.assertEqual(training.training_data_summary["end"], TRAINING_DATA_FILE["end"])
        self.assertEqual(training.training_data_summary["steps"], TRAINING_DATA_FILE["steps"])
        self.assertEqual(training.training_data_summary["price_change_requirement"], TRAINING_DATA_FILE["price_change_requirement"])
        self.assertEqual(training.training_data_summary["include_rsi"], TRAINING_DATA_FILE["include_rsi"])
        self.assertEqual(training.training_data_summary["include_aroon"], TRAINING_DATA_FILE["include_aroon"])
        self.assertEqual(training.training_data_summary["features_num"], TRAINING_DATA_FILE["features_num"])
        expected_keras_model: IKerasModelConfig = deepcopy(config["keras_model"])
        expected_keras_model["features_num"] = TRAINING_DATA_FILE["features_num"]
        self.assertDictEqual(training.keras_model, expected_keras_model)






# Test Execution
if __name__ == '__main__':
    main()
