from typing import List, TypedDict, Union
import unittest
from modules.types import IModelType, ITrainableModelType, IModelIDPrefix
from modules.database.Database import Database
from modules.model.ModelType import validate_id, get_model_type, get_trainable_model_type, get_prefix






## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")





# TEST DATA


# Valid Test Items
class ITestItem(TypedDict):
    id: str
    model_type: IModelType
    trainable_model_type: Union[ITrainableModelType, None]
    prefix: IModelIDPrefix
ITEMS: List[ITestItem] = [
    {
        "id": "A134",
        "model_type": "ArimaModel",
        "trainable_model_type": None,
        "prefix": "A"
    },
    {
        "id": "R_LSTM_S3_02940598-84ca-4870-a00a-78debde80db2",
        "model_type": "RegressionModel",
        "trainable_model_type": "keras_regression",
        "prefix": "R_"
    },
    {
        "id": "C_DNN_S2_86a440e8-6a9c-4979-b866-8b4157ecbe9c",
        "model_type": "ClassificationModel",
        "trainable_model_type": "keras_classification",
        "prefix": "C_"
    },
    {
        "id": "XGBR_SOME_MODEL_IDENTIFIER_02940598-84ca-4870-a00a-78debde80db2",
        "model_type": "XGBRegressionModel",
        "trainable_model_type": "xgb_regression",
        "prefix": "XGBR_"
    },
    {
        "id": "XGBC_SOME_MODEL_IDENTIFIER_02940598-84ca-4870-a00a-78debde80db2",
        "model_type": "XGBClassificationModel",
        "trainable_model_type": "xgb_classification",
        "prefix": "XGBC_"
    },
    {
        "id": "CON_3_5REG",
        "model_type": "ConsensusModel",
        "trainable_model_type": None,
        "prefix": "CON_"
    }
]



# Invalid IDS & Prefixes
INVALID_LIST: List[str] = [
    "ZA134", "RR_LSTM_S3_02940598-84ca-4870-a00a-78debde80db2", "XR_LSTM_S3_02940598-84ca-4870-a00a-78debde80db2",
    "CC_DNN_S2_86a440e8-6a9c-4979-b866-8b4157ecbe9c", "XC_DNN_S2_86a440e8-6a9c-4979-b866-8b4157ecbe9c",
    "XXGBR_SOME_MODEL_IDENTIFIER_02", "CXGBR_SOME_MODEL_IDENTIFIER_02", "XGBCSOME_MODEL_IDENTIFIER_02",
    "CONS_3_5REG", "CON3_5REG", "Z", "5", "R", "RR", "CR", "C", "XGB", "_", "XGBC-", "R-", "C-", "78", "CUN_", "CCON_", "ZA"
]



## Test Class ##
class ModelTypeTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass






    # Can identify model ids and prefixes as well as generating all required data
    def testModelTypeFlow(self):
        #Iterate over each test item
        for item in ITEMS:
            # Validate the ID
            validate_id(item["model_type"], item["id"])

            # Extract the model prefix from the id or the prefix itself
            self.assertEqual(get_prefix(item["id"]), item["prefix"])
            self.assertEqual(get_prefix(item["prefix"]), item["prefix"])

            # Extract the model type from the model type
            self.assertEqual(get_model_type(item["id"]), item["model_type"])
            self.assertEqual(get_model_type(item["prefix"]), item["model_type"])

            # Can extract the trainable model type if applies
            if isinstance(item["trainable_model_type"], str):
                self.assertEqual(get_trainable_model_type(item["id"]), item["trainable_model_type"])
                self.assertEqual(get_trainable_model_type(item["prefix"]), item["trainable_model_type"])
            
            # Otherwise, it should raise a value error
            else:
                with self.assertRaises(ValueError):
                    get_trainable_model_type(item["id"])
                with self.assertRaises(ValueError):
                    get_trainable_model_type(item["prefix"])





    # Cannot extract data with invalid prefixes/ids
    def testDataExtractionWithInvalidData(self):
        for id_or_prefix in INVALID_LIST:
            # The validation should raise an error
            with self.assertRaises(ValueError):
                validate_id("RegressionModel", id_or_prefix)

            # The prefix retriever should raise an error
            with self.assertRaises(ValueError):
                get_prefix(id_or_prefix)

            # The model type retriever should raise an error
            with self.assertRaises(ValueError):
                get_model_type(id_or_prefix)

            # The trainable_model_type retriever should raise an error
            with self.assertRaises(ValueError):
                get_trainable_model_type(id_or_prefix)






# Test Execution
if __name__ == '__main__':
    unittest.main()
