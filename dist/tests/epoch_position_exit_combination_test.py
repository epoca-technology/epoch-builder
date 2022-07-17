import unittest
from modules.database.Database import Database
from modules.epoch.PositionExitCombination import PositionExitCombination






## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")





# TEST DATA






## Test Class ##
class PositionExitCombinationTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass





    # Can interact over each combination id and retrieve all the required data
    def testDataExtraction(self):
        for key, val in PositionExitCombination.DB.items():
            # Can extract the ID from the combination
            self.assertEqual(PositionExitCombination.get_id(val['take_profit'], val['stop_loss']), key)

            # Can extract the path from the combination
            self.assertEqual(PositionExitCombination.get_path(val['take_profit'], val['stop_loss']), val['path'])



    # Cannot extract data with an invalid combination
    def testCannotExtractDataWithInvalidCombination(self):
        with self.assertRaises(ValueError):
            PositionExitCombination.get_id(1.75, 1)
        with self.assertRaises(ValueError):
            PositionExitCombination.get_id(2, 3.9154)
        with self.assertRaises(ValueError):
            PositionExitCombination.get_path(1.75, 1)
        with self.assertRaises(ValueError):
            PositionExitCombination.get_path(2, 3.9154)





# Test Execution
if __name__ == '__main__':
    unittest.main()
