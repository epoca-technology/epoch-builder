import unittest
from modules.database import Database


# Test Class
class DatabaseTestCase(unittest.TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass


    # Can return table names based on the current test mode
    def testTableName(self):
        # The initial state should be true since this is an unit test
        self.assertTrue(Database.TEST_MODE)

        # Since testmode is enabled, the table should be preffixed
        self.assertEqual(Database.tn("arima_predictions"), "test_arima_predictions")

        # The testmode can be disabled during runtime
        Database.TEST_MODE = False
        self.assertFalse(Database.TEST_MODE)
        self.assertEqual(Database.tn("arima_predictions"), "arima_predictions")

        # Renable testmode and make sure the tables are returned correctly
        Database.TEST_MODE = True
        self.assertEqual(Database.tn("arima_predictions"), "test_arima_predictions")








# Test Execution
if __name__ == '__main__':
    unittest.main()