from typing import List, Any
from unittest import TestCase, main
import time
from modules.database.Database import Database
from modules.utils.Utils import Utils


## ONLY RUN WHEN THE DATABASE TEST MODE IS ENABLED ##
if not Database.TEST_MODE:
    raise RuntimeError("Unit tests can only be performed when the Database is in test mode.")



# Test Class
class UtilsTestCase(TestCase):
    # Before Tests
    def setUp(self):
        pass

    # After Tests
    def tearDown(self):
        pass




    ## Number Helpers ##



    # Can alter a number by percentage
    def testAlterNumberByPercentage(self):
        self.assertEqual(Utils.alter_number_by_percentage(100, 50), 150)
        self.assertEqual(Utils.alter_number_by_percentage(100, -50), 50)
        self.assertEqual(Utils.alter_number_by_percentage(100, -90), 10)
        self.assertEqual(Utils.alter_number_by_percentage(100, 90), 190)
        self.assertEqual(Utils.alter_number_by_percentage(42555.63, -0.540892), 42325.45)
        self.assertEqual(Utils.alter_number_by_percentage(46855.49, -1.5182), 46144.13)
        self.assertEqual(Utils.alter_number_by_percentage(40222.54, 22.7342), 49366.81)




    # Can calculate the percentage change of a number
    def testGetPercentageChange(self):
        self.assertEqual(Utils.get_percentage_change(100, 150), 50)
        self.assertEqual(Utils.get_percentage_change(100, 50), -50)
        self.assertEqual(Utils.get_percentage_change(155.89, 199.63), 28.06)
        self.assertEqual(Utils.get_percentage_change(8559.63, 12455.87), 45.52)
        self.assertEqual(Utils.get_percentage_change(44785.12, 44521.33), -0.59)
        self.assertEqual(Utils.get_percentage_change(799415.88, 55121), -93.1)
        self.assertEqual(Utils.get_percentage_change(255446.69, 463551.11), 81.47)




    # Can calculate the percentage representation out of a total
    def testGetPercentageOutOfTotal(self):
        self.assertEqual(Utils.get_percentage_out_of_total(50, 100), 50)
        self.assertEqual(Utils.get_percentage_out_of_total(20, 200), 10)
        self.assertEqual(Utils.get_percentage_out_of_total(300, 1500), 20)
        self.assertEqual(Utils.get_percentage_out_of_total(300, 1000), 30)











    ## Time Helpers ##




    # Can retrieve the current time in milliseconds
    def testGetTime(self):
        current_time: float = time.time()
        self.assertAlmostEqual(Utils.get_time(), Utils.from_seconds_to_milliseconds(current_time), 10)





    # Can convert Milli Seconds into Seconds
    def testFromMilliSecondsToSeconds(self):
        self.assertEqual(Utils.from_milliseconds_to_seconds(1647469003036), 1647469003)
        self.assertEqual(Utils.from_milliseconds_to_seconds(1647469088126), 1647469088)
        self.assertEqual(Utils.from_milliseconds_to_seconds(1647469123651), 1647469123)
        self.assertEqual(Utils.from_milliseconds_to_seconds(1647469123651.5412), 1647469123)




    # Can convert Seconds into Milli Seconds
    def testFromSecondsToMilliSeconds(self):
        self.assertEqual(Utils.from_seconds_to_milliseconds(1647469003), 1647469003000)
        self.assertEqual(Utils.from_seconds_to_milliseconds(1647469088), 1647469088000)
        self.assertEqual(Utils.from_seconds_to_milliseconds(1647469123), 1647469123000)
        self.assertEqual(Utils.from_seconds_to_milliseconds(1647469123.1234), 1647469123123)


    # Can convert MilliSeconds into Minutes
    def testFromMilliSecondsToMinutes(self):
        self.assertEqual(Utils.from_milliseconds_to_minutes(50000), 1)
        self.assertEqual(Utils.from_milliseconds_to_minutes(60000), 1)
        self.assertEqual(Utils.from_milliseconds_to_minutes(120000), 2)
        self.assertEqual(Utils.from_milliseconds_to_minutes(115000), 2)


    # Can convert a Date String into a ms timestamp
    def testFromDateStringToMilliSeconds(self):
        self.assertAlmostEqual(Utils.from_date_string_to_milliseconds('1/1/2022'), 1641009600000, delta=14400000)
        self.assertAlmostEqual(Utils.from_date_string_to_milliseconds('25/04/2019'), 1556164800000, delta=14400000)
        self.assertAlmostEqual(Utils.from_date_string_to_milliseconds('18/09/2017'), 1505707200000, delta=14400000)
        self.assertAlmostEqual(Utils.from_date_string_to_milliseconds('3/12/2017'), 1512273600000, delta=14400000)
        self.assertAlmostEqual(Utils.from_date_string_to_milliseconds('19/07/1990'), 648360000000, delta=14400000)
        self.assertAlmostEqual(Utils.from_date_string_to_milliseconds('7/5/1992'), 705211200000, delta=14400000)




    # Can add minutes to a timestamp
    def testAddMinutes(self):
        self.assertEqual(Utils.add_minutes(1649776140000, 1), 1649776200000)
        self.assertEqual(Utils.add_minutes(1502942520000, 1), 1502942580000)
        self.assertEqual(Utils.add_minutes(1502942880000, 2), 1502943000000)
        self.assertEqual(Utils.add_minutes(1502944020000, 4), 1502944260000)
        self.assertEqual(Utils.add_minutes(1502945340000, 30), 1502947140000)






    ## UUID Helpers ##



    # Can generate valid UUIDs
    def testGenerateUUID(self):
        # Generate a list of uuids
        uuids: List[str] = [
            Utils.generate_uuid4(),
            Utils.generate_uuid4(),
            Utils.generate_uuid4(),
            Utils.generate_uuid4(),
            Utils.generate_uuid4(),
        ]
        
        # Iterate over them and make sure they are valid
        for id in uuids:
            self.assertTrue(Utils.is_uuid4(id) and isinstance(id, str) and len(id) > 0)




    # Can identify invalid UUIDs
    def testValidateUUIDs(self):
        # Generate a list of invalid uuids
        uuids: List[Any] = [
            'some-random-string',
            'd9428888-122b-11e1-b85c-61cd3cbb3210', # v1 uuid
            123.45,
            True,
            {'foo': 'bar'},
            (123456, 321654),
        ]
        
        # Iterate over them and make sure they are valid
        for id in uuids:
            self.assertFalse(Utils.is_uuid4(id))











# Test Execution
if __name__ == '__main__':
    main()