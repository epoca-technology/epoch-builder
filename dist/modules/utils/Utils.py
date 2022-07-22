from typing import Union, List
from time import time
from datetime import datetime
from uuid import UUID, uuid4





class Utils:
    """Utils Class

    This singleton provides a series of functionalities that simplify development and 
    provide consistency among modules.
    """










    ## Number Helpers ##






    @staticmethod
    def alter_number_by_percentage(value: float, percent: float) -> float:
        """Alters a number based on given percentage. For example, if value is 100 and percent
        is 50 it will return 150. On the other hand, if value is 100 and percent is -50 it will
        return 50.

        Args:
            value: float
                The number that will be altered.
            percent: float 
                The percentage that will be applied to the value.

        Returns:
            float
        """
        # Init the new value
        new_value = value

        # Handle an increase
        if percent > 0:
            new_value = ((percent / 100) + 1) * value

        # Handle a decrease
        elif percent < 0:
            new_value = -(((percent * -1) / 100) - 1) * value
        
        # Return the altered number
        return round(new_value, 2)







    @staticmethod
    def get_percentage_change(old_value: float, new_value: float) -> float:
        """Calculates the percentage change a value has experienced.

        Args:
            old_value: float
                The original number to calculate the % change for.
            new_value: float
                The new state of the original number.

        Returns:
            float
        """
        # Init the change
        change: float = 0.0

        # Handle an increase
        if new_value > old_value:
            increase: float = new_value - old_value
            change = (increase / old_value) * 100

        # Handle a decrease
        elif old_value > new_value:
            decrease: float = old_value - new_value
            change = -((decrease / old_value) * 100)

        # Return the change
        return round(change if change >=-100 else -100, 2)









    @staticmethod
    def get_percentage_out_of_total(value: float, total: float) -> float:
        """Calculates the percentage representation of a value based on the total.
        For example, if value is 20 and total is 200 the result is 10%

        Args:
            value: float
                The value to calculate the % representation for.
            total: float
                The total number of existing elements

        Returns:
            float
        """
        return round((value * 100) / total, 2)














    ## Time Helpers ##







    @staticmethod
    def get_time() -> int:
        """Retrieves the current time in milliseconds. Equivalent of Javascript's Date.now().

        Args:
            None

        Returns:
            int
        """
        return Utils.from_seconds_to_milliseconds(time())








    @staticmethod
    def from_milliseconds_to_seconds(milliseconds: Union[int, float]) -> int:
        """Converts a milli seconds value into seconds. Notice that it will round
        decimals downwards in case of a float.

        Args:
            milliseconds: Union[int, float]
                The timestamp in milliseconds to be converted to seconds.

        Returns:
            int
        """
        return int(milliseconds / 1000)






    @staticmethod
    def from_seconds_to_milliseconds(seconds: Union[int, float]) -> int:
        """Converts a seconds value into milliseconds. Notice that it will round 
        decimals downwards in case of a float.

        Args:
            seconds: Union[int, float]
                The seconds timestamp to be converted to milliseconds.

        Returns:
            int
        """
        return int(seconds * 1000)





    @staticmethod
    def from_milliseconds_to_minutes(ms: Union[int, float]) -> int:
        """Converts milliseconds into minutes. Notice that it will round 
        decimals downwards in case of a float.

        Args:
            ms: Union[int, float]
                The milliseconds to be converted to minutes.

        Returns:
            int
        """
        return round(ms / 60000)






    @staticmethod
    def from_date_string_to_milliseconds(date_str: str) -> int:
        """Converts a date string into a milliseconds timestamp format.

        Args:
            date_str: str
                The format must be as follows: DD/MM/YYYY. For example: '30/02/2020'

        Returns:
            int
        """
        # Split the date arguments
        date_split: List[str] = date_str.split('/')

        # Initialize the DateTime Instance
        dt: datetime = datetime(int(date_split[2]), int(date_split[1]), int(date_split[0]))

        # Return the timestamp in milliseconds
        return Utils.from_seconds_to_milliseconds(dt.timestamp())







    @staticmethod
    def from_milliseconds_to_date_string(ms: int) -> str:
        """Converts a timestamp in milliseconds into a readeable string.

        Args:
            ms: int
                Timestamp in milliseconds.

        Returns:
            str
        """
        return datetime.fromtimestamp(Utils.from_milliseconds_to_seconds(ms)).strftime("%d/%m/%Y, %H:%M:%S")






    @staticmethod
    def add_minutes(timestamp_ms: Union[int, float], minutes: int) -> int:
        """Adds minutes to a given timestamp. The output of this function is a 
        timestamp in milliseconds with the added minutes.

        Args:
            timestamp_ms: Union[int, float]
                The original timestamp that will be incremented.
            minutes: int 
                The number of minutes that will be added to the timestamp.

        Returns:
            int
        """
        return int(timestamp_ms + (Utils.from_seconds_to_milliseconds(60) * minutes))
    







    ## UUID Helpers ##




    @staticmethod
    def generate_uuid4() -> str:
        """Generates a random Universally Unique Identifier.

        Returns:
            str
        """
        return str(uuid4())





    @staticmethod
    def is_uuid4(uuid: str) -> bool:
        """Verifies if a provided uuid is valid.

        Returns:
            bool
        """
        uuid_obj: Union[UUID, None] = None
        try:
            uuid_obj = UUID(str(uuid), version=4)
        except ValueError:
            return False
        return str(uuid_obj) == uuid








    ## Model Helpers ##




    @staticmethod
    def prettify_model_id(id: str) -> str:
        """Given a model id, it will prettify it so it can be viewed
        in the console properly.

        Returns:
            str
        """
        return id if len(id) < 13 else f"{id[0:10]}..."