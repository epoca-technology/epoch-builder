from typing import List
from modules.types import IPositionExitCombinationDatabase, IPositionExitCombinationID, IPositionExitCombinationPath,\
    IPositionExitCombinationRecord




# Class
class PositionExitCombination:
    """PositionExitCombination Class

    This singleton handles the everything related to the 

    Class Properties:
        DB: IPositionExitCombinationDatabase
            Dict containing all combination records.
    """
    # Database containing all position exit combinations
    DB: IPositionExitCombinationDatabase = {
        "TP10_SL10": { "take_profit": 1,    "stop_loss": 1,     "path": "01_TP10_SL10"},
        "TP10_SL15": { "take_profit": 1,    "stop_loss": 1.5,   "path": "02_TP10_SL15"},
        "TP15_SL10": { "take_profit": 1.5,  "stop_loss": 1,     "path": "03_TP15_SL10"},
        "TP15_SL15": { "take_profit": 1.5,  "stop_loss": 1.5,   "path": "04_TP15_SL15"},
        "TP20_SL10": { "take_profit": 2,    "stop_loss": 1,     "path": "05_TP20_SL10"},
        "TP20_SL15": { "take_profit": 2,    "stop_loss": 1.5,   "path": "06_TP20_SL15"},
        "TP10_SL20": { "take_profit": 1,    "stop_loss": 2,     "path": "07_TP10_SL20"},
        "TP15_SL20": { "take_profit": 1.5,  "stop_loss": 2,     "path": "08_TP15_SL20"},
        "TP20_SL20": { "take_profit": 2,    "stop_loss": 2,     "path": "09_TP20_SL20"},
        "TP20_SL25": { "take_profit": 2,    "stop_loss": 2.5,   "path": "10_TP20_SL25"},
        "TP25_SL20": { "take_profit": 2.5,  "stop_loss": 2,     "path": "11_TP25_SL20"},
        "TP25_SL25": { "take_profit": 2.5,  "stop_loss": 2.5,   "path": "12_TP25_SL25"},
        "TP25_SL30": { "take_profit": 2.5,  "stop_loss": 3,     "path": "13_TP25_SL30"},
        "TP30_SL25": { "take_profit": 3,    "stop_loss": 2.5,   "path": "14_TP30_SL25"},
        "TP20_SL30": { "take_profit": 2,    "stop_loss": 3,     "path": "15_TP20_SL30"},
        "TP30_SL20": { "take_profit": 3,    "stop_loss": 2,     "path": "16_TP30_SL20"},
        "TP30_SL30": { "take_profit": 3,    "stop_loss": 3,     "path": "17_TP30_SL30"},
        "TP30_SL35": { "take_profit": 3,    "stop_loss": 3.5,   "path": "18_TP30_SL35"},
        "TP35_SL30": { "take_profit": 3.5,  "stop_loss": 3,     "path": "19_TP35_SL30"},
        "TP35_SL35": { "take_profit": 3.5,  "stop_loss": 3.5,   "path": "20_TP35_SL35"},
        "TP35_SL40": { "take_profit": 3.5,  "stop_loss": 4,     "path": "21_TP35_SL40"},
        "TP40_SL35": { "take_profit": 4,    "stop_loss": 3.5,   "path": "22_TP40_SL35"},
        "TP30_SL40": { "take_profit": 3,    "stop_loss": 4,     "path": "23_TP30_SL40"},
        "TP40_SL30": { "take_profit": 4,    "stop_loss": 3,     "path": "24_TP40_SL30"},
        "TP40_SL40": { "take_profit": 4,    "stop_loss": 4,     "path": "25_TP40_SL40"}
    }





    @staticmethod
    def get_id(take_profit: float, stop_loss: float) -> IPositionExitCombinationID:
        """Retrieves the ID based on a TP/SL Combination.

        Args:
             take_profit: float
             stop_loss: float
                The position exit combination.

        Returns:
            IPositionExitCombinationID
        """
        # Initialize the ID
        def _format_value(value: float) -> str:
            """Formats a float into a string compatible with filesystems.
            For example: 1 -> 10, 1.5 -> 15, 3.5 -> 35.

            Args:
                value: float
                    The take profit or the stop loss value

            Returns: 
                str
            """
            return str(round(float(value), 1)).replace(".", "")
        id: IPositionExitCombinationID = f"TP{_format_value(take_profit)}_SL{_format_value(stop_loss)}"

        # Make sure it exists, otherwise raise an error
        if PositionExitCombination.DB.get(id) is None:
            raise ValueError(f"The Position Exit Combination ID {id} is not in the Database.")

        # Finally, return the ID
        return id






    @staticmethod
    def get_path(take_profit: float, stop_loss: float) -> IPositionExitCombinationPath:
        """Retrieves the path based on a TP/SL Combination.

        Args:
             take_profit: float
             stop_loss: float
                The position exit combination.

        Returns:
            IPositionExitCombinationPath
        """
        # Initialize the ID
        id: IPositionExitCombinationID = PositionExitCombination.get_id(take_profit, stop_loss)

        # Finally, return the ID
        return PositionExitCombination.DB[id]["path"]







    @staticmethod
    def get_records() -> List[IPositionExitCombinationRecord]:
        """Retrieves a list with all the combination records.

        Returns:
            List[IPositionExitCombinationRecord]
        """
        return PositionExitCombination.DB.values()