from typing import Dict
from inquirer import Text, prompt
from modules.epoch.Epoch import Epoch
from modules.regression_selection.RegressionSelection import RegressionSelection


# EPOCH INIT
Epoch.init()



# Configuration Input
print("REGRESSION SELECTION\n")
limit_answer: Dict[str, str] = prompt([Text("value", "Number of models to be selected")])
if not limit_answer["value"].isdigit():
    raise ValueError("The number of models to be selected must be a valid integer.")
limit: int = int(limit_answer["value"])



# Initialize the instance and execute the selection
rs: RegressionSelection = RegressionSelection(limit)
rs.run()
print("\n\nREGRESSION SELECTION COMPLETED")