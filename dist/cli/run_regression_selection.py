from typing import Dict, List
from inquirer import Text, prompt
from modules.epoch.Epoch import Epoch
from modules.regression_selection.RegressionSelection import RegressionSelection


# EPOCH INIT
Epoch.init()



# Configuration Input
print("REGRESSION SELECTION\n")
selected_models: Dict[str, str] = prompt([Text("ids", "Enter the list of regression ids")])
model_ids: List[str] = selected_models["ids"].split(",")



# Initialize the instance and execute the selection
rs: RegressionSelection = RegressionSelection(model_ids)
rs.run()
print("\n\nREGRESSION SELECTION COMPLETED")