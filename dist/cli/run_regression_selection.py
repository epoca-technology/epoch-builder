from typing import Dict
from inquirer import Text, List as InquirerList, prompt
from modules.regression_selection.RegressionSelection import RegressionSelection

# Configuration Input
print("REGRESSION SELECTION")
answers: Dict[str, str] = prompt([
    Text("models_limit", "Number of models to be selected"),
    InquirerList("clean_results_dir", message="Delete Backtest Result Files on completion?", choices=["No", "Yes"])
])



# Initialize the instance and execute the selection
rs: RegressionSelection = RegressionSelection(int(answers["models_limit"]), answers["clean_results_dir"] == "Yes")
rs.run()
print("\nREGRESSION COMPLETED")