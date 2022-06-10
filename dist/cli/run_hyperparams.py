from typing import List, Dict
from inquirer import Text, List as InquirerList, prompt
from modules.hyperparams import KerasClassificationHyperparams

# Configuration Input
print("HYPERPARAMS")
answers: List[Dict[str, str]] = prompt([
    InquirerList("model_type", message="Select the type of model", choices=["Regression", "Classification"]),
    InquirerList("network_type", message="Select the type of network", choices=["DNN", "CNN", "LSTM", "CLSTM"]),
    Text("training_data_id", "Enter the Training Data ID (Ignore if it is a regression)"),
])



# Generate the batched config based on the type of model
if answers["model_type"] == "Regression":
    raise NotImplementedError("The regression hyperparams has not yet been implemented.")
elif answers["model_type"] == "Classification":
    KerasClassificationHyperparams.generate(answers["network_type"], answers["training_data_id"])
print("\nHYPERPARAMS COMPLETED")