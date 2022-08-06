from typing import Dict
from inquirer import List as InquirerList, Text, prompt
from modules.epoch.Epoch import Epoch




## WELCOME ##
print("EPOCH MANAGEMENT\n")
action_answer: Dict[str, str] = prompt([InquirerList("action", message="Select an action to execute", choices=[ 
    "Create Epoch", 
    "Classification Training Data for Unit Tests",
    "Export Epoch"
])])
action: str = action_answer["action"]


# Create Epoch
# Creates a brand new Epoch with all the required directories and files.
if action == "Create Epoch":
    print("\n\nCREATE EPOCH RUNNING\n")
    epoch_answers: Dict[str, str] = prompt([
        Text("id", "Enter the ID of the Epoch prefixed with '_'. F.e: _MYEPOCH"),
        Text("epoch_width", f"Enter the width of the Epoch in months (Defaults to {Epoch.DEFAULTS['epoch_width']})"),
        Text("seed", f"Enter the seed of the Epoch (Defaults to {Epoch.DEFAULTS['seed']})"),
        Text("train_split", f"Enter the Train Split (Defaults to {Epoch.DEFAULTS['train_split']})"),
        Text("backtest_split", f"Enter the Backtest Split (Defaults to {Epoch.DEFAULTS['backtest_split']})"),
        Text("model_discovery_steps", f"Enter the Model Discovery Steps (Defaults to {Epoch.DEFAULTS['model_discovery_steps']})"),
        Text("idle_minutes_on_position_close", f"Enter the number of minutes the model will idle on position close (Defaults to {Epoch.DEFAULTS['idle_minutes_on_position_close']})"),
    ])
    epoch_width: int = int(epoch_answers["epoch_width"]) if epoch_answers["epoch_width"].isdigit() else Epoch.DEFAULTS['epoch_width']
    seed: int = int(epoch_answers["seed"]) if epoch_answers["seed"].isdigit() else Epoch.DEFAULTS['seed']
    train_split: float = float(epoch_answers["train_split"]) if epoch_answers["train_split"].isdigit() else Epoch.DEFAULTS['train_split']
    backtest_split: float = float(epoch_answers["backtest_split"]) if epoch_answers["backtest_split"].isdigit() else Epoch.DEFAULTS['backtest_split']
    model_discovery_steps: int = int(epoch_answers["model_discovery_steps"]) if epoch_answers["model_discovery_steps"].isdigit() else Epoch.DEFAULTS['model_discovery_steps']
    idle_minutes_on_position_close: int = int(epoch_answers["idle_minutes_on_position_close"]) if epoch_answers["idle_minutes_on_position_close"].isdigit() else Epoch.DEFAULTS['idle_minutes_on_position_close']
    print(" ")
    Epoch.create(
        id=epoch_answers["id"],
        epoch_width=epoch_width,
        seed=seed,
        train_split=train_split,
        backtest_split=backtest_split,
        model_discovery_steps=model_discovery_steps,
        idle_minutes_on_position_close=idle_minutes_on_position_close,
    )



# Classification Training Data for Unit Tests
# Alters the current Epoch's configuration and sets the classification training data ID that will be used by
# the Unit Tests
elif action == "Classification Training Data for Unit Tests":
    print("\n\nCLASSIFICATION TRAINING DATA ID RUNNING\n")
    epoch_answers: Dict[str, str] = prompt([
        Text("id", "Enter the ID of the Classification Training Data for Unit Tests"),
    ])
    Epoch.set_classification_training_data_id_ut(epoch_answers["id"])



# Export Epoch
# Compiles all the neccessary data and outputs a zip file which is then processed by Epoca's Infrastructure.
elif action == "Export Epoch":
    print("\n\nEXPORT EPOCH RUNNING\n")



# Unknown Action
else:
    raise ValueError(f"The provided action could not be processed: {str(action)}")



print("\n\nEPOCH MANAGEMENT COMPLETED")