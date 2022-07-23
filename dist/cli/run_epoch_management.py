from typing import Dict
from inquirer import List as InquirerList, Text, prompt
from modules.epoch.Epoch import Epoch




## WELCOME ##
print("EPOCH MANAGEMENT\n")
action_answer: Dict[str, str] = prompt([InquirerList("action", message="Select an action to execute", choices=[ 
    "Create Epoch", 
    "Set Class. Training Data for Unit Tests",
    "Set Position Exit Combination",
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
        Text("regression_price_change_requirement", f"Enter the Price Change Requirement for evaluating Regressions (Defaults to {Epoch.DEFAULTS['regression_price_change_requirement']})"),
        Text("idle_minutes_on_position_close", f"Enter the number of minutes the model will idle on position close (Defaults to {Epoch.DEFAULTS['idle_minutes_on_position_close']})"),
    ])
    epoch_width: int = int(epoch_answers["epoch_width"]) if epoch_answers["epoch_width"].isdigit() else Epoch.DEFAULTS['epoch_width']
    seed: int = int(epoch_answers["seed"]) if epoch_answers["seed"].isdigit() else Epoch.DEFAULTS['seed']
    regression_price_change_requirement: float = float(epoch_answers["regression_price_change_requirement"]) if epoch_answers["regression_price_change_requirement"].isdigit() else Epoch.DEFAULTS['regression_price_change_requirement']
    idle_minutes_on_position_close: int = int(epoch_answers["idle_minutes_on_position_close"]) if epoch_answers["idle_minutes_on_position_close"].isdigit() else Epoch.DEFAULTS['idle_minutes_on_position_close']
    print(" ")
    Epoch.create(
        id=epoch_answers["id"],
        epoch_width=epoch_width,
        seed=seed,
        regression_price_change_requirement=regression_price_change_requirement,
        idle_minutes_on_position_close=idle_minutes_on_position_close,
    )



# Set Class. Training Data for Unit Tests
# Alters the current Epoch's configuration and sets the new ID. This ID is required by the Unit Test.
elif action == "Set Class. Training Data for Unit Tests":
    print("\n\nSET CLASS. TRAINING DATA RUNNING\n")
    epoch_answers: Dict[str, str] = prompt([
        Text("id", "Enter the ID of the Classification Training Data for Unit Tests"),
    ])
    Epoch.set_ut_class_training_data_id(epoch_answers["id"])



# Set Position Exit Combination
# Alters the current Epoch's configuration and sets the provided Position Exit Combination. 
# These values will be used when generating the training data and evaluating classification models.
elif action == "Set Position Exit Combination":
    print("\n\nSET POSITION EXIT COMBINATION RUNNING\n")
    epoch_answers: Dict[str, str] = prompt([
        Text("take_profit", "Enter the Take Profit"),
        Text("stop_loss", "Enter the Stop Loss"),
    ])
    Epoch.set_position_exit_combination(take_profit=float(epoch_answers["take_profit"]), stop_loss=float(epoch_answers["stop_loss"]))



# Export Epoch
# Compiles all the neccessary data and outputs a zip file which is then processed by Epoca's Infrastructure.
elif action == "Export Epoch":
    print("\n\nEXPORT EPOCH RUNNING\n")



# Unknown Action
else:
    raise ValueError(f"The provided action could not be processed: {str(action)}")



print("\n\nEPOCH MANAGEMENT COMPLETED")