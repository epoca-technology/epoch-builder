from typing import List, Dict
import os
from inquirer import List as InquirerList, prompt
from subprocess import Popen

# Processes
processes: Dict[str, str] = {
    "Arima Combinations": "run_arima_combinations.py",
    "Backtest": "run_backtest.py",
    "Classification Training Data": "run_classification_training_data.py",
    "Classification Training": "run_classification_training.py",
    "Database Management": "run_db_management.py",
    "Hyperparams": "run_hyperparams.py",
    "Merge Training Certificates": "run_merge_training_certificates.py",
    "Regression Selection": "run_regression_selection.py",
    "Regression Training": "run_regression_training.py"
}


# Configuration Input
os.system("cls" if os.name == "nt" else "clear")
print("PREDICTION BACKTESTING")
answer: List[Dict[str, str]] = prompt([
    InquirerList("process", message="Select the process to run", choices=processes.keys())
])


# Execute the process
os.system("cls" if os.name == "nt" else "clear")
proc = Popen(f"python3 dist/cli/{processes[answer['process']]}", shell=True)
proc.wait()