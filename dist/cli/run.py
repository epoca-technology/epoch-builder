from typing import List, Dict
import os
from inquirer import List as InquirerList, Text, prompt
from subprocess import Popen



# Welcome
os.system("cls" if os.name == "nt" else "clear")
print("EPOCH BUILDER\n")


# Host IP
path: str = "config/host_ip.txt"
if not os.path.exists("config"):
    os.makedirs("config")
if not os.path.isfile(path):
    host_ip_answer: Dict[str, str] = prompt([Text("ip", f"Enter the Host IP")])
    host_ip: str = host_ip_answer["ip"]
    with open(path, "w") as file_wrapper:
        file_wrapper.write(host_ip)



# Processes
print(" ")
processes: Dict[str, str] = {
    "Backtest": "run_backtest.py",
    "Classification Training Data": "run_classification_training_data.py",
    "Classification Training": "run_classification_training.py",
    "Database Management": "run_db_management.py",
    "Epoch Management": "run_epoch_management.py",
    "Hyperparams": "run_hyperparams.py",
    "Merge Training Certificates": "run_merge_training_certificates.py",
    "Regression Selection": "run_regression_selection.py",
    "Regression Training": "run_regression_training.py",
    "Update Host IP": "run_update_host_ip.py"
}
processes_answer: List[Dict[str, str]] = prompt([
    InquirerList("process", message="Select the process to run", choices=processes.keys())
])


# Execute the process
os.system("cls" if os.name == "nt" else "clear")
proc = Popen(f"python3 dist/cli/{processes[processes_answer['process']]}", shell=True)
proc.wait()