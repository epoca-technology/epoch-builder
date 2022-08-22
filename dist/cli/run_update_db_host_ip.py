from typing import Dict
from modules.configuration.Configuration import Configuration
from inquirer import Text, prompt



# Welcome
print("UPDATE DATABASE HOST IP\n")



# Update Host IP
db_host: Dict[str, str] = prompt([Text("ip", f"Enter the Database Host IP")])
Configuration.set_db_host_ip(db_host["ip"])



print("\n\nUPDATE DATABASE HOST IP COMPLETED")