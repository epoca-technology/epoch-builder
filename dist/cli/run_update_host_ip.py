from typing import Dict
from os import makedirs
from os.path import exists
from inquirer import Text, prompt



# Welcome
print("UPDATE HOST IP\n")



# Update Host IP
path: str = "config/host_ip.txt"
if not exists("config"):
    makedirs("config")
host_ip_answer: Dict[str, str] = prompt([Text("ip", f"Enter the Host IP")])
host_ip: str = host_ip_answer["ip"]
with open(path, "w") as file_wrapper:
    file_wrapper.write(host_ip)


print("\n\nUPDATE HOST IP COMPLETED")