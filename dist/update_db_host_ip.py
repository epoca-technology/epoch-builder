from argparse import ArgumentParser
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration




# UPDATE DATABASE HOST IP
# Args:
#   --ip "192.168.1.236"
endpoint_name: str = "UPDATE DATABASE HOST IP"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)


# Extract the args
parser = ArgumentParser()
parser.add_argument("--ip", dest="ip")
args = parser.parse_args()


# Update Host IP
print("1/1) Setting the IP on the configuration file...")
Configuration.set_db_host_ip(args.ip)


# End of Script
Utils.endpoint_footer(endpoint_name)