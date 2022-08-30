from argparse import ArgumentParser
from subprocess import Popen
from modules._types import IDatabaseSummary
from modules.utils.Utils import Utils
from modules.configuration.Configuration import Configuration
from modules.epoch.Epoch import Epoch
from modules.database.Database import Database, DB_CONNECTION_CONFIG


# DATABASE MANAGEMENT
# Args:
#   --action "summary"|"backup"|"restore"|"update_host_ip"
#
# Update Host IP Args:
#   --ip? "192.168.1.236" -> Mandatory arg when updating the host ip
endpoint_name: str = "DATABASE MANAGEMENT"
Utils.endpoint_header(Configuration.VERSION, endpoint_name)




# Initialize the Epoch
Epoch.init()





# Extract the args
parser = ArgumentParser()
parser.add_argument("--action", dest="action")
parser.add_argument("--ip", dest="ip", nargs='?')
args = parser.parse_args()





# Database Summary
# Extracts a summary of the tables and the data stored so far.
if args.action == "summary":
    # Retrieve the summary
    s: IDatabaseSummary = Database.get_summary()

    # Create the function that will print the sizes
    def format_size(size: float) -> str:
        return str(round(size/1073741824, 4)) + " Gbs"

    # Output it
    print("\nDATABASE SUMMARY\n")
    print(s["version"])
    print(f"Name: {s['connection_config']['database']}")
    print(f"User: {s['connection_config']['user']}")
    print(f"Host IP: {s['connection_config']['host_ip']}")
    print(f"Port: {s['connection_config']['port']}")
    print(f"Password: {s['connection_config']['password']}")
    print(f"Total Size: {format_size(s['size'])}")
    print(f"\nTables ({len(s['tables'])}):")
    for t in s["tables"]:
        print(f"{t['name']}: {format_size(t['size'])}")
    print(f"\nTest Tables ({len(s['test_tables'])}):")
    for tt in s["test_tables"]:
        print(f"{tt['name']}: {format_size(tt['size'])}")





# Database Backup
# Generates a compressed backup file and places it in the backup directory.
elif args.action == "backup":
    # Init the backup directory
    backup_path: str = Database.DB_MANAGEMENT_PATH + "/backup"
    backup_name: str = str(Utils.get_time()) + ".dump"

    # Create the backup directory if it does not exist
    Utils.make_directory(backup_path)

    # Print the progress
    print(f"1/1) Creating database backup: {backup_name}...")

    # Init the command
    command: str = f"pg_dump \
        -U {DB_CONNECTION_CONFIG['user']} \
            -h {DB_CONNECTION_CONFIG['host_ip']} \
                -p {DB_CONNECTION_CONFIG['port']} \
                    -d {DB_CONNECTION_CONFIG['database']} \
                        -f {backup_path}/{backup_name} -Fc"

    # Add the required env vars and execute the command
    proc = Popen(command, shell=True, env={
        'PGPASSWORD': DB_CONNECTION_CONFIG['password']
    })
    proc.wait()





# Database Restore
# Reads the restore directory and restores the compressed backup file located in it.
elif args.action == "restore":
    # Init the restore path
    restore_path: str = Database.DB_MANAGEMENT_PATH + "/restore"

    # Make sure the restore path exists, otherwise create it and provide information
    if not Utils.directory_exists(restore_path):
        Utils.make_directory(restore_path)
        raise ValueError(f"The backup dump to be restored must be placed in {restore_path}")

    # Extract the names of the databases in the path. Make sure there is exactly 1 dump
    _, backup_names = Utils.get_directory_content(restore_path, only_file_ext=".dump")
    if len(backup_names) != 1:
        raise ValueError(f"There must be 1 backup dump file to be restored in {restore_path}")

    # Print the progress
    print(f"1/1) Restoring database backup: {backup_names[0]}...")

    # Clean all the tables and reinitialize them fresh
    Database.delete_tables()
    Database.initialize_tables()

    # Init the command
    command: str = f"pg_restore --clean \
        -U {DB_CONNECTION_CONFIG['user']} \
            -h {DB_CONNECTION_CONFIG['host_ip']} \
                -p {DB_CONNECTION_CONFIG['port']} \
                    -d {DB_CONNECTION_CONFIG['database']} {restore_path}/{backup_names[0]}"

    # Add the required env vars and execute the command
    proc = Popen(command, shell=True, env={
        'PGPASSWORD': DB_CONNECTION_CONFIG['password']
    })
    proc.wait()




# Update Database Host IP
# Updates the host ip valuye in the db_host_ip.txt file within the root config directory.
elif args.action == "update_host_ip":
    # Update Host IP
    print("1/1) Setting the IP on the configuration file...")
    Configuration.set_db_host_ip(args.ip)




# Unknown Action 
else:
    raise ValueError(f"An invalid Database Management Action was provided: {args.action}")



# End of Script
Utils.endpoint_footer(endpoint_name)