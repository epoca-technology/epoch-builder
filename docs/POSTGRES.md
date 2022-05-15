[< Back](../README.md)

# POSTGRES



# Installation

https://www.postgresql.org/download/linux/ubuntu/

### Script

sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update
sudo apt-get -y install postgresql-14

#
# Commands

PSQL TOOL: `sudo -u postgres psql`

List Databases: `\l`

Select Database: `\c postgres;`

Start Status: `sudo service postgresql status`

Start Service: `sudo service postgresql start`

Stop Service: `sudo service postgresql stop`





#
# Database User

Setup the password on the postgres user through PSQL: 

`ALTER USER postgres WITH PASSWORD 'SOME_PASSWORD';`

*Important: The password specified above will be the one used for dev. The production machine has a different password.





#
# Local Network

For the hosts to be able to communicate with the Postgres Server, a few configurations must be modified.

- Firstly, edit the **postgresql.conf** file and set **listen_addresses** to *

- Finally, edit the file **pg_hba.conf** and insert the following entry:

`host    postgres        postgres          192.168.2.0/24          md5`

The IP to be set depends on the network's gateway. For more information visit:

https://stackoverflow.com/questions/22080307/access-postgresql-server-from-lan









#
# PGADMIN4


Install the public key for the repository (if not done previously):

`sudo curl https://www.pgadmin.org/static/packages_pgadmin_org.pub | sudo apt-key add`

#

Create the repository configuration file:

`sudo sh -c 'echo "deb https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" > /etc/apt/sources.list.d/pgadmin4.list && apt update'`

#


Install for both desktop and web modes: `sudo apt install pgadmin4`

Install for desktop mode only: `sudo apt install pgadmin4-desktop`


Install for web mode only: `sudo apt install pgadmin4-web `


Configure the webserver, if you installed pgadmin4-web: `sudo /usr/pgadmin4/bin/setup-web.sh`