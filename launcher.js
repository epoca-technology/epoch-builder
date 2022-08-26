import { spawn } from "child_process";
import { readFileSync, existsSync, lstatSync } from "fs";
import { extname } from "path";
import inquirer from "inquirer";



 /**
  * Cluster Manager
  * This class manages the user input as well as the execution of the process.
  * 
  * Constant Properties:
  * 	CONFIG_PATH: string
  * 	CLUSTER_CONFIG_PATH: string
  * 	EPOCH_CONFIG_PATH: string
  * 
  * Instance Properties
  * 	config: object
  * 	epoch_config: object|undefined
  * 	cluster_servers: ClusterServers
  * 
  */
 class ClusterManager {
	// Path to configuration files
	CONFIG_PATH = "config"
	CLUSTER_CONFIG_PATH = `${this.CONFIG_PATH}/cluster.json`;
	EPOCH_CONFIG_PATH = `${this.CONFIG_PATH}/epoch.json`;



	constructor() {
		// Read the cluster's configuration from the config file
		this.config = FileSystem.read_json_file(this.CLUSTER_CONFIG_PATH);

		// Read the epoch's configuration from the config file (if exists)
		this.epoch_config = undefined;
		try { this.epoch_config = FileSystem.read_json_file(this.EPOCH_CONFIG_PATH) } catch(e){ }

		// Initialize the Cluster Servers Instance
		this.cluster_servers = new ClusterServers(this.config.ssh_private_key_path, this.config.servers);

		// Initialize the Epoch Path Instance
		this.epoch_path = new EpochPath(this.epoch_config ? this.epoch_config.id: undefined)

		// Initialize the User Input Instance
		this.user_input = new UserInput(this.cluster_servers, this.epoch_path);
	}



	/**
	 * Runs the Cluster Manager. Firstly, asks the user for a process to
	 * execute, followed by any additional required or optional params.
	 * @returns Promise<void>
	 */
	async run() {
		// Ask the user for the process to execute
		const process_id = await this.user_input.process_id({
			"Server": [
				"connect_to_a_server",
				"view_server_status",
				"subscribe_to_server_logs",
				"reboot_server",
				"shutdown_server"
			],
			"Epoch Builder": [
				"hyperparams",
				"regression_training",
				"regression_selection",
				"classification_training_data",
				"classification_training",
				"backtest",
				"merge_training_certificates",
				"epoch_management",
				"database_management",
				"unit_tests"
			],
			"Push": [
				"push_configuration",
				"push_database_management",
				"push_candlesticks",
				"push_active_models",
				"push_regression_selection",
				"push_classification_training_data",
				"push_training_configurations",
				"push_backtest_configurations",
				"push_backtest_results",
				"push_epoch",
				"push_epoch_builder"
			],
			"Pull": [
				"pull_database_management",
				"pull_regression_selection",
				"pull_classification_training_data",
				"pull_backtest_results",
				"pull_trained_models"
			]
		});

		// Finally, execute the process
		await this[process_id]();
	}






	/* Server */





	/**
	 * Prompts the user with a list of online servers and establishes a ssh connection
	 * to one of them.
	 * @returns Promise<void>
	 */
	async connect_to_a_server() {
		const server = await this.user_input.server(false, false, true);
		await this.cluster_servers.execute("ssh", ["-i", this.config.ssh_private_key_path, `${server.name}@${server.ip}`], "inherit");
	}




	/**
	 * Displays the status of a specific server or all of them. This function can only
	 * be invoked in online servers.
	 * @returns Promise<void>
	 */
	async view_server_status() {
		// Retrieve the server to visualize the status of
		const server = await this.user_input.server(true, true, true);

		// Check if it has to display all servers
		if (server.name == "all") {
			// List all the servers
			const servers = await this.cluster_servers.list_servers(true);

			// Iterate over each server and print its status
			for (var i = 0; i < servers.length; i++) {
				// Handle localhost
				if (servers[i].name == "localhost") {
					console.log("\n\nlocalhost:\n");await this.cluster_servers.execute("landscape-sysinfo", [], "inherit");
				}

				// Handle external servers
				else {
					const payload = await this.cluster_servers.execute("ssh", ["-i", this.config.ssh_private_key_path, `${servers[i].name}@${servers[i].ip}`, "landscape-sysinfo"], "pipe");
					console.log(`\n\n${servers[i].name}:\n`);console.log(payload);
				}
			}
		}

		// Check if it has to display localhost
		else if (server.name == "localhost") {
			console.log(" ");await this.cluster_servers.execute("landscape-sysinfo", [], "inherit");
		}

		// Otherwise, just display the specific server
		else {
			const payload = await this.cluster_servers.execute("ssh", ["-i", this.config.ssh_private_key_path, `${server.name}@${server.ip}`, "landscape-sysinfo"], "pipe");
			console.log(" ");console.log(payload);
		}
	}


	
 }






 /**
  * Cluster Servers
  * This class handles the server listing and general info extraction.
  * 
  * Class Constants:
  * 	LOCALHOST_SERVER: object
  * 	ALL_SERVER: object
  * 
  * Instance Properties
  * 	ssh_private_key_path: string
  * 	servers: object[]
  */
class ClusterServers {
	// Server objects of localhost and all
	LOCALHOST_SERVER = {name: "localhost", ip: "", is_master: false, is_online: true, is_available: true};
	ALL_SERVER = {name: "all", ip: "", is_master: false, is_online: true, is_available: true};





	/**
	 * Initializes a Cluster Servers Instance
	 * @param ssh_private_key_path: string
	 * @param servers: object[]
	 */
	constructor(ssh_private_key_path, servers) { 
		// Init the SSH Private key path
		this.ssh_private_key_path = ssh_private_key_path;

		// Init the list of servers
		this.servers = servers;
	}







	/**
	 * Given an id, it will return information regarding a specific 
	 * server. Throws an error if no servers or more than 1 is found.
	 * IMPORTANT: The object returned by this function does not include
	 * the online and available properties.
	 * @param id: string
	 * @returns object
	 */
	get_server(id) {
		// Check if it is localhost
		if (id == "localhost") { return this.LOCALHOST_SERVER };

		// Check if it is all
		if (id == "all") { return this.ALL_SERVER };

		// Check against the list of servers
		const server = this.servers.filter((s) => { return s.name == id });
		if (server.length == 1) { return server[0] }
		else { throw new Error(`The server ${id} could not be found.`) };
	}




	/**
	 * Retrieves the server that is set as the master of the cluster.
	 * Throws an error if no servers or more than 1 is found.
	 * IMPORTANT: The object returned by this function does not include
	 * the online and available properties.
	 * @returns object
	 */
	get_server_master() {
		const server = this.servers.filter((s) => { return s.is_master })
		if (server.length == 1) { return server[0] }
		else { throw new Error(`The server master could not be found.`) }
	}





	/**
	 * Lists all the servers within the cluster, as well as the online 
	 * status and availability.
	 * If the include_localhost param is provided, the item will be 
	 * prepended to the list.
	 * If the include_all param is provided, the item will appended 
	 * to the list.
	 * @param include_localhost: boolean
	 * @param include_all: boolean
	 * @returns Promise<object[]>
	 */
	async list_servers(include_localhost = false, include_all = false) {
		// Init the full list of servers
		var servers = [];

		// Iterate over each server and populate the additional values
		for (var i = 0; i < this.servers.length; i++) {
			// Check if the server is online
			const is_online = await this.is_server_online(this.servers[i].ip);

			// Append the value to the list
			servers.push({
				name: this.servers[i].name,
				ip: this.servers[i].ip,
				is_master: this.servers[i].is_master,
				is_online: is_online,
				is_available: await this.is_server_available(is_online, this.servers[i].name, this.servers[i].ip)
			})
		}

		// Check if localhost should be prepended
		if (include_localhost) servers = [ this.LOCALHOST_SERVER ].concat(servers);

		// Check if all should be appended
		if (include_all) servers.push(this.ALL_SERVER)

		// Finally, return the list
		return servers;
	}






	/**
	 * Verifies if a given server is currently online with a ping.
	 * @param ip 
	 * @returns Promise<boolean>
	 */
	async is_server_online(ip) {
		try {
			await this.execute("ping", ["-c", "1", ip], "pipe");
			return true;
		} catch(e) { return false }
	}





	/**
	 * Verifies if a server is available. A server that is available,
	 * is not currently running any Epoch Builder process.
	 * If the server is not online, it will return undefined instead
	 * of a boolean.
	 * @param name 
	 * @param ip 
	 * @returns boolean|undefined
	 */
	async is_server_available(is_online, name, ip) {
		// Make sure the server is online
		if (is_online) {
			// Attempt to retrieve the state of the server
			try {
				// @TODO
				return true
			} catch (e) {
				console.error(e);
				return undefined;
			}
		} 
		
		// Otherwise, return undefined
		else { return undefined }
	}








	/* SSH Args */




	ssh_address(server, ssh_address_suffix = undefined) {
		return `${server.name}@${server.ip}`
	}




	/**
	 * Builds the list of args to be used when executing a ssh command.
	 * If provided, the ssh_address_suffix will be added to the end of the
	 * address, otherwise, it will remain unaltered.
	 * 
	 * @param name 
	 * @param ip 
	 * @param ssh_address_suffix?
	 * @returns 
	 */
	ssh_args(name, ip, ssh_address_suffix = undefined, post_address_commands) {
		// Init the ssh address
		let ssh_address = `${name}@${ip}`;
		
		// Append the address suffix if provided
		if (typeof ssh_address_suffix == "string") ssh_address = ssh_address + ssh_address_suffix;

		// Finally, return the list of args
		return ["-i", this.ssh_private_key_path, ssh_address]
	}











	/* Command Execution */



	/**
	 * Executes a given command and subscribes to its events. The promise is 
	 * resolved once the process indicates it and all the accumulated data is
	 * returned (if any). In case no data is accumulated, undefined will be returned.
	 * @param command: string
	 * @param args: string[]
	 * @param mode: string "inherit"|"pipe"
	 * @returns Promise<string|undefined>
	 */
	 execute(command, args, mode) {
		return new Promise((resolve, reject) => {
			// Init the options based on the provided mode
			var options = {};
			if (mode == "inherit") { options.stdio = "inherit" }
			else if (mode == "pipe") { options.stdio = [null, null, null, "pipe"] }
			else { throw new Error(`An invalid command execution mode was provided: ${mode}`) }

			// Start the process
			const ls = spawn(command, args, options);

			// Init the data
			let data = "";

			// Subscribe to the stdout data event if possible
			if (ls.stdout) ls.stdout.on("data", stdout_data => { data += stdout_data});
			
			// Subscribe to the stdeer data event if possible
			if (ls.stderr) ls.stderr.on("data", stderr_data => { data += stderr_data});
			
			// Subscribe to the error event
			ls.on("error", (error) => { reject(error) });
			
			// Subscribe to the close event
			ls.on("close", code => {
				// Make sure the process exited with code 0
				if (code == 0) { resolve(data.length > 0 ? data: undefined) } 
				
				// Otherwise, handle the error
				else { reject(`The ${command} process exited with the error code: ${code}`) }
			});
		})
	}
}








/**
 * Epoch Path
 * This class handles the retrieval of Epoch Paths in order to simplify many
 * processes.
 * IMPORTANT: If this class is initialized with an undefined epoch_id, most 
 * of its methods will throw an error.
 * 
 * Instance Properties
 * 	epoch_id: string|undefined
 */
class EpochPath {


	/**
	 * Initializes a Epoch Path Instance
	 * @param epoch_id: string|undefined
	 */
	constructor(epoch_id) { this.epoch_id = epoch_id }




	some_path() {
		if (typeof this.epoch_id != "string") throw new Error("The Epoch's ID was not found.");
	}
}









 /**
  * User Input
  * This class handles all the user input that is required in the Cluster Manager.
  * 
  * Instance Properties
  * 	epoch_id: string|undefined
  */
class UserInput {


	/**
	 * Initializes a User Input Instance
	 * @param cluster_servers: ClusterServers
	 * @param epoch_path: EpochPath
	 */
	constructor(cluster_servers, epoch_path) { 		
		// Initialize the servers instance
		this.cluster_servers = cluster_servers;

		// Initialize the epoch path instance
		this.epoch_path = epoch_path;
	}







	/**
	 * Displays the process menu and takes the user through the selection of 
	 * a process id to run.
	 * @param process_menu: object
	 * @returns Promise<string>
	 */
	 async process_id(process_menu) {
		// Retrieve the category of the process
		const category = await inquirer.prompt([
			{ type: "list", name: "value", message: "Select a category", loop: false, choices: Object.keys(process_menu) }
		]);

		// Retrieve the list of processes within the category
		console.log(" ");
		const process = await inquirer.prompt([
			{ type: "list", name: "value", message: "Select a process", loop: false, choices: process_menu[category["value"]] }
		]);

		// Finally, return the process id
		return process["value"];
	}





	/**
	 * Retrieves the list of servers and displays the list
	 * according to the provided configuration. Once a server
	 * is selected, the server object is returned.
	 * @param include_localhost: boolean
	 * @param include_all: boolean
	 * @param online_only: boolean
	 * @param available_only: boolean
	 * @returns Promise<object>
	 */
	async server(include_localhost = false, include_all = false, online_only = false, available_only = false) {
		// Retrieve the list of servers
		const servers = await this.cluster_servers.list_servers(include_localhost, include_all);

		// Iterate over each server and populate the list of choices
		let choices = [];
		for (let server of servers) {
			// Populate the state of the choice
			let disabled_state = undefined;
			if (online_only && !server.is_online) { disabled_state = "Offline" }
			else if (available_only && !server.is_available) { disabled_state = "Busy" }

			// Push the choice to the list
			choices.push({ name: server.name, disabled: disabled_state })
		}

		// Present the list
		console.log(" ");
		const server_answer = await inquirer.prompt([{type: "list", name: "value", message: "Select a server", loop: false, choices: choices}]);

		// Finally, return the server
		return this.cluster_servers.get_server(server_answer["value"]);
	}


}













/**
 * File System
 * This singleton handles all the interactions with the local file system.
 */
class FileSystem {

	/* Existance */


	/**
	 * Checks if a file exists in a given path.
	 * @param path: string
	 * @returns boolean
	 */
	static file_exists(path) {
		try {
			return lstatSync(path).isFile();
		} catch (e) { return false }
	}



	/**
	 * Checks if a directory exists in a given path.
	 * @param path: string
	 * @returns boolean
	 */
	 static dir_exists(path) {
		try {
			return lstatSync(path).isDirectory();
		} catch (e) { return false }
	}






	/* Reads */



	/**
	 * Reads a json file located an any given path.
	 * @param path: string
	 * @returns object
	 */
	 static read_json_file(path) {
		// Make sure it is a json file
		if (extname(path) != ".json") throw new Error(`The provided file is not json format: ${path}`);

		// Make sure the file exists
		if (!FileSystem.file_exists(path)) throw new Error(`The file could not loaded because it does not exist at: ${path}`);

		// Finally, return the configuration
		return JSON.parse(readFileSync(path));
	}
}















/**
 * Process Execution
 * If successful, the process will terminate with a status of 0. Otherwise, 
 * prints the error and terminates with a status of 1.
 */
console.clear();
console.log("EPOCH BUILDER LAUNCHER\n\n");
new ClusterManager().run().then(() => { process.exit(0) }).catch(e => { console.error(e); process.exit(1); })