import { FileSystem } from "./FileSystem.js";
import { ClusterPath } from "./ClusterPath.js"
import { ClusterCommand } from "./ClusterCommand.js"
import { ClusterServer } from "./ClusterServer.js"
import { ClusterInput } from "./ClusterInput.js"



 /**
  * Cluster
  * This class initializes all the requirements and runs any cluster process on any server.
  * 
  * Constant Properties:
  * 	CONFIG_PATH: string
  * 	CLUSTER_CONFIG_PATH: string
  * 	EPOCH_CONFIG_PATH: string
  * 
  * Instance Properties
  * 	config: object
  * 	epoch_config: object|undefined
  * 	cluster_path: ClusterPath
  * 	cluster_command: ClusterCommand
  * 	cluster_server: ClusterServer
  * 	cluster_input: ClusterInput
  */
 class Cluster {
	// Path to configuration files
	CONFIG_PATH = "config"
	CLUSTER_CONFIG_PATH = `${this.CONFIG_PATH}/cluster.json`;
	EPOCH_CONFIG_PATH = `${this.CONFIG_PATH}/epoch.json`;



	/**
	 * Initializes the Cluster's Instance. Notice that the only public 
	 * method is run().
	 */
	constructor() {
		// Read the cluster's configuration from the config file
		this.config = FileSystem.read_json_file(this.CLUSTER_CONFIG_PATH);

		// Read the epoch's configuration from the config file (if exists)
		this.epoch_config = undefined;
		try { this.epoch_config = FileSystem.read_json_file(this.EPOCH_CONFIG_PATH) } catch(e) { }

		// Initialize the Cluster Path Instance
		this.cluster_path = new ClusterPath(this.config.local_path, this.epoch_config ? this.epoch_config.id: undefined);

        // Initialize the Cluster Command Instance
        this.cluster_command = new ClusterCommand(this.config.ssh_private_key_path, this.cluster_path);

		// Initialize the Cluster Server Instance
		this.cluster_server = new ClusterServer(this.cluster_command, this.config.servers);

		// Initialize the Cluster Input Instance
		this.cluster_input = new ClusterInput(this.cluster_server, this.cluster_path);
	}



	/**
	 * Runs the Cluster Manager. Firstly, asks the user for a process to
	 * execute, followed by any additional required or optional params.
	 * @returns Promise<void>
	 */
	async run() {
		// Ask the user for the process to execute
		const process_id = await this.cluster_input.process_id({
			"Server": [
				"connect_to_a_server",
				"view_server_status",
				"subscribe_to_server_logs",
				"reboot_server",
				"shutdown_server",
				"kill_process",
				"install_ssh_key_on_a_server"
			],
			"Epoch Builder": [
				"create_epoch",
				"generate_regression_training_configs",
				"train_regression_batch",
				"build_prediction_models",
				"export_epoch",
				"unit_tests"
			],
			"Push": [
				"push_root_files",
				"push_configuration",
				"push_candlesticks",
				"push_dist",
				"push_regression_training_configs",
				"push_epoch_builder"
			],
			"Pull": [
				"pull_trained_regressions",
			]
		});

		// Finally, execute the process
		await this[process_id]();
	}






	/*****************************************************************************
     * SERVER												   					 *
     * Server stands for the general server actions that can be performed on the * 
	 * local machine, a cluster machine or an external machine.					 *
	 * 																		     *
	 * Processes:															     *
	 * 	connect_to_a_server													     *
	 * 	view_server_status											     		 *
	 * 	subscribe_to_server_logs											     *
	 * 	reboot_server											                 *
	 * 	shutdown_server											                 *
	 * 	kill_process											                 *
	 * 	install_ssh_key_on_a_server											     *
     *****************************************************************************/





	/**
	 * Prompts the user with a list of online servers and establishes a ssh connection
	 * to one of them.
	 * @returns Promise<void>
	 */
	async connect_to_a_server() { await this.cluster_command.connect(await this.cluster_input.server(false, false, true)) }





	/**
	 * Displays the status of a specific server or all of them. This function can only
	 * be invoked in online servers.
	 * @returns Promise<void>
	 */
	async view_server_status() {
		// Retrieve the server to visualize the status of
		const server = await this.cluster_input.server(true, true, true);

		// Check if it has to display all servers
		if (server.name == "all") { 
            for (let s of await this.cluster_server.list_servers(true)) { await this._display_server_status(s) } 
        }

		// Otherwise, print the specific server
		else { 
			server.is_online = server.name == "localhost" ? true: await this.cluster_command.is_server_online(server.ip);
			server.is_available = await this.cluster_command.is_server_available(server.is_online, server);
			await this._display_server_status(server);
		}
	}



	/**
	 * Handles the display of the status for a given server.
	 * @param server 
	 * @returns Promise<void>
	 */
	async _display_server_status(server) {
		// Print the heading
		console.log(`\n\n${server.name}:`);

		// Check if the server is online
		if (server.is_online) {
			// Print the state of the server
			if (server.is_available) {
				console.log("Not running.");
			} else {
				const { pid, command } = await this.cluster_command.get_process(server);
				console.log(`Running: ${pid} (${command})`);
			}

			// Print the system info
			console.log(await this.cluster_command.get_landscape_sysinfo(server));
		} else {
			console.log("Offline");
		}
	}






	/**
	 * Subscribes to the nuhup logs on a selected server.
	 * @returns Promise<void>
	 */
	async subscribe_to_server_logs() {
		// Retrieve the server to visualize the logs of
		const server = await this.cluster_input.server(false, false, true);

		// Subscribe to the logs
		await this.cluster_command.subscribe_to_logs(server, this.cluster_path.nohup_logs(false))
	}




    
	/**
	 * Reboots a specific server.
	 * @returns Promise<void>
	 */
	async reboot_server() { 
		// Select the server
		const server = await this.cluster_input.server(false, true, true);

		console.log(`1/1) Rebooting ${server.name}...`);
		await this.cluster_command.reboot(server);
	}





	/**
	 * Shuts down a specific server or all the servers within the
	 * cluster.
	 * @returns Promise<void>
	 */
	async shutdown_server() { 
		// Select the server
		const server = await this.cluster_input.server(false, true, true);

		// Check if it has to display all servers
		if (server.name == "all") { 
            for (let s of await this.cluster_server.list_servers()) { 
				if (s.is_online && s.is_cluster) {
					console.log(`\n\nShutting ${s.name} down...\n`);
					try {
						await this.cluster_command.shutdown(s);
					} catch (e) { }
				}
			} 
        }

		// Otherwise, print the specific server
		else { 
			console.log(`\n1/1) Shutting ${server.name} down...`);
			await this.cluster_command.shutdown(server);
		}
	}




	/**
	 * Kills the python3 process running on a server.
	 * @returns Promise<void>
	 */
	async kill_process() { 
		// Select the server
		const server = await this.cluster_input.server(true, false, true, false, true);

		// Kill the process
		console.log(`\n1/1) Killing python3 process on ${server.name}...`);
		await this.cluster_command.kill_process(server) 
	}





	/**
	 * Installs the SSH Public Key on a selected server.
	 * @returns Promise<void>
	 */
	async install_ssh_key_on_a_server() { await this.cluster_command.install_ssh_key(await this.cluster_input.server(false, false, true)) }









	/********************************************************************************
     * EPOCH BUILDER												   				*
     * Epoch Builder stands for all the actions related to the generation of assets * 
	 * of the new epoch.							   								*
	 * 																		   		*
	 * Processes:															   		*
	 * 	create_epoch													   		    *
	 * 	generate_regression_training_configs									    *
	 * 	train_regression_batch											           	*
	 * 	build_prediction_models											    		*
	 * 	export_epoch											           			*
	 * 	unit_tests											   						*
     ********************************************************************************/




	/**
	 * Collects all the required data and runs the epoch creation process
	 * @returns Promise<void>
	 */
	async create_epoch() { 
		// Retrieve the server
		const server = this.cluster_server.get_server("localhost");

		// Collect the args
		const args = await this.cluster_input.create_epoch();

		// Finally, Run the command
		await this.cluster_command.create_epoch(server, args); 
	}




	/**
	 * Generates the regression training configurations.
	 * @returns Promise<void>
	 */
	async generate_regression_training_configs() { 
		// Retrieve the server
		const server = this.cluster_server.get_server("localhost");

		// Finally, Run the command
		await this.cluster_command.generate_regression_training_configs(server); 
	}






	/**
	 * Initializes the training process for a selected batch in any server.
	 * @returns Promise<void>
	 */
	async train_regression_batch() { 
		// Retrieve the server
		const server = await this.cluster_input.server(true, false, true, true);

		// Retrieve and unpack the category and the config file
		const { category, batch_file_name } = await this.cluster_input.regression_training_configs();

		// Finally, Run the command
		await this.cluster_command.train_regression_batch(server, category, batch_file_name); 
	}






	/**
	 * Initializes the build of the prediction models in the local machine.
	 * @returns Promise<void>
	 */
	async build_prediction_models() { 
		// Retrieve the server
		const server = this.cluster_server.get_server("localhost");

		// Retrieve and unpack the unpacked args
		const { regression_ids, max_combinations } = await this.cluster_input.build_prediction_models();

		// Finally, Run the command
		await this.cluster_command.build_prediction_models(server, regression_ids, max_combinations); 
	}






	/**
	 * Initializes the build of the epoch export process in the local machine.
	 * @returns Promise<void>
	 */
	 async export_epoch() { 
		// Retrieve the server
		const server = this.cluster_server.get_server("localhost");

		// Retrieve and unpack the unpacked args
		const model_id = await this.cluster_input.export_epoch();

		// Finally, Run the command
		await this.cluster_command.export_epoch(server, model_id); 
	}







	/**
	 * Runs the unit tests for a selected server.
	 * @returns Promise<void>
	 */
	async unit_tests() { 
		// Retrieve the server
		const server = this.cluster_server.get_server("localhost");

		// Finally, Run the command
		await this.cluster_command.unit_tests(server); 
	}









	/***************************************************************************
     * PUSH												   					   *
     * Push stands for the transfer of data from the local machine to a server * 
	 * that can live in the cluster or anywhere.							   *
	 * In a push action, the directory in the server is completely removed 	   *
	 * prior to pushing the data in order to guarantee the server is always	   *
	 * running on the latest version.	   									   *
	 * 																		   *
	 * Processes:															   *
	 * 	push_root_files													   	   *
	 * 	push_configuration													   *
	 * 	push_candlesticks											   		   *
	 * 	push_dist											           		   *
	 * 	push_regression_training_configs									   *
	 * 	push_epoch_builder											   		   *
     ***************************************************************************/



	/* Root Files */


	/**
	 * Pushes the root files from the local machine to a selected server.
	 * @param server?: object
	 * @returns Promise<string>
	 */
	async push_root_files(server = undefined) {
		// Check if the server has been provided
		server = server ? server: await this.cluster_input.server(false, false, true, true); 

		// Initialize the root path in case it hadn't been
		await this.cluster_command.init_root_path(server);

		// Push the requirements.txt file
		console.log(`\n1/2) REQUIREMENTS.TXT:`);
		await this.push_file(this.cluster_path.requirements(true), this.cluster_path.requirements(false), server);

		// Push the package.json file
		console.log(`\n\n2/2) PACKAGE.JSON:`);
		await this.push_file(this.cluster_path.package_json(true), this.cluster_path.package_json(false), server);
	}






	/* Root Directories */



	/**
	 * Pushes the config directory from the local machine to a selected server.
	 * @param server?: object
	 * @returns Promise<string>
	 */
	push_configuration(server = undefined) {
		return this.push(this.cluster_path.config(true), this.cluster_path.path(false), this.cluster_path.config(false), server);
	}





	/**
	 * Pushes the candlesticks directory from the local machine to a selected server.
	 * @param server?: object
	 * @returns Promise<string>
	 */
	push_candlesticks(server = undefined) {
		return this.push(this.cluster_path.candlesticks(true), this.cluster_path.path(false), this.cluster_path.candlesticks(false), server);
	}




	/**
	 * Pushes the dist directory from the local machine to a selected server.
	 * @param server?: object
	 * @returns Promise<string>
	 */
	push_dist(server = undefined) {
		return this.push(this.cluster_path.dist(true), this.cluster_path.path(false), this.cluster_path.dist(false), server);
	}




	
	
    /* Epoch Directories */







	/**
	 * Pushes the regression training configs directory from the local machine to a selected server.
	 * @param server?: object
	 * @returns Promise<string>
	 */
	push_regression_training_configs(server = undefined) {
		return this.push(
			this.cluster_path.regression_training_configs(true), 
			this.cluster_path.regression_training_configs(false), 
			this.cluster_path.regression_training_configs(false), 
			server
		);
	}






	/* Full Push */



	/**
	 * Asks the user of the destination server and then performs a push on each
	 * directory within the epoch-builder, including the current Epoch. 
	 * @param server?: object
	 * @param trainable_model_type?: string
	 * @returns Promise<string>
	 */
	async push_epoch_builder() {
		// Retrieve the server
		const server = await this.cluster_input.server(false, false, true, true);

		// Push the root files
		console.log(`\nROOT FILES`);
		await this.push_root_files(server);

		// Push the root directories
		console.log(`\n\nROOT DIRECTORIES`);

		// Push the configurations
		console.log(`\nCONFIGURATION:`);
		await this.push_configuration(server);

		// Push the candlesticks
		console.log(`\nCANDLESTICKS:`);
		await this.push_candlesticks(server);

		// Push the dist
		console.log(`\nDIST:`);
		await this.push_dist(server);

		// Push the epoch directories
		console.log(`\n\nEPOCH DIRECTORIES`);
		await this.cluster_command.init_epoch_path(server);
		await this.push_regression_training_configs(server);
	}







	/* Push Handlers */





	/**
	 * Executes a push action on a file based on the provided parameters.
	 * @param origin: string 
	 * @param destination: string 
	 * @param server?: object 
	 * @param print_payload?: boolean
	 * @returns Promise<string>
	 */
	async push_file(origin, destination, server = undefined, print_payload = false) {
		// Retrieve the server in case it wasn't provided
		server = server ?  server : await this.cluster_input.server(false, false, true, true);

		// Removing the directory on the server if exists
		console.log(`\n1/2) Removing ${this.cluster_command.ssh_addr(server)}:${destination}...`);
		try { await this.cluster_command.remove_server_file(server, destination) } catch (e) { }
		
		// Push the directory
		console.log(`2/2) Pushing ${origin}...`);
		const push_payload = await this.cluster_command.push_file(server, origin, destination);
		if (print_payload) console.log(push_payload);
		return push_payload;
	}




	/**
	 * Executes a push action on a directory based on the provided parameters.
	 * @param origin: string 
	 * @param destination: string 
	 * @param destination_dir_path: string
	 * @param server?: object 
	 * @param print_payload?: boolean
	 * @returns Promise<string>
	 */
	async push(origin, destination, destination_dir_path, server = undefined, print_payload = false) {
		// Retrieve the server in case it wasn't provided
		server = server ?  server : await this.cluster_input.server(false, false, true, true);

		// Removing the directory on the server if exists
		console.log(`\n1/2) Removing ${this.cluster_command.ssh_addr(server)}:${destination_dir_path}...`);
		try { await this.cluster_command.remove_server_dir(server, destination_dir_path) } catch (e) { }
		
		// Push the directory
		console.log(`2/2) Pushing ${origin}...`);
		const push_payload = await this.cluster_command.push_dir(server, origin, destination);
		if (print_payload) console.log(push_payload);
		return push_payload;
	}










	/***************************************************************************
     * PULL												   					   *
     * Pull stands for the transfer of data from a server within the cluster   * 
	 * or hosted by a third party to the local machine. Some pull actions 	   *
	 * clean the server's directory once the data has been transferred. 	   *
	 * 																		   *
	 * Processes:															   *
	 * 	pull_trained_regressions											   *
     ***************************************************************************/





	/**
	 * Pulls all the batched training certificates as well as the trained regression
	 * files with individual certificates.
	 * @returns Promise<void>
	 */
	async pull_trained_regressions() {
		// Retrieve the server which data will be pulled from
		const server = await this.cluster_input.server(false, false, true, true);
		
		// Pull & Clean the Batched Training Certificates
		console.log("\n1/2) REGRESSION BATCHED CERTIFICATES:");
		await this.pull(
			this.cluster_path.regression_batched_certificates(false), 
			this.cluster_path.epoch_path(true), 
			server, 
			true
		);

		// Pull & Clean the Trained Regressions
		console.log("\n\n2/2) TRAINED REGRESSIONS:");
		await this.pull(
			this.cluster_path.regressions(false), 
			this.cluster_path.epoch_path(true), 
			server, 
			true
		);
	}








	/* Pull Handlers */


	
	/**
	 * Executes a push action on a directory based on the provided parameters.
	 * @param origin: string 
	 * @param destination: string 
	 * @param server?: object 
	 * @param clean_origin_on_complete?: boolean 
	 * @param print_payload?: boolean
	 * @returns Promise<string>
	 */
	async pull(origin, destination, server = undefined, clean_origin_on_complete = false, print_payload = false) {
		// Retrieve the server in case it wasn't provided
		server = server ?  server : await this.cluster_input.server(false, false, true, true);

		// Pull the directory
		console.log(`\n1/2) Pulling ${this.cluster_command.ssh_addr(server)}:${origin}...`);
		const pull_payload = await this.cluster_command.pull_dir(server, origin, destination);

		// Clean the origin directory if applies
		if (clean_origin_on_complete) {
			console.log(`2/2) Cleaning origin...`);
			await this.cluster_command.clean_server_dir(server, origin);
		} else {
			console.log(`2/2) Cleaning origin: Skipped`);
		}
		
		// Finally, return the payload
		if (print_payload) console.log(pull_payload);
		return pull_payload;
	}
}




// Export the modules
export { Cluster };