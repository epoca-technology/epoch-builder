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
        this.cluster_command = new ClusterCommand(this.config.ssh_private_key_path);

		// Initialize the Cluster Server Instance
		this.cluster_server = new ClusterServer(this.cluster_command, this.config.servers);

		// Initialize the Cluster Input Instance
		this.cluster_input = new ClusterInput(this.cluster_server, this.cluster_path, this.config.trainable_model_types);
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
				"install_ssh_key_on_a_server"
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
				"push_dist",
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
				"pull_trained_models",
				"pull_regression_selection",
				"pull_classification_training_data",
				"pull_backtest_results",
				"pull_database_management"
			]
		});

		// Finally, execute the process
		await this[process_id]();
	}






	
	/**
     * Server
     * All server related processes as well as the helpers.
     */





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
            for (let s of await this.cluster_server.list_servers(true)) { 
				console.log(`\n\n${s.name}:\n`);
				console.log(await this.cluster_command.get_server_status(s));
			} 
        }

		// Otherwise, print the specific server
		else { console.log(" "); console.log(await this.cluster_command.get_server_status(server)); }
	}






    // subscribe_to_server_logs



    // reboot_server



	// shutdown_server





	/**
	 * Installs the SSH Public Key on a selected server.
	 * @returns Promise<void>
	 */
	async install_ssh_key_on_a_server() { await this.cluster_command.install_ssh_key(await this.cluster_input.server(false, false, true)) }







	/**
     * Epoch Builder
     * All epoch builder related processes as well as the helpers.
     */



    // hyperparams


    // regression_training




    // regression_selection


    // classification_training_data


    // classification_training


    // backtest


    // merge_training_certificates


    // epoch_management


    // database_management


    // unit_tests










	/***************************************************************************
     * PUSH												   					   *
     * Push stands for the transfer of data from the local machine to a server * 
	 * that can live in the cluster or anywhere.							   *
	 * 																		   *
	 * Processes:															   *
	 * 	push_configuration													   *
	 * 	push_database_management											   *
	 * 	push_candlesticks											           *
	 * 	push_dist											                   *
	 * 	push_active_models											           *
	 * 	push_regression_selection											   *
	 * 	push_classification_training_data									   *
	 * 	push_training_configurations									       *
	 * 	push_backtest_configurations									       *
	 * 	push_backtest_results									      		   *
	 * 	push_epoch									      		   			   *
	 * 	push_epoch_builder									      		   	   *
     ***************************************************************************/



	/* Root Directories */


	/**
	 * Pushes the config directory from the local machine to a selected server.
	 * @param server?: object
	 * @returns Promise<string>
	 */
	push_configuration(server = undefined) {
		return this.push(this.cluster_path.config(true), this.cluster_path._path(false), this.cluster_path.config(false), server);
	}




	/**
	 * Pushes the db_management directory from the local machine to a selected server.
	 * @param server?: object
	 * @returns Promise<string>
	 */
	push_database_management(server = undefined) {
		return this.push(this.cluster_path.db_management(true), this.cluster_path._path(false), this.cluster_path.db_management(false), server);
	}





	/**
	 * Pushes the candlesticks directory from the local machine to a selected server.
	 * @param server?: object
	 * @returns Promise<string>
	 */
	push_candlesticks(server = undefined) {
		return this.push(this.cluster_path.candlesticks(true), this.cluster_path._path(false), this.cluster_path.candlesticks(false), server);
	}




	/**
	 * Pushes the dist directory from the local machine to a selected server.
	 * @param server?: object
	 * @returns Promise<string>
	 */
	push_dist(server = undefined) {
		return this.push(this.cluster_path.dist(true), this.cluster_path._path(false), this.cluster_path.dist(false), server);
	}




	
    /* Epoch Directories */



	/**
	 * Pushes the active models directory from the local machine to a selected server.
	 * @param server?: object
	 * @returns Promise<string>
	 */
	push_active_models(server = undefined) {
		return this.push(this.cluster_path.models(true), this.cluster_path._epoch_path(false), this.cluster_path.models(false), server);
	}


	/**
	 * Pushes the regression selection directory from the local machine to a selected server.
	 * @param server?: object
	 * @returns Promise<string>
	 */
	push_regression_selection(server = undefined) {
		return this.push(
			this.cluster_path.regression_selection(true), 
			this.cluster_path._epoch_path(false), 
			this.cluster_path.regression_selection(false), 
			server
		);
	}



	/**
	 * Pushes the classification training data directory from the local machine to a selected server.
	 * @param server?: object
	 * @returns Promise<string>
	 */
	push_classification_training_data(server = undefined) {
		return this.push(
			this.cluster_path.classification_training_data(true), 
			this.cluster_path._epoch_path(false), 
			this.cluster_path.classification_training_data(false), 
			server
		);
	}



    // push_training_configurations



	/**
	 * Pushes the backtest configurations directory from the local machine to a selected server.
	 * @param server?: object
	 * @returns Promise<string>
	 */
	push_backtest_configurations(server = undefined) {
		return this.push(
			this.cluster_path.backtest_configurations(true), 
			this.cluster_path._epoch_path(false), 
			this.cluster_path.backtest_configurations(false), 
			server
		);
	}



	/**
	 * Pushes the backtest results directory from the local machine to a selected server.
	 * @param server?: object
	 * @returns Promise<string>
	 */
	push_backtest_results(server = undefined) {
		return this.push(
			this.cluster_path.backtest_results(true), 
			this.cluster_path._epoch_path(false), 
			this.cluster_path.backtest_results(false), 
			server
		);
	}


    // push_epoch


    // push_epoch_builder






	/**
	 * Executes a push action based on the provided parameters.
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
		console.log(`\n1/2) Removing ${this.cluster_command.ssh_addr(server)}@${destination_dir_path}...`);
		try { await this.cluster_command.remove_server_dir(server, destination_dir_path) } catch (e) { }
		
		// Push the directory
		console.log(`2/2) Pushing ${origin}...`);
		const push_payload = await this.cluster_command.push_dir(server, origin, destination);
		if (print_payload) console.log(push_payload);
		return push_payload;
	}









	/**
     * Pull
     * All pull related processes as well as the helpers.
     */





	/**
	 * Pulls all the batched training certificates and the models. It also cleans
	 * the server's directories on completion.
	 * @returns Promise<void>
	 */
	 async pull_trained_models() {
		// Retrieve the server which data will be pulled from
		const server = await this.cluster_input.server(false, false, true, true);

		// Retrieve the type of model
		const trainable_model_type = await this.cluster_input.trainable_model_type("all");
		
		// Pull & Clean the Batched Training Certificates
		console.log(`\n1/4) Pulling ${trainable_model_type} batched training certificates...`);
		const btc_origin = this.cluster_path.batched_training_certificates(false, trainable_model_type);
		const btc_destination = this.cluster_path.batched_training_certificates(true);
		const btc_pull_payload = await this.cluster_command.pull_dir(server, btc_origin, btc_destination);

		console.log(`2/4) Cleaning ${trainable_model_type} batched training certificates...`);
		await this.cluster_command.clean_server_dir(server, btc_origin);

		// Pull & Clean the Models Bank
		console.log(`3/4) Pulling ${trainable_model_type} models...`);
		const models_bank_origin = this.cluster_path.models_bank(false, trainable_model_type);
		const models_bank_destination = this.cluster_path.models_bank(true);
		const models_bank_pull_payload = await this.cluster_command.pull_dir(server, models_bank_origin, models_bank_destination);

		console.log(`4/4) Cleaning ${trainable_model_type} models bank...`);
		await this.cluster_command.clean_server_dir(server, models_bank_origin);
	}




	/**
	 * Pulls the database management directory from a selected server
	 * @returns Promise<void>
	 */
	 async pull_database_management() {
		// Retrieve the server which data will be pulled from
		const server = await this.cluster_input.server(false, false, true, true);
		
		// Pull the database management dir
		console.log(`\n1/2) Pulling the database management directory...`);
		const origin = this.cluster_path.db_management(false);
		const destination = this.cluster_path._path(true);
		const pull_payload = await this.cluster_command.pull_dir(server, origin, destination);

		// Clean the directory
		console.log(`2/2) Cleaning the database management directory...`);
		await this.cluster_command.remove_server_dir(server, origin);
	}


    // pull_regression_selection


    // pull_classification_training_data


    // pull_backtest_results



	

}




// Export the modules
export { Cluster };