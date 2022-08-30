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
				"kill_process",
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
				"push_root_files",
				"push_configuration",
				"push_database_management",
				"push_candlesticks",
				"push_dist",
				"push_active_models",
				"push_regression_selection",
				"push_classification_training_data",
				"push_training_configurations",
				"push_backtest_configurations",
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
            for (let s of await this.cluster_server.list_servers(true)) { 
				console.log(`\n\n${s.name}:\n`);
				console.log(await this.cluster_command.get_server_status(s));
			} 
        }

		// Otherwise, print the specific server
		else { console.log(" "); console.log(await this.cluster_command.get_server_status(server)); }
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

		// Check if it has to shutdown all servers in the cluster
		if (server.name == "all") { 
			// Shutdown all the servers that are online and are a part of the cluster
            for (let s of await this.cluster_server.list_servers()) { 
				if (s.is_cluster && s.is_online) {
					console.log(`\nShutting ${server.name} down...`);
					await this.cluster_command.shutdown(a);
				}
			} 
        }

		// Otherwise, shutdown the specific server
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
	 * 	hyperparams													   				*
	 * 	regression_training											   				*
	 * 	regression_selection											           	*
	 * 	classification_training_data											    *
	 * 	classification_training											           	*
	 * 	backtest											   						*
	 * 	merge_training_certificates									   				*
	 * 	epoch_management									       					*
	 * 	database_management									       					*
	 * 	unit_tests									      		   			   		*
     ********************************************************************************/





	/**
	 * Builds the hyperparams based on the provided config on the selected 
	 * server.
	 * @returns Promise<void>
	 */
	 async hyperparams() { 
		// Retrieve the server
		const server = await this.cluster_input.server(true, false, true, true);

		// Retrieve and unpack the category and the config file
		const { model_type, training_data_file_name, batch_size } = await this.cluster_input.hyperparams();

		// Finally, Run the command
		await this.cluster_command.hyperparams(server, model_type, training_data_file_name, batch_size); 
	}





	/**
	 * Initializes the training process for a selected model type and server.
	 * @returns Promise<void>
	 */
	async regression_training() { 
		// Retrieve the server
		const server = await this.cluster_input.server(true, false, true, true);

		// Retrieve the model type
		const trainable_model_type = await this.cluster_input.trainable_model_type("regression");

		// Retrieve and unpack the category and the config file
		const { category, config_file_name } = await this.cluster_input.training_config(trainable_model_type);

		// Finally, Run the command
		await this.cluster_command.regression_training(server, trainable_model_type, category, config_file_name); 
	}




	/**
	 * Initializes the regression selection process for the selected list of models and server.
	 * @returns Promise<void>
	 */
	async regression_selection() { 
		// Retrieve the server
		const server = await this.cluster_input.server(true, false, true, true);

		// Retrieve the model ids string
		const model_ids = await this.cluster_input.selected_model_ids();

		// Finally, Run the command
		await this.cluster_command.regression_selection(server, model_ids); 
	}





	/**
	 * Initializes the classification training data process for the selected server.
	 * @returns Promise<void>
	 */
	async classification_training_data() { 
		// Retrieve the server
		const server = await this.cluster_input.server(true, false, true, true);

		// Retrieve the configuration values
		const {
			regression_selection_file_name,
			description,
			steps,
			include_rsi,
			include_aroon
		} = await this.cluster_input.classification_training_data();

		// Finally, Run the command
		await this.cluster_command.classification_training_data(
			server, 
			regression_selection_file_name,
			description,
			steps,
			include_rsi,
			include_aroon
		); 
	}





	/**
	 * Initializes the training process for a selected model type and server.
	 * @returns Promise<void>
	 */
	async classification_training() { 
		// Retrieve the server
		const server = await this.cluster_input.server(true, false, true, true);

		// Retrieve the model type
		const trainable_model_type = await this.cluster_input.trainable_model_type("classification");

		// Retrieve and unpack the category and the config file
		const { category, config_file_name } = await this.cluster_input.training_config(trainable_model_type);

		// Finally, Run the command
		await this.cluster_command.classification_training(server, trainable_model_type, category, config_file_name); 
	}





	/**
	 * Initializes the backtest process on a selected server.
	 * @returns Promise<void>
	 */
	async backtest() { 
		// Retrieve the server
		const server = await this.cluster_input.server(true, false, true, true);

		// Retrieve the backtest config file name
		const config_file_name = await this.cluster_input.backtest_config();

		// Finally, Run the command
		await this.cluster_command.backtest(server, config_file_name); 
	}



    

	/**
	 * Merges the training certificates for a selected model type and server.
	 * @returns Promise<void>
	 */
	async merge_training_certificates() { 
		// Retrieve the server
		const server = await this.cluster_input.server(true, false, true, true);

		// Retrieve the model type
		const trainable_model_type = await this.cluster_input.trainable_model_type("all");

		// Finally, Run the command
		await this.cluster_command.merge_training_certificates(server, trainable_model_type); 
	}





	/**
	 * Runs the Epoch Builder's Epoch Management tool on a selected server.
	 * @returns Promise<void>
	 */
	async epoch_management() { 
		// Retrieve the server
		const server = await this.cluster_input.server(true, false, true, true);

		// Retrieve the epoch management args
		const args = await this.cluster_input.epoch_management();

		// Finally, Run the command
		await this.cluster_command.epoch_management(server, args); 
	}





	/**
	 * Runs the Epoch Builder's Database Management tool on a selected server.
	 * @returns Promise<void>
	 */
	async database_management() { 
		// Retrieve the server
		const server = await this.cluster_input.server(true, false, true, true);

		// Retrieve the db management args
		const { action, ip } = await this.cluster_input.db_management(server);

		// Finally, Run the command
		await this.cluster_command.db_management(server, action, ip); 
	}





	/**
	 * Runs the unit tests for a selected server.
	 * @returns Promise<void>
	 */
	async unit_tests() { 
		// Retrieve the server
		const server = await this.cluster_input.server(true, false, true, true);

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
	 * 	push_database_management											   *
	 * 	push_candlesticks											           *
	 * 	push_dist											                   *
	 * 	push_active_models											           *
	 * 	push_regression_selection											   *
	 * 	push_classification_training_data									   *
	 * 	push_training_configurations									       *
	 * 	push_backtest_configurations									       *
	 * 	push_epoch									      		   			   *
	 * 	push_epoch_builder									      		   	   *
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





	/**
	 * Pushes the training configurations directory from the local machine to a selected server.
	 * Asks the user for the server and the trainable model type if they werent provided.
	 * @param server?: object
	 * @param trainable_model_type?: string
	 * @returns Promise<string>
	 */
	async push_training_configurations(server = undefined, trainable_model_type = undefined) {
		// Check if the server has been provided
		server = server ? server: await this.cluster_input.server(false, false, true, true); 

		// Check if the type of model has been provided
		trainable_model_type = typeof trainable_model_type == "string" ? trainable_model_type: 
			await this.cluster_input.trainable_model_type("all")

		// Finally, perform the push
		return this.push(
			this.cluster_path.training_configs(true, trainable_model_type), 
			this.cluster_path.training_configs(false), 
			this.cluster_path.training_configs(false, trainable_model_type), 
			server
		);
	}





	/**
	 * Pushes the backtest configurations directory from the local machine to a selected server.
	 * @param server?: object
	 * @returns Promise<string>
	 */
	push_backtest_configurations(server = undefined) {
		return this.push(
			this.cluster_path.backtests(true, "configurations"), 
			this.cluster_path.backtests(false), 
			this.cluster_path.backtests(false, "configurations"), 
			server
		);
	}






	/**
	 * Asks the user of the destination server and then performs a push on each
	 * epoch directory.
	 * @param server?: object
	 * @returns Promise<void>
	 */
	async push_epoch(server = undefined) {
		// Check if the server has been provided
		server = server ? server: await this.cluster_input.server(false, false, true, true); 

		// Initialize the epoch path in case it hadn't been
		await this.cluster_command.init_epoch_path(server);

		// Push the active models
		console.log(`\nACTIVE MODELS:`);
		await this.push_active_models(server);

		// Push the regression selection
		console.log(`\n\nREGRESSION SELECTION:`);
		await this.push_regression_selection(server);

		// Push the classification training data
		console.log(`\n\nCLASSIFICATION TRAINING DATA:`);
		await this.push_classification_training_data(server);

		// Push the training configurations
		console.log(`\n\nTRAINING CONFIGURATION:`);
		for (var model_type of this.config.trainable_model_types.all) {
			await this.push_training_configurations(server, model_type);
		}

		// Push the backtest configurations
		console.log(`\n\nBACKTEST CONFIGURATION:`);
		await this.push_backtest_configurations(server);
	}






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

		// Push the db_management if applies
		if (FileSystem.dir_exists(this.cluster_path.db_management(true))) {
			console.log(`\nDATABASE MANAGEMENT:`);
			await this.push_database_management(server)
		} else {
			console.log(`\nDATABASE MANAGEMENT: Skipped`);
		}

		// Push the candlesticks
		console.log(`\nCANDLESTICKS:`);
		await this.push_candlesticks(server);

		// Push the dist
		console.log(`\nDIST:`);
		await this.push_dist(server);

		// Push the epoch directories
		console.log(`\n\nEPOCH DIRECTORIES`);
		await this.push_epoch(server);
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
	 * 	pull_trained_models													   *
	 * 	pull_regression_selection											   *
	 * 	pull_classification_training_data									   *
	 * 	pull_backtest_results											       *
	 * 	pull_database_management											   *
     ***************************************************************************/





	/**
	 * Pulls all the batched training certificates and the models bank. It also cleans
	 * the server's directories on completion.
	 * @returns Promise<void>
	 */
	async pull_trained_models() {
		// Retrieve the server which data will be pulled from
		const server = await this.cluster_input.server(false, false, true, true);

		// Retrieve the type of model
		const trainable_model_type = await this.cluster_input.trainable_model_type("all");
		
		// Pull & Clean the Batched Training Certificates
		console.log("\n1/2) BATCHED TRAINING CERTIFICATES:");
		await this.pull(
			this.cluster_path.batched_training_certificates(false, trainable_model_type), 
			this.cluster_path.batched_training_certificates(true), 
			server, 
			true
		);

		// Pull & Clean the Models Bank
		console.log("\n\n2/2) MODELS BANK:");
		await this.pull(
			this.cluster_path.models_bank(false, trainable_model_type), 
			this.cluster_path.models_bank(true), 
			server, 
			true
		);
	}



	/**
	 * Pulls the regression selection directory from a selected server.
	 * @returns Promise<void>
	 */
	pull_regression_selection() {
		return this.pull(this.cluster_path.regression_selection(false), this.cluster_path._epoch_path(true));
	}





	/**
	 * Pulls the classification training data directory from a selected server.
	 * @returns Promise<void>
	 */
	pull_classification_training_data() {
		return this.pull(this.cluster_path.classification_training_data(false), this.cluster_path._epoch_path(true));
	}




	/**
	 * Pulls the backtest results directory from a selected server.
	 * @returns Promise<void>
	 */
	pull_backtest_results() {
		return this.pull(this.cluster_path.backtests(false, "results"), this.cluster_path.backtests(true), undefined, true);
	}




	/**
	 * Pulls the database management directory from a selected server.
	 * @returns Promise<void>
	 */
	pull_database_management() {
		return this.pull(this.cluster_path.db_management(false), this.cluster_path._path(true), undefined, true);
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