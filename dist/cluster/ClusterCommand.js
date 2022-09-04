import { spawn } from "child_process";





/**
 * Cluster Command
 * This class handles the execution of commands in local and external servers.
 * 
 * Instance Properties
 * 	ssh_private_key_path: string
 */
 class ClusterCommand {


	/**
	 * Initializes a Cluster Command Instance
	 * @param ssh_private_key_path: string
	 * @param cluster_path: ClusterPath
	 */
	constructor(ssh_private_key_path, cluster_path) { 
		// Init the SSH PK Path
		this.ssh_private_key_path = ssh_private_key_path;

		// Init the instance of the cluster path
		this.cluster_path = cluster_path;
	}







	/* Server Helpers & Processes */






	/**
	 * Verifies if a given server is currently online with a ping.
	 * @param ip: string
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
	 * @param server: object
	 * @returns boolean|undefined
	 */
	async is_server_available(is_online, server) {
		// Make sure the server is online
		if (is_online) {
			// Attempt to retrieve the state of the server
			try {
				// Retrieve the pid, if it exists, it means the server is unavailable
				const pid = await this.get_process_id(server);

				// If it exists, it means the server is unavailable.
				return pid == undefined;
			} catch (e) { console.error(e); return undefined; }
		} 
		
		// Otherwise, return undefined
		else { return undefined }
	}






	/**
	 * Attempts to extract the python3 process id from a server. If found, it will return it
	 * in string format. Otherwise, returns undefined.
	 * @param server: object
	 * @returns Promise<string|undefined>
	 */
	async get_process_id(server) {
		// Init the payload
		let payload = "";

		// Check if the server is localhost
		if (server.name == "localhost") { payload = await this.execute("ps", ["aux"], "pipe") }

		// Otherwise, check the server status with ssh
		else { 
			payload = await this.execute("ssh", this.ssh_args([server, "ps", "aux"]), "pipe") 
		}

		// Check if a payload was successfully retrieved
		if (typeof payload == "string" && payload.length) {
			// Extract the line that contains the process
			const process = payload.split("\n").filter((line) => { 
				return (line.includes("python3 ./dist") && !line.includes("ssh")) || 
					   (line.includes("python3 dist") && !line.includes("ssh"))
			});
			
			// If there is exactly one match, extract the pid
			if (process.length) {
				// Split the process into single elements
				const process_items = process[0].split(" ");

				// Iterate over each item the first valid number is found (PID)
				var i = 0;
				var pid = undefined;
				while (pid == undefined && i < process_items.length) {
					// Check if the current item is a valid number
					if (Number(process_items[i])) pid =  process_items[i];

					// Increment the counter
					i += 1;
				}

				// Finally, return the process id
				return pid;
			} 
			
			// Otherwise, no process was found
			else { return undefined }
		}

		// Otherwise, return undefined
		else { return undefined }
	}






    /**
     * Prints the server status (landscape-sysinfo) from a given server.
     * @param server: object
     * @returns Promise<string>
     */
    async get_server_status(server) {
        // Init the payload
        var payload = "";

        // Check if the server is localhost
        if (server.name == "localhost") { payload = await this.execute("landscape-sysinfo", [], "pipe") }
        
        // Otherwise, retrieve the specific server
        else {
			payload = await this.execute("ssh", this.ssh_args([server, "landscape-sysinfo"]), "pipe");
        }

        // Finally, return the payload
		return payload;
    }





	/**
	 * Establishes a SSH Connection with a given server
	 * @param server
	 * @returns Promise<void>
	 */
	async connect(server) { await this.execute("ssh", this.ssh_args([server]), "inherit") }






	/**
	 * Subscribes to server logs on a given path.
	 * @param server
	 * @param logs_path
	 * @returns Promise<void>
	 */
	async subscribe_to_logs(server, logs_path) {
        // Check if the server is localhost
        if (server.name == "localhost") { throw new Error("The command subscribe_to_logs is not supported on localhost.") }
        
        // Otherwise, subscribe to the specific server
        else {
			await this.execute("ssh", this.ssh_args([server, "tail", "-n", "500000", "-f", logs_path]), "inherit");
        }
    }




	/**
	 * Reboots a given server.
	 * @param server
	 * @returns Promise<void>
	 */
	 async reboot(server) {
        // Check if the server is localhost
        if (server.name == "localhost") { await this.execute("sudo", ["reboot"], "inherit") }
        
        // Otherwise, subscribe to the specific server
        else {
			await this.execute("ssh", this.ssh_args(["-t", server, "sudo", "reboot"]), "inherit");
        }
    }





	/**
	 * Turns a given server off.
	 * @param server
	 * @returns Promise<void>
	 */
	 async shutdown(server) {
        // Check if the server is localhost
        if (server.name == "localhost") { await this.execute("sudo", ["poweroff"], "inherit") }
        
        // Otherwise, subscribe to the specific server
        else {
			await this.execute("ssh", this.ssh_args(["-t", server, "sudo", "poweroff"]), "inherit");
        }
    }






	/**
	 * Kills the python3 process on a given server. It throws an error if
	 * the process is not currently running
	 * @param server
	 * @returns Promise<void>
	 */
	async kill_process(server) {
		// Retrieve the process id (if any)
		let pid = await this.get_process_id(server);
		if (typeof pid != "string") { throw new Error(`The python3 process is not running on ${server.name}`) }

        // Check if the server is localhost
        if (server.name == "localhost") { await this.execute("kill", ["-9", pid], "pipe") }
        
        // Otherwise, subscribe to the specific server
        else {
			// Kill the process
			await this.execute("ssh", this.ssh_args([server, "kill", "-9", pid]), "pipe");

			// When a process is executed with nohup, a second process is created
			pid = await this.get_process_id(server);
			if (typeof pid == "string") await this.execute("ssh", this.ssh_args([server, "kill", "-9", pid]), "pipe");
        }
    }







	/**
	 * Installs the SSH Public Key on a given server.
	 * @param server 
	 * @returns Promise<void>
	 */
	async install_ssh_key(server) { await this.execute("ssh-copy-id", this.ssh_args([server]), "inherit") }









	/* Epoch Builder Processes */





	/**
	 * Runs the Hyperparams Process.
	 * @param server: object 
	 * @param model_type: string
	 * @param training_data_file_name: string
	 * @param batch_size: string
	 * @returns Promise<void>
	 */
	hyperparams(server, model_type, training_data_file_name, batch_size) {
		return this.execute_eb(
			server, "hyperparams.py",
			[
				"--model_type", model_type, 
				"--training_data_file_name", training_data_file_name, 
				"--batch_size", batch_size
			]
		);
	}




	/**
	 * Initializes the Regression Training Process. If running on localhost, it will
	 * run it in inherited mode. Otherwise, it will run it in deatached mode.
	 * @param server: object 
	 * @param trainable_model_type: string
	 * @param hyperparams_category: string
	 * @param config_file_name: string
	 * @returns Promise<void>
	 */
	regression_training(server, trainable_model_type, hyperparams_category, config_file_name) {
		return this.execute_eb(
			server, "regression_training.py",
			[
				"--model_type", trainable_model_type, 
				"--hyperparams_category", hyperparams_category, 
				"--config_file_name", config_file_name
			],
			true
		);
	}




	/**
	 * Runs the Regression Selection Process.
	 * @param server: object 
	 * @param model_ids: string
	 * @returns Promise<void>
	 */
	regression_selection(server, model_ids) {
		return this.execute_eb(
			server, "regression_selection.py",
			[
				"--model_ids", model_ids
			]
		);
	}




	/**
	 * Initializes the Classification Training Data Process. If running on localhost, it will
	 * run it in inherited mode. Otherwise, it will run it in deatached mode.
	 * @param server: object 
	 * @param regression_selection_file_name: string
	 * @param description: string
	 * @param steps: string
	 * @param include_rsi: string
	 * @param include_aroon: string
	 * @returns Promise<void>
	 */
	classification_training_data(
		server,
		regression_selection_file_name, 
		description, 
		steps, 
		include_rsi, 
		include_aroon
	) {
		return this.execute_eb(
			server, "classification_training_data.py",
			[
				"--regression_selection_file_name", regression_selection_file_name,
				"--description", `'${description}'`,
				"--steps", steps,
				"--include_rsi", include_rsi,
				"--include_aroon", include_aroon
			],
			true
		);
	}





	/**
	 * Initializes the Classification Training Process. If running on localhost, it will
	 * run it in inherited mode. Otherwise, it will run it in deatached mode.
	 * @param server: object 
	 * @param trainable_model_type: string
	 * @param hyperparams_category: string
	 * @param config_file_name: string
	 * @returns Promise<void>
	 */
	classification_training(server, trainable_model_type, hyperparams_category, config_file_name) {
		return this.execute_eb(
			server, "classification_training.py",
			[
				"--model_type", trainable_model_type, 
				"--hyperparams_category", hyperparams_category, 
				"--config_file_name", config_file_name
			],
			true
		);
	}





	/**
	 * Initializes the Backtest Process. If running on localhost, it will
	 * run it in inherited mode. Otherwise, it will run it in deatached mode.
	 * @param server: object 
	 * @param config_file_name: string
	 * @returns Promise<void>
	 */
	 backtest(server, config_file_name) {
		return this.execute_eb(
			server, "backtest.py",
			[
				"--config_file_name", config_file_name
			],
			true
		);
	}





	/**
	 * Merges the training certificates from the model's that went through the entire process.
	 * Later, the models are moved from the active folder into the bank and finally, updates
	 * the configuration file in the root config directory so the training can be resumed.
	 * @param server: object
	 * @param trainable_model_type: string
	 * @returns Promise<void>
	 */
	merge_training_certificates(server, trainable_model_type) {
		return this.execute_eb(
			server, "merge_training_certificates.py",
			[
				"--model_type", trainable_model_type
			]
		);
	}






	/**
	 * Runs the Epoch Management Process.
	 * @param server: object 
	 * @param args
	 * @returns Promise<void>
	 */
	epoch_management(server, args) {
		return this.execute_eb(
			server, "epoch_management.py",
			[
				"--action", args.action,
				"--id", args.id,
				"--epoch_width", args.epoch_width,
				"--seed", args.seed,
				"--train_split", args.train_split,
				"--backtest_split", args.backtest_split,
				"--regression_lookback", args.regression_lookback,
				"--regression_predictions", args.regression_predictions,
				"--model_discovery_steps", args.model_discovery_steps,
				"--idle_minutes_on_position_close", args.idle_minutes_on_position_close,
				"--training_data_file_name", args.training_data_file_name,
				"--model_ids", args.model_ids
			]
		);
	}






	/**
	 * Runs the DB Management Process.
	 * @param server: object 
	 * @param action: string
	 * @param ip: string
	 * @returns Promise<void>
	 */
	db_management(server, action, ip) {
		return this.execute_eb(
			server, "db_management.py",
			[
				"--action", action,
				"--ip", ip
			]
		);
	}







	/**
	 * Runs the unit tests for a given server.
	 * @param server: object
	 * @returns Promise<void>
	 */
	async unit_tests(server) {
        // Check if the server is localhost
        if (server.name == "localhost") { 
			await this.execute(
				"python3", 
				["-m", "unittest", "discover", "-s", this.cluster_path.unit_tests(true), "-p", "*_test.py"], 
				"inherit"
			); 
		}
        
        // Otherwise, subscribe to the specific server
        else {
			await this.execute("ssh", this.ssh_args([
				server, 
				"cd", "epoch-builder", "&&", 
				"python3", "-m", "unittest", "discover", "dist", "-p", "*_test.py"
			]), "inherit");
        }
	}









	/* Epoch Builder Endpoint Execution */



	/**
	 * Executes an Epoch Builder endpoint on a given server.
	 * If it is not localhost, the process can be started in
	 * nohup mode.
	 * @param server: object
	 * @param endpoint: string
	 * @param args: string[]
	 * @param nohup_process: boolean
	 * @returns Promise<void> 
	 */
	 async execute_eb(server, endpoint, args = [], nohup_process = false) {
		// Sanitize the args
		args = args.map((a) => { return typeof a == "string" ? a: ""});

        // Check if the server is localhost
        if (server.name == "localhost") { 
			await this.execute("python3", [`dist/${endpoint}`].concat(args), "inherit")
		}
        
        // Otherwise, execute the process on the specific server
        else {
			// Execute a nohup process
			if (nohup_process) {
				// Initialize the logs path
				const logs_path = this.cluster_path.nohup_logs(false);

				// Create the log file in case it doesn't exist
				await this.execute("ssh", this.ssh_args([server, "touch", logs_path]), "pipe");

				// Execute the action without waiting for it to resolve
				this.execute("ssh", this.ssh_args([
					server, 
					"cd", "epoch-builder", "&&", 
					"nohup", `python3 dist/${endpoint} ${args.join(" ")}`, 
					">", "nohup.out", "2>&1", "&", 
				]), "pipe");

				// Finally, subscribe to the logs
				await this.subscribe_to_logs(server, logs_path);
			}

			// Otherwise, execute a normal process
			else {
				await this.execute("ssh", this.ssh_args([
					server, 
					"cd", "epoch-builder", "&&", 
					"python3", `dist/${endpoint}`
				].concat(args)), "inherit");
			}
        }
	}








	/* Server File Management */







	/**
	 * Pushes a file from the localhost machine to a selected server.
	 * @param server: object
	 * @param origin_path: string
	 * @param destination_path: string
	 * @returns Promise<string|undefined>
	 */
	push_file(server, origin_path, destination_path) {
		return this.execute("scp", this.ssh_args([origin_path, `${this.ssh_addr(server)}:${destination_path}`]), "inherit");
	}




	/**
	 * Removes a file from the server. This function can also be executed
	 * as sudo if specified in the params.
	 * @param server: object
	 * @param path: string
	 * @param as_sudo?: boolean
	 * @returns Promise<string|undefined>
	 */
	remove_server_file(server, path, as_sudo = false) {
		if (as_sudo) {
			return this.execute("ssh", this.ssh_args(["-t", server, "sudo", "rm", path]), "inherit");
		} else {
			return this.execute("ssh", this.ssh_args([server, "rm", path]), "pipe");
		}
	}









	/* Server Directory Management */




	/**
	 * Initializes the root path safely in case it hadn't been.
	 * @param server: object
	 * @returns Promise<void>
	 */
	async init_root_path(server) {
		try {
			await this.execute("ssh", this.ssh_args([server, "mkdir", this.cluster_path._path(false)]), "pipe");
		} catch (e) { }
	}





	/**
	 * Initializes the Epoch's path safely in case it hadn't been.
	 * @param server: object
	 * @returns Promise<void>
	 */
	 async init_epoch_path(server) {
		// The root epoch paths required to operate
		const root_epoch_paths = [
			this.cluster_path._path(false),
			this.cluster_path._epoch_path(false),
			this.cluster_path.backtests(false),
			this.cluster_path.batched_training_certificates(false),
			this.cluster_path.models(false),
			this.cluster_path.models_bank(false),
			this.cluster_path.regression_selection(false),
			this.cluster_path.classification_training_data(false),
			this.cluster_path.training_configs(false)
		];

		// Iterate over each path and create it safely
		for (var path of root_epoch_paths) {
			try {
				await this.execute("ssh", this.ssh_args([server, "mkdir", path]), "inherit");
			} catch (e) { }
		}
	}






	/**
	 * Pushes a directory and its contents from the localhost machine to a selected server.
	 * @param server: object
	 * @param origin_path: string
	 * @param destination_path: string
	 * @returns Promise<string|undefined>
	 */
	async push_dir(server, origin_path, destination_path) {
		return this.execute("scp", this.ssh_args(["-r", origin_path, `${this.ssh_addr(server)}:${destination_path}`]), "inherit");
	}





	/**
	 * Pulls a directory and its contents from a selected server to the localhost machine.
	 * @param server: object
	 * @param origin_path: string
	 * @param destination_path: string
	 * @returns Promise<string|undefined>
	 */
	async pull_dir(server, origin_path, destination_path) {
		return this.execute("scp", this.ssh_args(["-r", `${this.ssh_addr(server)}:${origin_path}`, destination_path]), "inherit");
	}







	/**
	 * Forces the removal of a given directory and then creates a fresh one.
	 * @param server: object
	 * @param path: string
	 * @param as_sudo?: boolean
	 * @returns Promise<void>
	 */
	async clean_server_dir(server, path, as_sudo = false) {
		// Remove the entire directory
		await this.remove_server_dir(server, path, as_sudo);
		
		// Create a brand new directory
		await this.execute("ssh", this.ssh_args([server, "mkdir", path]), "pipe");
	}





	/**
	 * Removes a directory from the server. This function can also be executed
	 * as sudo if specified in the params.
	 * @param server: object
	 * @param path: string
	 * @param as_sudo?: boolean
	 * @returns Promise<void>
	 */
	async remove_server_dir(server, path, as_sudo = false) {
		if (as_sudo) {
			await this.execute("ssh", this.ssh_args(["-t", server, "sudo", "rm", "-r", path]), "inherit");
		} else {
			await this.execute("ssh", this.ssh_args([server, "rm", "-r", path]), "pipe");
		}
	}





	








	/* SSH Helpers */




	/**
	 * Builds the SSH Address based on a server object.
	 * @param server: object
	 * @returns string
	 */
	 ssh_addr(server) { return `${server.name}@${server.ip}` }




	 /**
	  * Given a list of arguments, it adds them to the base ssh args.
	  * The server can be provided as an object and it will be converted
	  * to address format.
	  * IMPORTANT: The args must be provided in the correct order.
	  * @param args: Array<string|object>
	  * @returns string[]
	  */
	 ssh_args(partial_args) {
		 // Init the args
		 let final_args = [ "-i", this.ssh_private_key_path ];
 
		 // Iterate over the provided args and add them accordingly
		 for (var arg of partial_args) {
			 // If it is an object, retrieve the address
			 if (arg && typeof arg == "object") { final_args.push(this.ssh_addr(arg)) }
 
			 // Otherwise, add it to the list
			 else { final_args.push(arg) }
		 }
 
		 // Return the final list of args
		 return final_args;
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









// Export the modules
export { ClusterCommand };