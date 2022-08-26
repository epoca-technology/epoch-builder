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
	 */
	constructor(ssh_private_key_path) { this.ssh_private_key_path = ssh_private_key_path }




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









	/* Server Helpers */





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
                // Init the payload
                let payload = "";

                // Check if the server is localhost
                if (server.name == "localhost") { payload = await this.execute("ps", ["aux"], "pipe") }

                // Otherwise, check the server status with ssh
                else { 
					payload = await this.execute("ssh", this.ssh_args([server, "ps", "aux"]), "pipe") 
				}
				
                /**
				 * Make sure the payload is a valid string and that the python3 process isn't running 
				 * on the dist directory
				 */
                if (typeof payload == "string" && payload.length) { 
					return !payload.includes("python3 ./dist") 
				}

                // Otherwise, the availability is unknown
                else { return undefined }
			} catch (e) { console.error(e); return undefined; }
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

        // Check if localhost should be printed
        if (server.name == "localhost") { payload = await this.execute("landscape-sysinfo", [], "pipe") }
        
        // Otherwise, print the specific server
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
	 * Installs the SSH Public Key on a given server.
	 * @param server 
	 * @returns Promise<void>
	 */
	async install_ssh_key(server) { await this.execute("ssh-copy-id", this.ssh_args([server]), "inherit") }













	/* General Helpers */





	/**
	 * Pushes a directory and its contents from the localhost machine to a selected server.
	 * @param server: object
	 * @param origin_path: string
	 * @param destination_path: string
	 * @returns Promise<string>
	 */
	async push_dir(server, origin_path, destination_path) {
		return this.execute("scp", this.ssh_args(["-r", origin_path, `${this.ssh_addr(server)}:${destination_path}`]), "pipe");
	}





	/**
	 * Pulls a directory and its contents from a selected server to the localhost machine.
	 * @param server: object
	 * @param origin_path: string
	 * @param destination_path: string
	 * @returns Promise<string>
	 */
	async pull_dir(server, origin_path, destination_path) {
		return this.execute("scp", this.ssh_args(["-r", `${this.ssh_addr(server)}:${origin_path}`, destination_path]), "pipe");
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
			throw new Error("The remove_server_dir as sudo has not yet been implemented.")
		} else {
			await this.execute("ssh", this.ssh_args([server, "rm", "-r", path]), "pipe");
		}
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