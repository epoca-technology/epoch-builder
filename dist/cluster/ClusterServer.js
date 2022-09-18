


 /**
  * Cluster Server
  * This class handles the server listing and general info extraction.
  * 
  * Class Constants:
  * 	LOCALHOST_SERVER: object
  * 	ALL_SERVER: object
  * 
  * Instance Properties
  * 	cluster_command: ClusterCommand
  * 	servers: object[]
  */
class ClusterServer {
	// Server objects of localhost and all
	LOCALHOST_SERVER = {name: "localhost", ip: "", is_cluster: true, is_online: true, is_available: true};
	ALL_SERVER = {name: "all", ip: "", is_cluster: false, is_online: true, is_available: true};





	/**
	 * Initializes a Cluster Servers Instance
	 * @param cluster_command: ClusterCommand
	 * @param servers: object[]
	 */
	constructor(cluster_command, servers) { 
		// Init the Cluster Command Instance
		this.cluster_command = cluster_command;

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
			const is_online = await this.cluster_command.is_server_online(this.servers[i].ip);

			// Append the value to the list
			servers.push({
				name: this.servers[i].name,
				ip: this.servers[i].ip,
				is_cluster: this.servers[i].is_cluster,
				is_online: is_online,
				is_available: await this.cluster_command.is_server_available(is_online, this.servers[i])
			})
		}

		// Check if localhost should be prepended
		if (include_localhost) {
            servers = [{ 
                ...this.LOCALHOST_SERVER, 
                is_available: await this.cluster_command.is_server_available(true, this.LOCALHOST_SERVER) 
            }].concat(servers);
        }

		// Check if all should be appended
		if (include_all) servers.push(this.ALL_SERVER)

		// Finally, return the list
		return servers;
	}
}




// Export the modules
export { ClusterServer };