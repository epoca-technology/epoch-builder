import { lstatSync, readFileSync, readdirSync } from "fs";
import { extname } from "path";





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
	static file_exists(path) { try { return lstatSync(path).isFile() } catch (e) { return false } }



	/**
	 * Checks if a directory exists in a given path.
	 * @param path: string
	 * @returns boolean
	 */
	 static dir_exists(path) { try { return lstatSync(path).isDirectory() } catch (e) { return false } }






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




	/**
	 * Retrieves the directories and files located in a given path.
	 * @param path: string
	 * @returns object { directories: string[], files: string[] }
	 */
	static get_path_content(path) {
		// Init values
		let directories = [];
		let files = [];

		// Make sure the path is a directory
		if (!FileSystem.dir_exists(path)) throw new Error(`The provided path ${path} is not an existing directory.`);

		// Retrieve the raw path contents
		const path_content = readdirSync(path);

		// Iterate over each item and add them to the lists accordingly
		for (var item of path_content) {
			// Check if it is a directory
			if (FileSystem.dir_exists(`${path}/${item}`)) { directories.push(item) }

			// Otherwise, it is a file
			else { files.push(item) }
		}

		// Sort the lists alphabetically
		directories.sort();
		files.sort();

		// Finally, return the packed values
		return { directories: directories, files: files }
	}
}




// Export the modules
export { FileSystem };