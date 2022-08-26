import { Cluster } from "./cluster/Cluster.js";



/**
 * Process Execution
 * Initializes an instance of the cluster and runs it.
 * If successful, the process will terminate with a status of 0. 
 * Otherwise, prints the error and terminates with a status of 1.
 */
console.clear();
console.log("EPOCH BUILDER CLUSTER\n\n");
new Cluster().run()
.then(() => { 
    console.log("\n\nEPOCH BUILDER CLUSTER COMPLETED");
    process.exit(0);
}).catch(e => { 
    console.error(e); 
    process.exit(1); 
})