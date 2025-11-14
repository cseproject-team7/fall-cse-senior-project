const eventHubService = require('../services/eventHubService');
const { BlobServiceClient } = require('@azure/storage-blob');

// Get connection string from .env
const connectionString = process.env.AZURE_STORAGE_CONNECTION_STRING;
if (!connectionString) {
    console.error("Azure Storage Connection String not found in .env file");
}
const blobServiceClient = BlobServiceClient.fromConnectionString(connectionString);

// Helper function to convert a stream to a string
async function streamToString(readableStream) {
    return new Promise((resolve, reject) => {
        const chunks = [];
        readableStream.on('data', (data) => {
            chunks.push(data.toString());
        });
        readableStream.on('end', () => {
            resolve(chunks.join(''));
        });
        readableStream.on('error', reject);
    });
}

// Get list of available personas
exports.getPersonas = async (req, res) => {
  try {
    const personas = await eventHubService.getPersonas();
    res.json({ success: true, personas });
  } catch (error) {
    console.error('Error fetching personas:', error.message);
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
};

// Get logs filtered by persona
exports.getLogsByPersona = async (req, res) => {
  try {
    const { persona } = req.params;
    const limit = parseInt(req.query.limit) || 50;
    
    const logs = await eventHubService.getLogsByPersona(persona, limit);
    
    res.json({ 
      success: true, 
      persona,
      count: logs.length,
      logs 
    });
  } catch (error) {
    console.error('Error fetching logs:', error.message);
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
};

// === FUNCTION 1: Get All Logs for Dashboard (Now sorted by time) ===
exports.getAllLogs = async (req, res) => {
    
    const containerName = "json-signin-logs"; // Your container name
    const containerClient = blobServiceClient.getContainerClient(containerName);

    try {
        let allLogs = []; // 1. Create a temporary array to hold raw logs

        for await (const blob of containerClient.listBlobsFlat()) {
            if (blob.name.endsWith('.json')) {
                const blobClient = containerClient.getBlobClient(blob.name);
                const downloadBlockBlobResponse = await blobClient.download(0);
                const downloadedContent = await streamToString(downloadBlockBlobResponse.readableStreamBody);
                
                const lines = downloadedContent.split('\n');

                lines.forEach(line => {
                    if (line.trim() === "") return;
                    try {
                        const outerLog = JSON.parse(line);
                        const innerLog = JSON.parse(outerLog.Body);
                        
                        // 2. Push the raw log data, including the sort key
                        allLogs.push({
                            userPrincipalName: innerLog.userPrincipalName,
                            userId: innerLog.userId,
                            appDisplayName: innerLog.appDisplayName,
                            createdDateTime: innerLog.createdDateTime // <-- ESSENTIAL for sorting
                        });
                    } catch (e) {
                        console.error(`Failed to parse a line: ${e.message}`);
                    }
                });
            }
        }

        // 3. --- THIS IS THE NEW SORTING LOGIC ---
        // We sort by 'createdDateTime' in descending order (newest first)
        allLogs.sort((a, b) => {
            return new Date(b.createdDateTime) - new Date(a.createdDateTime);
        });

        // 4. --- THIS IS THE NEW FORMATTING LOGIC ---
        // Loop over the now-sorted array and create the final strings
        const formattedLogs = allLogs.map(log => {
            const d = new Date(log.createdDateTime);
            const date = `${d.getMonth() + 1}/${d.getDate()}/${d.getFullYear()}`;
            const minutes = d.getMinutes().toString().padStart(2, '0');
            const time = `${d.getHours()}:${minutes}`;
            
            return {
                userPrincipalName: log.userPrincipalName,
                userId: log.userPrincipalName, // Corrected from your old code
                appDisplayName: log.appDisplayName,
                date: date,
                time: time
            };
        });

        // 5. Send the sorted and formatted logs
        res.json(formattedLogs);

    } catch (error) {
        console.error(error);
        res.status(500).send({ message: "Error fetching logs", error: error.message });
    }
};

// === 
//      FUNCTION 2: (REPLACE THIS) Get Log Patterns for Dashboard 
//      This is the new version that counts full sessions (3+ apps)
// ===
exports.getLogPatterns = async (req, res) => {
    
    const containerName = "json-signin-logs"; // Your container name
    const containerClient = blobServiceClient.getContainerClient(containerName);

    let rawLogs = [];

    try {
        // 1. Fetch all logs
        for await (const blob of containerClient.listBlobsFlat()) {
            if (blob.name.endsWith('.json')) {
                const blobClient = containerClient.getBlobClient(blob.name);
                const downloadBlockBlobResponse = await blobClient.download(0);
                const downloadedContent = await streamToString(downloadBlockBlobResponse.readableStreamBody);
                
                const lines = downloadedContent.split('\n');

                lines.forEach(line => {
                    if (line.trim() === "") return;
                    try {
                        const outerLog = JSON.parse(line);
                        const innerLog = JSON.parse(outerLog.Body);
                        rawLogs.push({
                            userId: innerLog.userId,
                            appDisplayName: innerLog.appDisplayName,
                            createdDateTime: innerLog.createdDateTime 
                        });
                    } catch (e) { /* skip bad lines */ }
                });
            }
        }

        // 2. Sort all logs by user, then by time (Essential for finding patterns)
        rawLogs.sort((a, b) => {
            if (a.userId < b.userId) return -1;
            if (a.userId > b.userId) return 1;
            return new Date(a.createdDateTime) - new Date(b.createdDateTime);
        });

        // 3. Group logs by user into sessions
        const userSessions = {};
        rawLogs.forEach(log => {
            if (!userSessions[log.userId]) {
                userSessions[log.userId] = []; // Create a new list
            }
            
            let appName = log.appDisplayName;
            if (appName === "MyUSF (OASIS)") appName = "MyUSF";

            userSessions[log.userId].push(appName);
        });

        // 4. Count the frequency of full session patterns
        const patternCounts = {};
        
        Object.values(userSessions).forEach(sessionArray => {
            
            // Filter out consecutive duplicates (e.g., [A, A, B] -> [A, B])
            const filteredSession = sessionArray.filter((app, index) => {
                return index === 0 || app !== sessionArray[index - 1];
            });

            // If the user's session is less than 2 apps, skip it
            if (filteredSession.length < 2) {
                return;
            }

            // Create the full pattern string (e.g., "GitHub → Canvas → MATLAB")
            const pattern = filteredSession.join(' → '); 
            
            // Count this full pattern
            patternCounts[pattern] = (patternCounts[pattern] || 0) + 1;
        });

        // 5. Format for Recharts
        let chartData = Object.keys(patternCounts).map(patternName => ({
            name: patternName,
            count: patternCounts[patternName]
        }));

        // 6. Sort to rank by frequency and send just the TOP 10
        chartData.sort((a, b) => b.count - a.count);
        res.json(chartData.slice(0,10)); 
    } catch (error) {
        console.error(error);
        res.status(500).send({ message: "Error fetching patterns", error: error.message });
    }
};