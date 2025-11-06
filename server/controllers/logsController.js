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

// === NEW FUNCTION 1: Get All Logs for Dashboard ===
exports.getAllLogs = async (req, res) => {
    
    const containerName = "your-container-name"; // <--- ⚠️ UPDATE THIS
    const containerClient = blobServiceClient.getContainerClient(containerName);

    try {
        let formattedLogs = []; 

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
                        
                        const d = new Date(innerLog.createdDateTime);
                        const date = `${d.getMonth() + 1}/${d.getDate()}/${d.getFullYear()}`;
                        const minutes = d.getMinutes().toString().padStart(2, '0');
                        const time = `${d.getHours()}:${minutes}`;
                        
                        formattedLogs.push({
                            userPrincipalName: innerLog.userPrincipalName,
                            userId: innerLog.userId,
                            appDisplayName: innerLog.appDisplayName,
                            date: date,
                            time: time
                        });
                    } catch (e) {
                        console.error(`Failed to parse a line: ${e.message}`);
                    }
                });
            }
        }
        res.json(formattedLogs);
    } catch (error) {
        console.error(error);
        res.status(500).send({ message: "Error fetching logs", error: error.message });
    }
};


// === NEW FUNCTION 2: Get Log Patterns for Dashboard ===
exports.getLogPatterns = async (req, res) => {
    
    const containerName = "your-container-name"; // <--- ⚠️ UPDATE THIS
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

        // 2. Sort all logs by user, then by time
        rawLogs.sort((a, b) => {
            if (a.userId < b.userId) return -1;
            if (a.userId > b.userId) return 1;
            return new Date(a.createdDateTime) - new Date(b.createdDateTime);
        });

        // 3. Count the patterns
        const patternCounts = {};
        for (let i = 0; i < rawLogs.length - 1; i++) {
            if (rawLogs[i].userId === rawLogs[i + 1].userId) {
                let fromApp = rawLogs[i].appDisplayName;
                let toApp = rawLogs[i + 1].appDisplayName;

                if (fromApp === "MyUSF (OASIS)") fromApp = "MyUSF";
                if (toApp === "MyUSF (OASIS)") toApp = "MyUSF";
                
                if (fromApp !== toApp) {
                    const pattern = `${fromApp} → ${toApp}`;
                    patternCounts[pattern] = (patternCounts[pattern] || 0) + 1;
                }
            }
        }

        // 4. Format for Recharts
        let chartData = Object.keys(patternCounts).map(patternName => ({
            name: patternName,
            count: patternCounts[patternName]
        }));

        // 5. Sort and send top 15
        chartData.sort((a, b) => b.count - a.count);
        res.json(chartData.slice(0, 15));

    } catch (error) {
        console.error(error);
        res.status(500).send({ message: "Error fetching patterns", error: error.message });
    }
};