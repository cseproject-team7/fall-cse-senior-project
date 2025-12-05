const eventHubService = require('../services/eventHubService');
const { BlobServiceClient } = require('@azure/storage-blob');

// Get connection string from .env
const connectionString = process.env.AZURE_STORAGE_CONNECTION_STRING;
let blobServiceClient = null;

if (!connectionString) {
    console.warn("⚠️  Azure Storage Connection String not found in .env file - blob storage features will be disabled");
} else {
    blobServiceClient = BlobServiceClient.fromConnectionString(connectionString);
}

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

// === FUNCTION 1: Get All Logs for Dashboard (Sorted by Time) ===
exports.getAllLogs = async (req, res) => {
    if (!blobServiceClient) {
        return res.status(503).json({
            success: false,
            error: 'Azure Storage not configured'
        });
    }

    // Add pagination parameters
    const limit = parseInt(req.query.limit) || 1000; // Default 1000 logs
    const offset = parseInt(req.query.offset) || 0;

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

        // 3. --- Sort all logs by time ---
        allLogs.sort((a, b) => {
            return new Date(b.createdDateTime) - new Date(a.createdDateTime);
        });

        // 4. --- Apply pagination ---
        const totalLogs = allLogs.length;
        const paginatedLogs = allLogs.slice(offset, offset + limit);

        // 5. --- Format the logs *after* sorting and pagination ---
        const formattedLogs = paginatedLogs.map(log => {
            const d = new Date(log.createdDateTime);
            const date = `${d.getMonth() + 1}/${d.getDate()}/${d.getFullYear()}`;
            const minutes = d.getMinutes().toString().padStart(2, '0');
            const time = `${d.getHours()}:${minutes}`;

            return {
                userPrincipalName: log.userPrincipalName,
                userId: log.userPrincipalName,
                appDisplayName: log.appDisplayName,
                date: date,
                time: time
            };
        });

        // 6. Send the sorted and formatted logs with metadata
        res.json({
            logs: formattedLogs,
            total: totalLogs,
            limit: limit,
            offset: offset,
            hasMore: (offset + limit) < totalLogs
        });

    } catch (error) {
        console.error(error);
        res.status(500).send({ message: "Error fetching logs", error: error.message });
    }
};

// === 
//      FUNCTION 2: Get Log Patterns (MODIFIED)
//      This version splits patterns by a 1-hour time gap.
// ===
exports.getLogPatterns = async (req, res) => {
    if (!blobServiceClient) {
        return res.status(503).json({
            success: false,
            error: 'Azure Storage not configured'
        });
    }

    const containerName = "json-signin-logs"; // Your container name
    const containerClient = blobServiceClient.getContainerClient(containerName);
    const oneHourInMs = 60 * 60 * 1000; // 3,600,000 milliseconds
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

        // 2. Sort all logs by user, then by time (Essential!)
        rawLogs.sort((a, b) => {
            if (a.userId < b.userId) return -1;
            if (a.userId > b.userId) return 1;
            return new Date(a.createdDateTime) - new Date(b.createdDateTime);
        });

        // 3. --- NEW LOGIC: Build patterns based on time gaps ---
        const patternCounts = {};
        let currentPattern = [];

        for (let i = 0; i < rawLogs.length; i++) {
            const log = rawLogs[i];

            // Clean up the app name
            let appName = log.appDisplayName;
            if (appName === "MyUSF (OASIS)") appName = "MyUSF";

            // Add app to the current pattern
            currentPattern.push(appName);

            // Check for the end of a pattern
            let endOfPattern = false;

            if (i === rawLogs.length - 1) {
                // This is the very last log in the whole file
                endOfPattern = true;
            } else {
                const nextLog = rawLogs[i + 1];

                if (log.userId !== nextLog.userId) {
                    // The next log is for a new user, so this user's pattern is done
                    endOfPattern = true;
                } else {
                    // It's the same user, check the time gap
                    const time1 = new Date(log.createdDateTime);
                    const time2 = new Date(nextLog.createdDateTime);
                    const diffInMs = time2 - time1;

                    if (diffInMs > oneHourInMs) {
                        // The gap is > 1 hour, so this pattern is done
                        endOfPattern = true;
                    }
                }
            }

            // 4. --- NEW LOGIC: Process and count the finished pattern ---
            if (endOfPattern) {
                // Filter out consecutive duplicates (e.g., [A, A, B] -> [A, B])
                const filteredPattern = currentPattern.filter((app, index) => {
                    return index === 0 || app !== currentPattern[index - 1];
                });

                // Only count patterns with 2 or more apps
                if (filteredPattern.length >= 2) {
                    const patternString = filteredPattern.join(' → ');
                    patternCounts[patternString] = (patternCounts[patternString] || 0) + 1;
                }

                // Reset for the next pattern
                currentPattern = [];
            }
        }

        // 5. Format for Recharts
        let chartData = Object.keys(patternCounts).map(patternName => ({
            name: patternName,
            count: patternCounts[patternName]
        }));

        // 5.5 Aggregate similar patterns (group by first 2-3 apps)
        const aggregatedCounts = {};
        chartData.forEach(item => {
            const apps = item.name.split(' → ');
            // Take first 2-3 apps as the pattern signature
            const signature = apps.slice(0, Math.min(3, apps.length)).join(' → ');

            if (!aggregatedCounts[signature]) {
                aggregatedCounts[signature] = {
                    name: signature + (apps.length > 3 ? ' → ...' : ''),
                    count: 0,
                    variations: []
                };
            }
            aggregatedCounts[signature].count += item.count;
            aggregatedCounts[signature].variations.push(item.name);
        });

        // Convert back to array
        chartData = Object.values(aggregatedCounts).map(item => ({
            name: item.name,
            count: item.count
        }));

        // 6. Sort to rank by frequency and send Top 10
        chartData.sort((a, b) => b.count - a.count);
        res.json(chartData.slice(0, 10));

    } catch (error) {
        console.error(error);
        res.status(500).send({ message: "Error fetching patterns", error: error.message });
    }
};

// === FUNCTION 3: Get All Logs Grouped by User (For Predictions Page) ===
exports.getAllLogsGroupedByUser = async (req, res) => {
    if (!blobServiceClient) {
        return res.status(503).json({
            success: false,
            error: 'Azure Storage not configured'
        });
    }

    const containerName = "json-signin-logs";
    const containerClient = blobServiceClient.getContainerClient(containerName);

    try {
        let allLogs = [];

        // 1. Fetch all logs from Azure Blob Storage
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

                        allLogs.push({
                            userPrincipalName: innerLog.userPrincipalName,
                            userId: innerLog.userId,
                            appDisplayName: innerLog.appDisplayName,
                            createdDateTime: innerLog.createdDateTime,
                            hour: new Date(innerLog.createdDateTime).getHours(),
                            weekday: new Date(innerLog.createdDateTime).getDay()
                        });
                    } catch (e) {
                        console.error(`Failed to parse a line: ${e.message}`);
                    }
                });
            }
        }

        // 2. Group logs by user
        const logsByUser = {};
        const users = new Set();

        allLogs.forEach(log => {
            const user = log.userPrincipalName || log.userId;
            users.add(user);

            if (!logsByUser[user]) {
                logsByUser[user] = [];
            }

            logsByUser[user].push({
                appDisplayName: log.appDisplayName,
                hour: log.hour,
                weekday: log.weekday,
                createdDateTime: log.createdDateTime
            });
        });

        // 3. Sort logs for each user by time
        Object.keys(logsByUser).forEach(user => {
            logsByUser[user].sort((a, b) =>
                new Date(a.createdDateTime) - new Date(b.createdDateTime)
            );
        });

        // 4. Return grouped structure
        const result = {
            users: Array.from(users).sort(),
            logsByUser: logsByUser
        };

        res.json(result);

    } catch (error) {
        console.error('Error fetching grouped logs:', error);
        res.status(500).send({ message: "Error fetching grouped logs", error: error.message });
    }
};

// === FUNCTION 4: Get Pattern Chains from ML Model ===
exports.getPatternChains = async (req, res) => {
    try {
        const axios = require('axios');

        // Get ML server URL from environment or default to localhost
        const ML_SERVER_URL = process.env.ML_SERVER_URL || 'http://localhost:5001';
        const AZURE_ML_KEY = process.env.AZURE_ML_KEY || '';
        const isAzureML = ML_SERVER_URL.includes('inference.ml.azure.com');

        // If using Azure ML, pattern transitions are not available - return hardcoded data
        if (isAzureML) {
            console.log('📊 Fetching pattern transitions from Azure ML endpoint');

            // Azure ML endpoint - use POST with request_type
            const headers = { 'Content-Type': 'application/json' };
            if (AZURE_ML_KEY) {
                headers['Authorization'] = `Bearer ${AZURE_ML_KEY}`;
            }

            const response = await axios.post(
                ML_SERVER_URL,
                { request_type: 'pattern_transitions' },
                { headers, timeout: 30000 }
            );

            // Parse if string response
            let result = response.data;
            if (typeof result === 'string') {
                result = JSON.parse(result);
            }

            console.log('Azure ML response:', JSON.stringify(result).substring(0, 200));

            // Check for success (handle both Python True and JavaScript true)
            if (result.success !== true && result.success !== 'True') {
                console.error('Azure ML error response:', result);
                throw new Error(result.error || 'ML server returned error');
            }

            // Format response: convert probability to count-like display for consistency
            const formattedTransitions = result.transitions.map(item => {
                return {
                    pattern: item.pattern,
                    predecessors: item.predecessors.map(pred => ({
                        pattern: pred.pattern,
                        count: Math.round(pred.confidence), // Use confidence % as display count
                        probability: pred.probability
                    })),
                    successors: item.successors.map(succ => ({
                        pattern: succ.pattern,
                        count: Math.round(succ.confidence), // Use confidence % as display count
                        probability: succ.probability
                    }))
                };
            });

            // Return array directly to match local ML server format
            return res.json(formattedTransitions);
        }

        // Prepare headers for local ML server
        const headers = {};

        // Determine endpoint URL
        const endpointUrl = `${ML_SERVER_URL}/pattern-transitions`;

        // Call ML server to get model-learned pattern transitions
        const response = await axios.get(endpointUrl, { headers });

        if (!response.data.success) {
            throw new Error('ML server returned error');
        }

        // Format response: convert probability to count-like display for consistency
        const formattedTransitions = response.data.transitions.map(item => {
            return {
                pattern: item.pattern,
                predecessors: item.predecessors.map(pred => ({
                    pattern: pred.pattern,
                    count: Math.round(pred.confidence), // Use confidence % as display count
                    probability: pred.probability
                })),
                successors: item.successors.map(succ => ({
                    pattern: succ.pattern,
                    count: Math.round(succ.confidence), // Use confidence % as display count
                    probability: succ.probability
                }))
            };
        });

        res.json(formattedTransitions);

    } catch (error) {
        console.error('Error fetching pattern chains from ML model:', error);

        // Fallback: return empty transitions if ML server is down
        res.status(500).json({
            message: "ML server unavailable for pattern transitions",
            error: error.message,
            fallback: []
        });
    }
};