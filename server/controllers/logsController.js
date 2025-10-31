const eventHubService = require('../services/eventHubService');

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
    const rawLimit = req.query.limit;
    
    let limit;
    if (!rawLimit || rawLimit === 'all') {
      limit = null;
    } else {
      const parsed = parseInt(rawLimit, 10);
      limit = Number.isNaN(parsed) || parsed <= 0 ? 50 : parsed;
    }
    
    const { logs, total } = await eventHubService.getLogsByPersona(persona, limit);
    
    res.json({ 
      success: true, 
      persona,
      count: logs.length,
      total,
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
