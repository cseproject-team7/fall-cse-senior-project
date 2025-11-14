/**
 * Event Hub Service
 * TODO: Connect to Azure Event Hub to retrieve authentication logs
 * For now, returns mock data for development
 */

// Mock personas - test scenarios
const PERSONAS = [
  'cs_freshman',
  'night_owl',
  'nursing_student',
  'engineering_grad',
  'athlete'
];

// Mock log data - test scenarios with realistic patterns
const MOCK_LOGS = {
  cs_freshman: [
    { appDisplayName: 'Login', hour: 9, weekday: 1 },
    { appDisplayName: 'Outlook', hour: 9, weekday: 1 },
    { appDisplayName: 'Canvas', hour: 10, weekday: 1 },
    { appDisplayName: 'GitHub', hour: 14, weekday: 1 }
  ],
  night_owl: [
    { appDisplayName: 'Login', hour: 22, weekday: 3 },
    { appDisplayName: 'Canvas', hour: 22, weekday: 3 },
    { appDisplayName: 'GitHub', hour: 23, weekday: 3 },
    { appDisplayName: 'Canvas', hour: 0, weekday: 4 }
  ],
  nursing_student: [
    { appDisplayName: 'Login', hour: 7, weekday: 2 },
    { appDisplayName: 'Outlook', hour: 7, weekday: 2 },
    { appDisplayName: 'USF Health Portal', hour: 8, weekday: 2 },
    { appDisplayName: 'Canvas', hour: 12, weekday: 2 }
  ],
  engineering_grad: [
    { appDisplayName: 'Login', hour: 10, weekday: 1 },
    { appDisplayName: 'Outlook', hour: 10, weekday: 1 },
    { appDisplayName: 'MATLAB', hour: 11, weekday: 1 },
    { appDisplayName: 'GitHub', hour: 15, weekday: 1 }
  ],
  athlete: [
    { appDisplayName: 'Login', hour: 6, weekday: 1 },
    { appDisplayName: 'Outlook', hour: 6, weekday: 1 },
    { appDisplayName: 'Canvas', hour: 7, weekday: 1 }
  ]
};

exports.getPersonas = async () => {
  // TODO: Replace with Event Hub query to get unique personas
  return PERSONAS;
};

exports.getLogsByPersona = async (persona, limit = 50) => {
  // TODO: Replace with Event Hub consumer to fetch actual logs
  // This should:
  // 1. Connect to Event Hub
  // 2. Filter logs by persona
  // 3. Return recent logs (limited by 'limit' parameter)
  
  const logs = MOCK_LOGS[persona] || [];
  return logs.slice(0, limit);
};

