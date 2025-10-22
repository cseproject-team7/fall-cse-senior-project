/**
 * Event Hub Service
 * TODO: Connect to Azure Event Hub to retrieve authentication logs
 * For now, returns mock data for development
 */

// Mock personas - replace with actual Event Hub data
const PERSONAS = [
  'engineering_junior',
  'admin_employee',
  'arts_freshman',
  'medical_resident'
];

// Mock log data - replace with Event Hub integration
const MOCK_LOGS = {
  engineering_junior: [
    { appDisplayName: 'Canvas', hour: 14, weekday: 1 },
    { appDisplayName: 'MATLAB', hour: 11, weekday: 1 },
    { appDisplayName: 'GitHub', hour: 15, weekday: 2 },
  ],
  admin_employee: [
    { appDisplayName: 'Outlook', hour: 9, weekday: 1 },
    { appDisplayName: 'Microsoft Teams', hour: 10, weekday: 1 },
    { appDisplayName: 'MyUSF (GEMS/HR)', hour: 14, weekday: 2 },
  ],
  arts_freshman: [
    { appDisplayName: 'Canvas', hour: 10, weekday: 1 },
    { appDisplayName: 'JSTOR', hour: 14, weekday: 2 },
    { appDisplayName: 'Library Catalog', hour: 11, weekday: 3 },
  ],
  medical_resident: [
    { appDisplayName: 'USF Health Portal', hour: 7, weekday: 1 },
    { appDisplayName: 'PubMed', hour: 20, weekday: 2 },
    { appDisplayName: 'UpToDate', hour: 22, weekday: 3 },
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

