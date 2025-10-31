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
    { appDisplayName: 'Canvas', hour: 9, weekday: 0 },
    { appDisplayName: 'GitHub', hour: 11, weekday: 0 },
    { appDisplayName: 'MATLAB', hour: 14, weekday: 1 },
    { appDisplayName: 'JupyterHub', hour: 16, weekday: 1 },
    { appDisplayName: 'Canvas', hour: 10, weekday: 2 },
    { appDisplayName: 'Microsoft Teams', hour: 13, weekday: 2 },
    { appDisplayName: 'GitHub', hour: 15, weekday: 3 },
    { appDisplayName: 'Canvas', hour: 10, weekday: 4 },
  ],
  admin_employee: [
    { appDisplayName: 'Outlook', hour: 8, weekday: 0 },
    { appDisplayName: 'Microsoft Teams', hour: 9, weekday: 0 },
    { appDisplayName: 'ServiceNow', hour: 11, weekday: 0 },
    { appDisplayName: 'MyUSF (GEMS/HR)', hour: 14, weekday: 1 },
    { appDisplayName: 'Power BI', hour: 15, weekday: 1 },
    { appDisplayName: 'Microsoft Teams', hour: 9, weekday: 2 },
    { appDisplayName: 'Outlook', hour: 13, weekday: 2 },
    { appDisplayName: 'Workday', hour: 10, weekday: 3 },
  ],
  arts_freshman: [
    { appDisplayName: 'Canvas', hour: 10, weekday: 0 },
    { appDisplayName: 'YouTube', hour: 12, weekday: 0 },
    { appDisplayName: 'JSTOR', hour: 14, weekday: 1 },
    { appDisplayName: 'Library Catalog', hour: 11, weekday: 2 },
    { appDisplayName: 'Adobe Creative Cloud', hour: 16, weekday: 2 },
    { appDisplayName: 'Canvas', hour: 9, weekday: 3 },
    { appDisplayName: 'Spotify', hour: 17, weekday: 4 },
    { appDisplayName: 'Library Catalog', hour: 13, weekday: 5 },
  ],
  medical_resident: [
    {
      "appDisplayName": "Canvas",
      "hour": 13,
      "weekday": 4
    },
    {
      "appDisplayName": "MATLAB",
      "hour": 11,
      "weekday": 1
    },
    {
      "appDisplayName": "USF Health Portal",
      "hour": 12,
      "weekday": 1
    }
  ]
};

exports.getPersonas = async () => {
  // TODO: Replace with Event Hub query to get unique personas
  return PERSONAS;
};

exports.getLogsByPersona = async (persona, limit = null) => {
  // TODO: Replace with Event Hub consumer to fetch actual logs
  // This should:
  // 1. Connect to Event Hub
  // 2. Filter logs by persona
  // 3. Return recent logs (limited by 'limit' parameter)

  const logs = MOCK_LOGS[persona] || [];
  const total = logs.length;

  let subset = logs;
  if (typeof limit === 'number' && Number.isFinite(limit) && limit > 0) {
    subset = logs.slice(-limit);
  }

  const ordered = [...subset].reverse();

  return {
    logs: ordered,
    total
  };
};
