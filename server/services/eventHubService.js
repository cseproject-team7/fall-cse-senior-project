/**
 * Event Hub Service
 * TODO: Connect to Azure Event Hub to retrieve authentication logs
 * For now, returns mock data for development
 */

// Mock personas - test scenarios
const PERSONAS = [
  'user_1',
  'user_2',
  'user_3',
  'user_4',
  'user_5',
  'user_6'
];

// Mock log data - strategically designed to trigger different patterns and predictions
const MOCK_LOGS = {
  // USER 1: Heavy COURSEWORK pattern (Canvas + Productivity apps)
  user_1: [
    { appDisplayName: 'Login', hour: 9, weekday: 1, createdDateTime: '2025-11-21T09:00:00Z' },
    { appDisplayName: 'Canvas', hour: 9, weekday: 1, createdDateTime: '2025-11-21T09:05:00Z' },
    { appDisplayName: 'Word Online', hour: 9, weekday: 1, createdDateTime: '2025-11-21T09:30:00Z' },
    { appDisplayName: 'Canvas', hour: 10, weekday: 1, createdDateTime: '2025-11-21T10:15:00Z' },
    { appDisplayName: 'Excel Online', hour: 11, weekday: 1, createdDateTime: '2025-11-21T11:00:00Z' },
    { appDisplayName: 'PowerPoint Online', hour: 11, weekday: 1, createdDateTime: '2025-11-21T11:45:00Z' },
    { appDisplayName: 'Canvas', hour: 13, weekday: 1, createdDateTime: '2025-11-21T13:00:00Z' },
    { appDisplayName: 'OneNote', hour: 14, weekday: 1, createdDateTime: '2025-11-21T14:00:00Z' },
    { appDisplayName: 'Canvas', hour: 15, weekday: 1, createdDateTime: '2025-11-21T15:30:00Z' }
  ],

  // USER 2: ADMIN pattern (Registration, degree planning)
  user_2: [
    { appDisplayName: 'Login', hour: 8, weekday: 2, createdDateTime: '2025-11-22T08:00:00Z' },
    { appDisplayName: 'OASIS', hour: 8, weekday: 2, createdDateTime: '2025-11-22T08:05:00Z' },
    { appDisplayName: 'Schedule Planner', hour: 8, weekday: 2, createdDateTime: '2025-11-22T08:20:00Z' },
    { appDisplayName: 'DegreeWorks', hour: 9, weekday: 2, createdDateTime: '2025-11-22T09:00:00Z' },
    { appDisplayName: 'Archivum', hour: 9, weekday: 2, createdDateTime: '2025-11-22T09:30:00Z' },
    { appDisplayName: 'Navigate', hour: 10, weekday: 2, createdDateTime: '2025-11-22T10:00:00Z' },
    { appDisplayName: 'MyUSF', hour: 10, weekday: 2, createdDateTime: '2025-11-22T10:30:00Z' },
    { appDisplayName: 'OASIS', hour: 11, weekday: 2, createdDateTime: '2025-11-22T11:00:00Z' },
    { appDisplayName: 'Outlook', hour: 11, weekday: 2, createdDateTime: '2025-11-22T11:15:00Z' }
  ],

  // USER 3: RESEARCH pattern (Library databases + specialized tools)
  user_3: [
    { appDisplayName: 'Login', hour: 19, weekday: 3, createdDateTime: '2025-11-23T19:00:00Z' },
    { appDisplayName: 'USF Library', hour: 19, weekday: 3, createdDateTime: '2025-11-23T19:05:00Z' },
    { appDisplayName: 'JSTOR', hour: 19, weekday: 3, createdDateTime: '2025-11-23T19:30:00Z' },
    { appDisplayName: 'IEEE Xplore', hour: 20, weekday: 3, createdDateTime: '2025-11-23T20:15:00Z' },
    { appDisplayName: 'OneNote', hour: 21, weekday: 3, createdDateTime: '2025-11-23T21:00:00Z' },
    { appDisplayName: 'Word Online', hour: 21, weekday: 3, createdDateTime: '2025-11-23T21:30:00Z' },
    { appDisplayName: 'IEEE Xplore', hour: 22, weekday: 3, createdDateTime: '2025-11-23T22:00:00Z' },
    { appDisplayName: 'Adobe Creative Cloud', hour: 22, weekday: 3, createdDateTime: '2025-11-23T22:30:00Z' },
    { appDisplayName: 'Adobe Photoshop', hour: 23, weekday: 3, createdDateTime: '2025-11-23T23:00:00Z' }
  ],

  // USER 4: CAREER pattern (Job search + professional development)
  user_4: [
    { appDisplayName: 'Login', hour: 14, weekday: 4, createdDateTime: '2025-11-24T14:00:00Z' },
    { appDisplayName: 'Handshake', hour: 14, weekday: 4, createdDateTime: '2025-11-24T14:05:00Z' },
    { appDisplayName: 'LinkedIn Learning', hour: 14, weekday: 4, createdDateTime: '2025-11-24T14:45:00Z' },
    { appDisplayName: 'Word Online', hour: 15, weekday: 4, createdDateTime: '2025-11-24T15:30:00Z' },
    { appDisplayName: 'Handshake', hour: 16, weekday: 4, createdDateTime: '2025-11-24T16:00:00Z' },
    { appDisplayName: 'Outlook', hour: 16, weekday: 4, createdDateTime: '2025-11-24T16:30:00Z' },
    { appDisplayName: 'Teams', hour: 17, weekday: 4, createdDateTime: '2025-11-24T17:00:00Z' },
    { appDisplayName: 'LinkedIn Learning', hour: 17, weekday: 4, createdDateTime: '2025-11-24T17:30:00Z' },
    { appDisplayName: 'Handshake', hour: 18, weekday: 4, createdDateTime: '2025-11-24T18:00:00Z' }
  ],

  // USER 5: CODING pattern (Development tools + GitHub)
  user_5: [
    { appDisplayName: 'Login', hour: 10, weekday: 5, createdDateTime: '2025-11-25T10:00:00Z' },
    { appDisplayName: 'GitHub', hour: 10, weekday: 5, createdDateTime: '2025-11-25T10:05:00Z' },
    { appDisplayName: 'MATLAB Online', hour: 10, weekday: 5, createdDateTime: '2025-11-25T10:45:00Z' },
    { appDisplayName: 'GitHub', hour: 12, weekday: 5, createdDateTime: '2025-11-25T12:00:00Z' },
    { appDisplayName: 'Canvas', hour: 13, weekday: 5, createdDateTime: '2025-11-25T13:00:00Z' },
    { appDisplayName: 'MATLAB Online', hour: 14, weekday: 5, createdDateTime: '2025-11-25T14:00:00Z' },
    { appDisplayName: 'GitHub', hour: 15, weekday: 5, createdDateTime: '2025-11-25T15:00:00Z' },
    { appDisplayName: 'Teams', hour: 16, weekday: 5, createdDateTime: '2025-11-25T16:00:00Z' },
    { appDisplayName: 'GitHub', hour: 16, weekday: 5, createdDateTime: '2025-11-25T16:30:00Z' }
  ],

  // USER 6: SOCIAL/COLLABORATION pattern (Teams heavy + mixed productivity)
  user_6: [
    { appDisplayName: 'Login', hour: 11, weekday: 0, createdDateTime: '2025-11-26T11:00:00Z' },
    { appDisplayName: 'Teams', hour: 11, weekday: 0, createdDateTime: '2025-11-26T11:05:00Z' },
    { appDisplayName: 'Outlook', hour: 11, weekday: 0, createdDateTime: '2025-11-26T11:30:00Z' },
    { appDisplayName: 'Teams', hour: 12, weekday: 0, createdDateTime: '2025-11-26T12:00:00Z' },
    { appDisplayName: 'PowerPoint Online', hour: 13, weekday: 0, createdDateTime: '2025-11-26T13:00:00Z' },
    { appDisplayName: 'Teams', hour: 14, weekday: 0, createdDateTime: '2025-11-26T14:00:00Z' },
    { appDisplayName: 'Word Online', hour: 14, weekday: 0, createdDateTime: '2025-11-26T14:30:00Z' },
    { appDisplayName: 'Teams', hour: 15, weekday: 0, createdDateTime: '2025-11-26T15:00:00Z' },
    { appDisplayName: 'Yammer', hour: 15, weekday: 0, createdDateTime: '2025-11-26T15:30:00Z' }
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

