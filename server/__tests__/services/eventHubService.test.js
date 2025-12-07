const eventHubService = require('../../services/eventHubService');

describe('eventHubService', () => {
  describe('getPersonas', () => {
    it('should return array of persona IDs', async () => {
      const personas = await eventHubService.getPersonas();

      expect(Array.isArray(personas)).toBe(true);
      expect(personas.length).toBeGreaterThan(0);
      expect(personas).toContain('user_1');
      expect(personas).toContain('user_2');
    });

    it('should return consistent data', async () => {
      const personas1 = await eventHubService.getPersonas();
      const personas2 = await eventHubService.getPersonas();

      expect(personas1).toEqual(personas2);
    });
  });

  describe('getLogsByPersona', () => {
    it('should return logs for valid persona', async () => {
      const logs = await eventHubService.getLogsByPersona('user_1', 50);

      expect(Array.isArray(logs)).toBe(true);
      expect(logs.length).toBeGreaterThan(0);
      
      // Check log structure
      logs.forEach(log => {
        expect(log).toHaveProperty('appDisplayName');
        expect(log).toHaveProperty('hour');
        expect(log).toHaveProperty('weekday');
        expect(log).toHaveProperty('createdDateTime');
      });
    });

    it('should return empty array for invalid persona', async () => {
      const logs = await eventHubService.getLogsByPersona('invalid_user', 50);

      expect(Array.isArray(logs)).toBe(true);
      expect(logs).toEqual([]);
    });

    it('should respect limit parameter', async () => {
      const logs = await eventHubService.getLogsByPersona('user_1', 3);

      expect(logs.length).toBeLessThanOrEqual(3);
    });

    it('should use default limit when not specified', async () => {
      const logs = await eventHubService.getLogsByPersona('user_1');

      expect(Array.isArray(logs)).toBe(true);
      // Should not exceed default limit of 50
      expect(logs.length).toBeLessThanOrEqual(50);
    });

    it('should return different logs for different personas', async () => {
      const logs1 = await eventHubService.getLogsByPersona('user_1', 10);
      const logs2 = await eventHubService.getLogsByPersona('user_2', 10);

      // Different personas should have different patterns
      expect(logs1).not.toEqual(logs2);
    });

    it('should have valid hour values (0-23)', async () => {
      const logs = await eventHubService.getLogsByPersona('user_1', 50);

      logs.forEach(log => {
        expect(log.hour).toBeGreaterThanOrEqual(0);
        expect(log.hour).toBeLessThanOrEqual(23);
      });
    });

    it('should have valid weekday values (0-6)', async () => {
      const logs = await eventHubService.getLogsByPersona('user_3', 50);

      logs.forEach(log => {
        expect(log.weekday).toBeGreaterThanOrEqual(0);
        expect(log.weekday).toBeLessThanOrEqual(6);
      });
    });

    it('should have valid ISO datetime strings', async () => {
      const logs = await eventHubService.getLogsByPersona('user_4', 50);

      logs.forEach(log => {
        expect(log.createdDateTime).toBeTruthy();
        const date = new Date(log.createdDateTime);
        expect(date.toString()).not.toBe('Invalid Date');
      });
    });
  });
});

