/**
 * Test app factory for integration tests
 * Creates an Express app with API routes mounted
 */

const express = require('express');
const apiRoutes = require('../../routes/api');

function createTestApp() {
  const app = express();
  app.use(express.json());
  app.use('/api', apiRoutes);
  return app;
}

module.exports = createTestApp;

