import express from 'express';
import { apiRouter, modelInfo } from './api-router.js';

const app = express();
const PORT = Number(process.env.PORT || 3001);
const HOST = process.env.AXOM_HOST || '127.0.0.1';

app.use('/api', apiRouter);

app.get('/', (req, res) => {
  res.json({
    name: 'AXOM backend',
    status: 'online',
    provider: 'local',
    model: modelInfo.id,
    endpoints: { health: '/api/health', chat: 'POST /api/agent' },
  });
});

const server = app.listen(PORT, HOST, () => {
  console.log(`AXOM backend listening on http://${HOST}:${PORT}`);
  console.log(`Using local SLM: ${modelInfo.id} (Q4, CPU)`);
});

// Graceful shutdown: stop accepting new connections and wait for in-flight
// requests to finish before exiting.
function shutdown(signal) {
  console.log(`\nAXOM backend received ${signal}; shutting down gracefully.`);
  server.close((error) => {
    if (error) {
      console.error('Server close error:', error);
      process.exitCode = 1;
    }
    process.exit();
  });
  // Force-exit if graceful close takes too long (e.g. a stalled inference).
  setTimeout(() => {
    console.warn('AXOM backend force-exiting after shutdown timeout.');
    process.exit(1);
  }, 15_000).unref();
}

process.once('SIGTERM', () => shutdown('SIGTERM'));
process.once('SIGINT', () => shutdown('SIGINT'));
