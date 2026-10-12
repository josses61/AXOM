import express from 'express';
import { apiRouter, modelInfo } from './api-router.js';

const app = express();
const PORT = Number(process.env.PORT || 3001);

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

const HOST = process.env.AXOM_HOST || '127.0.0.1';

app.listen(PORT, HOST, () => {
  console.log(`AXOM backend listening on http://${HOST}:${PORT}`);
  console.log(`Using local SLM: ${modelInfo.id} (Q4, CPU)`);
});
