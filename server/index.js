import cors from 'cors';
import express from 'express';
import { apiRouter, modelInfo } from './api-router.js';

const app = express();
const PORT = Number(process.env.PORT || 3001);

app.use(cors());
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

app.listen(PORT, () => {
  console.log(`AXOM backend listening on http://localhost:${PORT}`);
  console.log(`Using local SLM: ${modelInfo.id} (Q4, CPU)`);
});
