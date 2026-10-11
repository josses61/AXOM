import { apiRouter } from './api-router.js';

export function axomApiPlugin() {
  return {
    name: 'axom-local-api',
    configureServer(server) {
      server.middlewares.use('/api', apiRouter);
    },
    configurePreviewServer(server) {
      server.middlewares.use('/api', apiRouter);
    },
  };
}
