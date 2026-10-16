import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
var apiTarget = process.env.AXOM_API_URL || 'http://127.0.0.1:3001';
export default defineConfig({
    plugins: [react()],
    server: {
        host: '0.0.0.0',
        port: 5173,
        proxy: {
            '/api': apiTarget,
        },
    },
    preview: {
        host: '0.0.0.0',
        port: 4173,
        proxy: {
            '/api': apiTarget,
        },
    },
});
