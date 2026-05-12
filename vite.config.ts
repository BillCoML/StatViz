import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  base: '/StatViz/',
  resolve: {
    alias: {
      '@shared': resolve(__dirname, 'src/shared'),
      '@lessons': resolve(__dirname, 'src/lessons'),
    },
  },
  build: {
    rollupOptions: {
      input: {
        roadmap:          resolve(__dirname, 'index.html'),
        em:               resolve(__dirname, 'lessons/em/index.html'),
        klJensen:         resolve(__dirname, 'lessons/kl-jensen/index.html'),
        elboVi:           resolve(__dirname, 'lessons/elbo-vi/index.html'),
        gaussianCookbook: resolve(__dirname, 'lessons/gaussian-cookbook/index.html'),
        vae:              resolve(__dirname, 'lessons/vae/index.html'),
      },
    },
  },
  test: {
    environment: 'node',
  },
});
