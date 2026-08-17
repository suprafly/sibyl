import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';
import { exec } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const projectRoot = (() => {
  let current = resolve(process.cwd());
  while (current !== dirname(current)) {
    if (existsSync(join(current, 'project.yml')) || existsSync(join(current, 'mix.exs'))) return current;
    current = dirname(current);
  }
  return resolve(process.cwd(), '..');
})();

const editor = {
  name: 'project-design-editor',
  configureServer(server: { middlewares: { use: (path: string, handler: (req: any, res: any) => void) => void } }) {
    server.middlewares.use('/__design-bench/edit', (req, res) => {
      if (req.method !== 'POST' && req.method !== 'GET') {
        res.statusCode = 405;
        res.end(JSON.stringify({ error: 'method_not_allowed' }));
        return;
      }

      let body = req.method === 'GET' ? JSON.stringify({ action: 'load' }) : '';
      req.setEncoding('utf8');
      req.on('data', (chunk: string) => { body += chunk; });
      req.on('end', () => {
        const command = process.env.DESIGN_BENCH_EDITOR_COMMAND || '';
        if (!command) {
          res.statusCode = 503;
          res.end(JSON.stringify({ error: 'editor_not_configured', message: 'Set DESIGN_BENCH_EDITOR_COMMAND for local editing.' }));
          return;
        }

        exec(command, { cwd: projectRoot, maxBuffer: 4 * 1024 * 1024, env: { ...process.env, DESIGN_BENCH_EDIT_JSON: body, DESIGN_BENCH_DESIGN_PATH: process.env.DESIGN_BENCH_DESIGN_PATH || join(projectRoot, 'design/system.yml'), DESIGN_BENCH_PROJECT_PATH: process.env.DESIGN_BENCH_PROJECT_PATH || join(projectRoot, 'project.yml') } }, (error, stdout, stderr) => {
          if (error) {
            res.statusCode = error.code === 10 ? 409 : 422;
            res.end(JSON.stringify({ error: 'design_edit_failed', detail: stderr || stdout || error.message }));
            return;
          }
          res.setHeader('content-type', 'application/json');
          res.end(stdout || JSON.stringify({ ok: true }));
        });
      });
    });
  }
};

export default defineConfig({
  plugins: [sveltekit(), editor],
  server: { port: 2739, strictPort: true },
  build: { chunkSizeWarningLimit: 1000 }
});
