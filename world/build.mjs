import { build, context } from 'esbuild';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const here = path.dirname(fileURLToPath(import.meta.url));
const opts = {
  entryPoints: [path.join(here, 'src/main.ts')],
  bundle: true, format: 'iife', target: 'es2020', minify: true, sourcemap: false, legalComments: 'none',
  outfile: path.join(here, '../web/world/world.js'),
  alias: { 'game-core': path.join(here, 'src/shim/game-core.ts') },
  logLevel: 'info',
};
if (process.argv.includes('--watch')) { (await context(opts)).watch(); } else { await build(opts); }
