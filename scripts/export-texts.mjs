// Prints the chant and dhamma-topic texts from www/app.js as JSON (used by generate_audio.py).
import { readFileSync } from 'node:fs';
const src = readFileSync(new URL('../www/app.js', import.meta.url), 'utf8');
const block = src.slice(src.indexOf('const TAGS=['), src.indexOf('/* ============ utils'));
const data = new Function(block + '\nreturn {CH, TOPICS};')();
const out = {
  chants: data.CH.map(c => ({ id: c.id, lines: c.l })),
  topics: data.TOPICS.map(t => ({ id: 'topic:' + t.id, lines: t.story })),
};
process.stdout.write(JSON.stringify(out));
