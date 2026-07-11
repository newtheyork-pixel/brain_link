// Tile-quality eval.  node eval/run.mjs [--core N] [--save TAG]
//
// Tile quality IS the product: a man who spends 40 seconds on one selection cannot
// afford a grid of words he doesn't need. And this is the measurement instrument for
// the research claim — you cannot report that an LLM prior beats a static baseline if
// you never measured either one.
//
// The judge refuses to report a number until it proves it can tell a good grid from a
// bad one. See judge.mjs.

import { readFile, writeFile } from 'node:fs/promises';
import { predictTiles } from '../lib/llm.mjs';
import { buildGrid } from '../lib/tiles.mjs';
import { judge, calibrate } from './judge.mjs';

const arg = (f, d) => (process.argv.includes(f) ? process.argv[process.argv.indexOf(f) + 1] : d);
const coreSlots = +arg('--core', 2);
const tag = arg('--save', null);

const profile = JSON.parse(await readFile(new URL('../data/profile.json', import.meta.url)));
const cases = JSON.parse(await readFile(new URL('./cases.json', import.meta.url)));

console.log('\n  Calibrating judge…');
const cal = await calibrate({ quiet: true });
console.log(`  worst good ${cal.good.toFixed(1)} · best bad ${cal.bad.toFixed(1)} · ${cal.fit ? 'FIT' : 'UNFIT'}`);
if (!cal.fit) {
  console.error('\n  JUDGE UNFIT — refusing to report scores. Fix judge.mjs first.\n');
  process.exit(1);
}

console.log(`\n  coreSlots=${coreSlots}  (${coreSlots} tiles pinned, ${8 - coreSlots} predicted)\n`);
const rows = [];
for (const c of cases) {
  const t0 = Date.now();
  const predicted = await predictTiles({ selected: c.selected, partner: c.partner, profile });
  // Mirror the server exactly: yes/no are pinned only while he is ANSWERING. Mid-phrase
  // they are dead tiles, and the eval is worthless if it grades a grid we don't ship.
  const answering = c.selected.length === 0;
  const tiles = buildGrid({
    core: answering ? profile.core : [],
    predicted,
    coreSlots: answering ? coreSlots : 0,
  });
  const ms = Date.now() - t0;
  const s = await judge({ ...c, tiles });
  rows.push({ id: c.id, ms, tiles, ...s });

  const flag = s.score >= 8 ? '   ' : s.score >= 6 ? ' ~ ' : ' ! ';
  console.log(`${flag}${s.score.toFixed(1).padStart(4)}  ${c.id.padEnd(12)} ${tiles.join(' · ')}`);
  if (s.dead_tiles?.length) console.log(`          dead: ${s.dead_tiles.join(', ')}`);
}

const avg = (k) => (rows.reduce((a, r) => a + r[k], 0) / rows.length).toFixed(1);
const mean = +avg('score');
const lat = rows.map((r) => r.ms).sort((a, b) => a - b);
console.log(`\n  OVERALL  ${mean} / 10`);
console.log(`  relevance ${avg('relevance')} · coverage ${avg('coverage')} · distinct ${avg('distinctness')} · natural ${avg('naturalness')}`);
console.log(`  median latency ${lat[Math.floor(lat.length / 2)]}ms · p90 ${lat[Math.floor(lat.length * 0.9)]}ms`);
const bad = rows.filter((r) => r.score < 6);
console.log(`  below 6.0: ${bad.length ? bad.map((r) => `${r.id}(${r.score})`).join(', ') : 'none'}\n`);

if (tag) {
  await writeFile(new URL(`./results-${tag}.json`, import.meta.url),
    JSON.stringify({ tag, coreSlots, mean, rows }, null, 2));
  console.log(`  saved → eval/results-${tag}.json\n`);
}
