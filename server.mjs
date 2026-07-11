// StillMe — local dev server.  node server.mjs  →  http://localhost:8000
//
// Zero dependencies. Ships to iPad as a native app later; this is the loop.

import { createServer } from 'node:http';
import { readFile, appendFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { predictTiles, composeSentence, health } from './lib/llm.mjs';
import { buildGrid } from './lib/tiles.mjs';
import { speak } from './lib/voice.mjs';

const PORT = process.env.PORT ?? 8000;
const ROOT = process.cwd();
const SESSIONS = path.join(ROOT, 'data', 'sessions.jsonl');

const profile = JSON.parse(await readFile(path.join(ROOT, 'data', 'profile.json'), 'utf8'));
const OPENERS = profile.openers;

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.wav': 'audio/wav', '.svg': 'image/svg+xml' };

const json = (res, code, body) => {
  res.writeHead(code, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
};

async function body(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  return JSON.parse(Buffer.concat(chunks).toString() || '{}');
}

/** Every sentence spoken is a data point. This file IS the research result. */
async function logEvent(ev) {
  await mkdir(path.dirname(SESSIONS), { recursive: true });
  await appendFile(SESSIONS, JSON.stringify({ ...ev, at: new Date().toISOString() }) + '\n');
}

const routes = {
  'GET /api/health': async (req, res) => {
    try { json(res, 200, { ok: true, llm: await health(), profile: profile.name }); }
    catch (e) { json(res, 503, { ok: false, error: String(e.message) }); }
  },

  // The grid he sees. core (pinned, never moves) + predicted (the contribution).
  'POST /api/tiles': async (req, res) => {
    const { selected = [], partner = '', predictive = true, coreSlots = 2 } = await body(req);

    // yes/no are ANSWERS, not continuations. Pin them while he is replying to a question;
    // free the slots once he is mid-phrase. The eval caught this: after he picked "tired",
    // the judge marked the pinned yes/no as dead tiles — two of his eight slots, wasted.
    const answering = selected.length === 0;
    const slots = answering ? coreSlots : 0;
    const core = answering ? (profile.core ?? []) : [];

    if (!selected.length && !partner) return json(res, 200, { tiles: OPENERS, source: 'openers' });
    if (!predictive) return json(res, 200, { tiles: OPENERS, source: 'static' });

    try {
      const t0 = Date.now();
      const predicted = await predictTiles({ selected, partner, profile });
      const tiles = buildGrid({ core, predicted, coreSlots: slots });
      json(res, 200, { tiles, source: 'predicted', ms: Date.now() - t0, coreSlots: slots });
    } catch (e) {
      // Never leave him staring at an empty grid because a model timed out.
      json(res, 200, { tiles: OPENERS, source: 'fallback', error: String(e.message) });
    }
  },

  // Always one selection away, from anywhere. If the grid is about Emma's graduation and
  // he suddenly can't breathe, a predicted grid traps him. No tile quality fixes that.
  'GET /api/urgent': async (req, res) => json(res, 200, { tiles: profile.urgent ?? [] }),

  'POST /api/compose': async (req, res) => {
    const { selected = [], partner = '', literal = false } = await body(req);
    try {
      const t0 = Date.now();
      const candidates = await composeSentence({ selected, partner, profile, literal });
      json(res, 200, { candidates, ms: Date.now() - t0, literal });
    } catch (e) {
      json(res, 200, { candidates: [selected.join(' ')], error: String(e.message), literal: true });
    }
  },

  'POST /api/speak': async (req, res) => {
    const { text, voice = 'placeholder' } = await body(req);
    if (!text) return json(res, 400, { error: 'no text' });
    try {
      const { wav, cached } = await speak(text, { voice, model: profile.voiceModel });
      res.writeHead(200, { 'content-type': 'audio/wav', 'content-length': wav.length,
        'x-cached': String(cached) });
      res.end(wav);
    } catch (e) { json(res, 500, { error: String(e.message) }); }
  },

  // selections_per_sentence and seconds_to_sentence are the two numbers the paper is about.
  'POST /api/log': async (req, res) => {
    await logEvent(await body(req));
    json(res, 200, { ok: true });
  },
};

createServer(async (req, res) => {
  const url = new URL(req.url, 'http://x');
  const route = routes[`${req.method} ${url.pathname}`];
  if (route) {
    try { return await route(req, res); }
    catch (e) { return json(res, 500, { error: String(e.message) }); }
  }

  const file = url.pathname === '/' ? '/index.html' : url.pathname;
  try {
    const buf = await readFile(path.join(ROOT, 'public', file));
    res.writeHead(200, { 'content-type': MIME[path.extname(file)] ?? 'text/plain' });
    res.end(buf);
  } catch { res.writeHead(404); res.end('not found'); }
}).listen(PORT, () => {
  console.log(`\n  StillMe → http://localhost:${PORT}`);
  console.log(`  user: ${profile.name}   voice: ${profile.voiceModel ? 'CLONED' : 'placeholder (not his voice yet)'}`);
  console.log(`  input: touch + scanning (arrow keys simulate EOG)\n`);
});
