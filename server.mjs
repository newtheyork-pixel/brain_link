// StillMe — local dev server.  node server.mjs  →  http://localhost:8000
//
// Zero dependencies. Ships to iPad as a native app later; this is the loop.

import { createServer } from 'node:http';
import { readFile, appendFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { predictTiles, composeSentence, health } from './lib/llm.mjs';
import { buildGrid, clusterOf, dedupe } from './lib/tiles.mjs';
import { speak } from './lib/voice.mjs';
import { transcribe, available as asrReady } from './lib/listen.mjs';

const PORT = process.env.PORT ?? 8000;
const ROOT = process.cwd();
const SESSIONS = path.join(ROOT, 'data', 'sessions.jsonl');

const profile = JSON.parse(await readFile(path.join(ROOT, 'data', 'profile.json'), 'utf8'));
const OPENERS = profile.openers;

// .mjs and .wasm matter: a module served as text/plain is REFUSED by the browser, and the whole
// import chain dies with "failed to fetch dynamically imported module" — which points at the
// importer, not the file that actually has the wrong type.
const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.wav': 'audio/wav',
  '.svg': 'image/svg+xml', '.wasm': 'application/wasm',
  '.task': 'application/octet-stream', '.bin': 'application/octet-stream',
  '.data': 'application/octet-stream',
};

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
    try {
      json(res, 200, {
        ok: true,
        llm: await health(),
        profile: profile.name,
        // The client used to hardcode voice:'placeholder' — so even with a trained clone
        // loaded, every utterance came out in a stranger's voice, forever.
        voice: profile.voiceModel ? 'cloned' : 'placeholder',
        // ONE instant map. The client kept a hand-copied duplicate behind a "kept in sync"
        // comment, and it wasn't: six of the eight URGENT tiles had no entry, so tapping
        // "nurse" or "suction" produced silence.
        instant: profile.instant ?? {},
      });
    } catch (e) { json(res, 503, { ok: false, error: String(e.message) }); }
  },

  // Where is the model actually running? Answers the question a judge WILL ask on camera,
  // and stops you from quietly filming a demo that depends on a tunnel.
  'GET /api/where': async (req, res) => {
    const b = await health();
    json(res, 200, {
      running_on: b.where,
      model: b.model,
      offline: b.offline,
      speech_recognition: (await asrReady()) ? 'whisper.cpp — on this machine' : 'MISSING (run asr/ setup)',
      note: b.offline
        ? 'Fully on-device. Pull the network cable and it keeps working.'
        : 'Remote GPU — best quality, but NOT offline. Switch to on-device before filming.',
    });
  },

  // The grid he sees. core (pinned, never moves) + predicted (the contribution).
  'POST /api/tiles': async (req, res) => {
    const { selected = [], partner = '', predictive = true, coreSlots = 2, mode = 'answer' } = await body(req);

    // yes/no are ANSWERS, not continuations — and they are certainly not questions.
    // Pin them only while he is replying to someone. The eval caught the first half of
    // this: after he picked "tired", the judge marked the pinned yes/no as dead tiles.
    // Pin yes/no ONLY when he is answering someone. It used to ignore mode, so an ASK grid
    // got "yes"/"no" pinned onto it — answers, to a question he is the one asking — and they
    // bypassed the very filter meant to remove them.
    const replying = mode === 'answer' && selected.length === 0 && !!partner;
    const slots = replying ? coreSlots : 0;
    const core = replying ? (profile.core ?? []) : [];

    if (!predictive && !partner) return json(res, 200, { tiles: OPENERS, source: 'static' });
    if (!selected.length && !partner && mode === 'answer') {
      return json(res, 200, { tiles: OPENERS, source: 'openers' });
    }

    try {
      const t0 = Date.now();
      // Ask for more than we need. Echo-stripping and synonym-dedupe both remove tiles,
      // and a half-empty grid is a worse failure than a slightly weaker 8th tile —
      // he only gets 8 chances to say anything at all.
      const predicted = await predictTiles({ selected, partner, profile, mode, n: 14 });
      let tiles = buildGrid({ core, predicted, selected, coreSlots: slots });

      // He is the one asking. He cannot answer his own question, and every "yes" tile on an
      // ASK grid is one of his eight slots spent on a word he can never use.
      // AFTER buildGrid, because echo-stripping can mint "okay" out of a longer tile — and via
      // the synonym clusters, not a hand-rolled regex that had already drifted from them.
      const YESNO = [clusterOf('yes'), clusterOf('no')];
      if (mode !== 'answer') tiles = tiles.filter((t) => !YESNO.includes(clusterOf(t)));

      // Never hand him an empty board. Echo-strip + dedupe can eat a whole grid, and a missing
      // tile is a thing he cannot say.
      if (tiles.length < 5) {
        tiles = [...tiles, ...dedupe(OPENERS, [...tiles, ...selected])].slice(0, 8);
      }
      json(res, 200, { tiles, source: mode === 'answer' ? 'predicted' : mode, ms: Date.now() - t0, coreSlots: slots });
    } catch (e) {
      // Never leave him staring at an empty grid because a model timed out.
      json(res, 200, { tiles: OPENERS, source: 'fallback', error: String(e.message) });
    }
  },

  // Always one selection away, from anywhere. If the grid is about Emma's graduation and
  // he suddenly can't breathe, a predicted grid traps him. No tile quality fixes that.
  'GET /api/urgent': async (req, res) => json(res, 200, { tiles: profile.urgent ?? [] }),

  // The mic, transcribed HERE. The browser sends 16kHz mono WAV; whisper.cpp runs on this
  // machine and the audio is deleted immediately. Nothing is uploaded to anyone.
  'POST /api/listen': async (req, res) => {
    const chunks = [];
    for await (const c of req) chunks.push(c);
    try {
      const text = await transcribe(Buffer.concat(chunks));
      json(res, 200, { text });
    } catch (e) {
      json(res, 500, { error: String(e.message) });
    }
  },

  'POST /api/compose': async (req, res) => {
    const { selected = [], partner = '', literal = false, mode = 'answer' } = await body(req);
    try {
      const t0 = Date.now();
      const candidates = await composeSentence({ selected, partner, profile, literal, mode });
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
  console.log(`  input: touch + scanning (arrow keys simulate EOG)`);
  asrReady().then((ok) => console.log(`  listening: ${ok ? 'whisper.cpp (local — audio never leaves this machine)' : 'MISSING — asr/ not built'}\n`));
});
