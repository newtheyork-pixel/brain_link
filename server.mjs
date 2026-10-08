// StillMe — local dev server.  node server.mjs  →  http://localhost:8000
//
// Zero dependencies. Ships to iPad as a native app later; this is the loop.

import { createServer } from 'node:http';
import { readFile, appendFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { predictTiles, composeSentence, health } from './lib/llm.mjs';
import { buildGrid, clusterOf, dedupe } from './lib/tiles.mjs';
import { speak, prerender } from './lib/voice.mjs';
import { transcribe, available as asrReady } from './lib/listen.mjs';

const PORT = process.env.PORT ?? 8000;
const ROOT = process.cwd();
const SESSIONS = path.join(ROOT, 'data', 'sessions.jsonl');

// Load the real per-user profile if present (gitignored); fall back to the committed template.
// A real family's profile must never be in the repo, so the app must run without it.
let profile;
try { profile = JSON.parse(await readFile(path.join(ROOT, 'data', 'profile.json'), 'utf8')); }
catch { profile = JSON.parse(await readFile(path.join(ROOT, 'data', 'profile.example.json'), 'utf8')); }
const OPENERS = profile.openers;
// yes/no are answers. On an Ask or Tell grid they are two of his eight slots spent on words he
// can never use, so every list that fills those grids (static, padding, fallback) leaves them out.
const YESNO = [clusterOf('yes'), clusterOf('no')];
const openersFor = (mode) => (mode === 'answer' ? OPENERS : OPENERS.filter((t) => !YESNO.includes(clusterOf(t))));

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

const httpError = (status, message) => Object.assign(new Error(message), { status });

// Every request body goes through one cap. Nothing real is big (a 20 s mic clip is ~640 KB, a
// record.html chunk ~0.9 MB), and an uncapped POST grows this process until it dies, and this is
// the same process that speaks the Urgent grid. Past the cap the rest is read and thrown away,
// not buffered, so the 413 can still be sent back.
function readCapped(req, max) {
  if (+(req.headers['content-length'] ?? 0) > max) return Promise.reject(httpError(413, 'body too large'));
  return new Promise((resolve, reject) => {
    const chunks = [];
    let n = 0, over = false;
    req.on('data', (c) => {
      if (over) return;
      n += c.length;
      if (n > max) { over = true; chunks.length = 0; reject(httpError(413, 'body too large')); }
      else chunks.push(c);
    });
    req.on('end', () => { if (!over) resolve(Buffer.concat(chunks)); });
    req.on('error', reject);
  });
}

// JSON only. A cross-site page can POST text/plain with no CORS preflight, and this used to parse
// it anyway, so any open tab could append rows to the research logs.
async function body(req, max = 1 << 20) {
  if (!/^application\/json\b/i.test(req.headers['content-type'] ?? '')) throw httpError(415, 'json only');
  return JSON.parse((await readCapped(req, max)).toString() || '{}');
}

/** Every sentence spoken is a data point. This file IS the research result. */
async function logEvent(ev) {
  await mkdir(path.dirname(SESSIONS), { recursive: true });
  await appendFile(SESSIONS, JSON.stringify({ ...ev, at: new Date().toISOString() }) + '\n');
}

const routes = {
  // Health is about THIS server, not the model. It used to fail with 503 whenever no model was
  // reachable, and boot() waits on it, so a missing Ollama left an empty grid and an Urgent grid
  // without its sentences, even though tiles and compose both have offline fallbacks.
  'GET /api/health': async (req, res) => {
    let llm;
    try { llm = { ok: true, ...(await health()) }; }
    catch (e) { llm = { ok: false, where: 'none (no model reachable)', offline: true, error: String(e.message) }; }
    try {
      json(res, 200, {
        ok: true,
        llm,
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
    let b, down = null;
    try { b = await health(); } catch (e) { down = String(e.message); b = { where: 'none (no model reachable)', model: null, offline: true }; }
    json(res, 200, {
      running_on: b.where,
      model: b.model,
      offline: b.offline,
      speech_recognition: (await asrReady()) ? 'whisper.cpp — on this machine' : 'MISSING (run asr/ setup)',
      note: down
        ? `No model. Tiles fall back to openers and sentences are spoken exactly as picked. ${down}`
        : b.offline
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

    if (!predictive && !partner) return json(res, 200, { tiles: openersFor(mode), source: 'static' });
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
      if (mode !== 'answer') tiles = tiles.filter((t) => !YESNO.includes(clusterOf(t)));

      // Never hand him an empty board. Echo-strip + dedupe can eat a whole grid, and a missing
      // tile is a thing he cannot say.
      if (tiles.length < 5) {
        tiles = [...tiles, ...dedupe(openersFor(mode), [...tiles, ...selected])].slice(0, 8);
      }
      json(res, 200, { tiles, source: mode === 'answer' ? 'predicted' : mode, ms: Date.now() - t0, coreSlots: slots });
    } catch (e) {
      // Never leave him staring at an empty grid because a model timed out.
      json(res, 200, { tiles: openersFor(mode), source: 'fallback', error: String(e.message) });
    }
  },

  // Always one selection away, from anywhere. If the grid is about Emma's graduation and
  // he suddenly can't breathe, a predicted grid traps him. No tile quality fixes that.
  'GET /api/urgent': async (req, res) => json(res, 200, { tiles: profile.urgent ?? [] }),

  // The mic, transcribed HERE. The browser sends 16kHz mono WAV; whisper.cpp runs on this
  // machine and the audio is deleted immediately. Nothing is uploaded to anyone.
  'POST /api/listen': async (req, res) => {
    // Raw audio, so body()'s JSON check does not cover it. mic.js always sends audio/wav.
    if (!/^audio\/wav\b/i.test(req.headers['content-type'] ?? '')) return json(res, 415, { error: 'audio/wav only' });
    const audio = await readCapped(req, 4 << 20);
    try {
      const text = await transcribe(audio);
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

  // Eye-tracking telemetry. The camera runs in HIS browser, not mine, so the app measures
  // itself and writes the result here. Calibration residual, live feature ranges, and a real
  // per-tile hit rate — a number, instead of "does the dot feel about right".
  'POST /api/gazelog': async (req, res) => {
    const ev = await body(req);
    await mkdir(path.dirname(SESSIONS), { recursive: true });
    await appendFile(path.join(ROOT, 'data', 'gaze.jsonl'),
      JSON.stringify({ ...ev, at: new Date().toISOString() }) + '\n');
    json(res, 200, { ok: true });
  },

  // Raw eye recordings from /record.html: every frame's landmarks + eye crops + where the dot
  // was. The tracker is tuned OFFLINE against these (eval/gaze/), not by trial on his face.
  'POST /api/gazerec': async (req, res) => {
    // A chunk is 60 frames (~0.9 MB); record.js re-sends a failed chunk with the next one, so
    // leave room for a few.
    const { id, lines } = await body(req, 8 << 20);
    if (!/^[\w-]{1,64}$/.test(id ?? '') || !Array.isArray(lines)) return json(res, 400, { error: 'bad recording chunk' });
    const dir = path.join(ROOT, 'data', 'recordings');
    await mkdir(dir, { recursive: true });
    await appendFile(path.join(dir, `${id}.jsonl`), lines.map((l) => JSON.stringify(l)).join('\n') + '\n');
    json(res, 200, { ok: true });
  },

  // selections_per_sentence and seconds_to_sentence are the two numbers the paper is about.
  'POST /api/log': async (req, res) => {
    await logEvent(await body(req));
    json(res, 200, { ok: true });
  },
};

// Loopback binding stops other machines, not other web pages in his own browser. A page that
// rebinds its DNS name to 127.0.0.1 arrives with ITS hostname in Host, and could then read
// /api/health (the profile name and phrase map). So only answer to the names this server is
// opened by. STILLME_ALLOWED_HOSTS (comma-separated host:port) adds more, kept out of the repo.
const ALLOWED_HOSTS = new Set([`localhost:${PORT}`, `127.0.0.1:${PORT}`,
  ...(String(PORT) === '80' ? ['localhost', '127.0.0.1'] : []),
  ...(process.env.STILLME_ALLOWED_HOSTS ?? '').split(',').map((h) => h.trim()).filter(Boolean)]);

function sameOrigin(req) {
  const o = req.headers.origin;
  if (!o) return true;                       // not a cross-site browser request
  try { return new URL(o).host === req.headers.host; } catch { return false; }   // "null" too
}

createServer(async (req, res) => {
  if (!ALLOWED_HOSTS.has(req.headers.host ?? '')) { res.writeHead(421); return res.end('wrong host'); }
  // Writes only from our own pages. Without this any open tab could forge rows in the logs the
  // paper rests on, or fill the disk through /api/gazerec.
  if (req.method !== 'GET' && req.method !== 'HEAD' && !sameOrigin(req)) return json(res, 403, { error: 'cross-origin' });

  const url = new URL(req.url, 'http://x');
  const route = routes[`${req.method} ${url.pathname}`];
  if (route) {
    try { return await route(req, res); }
    catch (e) {
      // After a 413 the client may still be sending; close the socket once the reply is out.
      if (e.status === 413) { res.setHeader('connection', 'close'); res.on('finish', () => req.destroy()); }
      return json(res, e.status ?? 500, { error: String(e.message) });
    }
  }

  const file = url.pathname === '/' ? '/index.html' : url.pathname;
  try {
    const buf = await readFile(path.join(ROOT, 'public', file));
    res.writeHead(200, { 'content-type': MIME[path.extname(file)] ?? 'text/plain' });
    res.end(buf);
  } catch { res.writeHead(404); res.end('not found'); }
// Bind loopback ONLY. Binding all interfaces exposed /api/speak and /api/listen to the whole LAN
// — anyone on the network could make the device speak, or send it audio. "Nothing leaves the
// device" has to be true at the socket, not just in the README.
}).listen(PORT, '127.0.0.1', () => {
  console.log(`\n  StillMe → http://localhost:${PORT}`);
  console.log(`  user: ${profile.name}   voice: ${profile.voiceModel ? 'CLONED' : 'placeholder (not his voice yet)'}`);
  console.log(`  input: touch + scanning (arrow keys simulate EOG)`);
  asrReady().then((ok) => console.log(`  listening: ${ok ? 'whisper.cpp (local — audio never leaves this machine)' : 'MISSING — asr/ not built'}\n`));
  // Warm the cache for the urgent and instant phrases, so the first "can't breathe" of the day
  // does not wait on synthesis. Same voice and model key as /api/speak uses. Not awaited.
  prerender(Object.values(profile.instant ?? {}).filter((t) => typeof t === 'string' && t.trim()),
    { voice: profile.voiceModel ? 'cloned' : 'placeholder', model: profile.voiceModel })
    .catch((e) => console.log(`  prerender failed: ${e.message}`));
});
