// The voice engine.
//
// This is the seam that matters. Today it is macOS `say` — a stranger's voice,
// a placeholder. When the clone is trained (Piper/ONNX, fine-tuned on ~60s of the
// user's own home-video audio), you swap ONE function and nothing else in the app
// changes. Keep it that way.
//
// Pre-rendering: the ~200 utterances a user says most get synthesized once and
// cached to disk. That is a real product feature (instant speech, no inference
// wait on a 4-minute-per-sentence input channel) AND it is demo insurance.

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, access } from 'node:fs/promises';
import path from 'node:path';

const exec = promisify(execFile);
const CACHE = path.join(process.cwd(), 'voices', 'cache');
const PLACEHOLDER_VOICE = process.env.STILLME_SAY_VOICE ?? 'Daniel';

// The model path is part of the key: without it, user B's cache would be served user A's cloned
// voice for the same words — a stranger's sentence in the wrong person's voice.
const key = (text, voice, model) => createHash('sha1').update(`${voice}::${model ?? ''}::${text}`).digest('hex');

async function exists(p) {
  try { await access(p); return true; } catch { return false; }
}

// A wav with no audio in it. `say` creates its output file with a 4 KB header (data length 0)
// a few hundred ms before any audio lands, and a failed or killed run leaves that, or 0 bytes,
// behind. Served from cache, that phrase would play silence on every use from then on.
function hasAudio(buf) {
  if (buf.length < 12 || buf.toString('ascii', 0, 4) !== 'RIFF') return false;
  for (let at = 12; at + 8 <= buf.length; ) {
    const size = buf.readUInt32LE(at + 4);
    if (buf.toString('ascii', at, at + 4) === 'data') return buf.length > at + 8;
    at += 8 + size + (size & 1);
  }
  return false;
}

/** ENGINE: macOS `say`. Placeholder only — this is NOT his voice. */
async function synthPlaceholder(text, outPath) {
  // '--' ends option parsing. Text starting with "-r 1 ..." was read as a speaking rate, and `say`
  // then waited forever on stdin. The timeout covers whatever else could hang it.
  await exec('say', ['-v', PLACEHOLDER_VOICE, '-o', outPath, '--data-format=LEI16@22050', '--', text],
    { timeout: 15000 });
}

/** ENGINE: the cloned voice. Wire this to the trained Piper model. */
async function synthCloned(text, outPath, model) {
  // NO SHELL. The old version piped text through `sh -c` with only JSON.stringify quoting, which
  // does not escape $ or backticks — a tile like `$(rm -rf ~)` would have executed. piper reads
  // its text from stdin, so hand it there directly and the argv never touches a shell.
  const child = execFile('piper', ['--model', model, '--output_file', outPath], { timeout: 30000 });
  child.stdin.end(text);
  await new Promise((resolve, reject) => {
    child.on('error', reject);
    child.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`piper exited ${code}`))));
  });
}

/**
 * Speak. Returns a wav buffer. Cached by (text, voice) — so the utterances he
 * says every day cost nothing after the first time.
 */
export async function speak(text, { voice = 'placeholder', model = null } = {}) {
  await mkdir(CACHE, { recursive: true });
  const k = key(text, voice, model);
  const out = path.join(CACHE, `${k}.wav`);

  if (await exists(out)) {
    const wav = await readFile(out);
    if (hasAudio(wav)) return { wav, cached: true, voice };
    await rm(out, { force: true });   // a stub from an older build or a killed run: make it again
  }

  // Two picks of the same new phrase at once (a double tap on Urgent) share one synthesis,
  // instead of the second one reading the first one's half-written file.
  if (!inflight.has(k)) {
    inflight.set(k, synth(text, out, voice, model).finally(() => inflight.delete(k)));
  }
  return { wav: await inflight.get(k), cached: false, voice };
}

const inflight = new Map();

// Write to a temp file and rename it into place only once it holds audio, so the cache path is
// never a partial file. The temp name must still end in .wav: `say` picks the format from it.
async function synth(text, out, voice, model) {
  const tmp = out.replace(/\.wav$/, `.${randomUUID()}.part.wav`);
  try {
    if (voice === 'cloned' && model) await synthCloned(text, tmp, model);
    else await synthPlaceholder(text, tmp);
    const wav = await readFile(tmp);
    if (!hasAudio(wav)) throw new Error('speech engine produced no audio');
    await rename(tmp, out);
    return wav;
  } finally {
    await rm(tmp, { force: true });
  }
}

/** Warm the cache for a user's most frequent utterances. Run this before filming. */
export async function prerender(utterances, opts) {
  const done = [];
  for (const u of utterances) {
    await speak(u, opts);
    done.push(u);
  }
  return done;
}

export const isCloneReady = async (model) => (model ? exists(model) : false);
