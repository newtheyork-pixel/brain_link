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
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile, access } from 'node:fs/promises';
import path from 'node:path';

const exec = promisify(execFile);
const CACHE = path.join(process.cwd(), 'voices', 'cache');
const PLACEHOLDER_VOICE = process.env.STILLME_SAY_VOICE ?? 'Daniel';

const key = (text, voice) => createHash('sha1').update(`${voice}::${text}`).digest('hex');

async function exists(p) {
  try { await access(p); return true; } catch { return false; }
}

/** ENGINE: macOS `say`. Placeholder only — this is NOT his voice. */
async function synthPlaceholder(text, outPath) {
  await exec('say', ['-v', PLACEHOLDER_VOICE, '-o', outPath, '--data-format=LEI16@22050', text]);
}

/** ENGINE: the cloned voice. Wire this to the trained Piper model. */
async function synthCloned(text, outPath, model) {
  // piper --model voices/<name>.onnx --output_file out.wav  <<< text
  await exec('sh', ['-c',
    `printf %s ${JSON.stringify(text)} | piper --model ${JSON.stringify(model)} --output_file ${JSON.stringify(outPath)}`]);
}

/**
 * Speak. Returns a wav buffer. Cached by (text, voice) — so the utterances he
 * says every day cost nothing after the first time.
 */
export async function speak(text, { voice = 'placeholder', model = null } = {}) {
  await mkdir(CACHE, { recursive: true });
  const out = path.join(CACHE, `${key(text, voice)}.wav`);

  if (await exists(out)) return { wav: await readFile(out), cached: true, voice };

  if (voice === 'cloned' && model) await synthCloned(text, out, model);
  else await synthPlaceholder(text, out);

  return { wav: await readFile(out), cached: false, voice };
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
