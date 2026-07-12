// Speech recognition — ON THIS MACHINE.
//
// Chrome's Web Speech API uploads the microphone to Google. In an app whose entire claim is
// "nothing ever leaves the device," that is not a shortcut, it is a lie: the most private
// audio in the house (a dying man's family talking to him) would be shipped to a third party.
//
// whisper.cpp, base.en, Metal. Measured: 0.15s for a short utterance. Nothing leaves the Mac.
// Ship path is the same shape: iOS SFSpeechRecognizer with requiresOnDeviceRecognition = true.

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { writeFile, unlink, access } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import os from 'node:os';

const exec = promisify(execFile);
const ASR = path.join(process.cwd(), 'asr');
const BIN = path.join(ASR, 'whisper-cli');
const MODEL = path.join(ASR, 'ggml-base.en.bin');

export async function available() {
  try { await access(BIN); await access(MODEL); return true; } catch { return false; }
}

/** wav: a Buffer of 16kHz mono PCM WAV (the browser encodes it; no ffmpeg anywhere). */
export async function transcribe(wav) {
  const tmp = path.join(os.tmpdir(), `stillme-${randomUUID()}.wav`);
  await writeFile(tmp, wav);
  try {
    const { stdout } = await exec(BIN, [
      '-m', MODEL,
      '-f', tmp,
      '-nt',            // no timestamps
      '-np',            // no progress spam
      '-t', '4',        // threads
      '--language', 'en',
    ], { cwd: ASR, timeout: 30000 });

    return stdout
      .replace(/\[.*?\]/g, '')          // stray timestamp brackets
      .replace(/\(.*?\)/g, '')          // (BLANK_AUDIO), (silence), etc.
      .replace(/\s+/g, ' ')
      .trim();
  } finally {
    unlink(tmp).catch(() => {});
  }
}
