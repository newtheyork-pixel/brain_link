// The microphone → a 16kHz mono WAV → our own server → whisper.cpp on this machine.
//
// Chrome's Web Speech API would have been three lines. It also uploads the microphone to
// Google — which, in an app whose whole claim is "nothing ever leaves the device," is not a
// shortcut but a lie. The most private audio in the house is a dying man's family talking
// to him. It does not go to a third party.
//
// We encode the WAV in the page (Web Audio, no ffmpeg, no dependency) and auto-stop on
// silence, so nobody has to press anything twice mid-conversation.

const TARGET_HZ = 16000;   // what whisper wants
const SILENCE = 0.012;     // RMS below this is "nobody is talking"
const HANG_MS = 1200;      // ...for this long, and we assume they finished the sentence
const MIN_MS = 400;        // ignore a cough
const MAX_MS = 15000;      // never hold the mic open forever

/** Float32 @ ctx rate → Float32 @ 16kHz. Linear resample is plenty for speech. */
function downsample(buf, from, to = TARGET_HZ) {
  if (from === to) return buf;
  const ratio = from / to;
  const out = new Float32Array(Math.round(buf.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(Math.floor((i + 1) * ratio), buf.length);
    let sum = 0;
    for (let j = start; j < end; j++) sum += buf[j];
    out[i] = sum / Math.max(1, end - start);   // average, not pick — avoids aliasing hiss
  }
  return out;
}

/** 16-bit PCM WAV. whisper.cpp reads this directly. */
function encodeWav(samples, rate = TARGET_HZ) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };

  str(0, 'RIFF');
  v.setUint32(4, 36 + samples.length * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  v.setUint32(16, 16, true);      // PCM chunk size
  v.setUint16(20, 1, true);       // format = PCM
  v.setUint16(22, 1, true);       // mono
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, 'data');
  v.setUint32(40, samples.length * 2, true);

  let off = 44;
  for (const s of samples) {
    const c = Math.max(-1, Math.min(1, s));
    v.setInt16(off, c < 0 ? c * 0x8000 : c * 0x7fff, true);
    off += 2;
  }
  return new Blob([buf], { type: 'audio/wav' });
}

export function createMic({ onTranscript, onError, onSpeechStart }) {
  let ctx = null, stream = null, node = null, source = null;
  let chunks = [], recording = false, sawSpeech = false;
  let quietSince = 0, startedAt = 0, busy = false;

  async function flush() {
    if (!chunks.length || busy) { chunks = []; sawSpeech = false; return; }
    const audio = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
    let o = 0;
    for (const c of chunks) { audio.set(c, o); o += c.length; }
    chunks = [];
    sawSpeech = false;

    if ((audio.length / TARGET_HZ) * 1000 < MIN_MS) return;

    busy = true;
    try {
      const res = await fetch('/api/listen', {
        method: 'POST',
        headers: { 'content-type': 'audio/wav' },
        body: encodeWav(audio),
      });
      const { text, error } = await res.json();
      if (error) throw new Error(error);
      const clean = (text ?? '').trim();
      // whisper emits "[BLANK_AUDIO]"-ish noise for silence; don't hand that to the model.
      if (clean && clean.length > 1 && !/^\W+$/.test(clean)) onTranscript(clean);
    } catch (e) {
      onError(`Could not transcribe: ${e.message}`);
    } finally {
      busy = false;
    }
  }

  return {
    get on() { return recording; },

    async start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
      } catch (e) {
        return onError(e.name === 'NotAllowedError'
          ? 'Microphone blocked. Allow it in the address bar, then tap Listen.'
          : `No microphone: ${e.message}`);
      }

      ctx = new AudioContext();
      source = ctx.createMediaStreamSource(stream);
      node = ctx.createScriptProcessor(4096, 1, 1);
      recording = true;
      startedAt = performance.now();
      quietSince = 0;

      node.onaudioprocess = (e) => {
        if (!recording) return;
        const raw = e.inputBuffer.getChannelData(0);

        let sum = 0;
        for (let i = 0; i < raw.length; i++) sum += raw[i] * raw[i];
        const rms = Math.sqrt(sum / raw.length);

        if (rms > SILENCE) {
          if (!sawSpeech) { sawSpeech = true; onSpeechStart?.(); }
          quietSince = 0;
          chunks.push(downsample(new Float32Array(raw), ctx.sampleRate));
        } else if (sawSpeech) {
          // Keep the tail — cutting on the first quiet frame clips the last word.
          chunks.push(downsample(new Float32Array(raw), ctx.sampleRate));
          const now = performance.now();
          if (!quietSince) quietSince = now;
          if (now - quietSince > HANG_MS) { flush(); startedAt = now; }
        }

        if (performance.now() - startedAt > MAX_MS && chunks.length) {
          flush();
          startedAt = performance.now();
        }
      };

      source.connect(node);
      node.connect(ctx.destination);
      return true;
    },

    stop() {
      recording = false;
      try { node?.disconnect(); source?.disconnect(); } catch {}
      stream?.getTracks().forEach((t) => t.stop());
      ctx?.close().catch(() => {});
      ctx = stream = node = source = null;
      chunks = [];
      sawSpeech = false;
    },
  };
}
