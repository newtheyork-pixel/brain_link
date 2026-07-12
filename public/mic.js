// The microphone → 16kHz mono WAV → our own server → whisper.cpp on this machine.
//
// Chrome's Web Speech API would have been three lines. It also uploads the microphone to
// Google — which, in an app whose whole claim is "nothing ever leaves the device," is not a
// shortcut but a lie. The most private audio in the house is a dying man's family talking
// to him. It does not go to a third party.
//
// LIVE: whisper re-runs on the growing buffer roughly every second, so the words appear as
// they are spoken instead of arriving in a lump after everyone stops talking. base.en on
// Metal does a short utterance in ~0.2s, so we can afford to keep re-reading the sentence
// as it grows.

const TARGET_HZ = 16000;
const INTERIM_MS = 900;    // re-transcribe the buffer this often while they are talking
const HANG_MS = 1100;      // this much quiet = the sentence is finished
const MIN_MS = 350;        // ignore a cough
const MAX_MS = 20000;      // never hold the mic open forever

/** Float32 @ ctx rate → Float32 @ 16kHz. Averaging (not picking) avoids aliasing hiss. */
function downsample(buf, from, to = TARGET_HZ) {
  if (from === to) return buf;
  const ratio = from / to;
  const out = new Float32Array(Math.round(buf.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const a = Math.floor(i * ratio);
    const b = Math.min(Math.floor((i + 1) * ratio), buf.length);
    let sum = 0;
    for (let j = a; j < b; j++) sum += buf[j];
    out[i] = sum / Math.max(1, b - a);
  }
  return out;
}

/** 16-bit PCM WAV. whisper.cpp reads this directly. */
function encodeWav(samples, rate = TARGET_HZ) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (const s of samples) {
    const c = Math.max(-1, Math.min(1, s));
    v.setInt16(o, c < 0 ? c * 0x8000 : c * 0x7fff, true);
    o += 2;
  }
  return new Blob([buf], { type: 'audio/wav' });
}

const flatten = (chunks) => {
  const all = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let o = 0;
  for (const c of chunks) { all.set(c, o); o += c.length; }
  return all;
};

export function createMic({ onInterim, onFinal, onError, onLevel }) {
  let ctx, stream, node, source;
  let chunks = [];          // the current sentence
  let recording = false;
  let speaking = false;
  let quietAt = 0, lastInterim = 0, startedAt = 0;
  let inFlight = false;

  // The old fixed threshold (0.012) was almost certainly above the noise floor of a real
  // laptop mic in a quiet room — so "is anyone talking?" was permanently false and nothing
  // was ever captured. Learn the room instead of guessing at it.
  let floor = 0.004, calibrating = true, calSamples = [], calUntil = 0;

  async function send(audio, final) {
    if (inFlight && !final) return;              // don't stack interim requests
    if ((audio.length / TARGET_HZ) * 1000 < MIN_MS) return;
    inFlight = true;
    try {
      const res = await fetch('/api/listen', {
        method: 'POST', headers: { 'content-type': 'audio/wav' }, body: encodeWav(audio),
      });
      const { text, error } = await res.json();
      if (error) throw new Error(error);
      const clean = (text ?? '').replace(/\[.*?\]/g, '').trim();
      if (!clean || clean.length < 2 || /^\W+$/.test(clean)) return;
      (final ? onFinal : onInterim)(clean);
    } catch (e) {
      if (final) onError(`Could not transcribe: ${e.message}`);
    } finally {
      inFlight = false;
    }
  }

  return {
    get on() { return recording; },

    async start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true,
                   autoGainControl: true },
        });
      } catch (e) {
        onError(e.name === 'NotAllowedError'
          ? 'Microphone blocked. Allow it in the address bar, then tap Listen.'
          : `No microphone: ${e.message}`);
        return false;
      }

      ctx = new AudioContext();
      if (ctx.state === 'suspended') await ctx.resume();   // Chrome starts it suspended
      source = ctx.createMediaStreamSource(stream);
      node = ctx.createScriptProcessor(4096, 1, 1);

      recording = true;
      startedAt = performance.now();
      calUntil = startedAt + 500;
      calibrating = true;
      calSamples = [];

      node.onaudioprocess = (e) => {
        if (!recording) return;
        const raw = e.inputBuffer.getChannelData(0);
        const now = performance.now();

        let sum = 0;
        for (let i = 0; i < raw.length; i++) sum += raw[i] * raw[i];
        const rms = Math.sqrt(sum / raw.length);
        onLevel?.(rms);   // so a human can SEE the mic is alive

        // First 500ms: listen to the room, don't judge it.
        if (calibrating) {
          calSamples.push(rms);
          if (now < calUntil) return;
          const med = calSamples.sort((a, b) => a - b)[Math.floor(calSamples.length / 2)] || 0.002;
          floor = Math.max(med * 3, 0.006);   // speech sits well above the room's own hum
          calibrating = false;
          return;
        }

        if (rms > floor) {
          speaking = true;
          quietAt = 0;
          chunks.push(downsample(new Float32Array(raw), ctx.sampleRate));
        } else if (speaking) {
          chunks.push(downsample(new Float32Array(raw), ctx.sampleRate)); // keep the tail
          if (!quietAt) quietAt = now;
          if (now - quietAt > HANG_MS) {
            send(flatten(chunks), true);      // the sentence is done
            chunks = []; speaking = false; quietAt = 0; startedAt = now;
            return;
          }
        }

        // LIVE: keep re-reading the sentence as it grows, so the words show up while they
        // are still being said.
        if (speaking && chunks.length && now - lastInterim > INTERIM_MS) {
          lastInterim = now;
          send(flatten(chunks), false);
        }

        if (now - startedAt > MAX_MS && chunks.length) {
          send(flatten(chunks), true);
          chunks = []; speaking = false; startedAt = now;
        }
      };

      source.connect(node);
      // ScriptProcessor only fires when connected onward. A zero-gain sink means it runs
      // without pushing the microphone back out of the speakers.
      const mute = ctx.createGain();
      mute.gain.value = 0;
      node.connect(mute);
      mute.connect(ctx.destination);
      return true;
    },

    stop() {
      if (recording && chunks.length) send(flatten(chunks), true);  // don't lose a half-sentence
      recording = false;
      speaking = false;
      chunks = [];
      try { node?.disconnect(); source?.disconnect(); } catch {}
      stream?.getTracks().forEach((t) => t.stop());
      ctx?.close().catch(() => {});
      ctx = stream = node = source = null;
    },
  };
}
