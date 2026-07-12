// StillMe — the loop.
//
// THE ARCHITECTURE THAT MATTERS: every input method emits the same six events.
//
//     LEFT · RIGHT · UP · DOWN · SELECT · UNDO
//
// Touch emits them. Arrow keys emit them. The camera emits them (gaze moves the cursor, a
// dwell selects). And when the EOG board arrives, its Bluetooth peripheral emits them too.
// Nothing below this line changes.
//
// The zone list is NOT just the tiles. A switch or eye user who can reach the tiles but not
// "Say it" can build a sentence and never speak it — so Say-it and Urgent are zones too, and
// when the confirm sheet is open the SAME six events drive the sheet instead of the grid
// underneath it.

import { createMic } from '/mic.js';
import { createGaze } from '/gaze.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const SAY = '__say__';
const URGENT = '__urgent__';

const state = {
  selected: [],
  tiles: [],
  cursor: 0,
  driver: 'touch',
  literal: false,
  predictive: true,
  urgent: false,
  listening: false,
  speaking: false,
  mode: 'answer',   // answer | ask | tell — the only way he gets to start a conversation
  coreSlots: 2,     // tiles that never move. Motor learning vs prediction — a measured knob.
  dwellMs: 900,
  pinned: 0,
  startedAt: null,
  selections: 0,
  profile: null,
  instant: {},      // tile -> full sentence, spoken the moment it is picked. From the server.
};

// Any async response that lands after the world moved on must be discarded, not applied.
// Without this a stale prediction overwrites the EMERGENCY grid, and a compose from a
// previous mode pops open a confirm sheet full of the wrong sentence.
let tilesSeq = 0, composeSeq = 0;

/* ---------- input bus ---------- */

const bus = {
  handlers: [],
  on(fn) { this.handlers.push(fn); },
  emit(e) { for (const h of this.handlers) h(e); },
};

/** Everything the cursor can land on — the tiles, then the two controls. */
const zones = () => [...state.tiles, SAY, URGENT];

function moveCursor(d) {
  const n = zones().length;
  if (!n) return;
  state.cursor = (state.cursor + d + n) % n;   // wrap by REAL length: grids shrink below 8
  renderGrid();
}

bus.on((event) => {
  // The sheet owns the six events while it is up. Without this, SELECT reaches through and
  // picks a tile on the hidden grid — corrupting the very sentence awaiting confirmation.
  if (!$('#confirm').hidden) return sheetEvent(event);
  if (!$('#calib').hidden) return;

  const cols = 4;
  if (event === 'LEFT')  return moveCursor(-1);
  if (event === 'RIGHT') return moveCursor(1);
  if (event === 'UP')    return moveCursor(-cols);
  if (event === 'DOWN')  return moveCursor(cols);
  if (event === 'UNDO')  return undo();
  if (event !== 'SELECT') return;

  const z = zones()[state.cursor];
  if (z === SAY) return compose();
  if (z === URGENT) return toggleUrgent();
  return pick(z);
});

/** Six events, driving the confirm sheet. A switch user must be able to speak. */
let sheetIdx = 0;
function sheetEvent(event) {
  const opts = [...$$('.candidate'), $('#cancel')];
  if (!opts.length) return;
  if (event === 'LEFT' || event === 'UP') sheetIdx = (sheetIdx - 1 + opts.length) % opts.length;
  else if (event === 'RIGHT' || event === 'DOWN') sheetIdx = (sheetIdx + 1) % opts.length;
  else if (event === 'UNDO') return closeConfirm();
  else if (event === 'SELECT') return opts[sheetIdx].click();
  opts.forEach((o, i) => o.classList.toggle('cursor', i === sheetIdx));
  opts[sheetIdx].focus();
}

// Keyboard → the six events. Only while the scanning driver is on, and NEVER while the
// caregiver is typing: a spacebar in the partner box used to fire SELECT on the pinned "yes"
// tile and make the device say "Yes." out loud, in the patient's voice.
window.addEventListener('keydown', (e) => {
  if (state.driver !== 'scan') return;
  const t = e.target;
  if (/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(t.tagName) || t.isContentEditable) return;
  if (!$('#settings').hidden) return;

  const map = { ArrowLeft: 'LEFT', ArrowRight: 'RIGHT', ArrowUp: 'UP', ArrowDown: 'DOWN',
    ' ': 'SELECT', Enter: 'SELECT', Backspace: 'UNDO' };
  const ev = map[e.key];
  if (!ev) return;
  e.preventDefault();
  bus.emit(ev);
});

/* ---------- the loop ---------- */

async function pick(tile) {
  if (!tile || state.speaking) return;
  if (!state.startedAt) state.startedAt = performance.now();

  // URGENT ALWAYS SPEAKS, IMMEDIATELY, AND KEEPS HIS SENTENCE.
  // This used to be gated on having nothing selected — so tapping "can't breathe" halfway
  // through a sentence said NOTHING and quietly appended the words to the sentence instead.
  // The one grid that exists so he is never trapped was itself a trap.
  if (state.urgent) {
    state.selections++;
    await say(state.instant[norm(tile)] ?? `${tile}.`, { instant: true, keep: true });
    return;
  }

  // "yes" needs no sentence built around it.
  const now = state.instant[norm(tile)];
  if (now && !state.selected.length) {
    state.selections++;
    await say(now, { instant: true });
    return;
  }

  state.selected.push(tile);
  state.selections++;
  renderSelected();
  renderHUD();
  await loadTiles();   // predictive narrowing: the next word he needs is now on screen
}

const norm = (s) => String(s).toLowerCase().trim();

function undo() {
  if (!state.selected.length) return;
  state.selected.pop();
  renderSelected();
  loadTiles();
}

async function loadTiles() {
  if (state.urgent) return;
  const seq = ++tilesSeq;
  const mode = state.mode;
  const partner = mode === 'answer' ? $('#partner-said').value.trim() : '';

  try {
    const res = await fetch('/api/tiles', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ selected: state.selected, partner, mode,
        predictive: state.predictive, coreSlots: state.coreSlots }),
    });
    if (!res.ok) throw new Error(`tiles ${res.status}`);
    const { tiles, source, ms, coreSlots } = await res.json();

    // The world may have moved while we waited: he hit Urgent, changed mode, picked another
    // word. Applying this now would overwrite the emergency grid or show him the wrong words.
    if (seq !== tilesSeq || state.urgent || mode !== state.mode) return;

    state.tiles = (tiles ?? []).slice(0, 8);
    state.pinned = coreSlots ?? 0;
    state.cursor = 0;
    $('#m-src').textContent = source === 'fallback' ? 'model down — fallback tiles'
      : source === 'predicted' ? `predicted ${ms}ms` : source;
    renderGrid();
  } catch (e) {
    toast(`Could not load words: ${e.message}`);
  }
}

async function toggleUrgent() {
  // Mutate state only AFTER the tiles arrive. Flipping first meant a failed fetch stranded
  // him in a fake urgent mode — stale tiles, and loadTiles permanently short-circuited.
  if (state.urgent) {
    state.urgent = false;
    document.body.classList.remove('in-urgent');
    $('#urgent').textContent = 'Urgent';
    await loadTiles();
    return;
  }
  try {
    const res = await fetch('/api/urgent');
    if (!res.ok) throw new Error(`urgent ${res.status}`);
    const { tiles } = await res.json();
    state.urgent = true;
    document.body.classList.add('in-urgent');
    $('#urgent').textContent = 'Back';
    state.tiles = tiles.slice(0, 8);
    state.cursor = 0;
    $('#m-src').textContent = 'urgent — fixed grid';
    renderGrid();
  } catch (e) {
    toast(`Urgent grid unavailable: ${e.message}`);
  }
}

async function compose() {
  if (!state.selected.length || state.speaking) return;
  const seq = ++composeSeq;
  const mode = state.mode;
  const selected = [...state.selected];

  $('#compose').disabled = true;
  try {
    const res = await fetch('/api/compose', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ selected, mode, literal: state.literal,
        partner: mode === 'answer' ? $('#partner-said').value.trim() : '' }),
    });
    if (!res.ok) throw new Error(`compose ${res.status}`);
    const { candidates } = await res.json();
    if (seq !== composeSeq || mode !== state.mode) return;   // he moved on
    showConfirm(candidates ?? []);
  } catch (e) {
    toast(`Could not build the sentence: ${e.message}`);
  } finally {
    $('#compose').disabled = !state.selected.length;
  }
}

/** answer = reply to them. ask = put a question to them. tell = say something unprompted. */
function setMode(mode) {
  state.mode = mode;
  state.selected = [];
  state.selections = 0;
  state.startedAt = null;
  for (const m of ['answer', 'ask', 'tell']) {
    const b = $(`#m-${m}`);
    b.classList.toggle('on', m === mode);
    b.setAttribute('aria-selected', String(m === mode));
  }
  $('#partner-block').hidden = mode !== 'answer';
  $('#compose').textContent = mode === 'ask' ? 'Ask it' : 'Say it';
  renderSelected();
  renderHUD();
  loadTiles();
}

/* ---------- speaking ---------- */

// He always chooses. The model proposes; it never speaks for him.
function showConfirm(candidates) {
  const box = $('#candidates');
  box.innerHTML = '';
  for (const c of candidates) {
    const b = document.createElement('button');
    b.className = 'candidate';
    b.textContent = c;
    b.onclick = () => say(c);
    box.appendChild(b);
  }
  sheetIdx = 0;
  $('#confirm').hidden = false;
  $$('.candidate')[0]?.classList.add('cursor');
  $$('.candidate')[0]?.focus();
}

function closeConfirm() {
  $('#confirm').hidden = true;
  $('#compose').focus();
}

let player = null;

/**
 * Speak.
 *
 * keep: do NOT clear his sentence afterwards. Urgent interjections and "wait, I'm talking"
 *       must not delete the sentence he was halfway through building.
 *
 * The old version treated a FAILED utterance as a successful one: on a 500 or a blocked
 * play() it toasted, then fell through and (a) logged the unsaid words as spoken — poisoning
 * selections_per_sentence and effective_wpm, the two numbers the whole paper rests on — and
 * (b) wiped his sentence, so "try again" was impossible. It must not have been said, so it
 * must not be logged, and his words must survive.
 */
async function say(text, { instant = false, keep = false } = {}) {
  if (state.speaking) return;          // double-tap used to stack two voices over each other
  state.speaking = true;
  closeConfirm();

  const startedAt = state.startedAt ?? performance.now();
  const wasListening = state.listening;
  if (wasListening) stopListening();   // or the mic hears his own voice and answers itself

  // Kill any previous audio before making a new one, and never let its onended fire late and
  // restart the mic in the middle of this utterance.
  if (player) { player.onended = null; player.pause(); player = null; }

  let url = null;
  try {
    const res = await fetch('/api/speak', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text, voice: state.profile?.voice ?? 'placeholder' }),
    });
    if (!res.ok) throw new Error(`speak ${res.status}`);

    url = URL.createObjectURL(await res.blob());
    player = new Audio(url);
    await player.play();               // a rejected promise here is the browser blocking sound

    const done = () => {
      if (url) { URL.revokeObjectURL(url); url = null; }
      state.speaking = false;
      if (wasListening) startListening();
    };
    player.onended = done;
    player.onerror = done;

    speaking(text);
    logUtterance(text, startedAt, instant);
    if (!keep) reset();
  } catch (e) {
    if (url) URL.revokeObjectURL(url);
    state.speaking = false;
    if (wasListening) startListening();
    toast(e.name === 'NotAllowedError'
      ? 'The browser blocked the sound. Tap anywhere on the page, then try again.'
      : `Could not speak: ${e.message}`);
    // NOT logged, NOT reset. His words stay on screen so he can try again without rebuilding.
    if (!instant && state.selected.length) $('#confirm').hidden = false;
  }
}

/** selections_per_sentence and seconds_to_sentence are the numbers the research is about. */
function logUtterance(text, startedAt, instant) {
  const elapsed = Math.max(0.4, (performance.now() - startedAt) / 1000);
  const words = text.trim().split(/\s+/).length;
  fetch('/api/log', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      spoken: text,
      tiles: instant ? [] : [...state.selected],
      mode: state.mode,
      urgent: state.urgent,
      instant,
      selections_per_sentence: state.selections,
      seconds_to_sentence: +elapsed.toFixed(1),
      effective_wpm: +(words / (elapsed / 60)).toFixed(1),
      predictive: state.predictive,
      literal: state.literal,
      driver: state.driver,
    }),
  }).catch(() => {});
}

function reset() {
  state.selected = [];
  state.selections = 0;
  state.startedAt = null;
  renderSelected();
  renderHUD();
  loadTiles();
}

/* ---------- listening ---------- */
//
// LOCAL. whisper.cpp on this machine, ~0.2s. Chrome's Web Speech API sends the microphone to
// Google — unacceptable in an app whose whole claim is that nothing leaves the device.

let mic = null;

function startListening() {
  // Guard on the OBJECT, synchronously. Guarding on state.listening (set only after start()
  // resolves) let a double-tap orphan a second, unstoppable microphone that transcribed
  // forever into a UI that said it was off.
  if (mic) return;

  const m = createMic({
    onInterim: (text) => {
      $('#partner-said').value = text;
      $('#partner-said').classList.add('interim');
    },
    onFinal: (text) => {
      $('#partner-said').value = text;
      $('#partner-said').classList.remove('interim');
      if (!state.selected.length && state.mode === 'answer') loadTiles();
    },
    onLevel: (rms) => {
      const pct = Math.min(100, Math.round(rms * 900));
      $('#level').style.width = `${pct}%`;
      $('#listen').classList.toggle('hearing', pct > 12);
    },
    onError: (msg) => { toast(msg); stopListening(); },
  });
  mic = m;

  m.start().then((ok) => {
    if (m !== mic) return;             // a stop() raced us
    if (!ok) { mic = null; return; }   // and NULL it, or one denied permission bricks the button
    state.listening = true;
    $('#listen').classList.add('on');
    $('#listen').textContent = 'Listening…';
    $('#listen').setAttribute('aria-pressed', 'true');
    $('#meter').hidden = false;
  }).catch(() => { if (m === mic) mic = null; });
}

function stopListening() {
  state.listening = false;
  mic?.stop();
  mic = null;
  $('#listen').classList.remove('on', 'hearing');
  $('#listen').textContent = 'Listen';
  $('#listen').setAttribute('aria-pressed', 'false');
  $('#partner-said').classList.remove('interim');
  $('#meter').hidden = true;
  $('#level').style.width = '0%';
}

/* ---------- eyes ---------- */
//
// The camera moves the SAME cursor the arrow keys move, and a dwell emits the SAME SELECT.
//
// Dwell, not blink-only: a webcam cannot tell a deliberate blink from a reflex, and a device
// that fires on a reflex is a device that puts words in a man's mouth. Blink is a SECOND way
// to confirm what he is already looking at — never the only one.

let gaze = null;
let dwellTile = -1, dwellStart = 0;

// TILE HYSTERESIS.
//
// A gaze point sitting near a border flickers between two tiles, and every flicker RESETS the
// dwell — so he stares at a word for ten seconds and it never picks. The cursor now stays where
// it is until the gaze is convincingly inside a different tile, for several frames running.
let candidateTile = -1, candidateFrames = 0;
const SWITCH_FRAMES = 4;      // frames of agreement before the cursor moves
const MARGIN = 0.72;          // must be this far inside the new tile, not just over the line

function tileUnder(x, y) {
  const tiles = $$('.tile');
  for (let i = 0; i < tiles.length; i++) {
    const r = tiles[i].getBoundingClientRect();
    // Shrink the hit box: being barely over the edge is not the same as looking AT it.
    const w = r.width * MARGIN / 2, h = r.height * MARGIN / 2;
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    if (Math.abs(x - cx) <= w && Math.abs(y - cy) <= h) return i;
  }
  return -1;
}

function tileAt(x, y) {
  const tiles = $$('.tile');
  for (let i = 0; i < tiles.length; i++) {
    const r = tiles[i].getBoundingClientRect();
    if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) return i;
  }
  return -1;
}

function resetDwell() {
  dwellTile = -1;
  dwellStart = 0;
  $$('.tile').forEach((t) => t.style.setProperty('--dwell', '0'));
}

function onGazePoint(x, y, locked) {
  const dot = $('#gaze-dot');
  dot.hidden = false;
  dot.style.transform = `translate(${x}px, ${y}px)`;
  dot.classList.toggle('locked', !!locked);   // he can see when the eye has actually landed

  if (!$('#confirm').hidden || !$('#calib').hidden || state.speaking) return resetDwell();

  const i = tileUnder(x, y);

  // Only accept a tile change once the gaze has agreed with itself for several frames. Without
  // this the cursor flickers across a border and the dwell restarts forever — he stares at a
  // word and it never picks.
  if (i !== dwellTile) {
    if (i === candidateTile) candidateFrames++;
    else { candidateTile = i; candidateFrames = 1; }
    if (candidateFrames < SWITCH_FRAMES) return;

    resetDwell();
    dwellTile = i;
    dwellStart = performance.now();
    if (i >= 0) { state.cursor = i; renderGrid(); }
    return;
  }
  candidateTile = i;
  candidateFrames = 0;
  if (i < 0) return;

  // The dwell only accumulates while the eye is HOLDING. If he is still moving, he has not
  // chosen yet — and a dwell that fills while the eye is in flight is a misfire waiting to
  // happen, which on this device means a word he did not mean, spoken aloud in his voice.
  if (!locked) { dwellStart = performance.now(); return; }

  const frac = Math.min(1, (performance.now() - dwellStart) / state.dwellMs);
  $$('.tile')[i]?.style.setProperty('--dwell', String(frac));
  if (frac >= 1) {
    resetDwell();
    dwellTile = -2;                    // refractory: don't instantly re-fire on the same tile
    bus.emit('SELECT');
  }
}

async function startGaze() {
  if (gaze) return;
  const g = createGaze({
    onGaze: onGazePoint,
    onBlink: () => {
      if (dwellTile >= 0 && $('#confirm').hidden && $('#calib').hidden && !state.speaking) {
        resetDwell();
        bus.emit('SELECT');
      }
    },
    onFace: (found) => {
      $('#gaze-dot').classList.toggle('lost', !found);
      if (!found) resetDwell();
    },
    onError: (msg) => { toast(msg); stopGaze(); },
    onCalibrationProgress: (p) => {
      if (p.state === 'done') {
        $('#calib').hidden = true;
        // The leave-one-out error, against half a tile. Anything worse and the wrong word gets
        // spoken in his voice — so say so, instead of congratulating him on a broken fit.
        // Which model won, and what each scored. If "head" wins, he is aiming with his head and
        // not his eyes — and that is a fact about him, not a bug to hide.
        const by = Object.entries(p.looByVariant ?? {}).map(([k, v]) => `${k} ${v}px`).join(' · ');
        $('#gaze-state').textContent =
          `X ±${p.errX}px / Y ±${p.errY}px (tile ${p.tile.w}x${p.tile.h}) · ${p.variant} · ${p.usable ? 'usable' : 'TOO LOOSE'}`;
        // Judge each axis against its OWN tile dimension. Horizontal was already fine while
        // vertical was failing, and a single blended number hid that completely.
        const okX = p.errX < p.tile.w * 0.45, okY = p.errY < p.tile.h * 0.45;
        const msg = (okX && okY)
          ? `Calibrated. Now run the test.`
          : !okX && !okY ? `Calibration is too loose in both directions. Sit still and try again.`
          : okX ? `Left and right is good, but up and down is too loose. Keep your head level and try again.`
          : `Up and down is good, but left and right is too loose. Try again.`;
        toast(msg);
        say(msg, { instant: true, keep: true });
        fetch('/api/gazelog', {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ kind: 'calibration', ...p,
            screen: { w: window.innerWidth, h: window.innerHeight }, probe: gaze?.probe() }),
        }).catch(() => {});
        return;
      }
      $('#calib').hidden = false;
      const d = $('#calib-dot');
      d.style.left = `${p.x * 100}%`;
      d.style.top = `${p.y * 100}%`;
      d.classList.toggle('sampling', p.state === 'sampling');

      const m = $('#calib-msg');
      m.textContent = p.state === 'sampling' ? 'Hold it…' : 'Look at the dot';
      // Words ride with the dot. He is looking at a corner; he cannot read the middle.
      m.style.left = `${Math.min(0.78, Math.max(0.22, p.x)) * 100}%`;
      m.style.top = `${Math.min(0.86, p.y + 0.12) * 100}%`;
      $('#calib-count').textContent = `${p.index + 1} of ${p.total}`;
      if (p.state === 'sampling') beep(880, 0.09);
    },
  });
  gaze = g;

  const ok = await g.start();
  if (g !== gaze) return;
  if (!ok) { gaze = null; return; }
  $('#gaze-row').hidden = false;
  $('#gaze-state').textContent = `Camera on (${g.backend}) — not calibrated yet.`;
  toast('Camera on. Calibrate before selecting with your eyes.');
}

/**
 * SIGNAL CHECK — run this BEFORE calibrating.
 *
 * "I'm not moving my eyes and it still moves" is a signal problem, not a calibration problem,
 * and no amount of recalibrating fixes it. This measures the two things that actually decide
 * whether eye tracking is possible on this camera, in this light, at this distance:
 *
 *   NOISE  — how much the iris reading wobbles while he holds still.
 *   TRAVEL — how far it moves when he deliberately looks left, then right.
 *
 * If TRAVEL is not comfortably larger than NOISE, the camera cannot see his eyes move, and
 * everything downstream is theatre. Better to say so than to hand him a cursor that lies.
 */
async function signalCheck() {
  if (!gaze?.running) return toast('Turn the camera on first.');

  const grab = async (ms) => {
    const out = [];
    const until = performance.now() + ms;
    while (performance.now() < until) {
      const s = gaze.raw();
      if (s) out.push(s);
      await new Promise((r) => setTimeout(r, 40));
    }
    return out;
  };
  const meanX = (a) => a.reduce((s, v) => s + v[0], 0) / (a.length || 1);
  const sdX = (a) => {
    const m = meanX(a);
    return Math.sqrt(a.reduce((s, v) => s + (v[0] - m) ** 2, 0) / (a.length || 1));
  };

  $('#calib').hidden = false;
  $('#calib-dot').style.display = '';

  // You cannot read an instruction in the middle of the screen while looking at the far edge of
  // it. That is the whole point of the test. So the app SAYS it out loud, and the target moves
  // to where he should be looking — he never has to look away to find out what to do next.
  const step = async (msg, x, y, settleMs, sampleMs) => {
    await prompt(msg, x, y);
    await new Promise((r) => setTimeout(r, settleMs));
    beep(880);
    $('#calib-dot').classList.add('sampling');
    const data = await grab(sampleMs);
    $('#calib-dot').classList.remove('sampling');
    beep(660, 0.08);
    return data;
  };

  const still = await step('Look at the dot in the middle. Hold still.', 0.5, 0.5, 900, 1600);
  const left  = await step('Now look at the dot on the far left.',       0.03, 0.5, 1200, 1300);
  const right = await step('Now the dot on the far right.',              0.97, 0.5, 1200, 1300);

  $('#calib').hidden = true;

  if (still.length < 10 || left.length < 6 || right.length < 6) {
    return toast('Could not see your face well enough. More light, and sit closer.');
  }

  const noise = sdX(still);
  const travel = Math.abs(meanX(right) - meanX(left));
  const snr = travel / (noise || 1e-6);
  const verdict = snr > 8 ? 'good' : snr > 4 ? 'usable' : 'too noisy';
  const p = gaze.probe();
  const cam = p.camera ? `${p.camera.w}x${p.camera.h}` : '?';

  $('#gaze-state').textContent =
    `${snr.toFixed(1)}x (${verdict}) · travel ${travel.toFixed(3)} / noise ${noise.toFixed(4)} · ${cam}, iris ${p.irisPx}px`;

  const verdictMsg = p.irisPx && p.irisPx < 12
    ? `Your iris is only ${p.irisPx} pixels wide. Sit closer to the screen.`
    : snr > 4
      ? `Signal is ${verdict}. Now calibrate.`
      : `Too noisy. Sit closer, put more light on your face, and raise the camera to eye level.`;
  toast(`${snr.toFixed(1)}x — ${verdictMsg}`);
  say(verdictMsg, { instant: true, keep: true });   // he does not have to read it either

  fetch('/api/gazelog', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ kind: 'signal', noise: +noise.toFixed(5), travel: +travel.toFixed(4),
      snr: +snr.toFixed(2), verdict, probe: p,
      samples: { still: still.length, left: left.length, right: right.length } }),
  }).catch(() => {});
}

/** Move the target to where he should look, put the words THERE, and say them aloud. */
async function prompt(msg, x, y) {
  const d = $('#calib-dot');
  d.style.left = `${x * 100}%`;
  d.style.top = `${y * 100}%`;

  // The words ride WITH the target — never in the middle of the screen when he is looking at
  // the edge of it. Flip to the inner side near an edge so they stay on screen.
  const m = $('#calib-msg');
  m.textContent = msg;
  m.style.left = `${Math.min(0.78, Math.max(0.22, x)) * 100}%`;
  m.style.top = `${Math.min(0.86, y + 0.12) * 100}%`;

  speakPrompt(msg);
  await new Promise((r) => setTimeout(r, 450));
}

/** Instructions, spoken. This is an app that talks; it should talk to HIM too. */
let promptPlayer = null;
async function speakPrompt(text) {
  try {
    const res = await fetch('/api/speak', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text, voice: 'placeholder' }),
    });
    if (!res.ok) return;
    const url = URL.createObjectURL(await res.blob());
    promptPlayer?.pause();
    promptPlayer = new Audio(url);
    promptPlayer.onended = () => URL.revokeObjectURL(url);
    await promptPlayer.play().catch(() => {});
  } catch {}
}

/** A tone when sampling starts, a lower one when it ends. He needs to know when to hold still. */
let actx = null;
function beep(hz = 880, len = 0.12) {
  try {
    actx = actx ?? new AudioContext();
    const o = actx.createOscillator(), g = actx.createGain();
    o.frequency.value = hz;
    o.type = 'sine';
    g.gain.setValueAtTime(0.0001, actx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.15, actx.currentTime + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, actx.currentTime + len);
    o.connect(g); g.connect(actx.destination);
    o.start(); o.stop(actx.currentTime + len + 0.02);
  } catch {}
}

/**
 * THE ACCURACY TEST.
 *
 * "Does the dot feel about right" is not a measurement. This lights each tile in turn, asks him
 * to look at it, records where the gaze actually lands, and scores itself: how often does the
 * gaze fall on the tile he was told to look at?
 *
 * That number decides whether the camera tier is real or a toy — and it is the number that goes
 * in the writeup. It also tells us WHICH tiles fail: if the corners miss but the middle is fine,
 * the fit needs more calibration spread, not a different algorithm.
 */
async function testGazeAccuracy() {
  if (!gaze?.calibrated) return toast('Calibrate first.');

  const tiles = $$('.tile');
  const results = [];
  $('#calib-msg').textContent = 'Look at the highlighted tile';
  $('#calib').hidden = false;
  $('#calib-dot').style.display = 'none';

  for (let i = 0; i < tiles.length; i++) {
    $('#calib').hidden = true;                       // let him actually see the grid
    tiles.forEach((t, j) => t.classList.toggle('target', i === j));
    $('#calib-count').textContent = `${i + 1} of ${tiles.length}`;
    await new Promise((r) => setTimeout(r, 1300));   // settle

    const pts = [];
    for (let k = 0; k < 12; k++) {
      const s = gaze.sample();
      if (s) pts.push(s);
      await new Promise((r) => setTimeout(r, 45));
    }
    if (!pts.length) { results.push({ tile: i, hit: false, reason: 'no face' }); continue; }

    const mx = pts.reduce((a, p) => a + p.x, 0) / pts.length;
    const my = pts.reduce((a, p) => a + p.y, 0) / pts.length;
    const r = tiles[i].getBoundingClientRect();
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;

    results.push({
      tile: i,
      label: state.tiles[i],
      hit: tileAt(mx, my) === i,
      landedOn: tileAt(mx, my),
      offsetPx: Math.round(Math.hypot(mx - cx, my - cy)),
      raw: pts[pts.length - 1].raw,
    });
  }

  tiles.forEach((t) => t.classList.remove('target'));
  $('#calib').hidden = true;
  $('#calib-dot').style.display = '';

  const hits = results.filter((r) => r.hit).length;
  const pct = Math.round((hits / results.length) * 100);
  toast(`Eye tracking hit the right tile ${hits} of ${results.length} times (${pct}%).`);
  $('#gaze-state').textContent = `Accuracy: ${hits}/${results.length} tiles (${pct}%).`;

  // Send it to the server so it can actually be read, instead of living in a toast.
  fetch('/api/gazelog', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      kind: 'accuracy',
      hits, total: results.length, pct,
      screen: { w: window.innerWidth, h: window.innerHeight },
      probe: gaze.probe(),
      results,
    }),
  }).catch(() => {});
}

function stopGaze() {
  gaze?.stop();
  gaze = null;
  resetDwell();
  $('#gaze-dot').hidden = true;
  $('#gaze-row').hidden = true;
}

/* ---------- feedback ---------- */

let toastTimer = null;
function toast(msg) {
  const el = $('#toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 6000);
}

/** What he just said, on screen — so the room can read him even if they missed the audio. */
function speaking(text) {
  const el = $('#spoken');
  el.textContent = `“${text}”`;
  el.hidden = false;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.hidden = true; }, 6000);
}

/* ---------- render ---------- */

function renderGrid() {
  const g = $('#grid');
  g.innerHTML = '';
  const aiming = state.driver === 'scan' || state.driver === 'gaze';

  state.tiles.forEach((t, i) => {
    const b = document.createElement('button');
    b.className = 'tile'
      + (aiming && i === state.cursor ? ' cursor' : '')
      + (!state.urgent && i < state.pinned ? ' pinned' : '')
      + (state.urgent ? ' urgent-tile' : '');
    b.textContent = t;
    b.onclick = () => pick(t);
    g.appendChild(b);
  });

  // The cursor can rest on Say-it and Urgent too, or a switch user could never speak.
  const n = state.tiles.length;
  $('#compose').classList.toggle('cursor', aiming && state.cursor === n);
  $('#urgent').classList.toggle('cursor', aiming && state.cursor === n + 1);
}

function renderSelected() {
  $('#compose').disabled = !state.selected.length || state.speaking;
  $('#composing').hidden = state.selected.length === 0;
  $('#composing-words').textContent = state.selected.join(' ');
}

function renderHUD() {
  $('#m-sel').textContent = state.selections;
  const s = state.startedAt ? (performance.now() - state.startedAt) / 1000 : 0;
  $('#m-time').textContent = s.toFixed(1);
}
setInterval(() => { if (state.startedAt) renderHUD(); }, 250);

/* ---------- wiring ---------- */

$('#m-answer').onclick = () => setMode('answer');
$('#m-ask').onclick = () => setMode('ask');
$('#m-tell').onclick = () => setMode('tell');
$('#compose').onclick = compose;
$('#urgent').onclick = toggleUrgent;
$('#undo').onclick = undo;
$('#cancel').onclick = closeConfirm;
$('#listen').onclick = () => (state.listening ? stopListening() : startListening());
$('#hold').onclick = () => {
  // keep:true — this is the button that BUYS him time to finish his sentence. It must not be
  // the button that deletes it.
  say("Wait — I'm saying something.", { instant: true, keep: true });
};
$('#confirm').onclick = (e) => { if (e.target.id === 'confirm') closeConfirm(); };
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !$('#confirm').hidden) closeConfirm();
});
$('#partner-said').onchange = () => { if (!state.selected.length) loadTiles(); };

$('#gear').onclick = () => {
  const open = $('#settings').hidden;
  $('#settings').hidden = !open;
  $('#gear').setAttribute('aria-expanded', String(open));
};
$('#close-settings').onclick = () => {
  $('#settings').hidden = true;
  $('#gear').setAttribute('aria-expanded', 'false');
};
$('#literal').onchange = (e) => { state.literal = e.target.checked; };
$('#predictive').onchange = (e) => { state.predictive = e.target.checked; loadTiles(); };
$('#driver').onchange = (e) => {
  state.driver = e.target.value;
  $('#scan-help').hidden = state.driver !== 'scan';
  if (state.driver === 'gaze') startGaze(); else stopGaze();
  renderGrid();
};
$('#calibrate').onclick = () => {
  // Calibrate on the REAL tile centres, plus the centre and a ring just outside them. He never
  // looks at the corner of the glass; he looks at words. Teach the model where the words are.
  const t = $$('.tile').map((el) => {
    const r = el.getBoundingClientRect();
    return [(r.left + r.width / 2) / window.innerWidth, (r.top + r.height / 2) / window.innerHeight];
  });
  const targets = t.length === 8
    ? [[0.5, 0.5], ...t,
       // a modest ring outside the grid, so the fit interpolates across his working area
       // instead of extrapolating past the last thing it has ever seen
       [0.06, 0.5], [0.94, 0.5], [0.5, 0.14], [0.5, 0.88]]
    : null;
  gaze?.calibrate(targets);
};
$('#test-gaze').onclick = testGazeAccuracy;
$('#signal-check').onclick = signalCheck;
$('#dwell').oninput = (e) => {
  state.dwellMs = +e.target.value;
  $('#dwell-label').textContent = `Hold a tile for ${(state.dwellMs / 1000).toFixed(1)}s to pick it.`;
};

/* ---------- boot ---------- */

async function boot() {
  try {
    const res = await fetch('/api/health');
    const h = await res.json();
    if (!h.ok) throw new Error(h.error ?? 'server not ready');

    state.profile = { name: h.profile, voice: h.voice };
    state.instant = h.instant ?? {};
    $('#who-name').textContent = h.profile;
    $('#voice-state').textContent = h.voice === 'cloned'
      ? 'his own voice' : 'placeholder voice — not his yet';
    await loadTiles();
  } catch (e) {
    // A blank grid with a silent console is how this app used to fail. Say so, and keep trying.
    toast(`Cannot reach StillMe: ${e.message} — retrying…`);
    setTimeout(boot, 3000);
  }
}
boot();
