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

function onGazePoint(x, y) {
  const dot = $('#gaze-dot');
  dot.hidden = false;
  dot.style.transform = `translate(${x}px, ${y}px)`;

  if (!$('#confirm').hidden || !$('#calib').hidden || state.speaking) return resetDwell();

  const i = tileAt(x, y);
  if (i < 0) return resetDwell();

  if (i !== dwellTile) {
    resetDwell();
    dwellTile = i;
    dwellStart = performance.now();
    state.cursor = i;
    renderGrid();
    return;
  }
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
        $('#gaze-state').textContent = `Calibrated (±${p.errorPx}px).`;
        if (p.errorPx > 180) toast('Calibration is rough — sit still, get more light, try again.');
        return;
      }
      $('#calib').hidden = false;
      const d = $('#calib-dot');
      d.style.left = `${p.x * 100}%`;
      d.style.top = `${p.y * 100}%`;
      d.classList.toggle('sampling', p.state === 'sampling');
      $('#calib-msg').textContent = p.state === 'sampling' ? 'Hold it…' : 'Look at the dot';
      $('#calib-count').textContent = `${p.index + 1} of ${p.total}`;
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
$('#calibrate').onclick = () => gaze?.calibrate();
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
