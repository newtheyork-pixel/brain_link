// StillMe — the loop.
//
// THE ARCHITECTURE THAT MATTERS: every input method emits the same six events.
//
//     LEFT · RIGHT · UP · DOWN · SELECT · UNDO
//
// Touch emits them. Arrow keys emit them. And when the EOG board arrives, its
// BLE peripheral emits them too — a left flick of the eye is LEFT, a blink is
// SELECT, a double-blink is UNDO. Nothing below this line changes.
//
// That is why you can build and test the entire eye-controlled interface today,
// with a keyboard, before a single electrode ships.

import { createMic } from '/mic.js';
import { createGaze } from '/gaze.js';

const $ = (s) => document.querySelector(s);
const GRID = 8; // 8 tiles + undo = 9 zones. Hard cap: EOG selection degrades past 9.

const state = {
  selected: [],
  tiles: [],
  cursor: 0,
  driver: 'touch',
  literal: false,
  predictive: true,
  urgent: false,
  listening: false,
  mode: 'answer',   // answer | ask | tell — the only way he gets to start a conversation
  coreSlots: 2,   // how many tiles never move. Motor learning vs prediction — a measured knob.
  dwellMs: 900,   // how long he must hold a tile with his eyes before it counts
  pinned: 0,      // how many the server actually pinned this turn
  startedAt: null,
  selections: 0,
  profile: null,
};

/* ---------- input bus ---------- */

const bus = {
  handlers: [],
  on(fn) { this.handlers.push(fn); },
  emit(event, payload) { for (const h of this.handlers) h(event, payload); },
};

// Driver: keyboard → the EOG alphabet. Replace with a WebSocket from the ESP32
// and this file does not change by one character.
window.addEventListener('keydown', (e) => {
  if (state.driver !== 'scan') return;
  const map = { ArrowLeft: 'LEFT', ArrowRight: 'RIGHT', ArrowUp: 'UP', ArrowDown: 'DOWN',
    ' ': 'SELECT', Backspace: 'UNDO' };
  const ev = map[e.key];
  if (!ev) return;
  e.preventDefault();
  bus.emit(ev);
});

bus.on((event) => {
  const cols = 4;
  if (event === 'LEFT')  state.cursor = (state.cursor - 1 + GRID) % GRID;
  if (event === 'RIGHT') state.cursor = (state.cursor + 1) % GRID;
  if (event === 'UP')    state.cursor = (state.cursor - cols + GRID) % GRID;
  if (event === 'DOWN')  state.cursor = (state.cursor + cols) % GRID;
  if (event === 'SELECT') return pick(state.tiles[state.cursor]);
  if (event === 'UNDO') return undo();
  renderGrid();
});

/* ---------- the loop ---------- */

async function pick(tile) {
  if (!tile) return;
  if (!state.startedAt) state.startedAt = performance.now();

  // "yes" needs no sentence built around it. Making him pick it, press Build, and choose
  // between three phrasings of the word "yes" costs three selections and a minute of
  // someone else's patience to say one syllable. Speak it now.
  const now = INSTANT[String(tile).toLowerCase().trim()];
  if (now && !state.selected.length) {
    state.selections++;
    await say(now, { instant: true });
    return;
  }

  state.selected.push(tile);
  state.selections++;
  renderSelected();
  renderComposing();  // the other person can now see he is mid-sentence
  renderHUD();
  await loadTiles(); // predictive narrowing: the next word he needs is now on screen
}

// Kept in sync with lib/llm.mjs INSTANT.
const INSTANT = {
  'yes': 'Yes.', 'no': 'No.', 'thank you': 'Thank you.', 'i love you': 'I love you.',
  'help me': 'Help me.', 'stop': 'Stop.', 'pain': 'I am in pain.',
  "can't breathe": "I can't breathe.", 'wait': "Wait — I am saying something.",
};

function undo() {
  if (!state.selected.length) return;
  state.selected.pop();
  renderSelected();
  loadTiles();
}

async function loadTiles() {
  if (state.urgent) return; // he is in the emergency grid; do not yank it away

  // When he is initiating, there IS no partner utterance — the whole point is that nobody
  // spoke to him. Never let a stale sentence in the box turn his question back into a reply.
  const partner = state.mode === 'answer' ? $('#partner-said').value.trim() : '';

  const res = await fetch('/api/tiles', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      selected: state.selected,
      partner,
      predictive: state.predictive,
      coreSlots: state.coreSlots,
      mode: state.mode,
    }),
  });
  const { tiles, source, ms, coreSlots } = await res.json();
  state.tiles = tiles.slice(0, GRID);
  state.pinned = coreSlots ?? 0;   // the SERVER decides; the client was inventing pins
  state.cursor = 0;
  $('#m-src').textContent = source === 'predicted' ? `predicted ${ms}ms`
    : source === 'fallback' ? 'model down — fallback tiles' : source;
  renderGrid();
}

// The escape hatch. One selection, from any screen, no model call, no waiting.
async function toggleUrgent() {
  state.urgent = !state.urgent;
  document.body.classList.toggle('in-urgent', state.urgent);
  $('#urgent').textContent = state.urgent ? 'Back' : 'Urgent';
  if (state.urgent) {
    const { tiles } = await (await fetch('/api/urgent')).json();
    state.tiles = tiles.slice(0, GRID);
    state.cursor = 0;
    $('#m-src').textContent = 'urgent — fixed grid';
    renderGrid();
  } else {
    await loadTiles();
  }
}

async function compose() {
  const res = await fetch('/api/compose', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      selected: state.selected,
      partner: state.mode === 'answer' ? $('#partner-said').value.trim() : '',
      literal: state.literal,
      mode: state.mode,
    }),
  });
  const { candidates } = await res.json();
  showConfirm(candidates);
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
  // Hide the "they said to you" box when he is the one starting — it isn't his turn to
  // react, and leaving it there quietly reframes his question as an answer.
  $('#partner-block').hidden = mode !== 'answer';
  $('#compose').textContent = mode === 'ask' ? 'Ask it' : 'Say it';
  renderSelected();
  renderComposing();
  renderHUD();
  loadTiles();
}

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
  $('#confirm').hidden = false;
  $('#cancel').focus();
}

const closeConfirm = () => { $('#confirm').hidden = true; $('#compose').focus(); };

// Hold the reference at module scope. A local `const audio` inside an async function can be
// garbage-collected the moment the function returns — the sound just never comes out, with
// no error anywhere. This is the classic silent-audio bug and it is why nothing was speaking.
let player = null;

async function say(text, { instant = false } = {}) {
  closeConfirm();
  const elapsed = (performance.now() - state.startedAt) / 1000;
  const words = text.trim().split(/\s+/).length;

  // Stop listening while he speaks, or the mic hears his own voice and treats it as the
  // other person talking — the app would start answering itself.
  const wasListening = state.listening;
  if (wasListening) stopListening();

  try {
    const res = await fetch('/api/speak', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text, voice: state.profile?.voiceModel ? 'cloned' : 'placeholder' }),
    });
    if (!res.ok) throw new Error(`speak ${res.status}`);

    const url = URL.createObjectURL(await res.blob());
    player = new Audio(url);
    player.onended = () => { URL.revokeObjectURL(url); if (wasListening) startListening(); };
    // A rejected play() promise is how the browser tells you it blocked the sound. Ignore it
    // and the app is simply mute forever with nothing in the console.
    await player.play().catch((e) => {
      toast(e.name === 'NotAllowedError'
        ? 'Browser blocked the audio — tap anywhere, then try again.'
        : `Could not play audio: ${e.message}`);
      throw e;
    });
    speaking(text);
  } catch (e) {
    toast(`Speech failed: ${e.message}`);
  }

  // The two numbers the research is about.
  fetch('/api/log', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      spoken: text,
      tiles: state.selected,
      selections_per_sentence: state.selections,
      seconds_to_sentence: +elapsed.toFixed(1),
      effective_wpm: +(words / (elapsed / 60)).toFixed(1),
      predictive: state.predictive,
      literal: state.literal,
      driver: state.driver,
      instant,
    }),
  });

  reset();
}

function reset() {
  state.selected = [];
  state.selections = 0;
  state.startedAt = null;
  renderSelected();
  renderComposing();
  renderHUD();
  loadTiles();
}

/* ---------- listening ---------- */
//
// LOCAL. whisper.cpp on this machine, ~0.15s. The audio is written to a temp file, read by
// whisper, and deleted. It is never uploaded. Chrome's Web Speech API (which we used before)
// sends the microphone to Google — unacceptable in an app whose entire claim is that nothing
// leaves the device. Ship path is the same shape: iOS SFSpeechRecognizer, on-device.

let mic = null;

function startListening() {
  if (state.listening) return;

  mic = createMic({
    // The words appear WHILE they are being said. A caregiver who has to wait for a lump of
    // text after a pause has no idea whether the thing is even on.
    onInterim: (text) => {
      $('#partner-said').value = text;
      $('#partner-said').classList.add('interim');
    },
    onFinal: (text) => {
      $('#partner-said').value = text;
      $('#partner-said').classList.remove('interim');
      // Re-predict only on a finished sentence, and never once he has started answering —
      // yanking the grid out from under his finger mid-selection is unforgivable.
      if (!state.selected.length && state.mode === 'answer') loadTiles();
    },
    // A visible level meter. Without it, a dead mic and a quiet room look identical, and
    // you are left tapping a button that never says anything back.
    onLevel: (rms) => {
      const pct = Math.min(100, Math.round(rms * 900));
      $('#level').style.width = `${pct}%`;
      $('#listen').classList.toggle('hearing', pct > 12);
    },
    onError: (msg) => { toast(msg); stopListening(); },
  });

  mic.start().then((ok) => {
    if (!ok) return;
    state.listening = true;
    $('#listen').classList.add('on');
    $('#listen').textContent = 'Listening…';
    $('#listen').setAttribute('aria-pressed', 'true');
    $('#meter').hidden = false;
  });
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
// That is the whole point of the event bus: the eye tier plugs in and nothing else changes.
//
// Dwell rather than blink-only, because a webcam sees an involuntary blink and a deliberate one
// identically, and a device that fires on a reflex would put words in his mouth. Blink is offered
// as a SECOND way to confirm, never the only one.

let gaze = null;
let dwellTile = -1, dwellStart = 0, dwellRaf = 0;

function tileAt(x, y) {
  const tiles = [...document.querySelectorAll('.tile')];
  for (let i = 0; i < tiles.length; i++) {
    const r = tiles[i].getBoundingClientRect();
    if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) return i;
  }
  return -1;
}

function resetDwell() {
  dwellTile = -1;
  dwellStart = 0;
  document.querySelectorAll('.tile').forEach((t) => t.style.setProperty('--dwell', '0'));
}

function onGazePoint(x, y) {
  const dot = $('#gaze-dot');
  dot.hidden = false;
  dot.style.transform = `translate(${x}px, ${y}px)`;

  // The confirm sheet and the calibration overlay own the screen when they are up.
  if (!$('#confirm').hidden || !$('#calib').hidden) return resetDwell();

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

  const held = performance.now() - dwellStart;
  const frac = Math.min(1, held / state.dwellMs);
  document.querySelectorAll('.tile')[i]?.style.setProperty('--dwell', String(frac));

  if (frac >= 1) {
    const tile = state.tiles[i];
    resetDwell();
    dwellStart = performance.now() + 600;   // brief refractory: don't instantly re-fire
    bus.emit('SELECT');
    if (!tile) return;
  }
}

async function startGaze() {
  if (gaze?.running) return;

  gaze = createGaze({
    onGaze: onGazePoint,
    // A blink is a SECOND way to confirm what he is already looking at — never a first.
    onBlink: () => {
      if (dwellTile >= 0 && $('#confirm').hidden && $('#calib').hidden) {
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
        // A fit that can't even reproduce its own calibration points will never hit a tile.
        if (p.errorPx > 180) toast('Calibration is rough — sit still, good light, and try again.');
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

  const ok = await gaze.start();
  if (!ok) return;
  $('#gaze-row').hidden = false;
  $('#gaze-state').textContent = 'Camera on — not calibrated yet.';
  toast('Camera on. Calibrate before using your eyes to select.');
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
  toastTimer = setTimeout(() => { el.hidden = true; }, 5000);
}

// Show what he just said, so the room can read it even if they missed the audio —
// and so YOU can see the app is working when the speakers are muted.
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
  state.tiles.forEach((t, i) => {
    const b = document.createElement('button');
    // Pinned tiles get a quiet marker: they are in the same place every single time,
    // which is what lets him stop reading the grid and start knowing it.
    const aiming = state.driver === 'scan' || state.driver === 'gaze';
    const pinned = !state.urgent && i < state.pinned;
    b.className = 'tile'
      + (aiming && i === state.cursor ? ' cursor' : '')
      + (pinned ? ' pinned' : '')
      + (state.urgent ? ' urgent-tile' : '');
    b.textContent = t;
    b.onclick = () => pick(t);
    g.appendChild(b);
  });
}

// His words live in ONE place — the same strip the other person reads. Two rows showing
// the same sentence was just chrome between him and the tiles.
function renderSelected() {
  $('#compose').disabled = !state.selected.length;
  renderComposing();
}

// What the OTHER person sees. Without it they are watching a man stare at a screen with
// no idea he is halfway through a sentence — so they talk over him, or they leave.
function renderComposing() {
  $('#composing').hidden = state.selected.length === 0;
  $('#composing-words').textContent = state.selected.join(' ');
}

function renderHUD() {
  $('#m-sel').textContent = state.selections;
  const s = state.startedAt ? (performance.now() - state.startedAt) / 1000 : 0;
  $('#m-time').textContent = s.toFixed(1);
}
setInterval(() => { if (state.startedAt) renderHUD(); }, 200);

/* ---------- boot ---------- */

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
$('#compose').onclick = compose;
$('#undo').onclick = undo;
$('#urgent').onclick = toggleUrgent;
$('#cancel').onclick = closeConfirm;
$('#listen').onclick = () => (state.listening ? stopListening() : startListening());
$('#gear').onclick = () => {
  const open = $('#settings').hidden;
  $('#settings').hidden = !open;
  $('#gear').setAttribute('aria-expanded', String(open));
};
$('#close-settings').onclick = () => {
  $('#settings').hidden = true;
  $('#gear').setAttribute('aria-expanded', 'false');
};
$('#m-answer').onclick = () => setMode('answer');
$('#m-ask').onclick = () => setMode('ask');
$('#m-tell').onclick = () => setMode('tell');
// He cannot raise a hand or clear his throat. Without this he can never ENTER a
// conversation — only ever answer one someone else started.
$('#hold').onclick = () => {
  if (!state.startedAt) state.startedAt = performance.now();
  say("Wait — I'm saying something.", { instant: true });
};
// Escape, or clicking the backdrop, always gets him out. Never trap him in a dialog.
$('#confirm').onclick = (e) => { if (e.target.id === 'confirm') closeConfirm(); };
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !$('#confirm').hidden) closeConfirm();
});
$('#partner-said').onchange = () => { if (!state.selected.length) loadTiles(); };

const h = await (await fetch('/api/health')).json();
state.profile = { name: h.profile, voiceModel: null };
$('#who-name').textContent = h.profile;
$('#voice-state').textContent = 'placeholder voice — not his yet';
await loadTiles();
