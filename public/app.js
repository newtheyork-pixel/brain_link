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
  coreSlots: 2,   // how many tiles never move. Motor learning vs prediction — a measured knob.
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

  const res = await fetch('/api/tiles', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      selected: state.selected,
      partner: $('#partner-said').value.trim(),
      predictive: state.predictive,
      coreSlots: state.coreSlots,
    }),
  });
  const { tiles, source, ms } = await res.json();
  state.tiles = tiles.slice(0, GRID);
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
      partner: $('#partner-said').value.trim(),
      literal: state.literal,
    }),
  });
  const { candidates } = await res.json();
  showConfirm(candidates);
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

async function say(text, { instant = false } = {}) {
  closeConfirm();
  const elapsed = (performance.now() - state.startedAt) / 1000;
  const words = text.trim().split(/\s+/).length;

  // Stop listening while he speaks, or the mic hears his own voice and treats it as the
  // other person talking — the app would start answering itself.
  const wasListening = state.listening;
  if (wasListening) stopListening();

  const res = await fetch('/api/speak', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text, voice: state.profile.voiceModel ? 'cloned' : 'placeholder' }),
  });
  const audio = new Audio(URL.createObjectURL(await res.blob()));
  audio.onended = () => { if (wasListening) startListening(); };
  audio.play();

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
// DEV ONLY: the browser's SpeechRecognition. In Chrome this sends audio to Google, which
// breaks the offline promise — so it is labelled on screen, not hidden.
// SHIP: iOS Speech framework with requiresOnDeviceRecognition = true. Genuinely on-device,
// free, no network. Same seam: transcript in, tiles out.

const SR = window.SpeechRecognition ?? window.webkitSpeechRecognition;
let recog = null;

function startListening() {
  if (!SR || state.listening) return;
  recog = new SR();
  recog.continuous = true;
  recog.interimResults = true;
  recog.lang = 'en-US';

  recog.onresult = (e) => {
    let final = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) final += e.results[i][0].transcript;
    }
    const partial = e.results[e.results.length - 1][0].transcript;
    $('#partner-said').value = (final || partial).trim();
    // Only re-predict on a completed sentence — not on every syllable, or the grid
    // thrashes under his finger while he is trying to aim at it.
    if (final && !state.selected.length) loadTiles();
  };
  recog.onerror = (e) => { if (e.error !== 'no-speech') stopListening(); };
  recog.onend = () => { if (state.listening) recog.start(); }; // Chrome times out; keep it alive

  recog.start();
  state.listening = true;
  $('#listen').classList.add('on');
  $('#listen').textContent = 'Listening…';
  $('#listen').setAttribute('aria-pressed', 'true');
}

function stopListening() {
  state.listening = false;
  try { recog?.stop(); } catch {}
  recog = null;
  $('#listen').classList.remove('on');
  $('#listen').textContent = 'Listen';
  $('#listen').setAttribute('aria-pressed', 'false');
}

/* ---------- render ---------- */

function renderGrid() {
  const g = $('#grid');
  g.innerHTML = '';
  state.tiles.forEach((t, i) => {
    const b = document.createElement('button');
    // Pinned tiles get a quiet marker: they are in the same place every single time,
    // which is what lets him stop reading the grid and start knowing it.
    const pinned = !state.urgent && i < state.coreSlots;
    b.className = 'tile'
      + (state.driver === 'scan' && i === state.cursor ? ' cursor' : '')
      + (pinned ? ' pinned' : '')
      + (state.urgent ? ' urgent-tile' : '');
    b.textContent = t;
    b.onclick = () => pick(t);
    g.appendChild(b);
  });
}

function renderSelected() {
  const box = $('#selected');
  box.innerHTML = '';
  for (const t of state.selected) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.setAttribute('role', 'listitem');
    chip.textContent = t;
    box.appendChild(chip);
  }
  $('#compose').disabled = !state.selected.length;
  $('#undo').disabled = !state.selected.length;
}

// What the OTHER person sees. Without it they are watching a man stare at a screen and
// they have no idea he is halfway through a sentence — so they talk over him, or leave.
function renderComposing() {
  const on = state.selected.length > 0;
  $('#composing').hidden = !on;
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
  renderGrid();
};
$('#compose').onclick = compose;
$('#undo').onclick = undo;
$('#urgent').onclick = toggleUrgent;
$('#cancel').onclick = closeConfirm;
$('#listen').onclick = () => (state.listening ? stopListening() : startListening());
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
