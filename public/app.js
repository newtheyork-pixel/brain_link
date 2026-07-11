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
  state.selected.push(tile);
  state.selections++;
  renderSelected();
  renderHUD();
  await loadTiles(); // predictive narrowing: the next word he needs is now on screen
}

function undo() {
  if (!state.selected.length) return;
  state.selected.pop();
  renderSelected();
  loadTiles();
}

async function loadTiles() {
  const res = await fetch('/api/tiles', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      selected: state.selected,
      partner: $('#partner-said').value.trim(),
      predictive: state.predictive,
    }),
  });
  const { tiles, source, ms } = await res.json();
  state.tiles = tiles.slice(0, GRID);
  state.cursor = 0;
  $('#m-src').textContent = source === 'predicted' ? `predicted ${ms}ms`
    : source === 'fallback' ? 'model down — fallback tiles' : source;
  renderGrid();
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
}

async function say(text) {
  $('#confirm').hidden = true;
  const elapsed = (performance.now() - state.startedAt) / 1000;
  const words = text.trim().split(/\s+/).length;

  const res = await fetch('/api/speak', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text, voice: state.profile.voiceModel ? 'cloned' : 'placeholder' }),
  });
  const audio = new Audio(URL.createObjectURL(await res.blob()));
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
    }),
  });

  reset();
}

function reset() {
  state.selected = [];
  state.selections = 0;
  state.startedAt = null;
  renderSelected();
  renderHUD();
  loadTiles();
}

/* ---------- render ---------- */

function renderGrid() {
  const g = $('#grid');
  g.innerHTML = '';
  state.tiles.forEach((t, i) => {
    const b = document.createElement('button');
    b.className = 'tile' + (state.driver === 'scan' && i === state.cursor ? ' cursor' : '');
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
$('#cancel').onclick = () => { $('#confirm').hidden = true; };
$('#partner-said').onchange = () => { if (!state.selected.length) loadTiles(); };

const h = await (await fetch('/api/health')).json();
state.profile = { name: h.profile, voiceModel: null };
$('#who-name').textContent = h.profile;
$('#voice-state').textContent = 'placeholder voice — not his yet';
await loadTiles();
