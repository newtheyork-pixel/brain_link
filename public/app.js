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
import { blinkAction, nextZone, stepMinMs } from '/blinkscan.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

// How long the eyes must stay shut to say the armed word. Natural blinks on the 2026-10-08
// recording measured 83-292 ms shut (eval/gaze/blink.mjs), so 350 ms is clearly deliberate but
// still easy. A caregiver can move it (Settings, Blink length); it is remembered on this device.
const BLINK_MS_KEY = 'stillme.blinkMs';
const loadBlinkMs = () => {
  try { const v = +localStorage.getItem(BLINK_MS_KEY); return v >= 250 && v <= 1200 ? v : 350; }
  catch { return 350; }
};
// Look-then-blink or look-and-hold. A caregiver picks it once for him; it must survive a reload,
// or the fallback for a man who cannot hold a blink silently turns itself off every morning.
const CONFIRM_KEY = 'stillme.confirmBy';
const loadConfirmBy = () => {
  try { return localStorage.getItem(CONFIRM_KEY) === 'dwell' ? 'dwell' : 'blink'; } catch { return 'blink'; }
};
const DWELL_MS_KEY = 'stillme.dwellMs';
const loadDwellMs = () => {
  try { const v = +localStorage.getItem(DWELL_MS_KEY); return v >= 400 && v <= 2500 ? v : 900; }
  catch { return 900; }
};
const SAY = '__say__';
const URGENT = '__urgent__';
const UNDO = '__undo__';
const HOLD = '__hold__';
const MODE = '__mode__';
// Each non-tile zone, its button, and its name as a gaze target. Undo, Wait and Mode live in the
// big actions row too: the small Undo in the composing strip and the header tabs are touch-sized,
// and the header is hidden in camera modes, so an eyes-only user could not reach them at all.
const ZONE_EL = { [SAY]: '#compose', [URGENT]: '#urgent', [UNDO]: '#undo-eye', [HOLD]: '#hold', [MODE]: '#mode-eye' };

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
  composing: false, // Say it is waiting on the model: the grid is paused until the sheet opens
  mode: 'answer',   // answer | ask | tell — the only way he gets to start a conversation
  coreSlots: 2,     // tiles that never move. Motor learning vs prediction — a measured knob.
  dwellMs: loadDwellMs(),
  blinkMs: loadBlinkMs(),
  confirmBy: loadConfirmBy(),   // 'blink' = look then blink to say; 'dwell' = look and hold
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

/**
 * Everything the cursor can land on: the tiles, then Say it and Urgent, then Undo, Wait and Mode.
 * The last three come after Urgent so the common path costs a blink user no extra steps. Say it,
 * Undo and Mode only appear when they would do something (Say it and Undo need a word; Mode would
 * wipe a half-built sentence), so a blink user never spends a step on a dead button.
 */
const zones = () => [...state.tiles,
  ...(!state.urgent && state.selected.length ? [SAY] : []), URGENT,
  ...(!state.urgent && state.selected.length ? [UNDO] : []), HOLD,
  ...(!state.urgent && !state.selected.length ? [MODE] : [])];

// The cursor is an index into zones(), and which zones exist depends on his words. So every change
// to state.selected shifts the buttons after the tiles: undoing the last word turned a highlighted
// Undo into Wait, and the next long blink said "Wait" out loud. This keeps the highlight on the
// same zone, or on nothing in Blinks mode if that zone is gone, and redraws at once so the
// highlight and what SELECT fires never disagree.
function changeSelected(change) {
  const was = zones()[state.cursor];
  change();
  const i = was === undefined ? -1 : zones().indexOf(was);
  state.cursor = state.driver === 'blink' ? i : Math.max(0, i);
  renderGrid();
}
const ZONE_ACT = { [SAY]: () => compose(), [URGENT]: () => toggleUrgent(), [UNDO]: () => undo(),
  [HOLD]: () => sayWait(), [MODE]: () => cycleMode() };

let navAt = 0;   // when he last moved the cursor himself (loadTiles keeps his place if he did)
function moveCursor(d) {
  const n = zones().length;
  if (!n) return;
  state.cursor = (state.cursor + d + n) % n;   // wrap by REAL length: grids shrink below 8
  navAt = performance.now();
  renderGrid();
}

// Blinks driver: a step goes forward only, so the order is the whole cost. Urgent comes first on
// the main grid, one step from a fresh grid instead of nine. Inside the urgent grid the urgent
// tiles come first and Back stays at the end, or his first step there would land on Back.
let blinkIdleTimer = null;
const BLINK_IDLE_MS = 15000;
function blinkStep() {
  const z = zones();
  state.cursor = nextZone(state.cursor, z.length, state.urgent ? -1 : z.indexOf(URGENT));
  navAt = performance.now();
  renderGrid();
  // A highlight he walked away from must not be fired by his eyes resting later. Left alone this
  // long, it clears, and he steps to it again.
  clearTimeout(blinkIdleTimer);
  blinkIdleTimer = setTimeout(() => {
    if (state.driver !== 'blink' || !$('#confirm').hidden || state.cursor < 0) return;
    state.cursor = -1;
    renderGrid();
  }, BLINK_IDLE_MS);
}

bus.on((event) => {
  // The sheet owns the six events while it is up. Without this, SELECT reaches through and
  // picks a tile on the hidden grid — corrupting the very sentence awaiting confirmation.
  if (!$('#confirm').hidden) return sheetEvent(event);
  if (!$('#calib').hidden || gazeTesting) return;

  const cols = 4;
  if (event === 'LEFT')  return moveCursor(-1);
  if (event === 'RIGHT') return moveCursor(1);
  if (event === 'UP')    return moveCursor(-cols);
  if (event === 'DOWN')  return moveCursor(cols);
  if (event === 'UNDO')  return undo();
  if (event !== 'SELECT') return;

  const z = zones()[state.cursor];
  if (ZONE_ACT[z]) return ZONE_ACT[z]();
  return pick(z);
});

/** Six events, driving the confirm sheet. A switch user must be able to speak. */
let sheetIdx = 0;
let gridGen = 0, sheetGen = 0;   // bumped on every re-render: what a blink snapshot is checked against
// Everything on the sheet he can land on: the numbered sentences, Back, and Urgent. Urgent is on
// the sheet because the sheet covers the whole screen and owns every input while it is up, so
// without it the emergency grid was behind a Back he first had to find and fire.
const sheetOptions = () => [...$$('.candidate'), $('#cancel'), $('#sheet-urgent')];
function sheetEvent(event) {
  const opts = sheetOptions();
  if (!opts.length) return;
  if (event === 'LEFT' || event === 'UP') sheetIdx = (sheetIdx - 1 + opts.length) % opts.length;
  else if (event === 'RIGHT' || event === 'DOWN') sheetIdx = (sheetIdx + 1) % opts.length;
  else if (event === 'UNDO') return closeConfirm();
  else if (event === 'SELECT') {
    // The sheet opens with nothing highlighted, so the first press only lands on option 1. It
    // used to call opts[-1].click() and throw.
    if (sheetIdx < 0) sheetIdx = 0;
    else return opts[sheetIdx].click();
  }
  highlightSheet();
}

// Keyboard → the six events. Only while the scanning driver is on, and NEVER while the
// caregiver is typing: a spacebar in the partner box used to fire SELECT on the pinned "yes"
// tile and make the device say "Yes." out loud, in the patient's voice.
// Buttons are NOT exempt: a focused button (a tapped tile, the gear, Say it after Back) used to
// swallow every arrow and Backspace, trapping a switch user. preventDefault below stops the
// button's own Space/Enter click, so nothing fires twice.
window.addEventListener('keydown', (e) => {
  if (state.driver !== 'scan') return;
  const t = e.target;
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName) || t.isContentEditable) return;
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
  if (!tile) return;

  // URGENT JUMPS THE QUEUE. Checked BEFORE the speaking guard — the old order let a still-playing
  // sentence block "can't breathe" entirely. It speaks immediately, interrupting whatever is
  // playing, and keeps his in-progress sentence. The grid that exists so he is never trapped must
  // never itself be blocked.
  // It is logged as its own one-pick utterance. Counting it into the kept sentence used to log
  // "I need the bathroom." as 16 selections and 730 s, the half-built sentence's numbers.
  if (state.urgent) {
    await say(state.instant[norm(tile)] ?? `${tile}.`,
      { instant: true, keep: true, interrupt: true, startedAt: performance.now(), selections: 1 });
    return;
  }

  // While Say it is building the sentence, the words it was sent are the sentence. A word picked
  // now never reached the model, so the spoken sentence left it out and reset() then deleted it.
  if (state.speaking || state.composing) return;
  if (!state.startedAt) state.startedAt = performance.now();

  // "yes" needs no sentence built around it.
  const now = state.instant[norm(tile)];
  if (now && !state.selected.length) {
    state.selections++;
    await say(now, { instant: true });
    return;
  }

  changeSelected(() => state.selected.push(tile));
  state.selections++;
  renderSelected();
  renderHUD();
  await loadTiles();   // predictive narrowing: the next word he needs is now on screen
}

const norm = (s) => String(s).toLowerCase().trim();

function undo() {
  if (state.composing) return;   // the words are with the model; see pick()
  retractIfRecent();   // he's correcting a pick — don't let the map learn the mistake
  if (!state.selected.length) return;
  changeSelected(() => state.selected.pop());
  renderSelected();
  loadTiles();
}

async function loadTiles() {
  if (state.urgent) return;
  const seq = ++tilesSeq;
  const mode = state.mode;
  const partner = mode === 'answer' ? $('#partner-said').value.trim() : '';
  const reqAt = performance.now();

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

    const was = zones()[state.cursor];
    state.tiles = (tiles ?? []).slice(0, 8);
    state.pinned = coreSlots ?? 0;
    state.cursor = cursorForNewTiles(navAt > reqAt ? was : undefined);
    $('#m-src').textContent = source === 'fallback' ? 'model down — fallback tiles'
      : source === 'predicted' ? `predicted ${ms}ms` : source;
    renderGrid();
  } catch (e) {
    toast(`Could not load words: ${e.message}`);
  }
}

// In Eyes mode the cursor is wherever his eye is. Jumping it to tile 0 on every new grid put the
// gold frame on a word he was not looking at, while the armed word sat somewhere else.
//
// In Blinks mode a new grid starts with NOTHING highlighted (-1), like the sentence sheet. Tile 0
// is usually "yes", which speaks at once, so starting there meant one eye-rest said "Yes." for
// him. keep: the zone he stepped to while this grid was loading. If it is still on screen the
// highlight follows it; if it is gone, nothing is highlighted rather than whatever took its place.
function cursorForNewTiles(keep) {
  if (state.driver === 'blink') return keep === undefined ? -1 : zones().indexOf(keep);
  return state.driver === 'gaze' && dwellTile >= 0 && dwellTile < state.tiles.length ? dwellTile : 0;
}

async function toggleUrgent() {
  // Mutate state only AFTER the tiles arrive. Flipping first meant a failed fetch stranded
  // him in a fake urgent mode — stale tiles, and loadTiles permanently short-circuited.
  if (state.urgent) {
    state.urgent = false;
    document.body.classList.remove('in-urgent');
    $('#urgent').textContent = 'Urgent';
    // Clear the urgent words now, as the way in does. Left on screen until the new words came
    // back, they looked urgent but picked like normal words: "suction" went quietly into his
    // half-built sentence instead of being said at once.
    state.tiles = [];
    state.cursor = state.driver === 'blink' ? -1 : 0;
    renderControls();
    renderGrid();
    await loadTiles();
    return;
  }
  try {
    const res = await fetch('/api/urgent');
    if (!res.ok) throw new Error(`urgent ${res.status}`);
    const { tiles } = await res.json();
    // A sentence still being built must not open its sheet over the emergency grid, and no sheet
    // may stay up over it: the sheet owns every input, so he could only reach his sentences.
    composeSeq++;
    if (state.composing) setComposing(false);   // or the urgent grid would sit greyed and paused
    if (!$('#confirm').hidden) closeConfirm();
    state.urgent = true;
    document.body.classList.add('in-urgent');
    $('#urgent').textContent = 'Back';
    state.tiles = tiles.slice(0, 8);
    state.cursor = cursorForNewTiles();
    $('#m-src').textContent = 'urgent — fixed grid';
    renderControls();
    renderGrid();
  } catch (e) {
    toast(`Urgent grid unavailable: ${e.message}`);
  }
}

async function compose() {
  if (!state.selected.length || state.speaking || state.composing || state.urgent) return;
  const seq = ++composeSeq;
  const mode = state.mode;
  const selected = [...state.selected];

  // A real model takes seconds. Until the sheet opens, picks and Undo wait (pick(), undo(), the
  // gaze and blink paths), and the screen says so, or his eye keeps picking words that are lost.
  setComposing(true);
  try {
    const res = await fetch('/api/compose', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ selected, mode, literal: state.literal,
        partner: mode === 'answer' ? $('#partner-said').value.trim() : '' }),
    });
    if (!res.ok) throw new Error(`compose ${res.status}`);
    const { candidates } = await res.json();
    // He moved on: Urgent, another mode, or (belt and braces) different words than were sent.
    if (seq !== composeSeq || state.urgent || mode !== state.mode
        || selected.join('\u0001') !== state.selected.join('\u0001')) return;
    showConfirm(candidates ?? []);
  } catch (e) {
    if (seq === composeSeq) toast(`Could not build the sentence: ${e.message}`);
  } finally {
    // Only the latest request owns the flag: Urgent already cleared it for this one, and a newer
    // Say it may be waiting now.
    if (seq === composeSeq) setComposing(false);
  }
}

function setComposing(on) {
  state.composing = on;
  document.body.classList.toggle('building', on);
  $('#compose').textContent = on ? 'Building your sentence…' : COMPOSE_LABEL[state.mode];
  renderControls();
  if (on) resetDwell();   // nothing half-armed on the paused grid fires when it comes back
}

// answer = reply to them. ask = put a question to them. tell = say something unprompted.
//
// The words he has picked survive a mode change. Only the grid and the sentence Say it builds
// depend on the mode. Clearing them meant one brush of a tab threw away minutes of picks, with
// no Undo to bring them back. Tapping the tab already on, or any tab in urgent, does nothing:
// urgent keeps his sentence, and the tabs are dimmed there.
const COMPOSE_LABEL = { answer: 'Say it', ask: 'Ask it', tell: 'Say it' };
function setMode(mode) {
  if (mode === state.mode || state.urgent) return;
  state.mode = mode;
  if (!state.selected.length) { state.selections = 0; state.startedAt = null; }
  for (const m of ['answer', 'ask', 'tell']) {
    const b = $(`#m-${m}`);
    b.classList.toggle('on', m === mode);
    b.setAttribute('aria-selected', String(m === mode));
  }
  $('#partner-block').hidden = mode !== 'answer';
  document.body.classList.toggle('partner-off', mode !== 'answer');   // keeps the top band (style.css)
  if (!state.composing) $('#compose').textContent = COMPOSE_LABEL[mode];
  $('#mode-eye').textContent = `Mode: ${MODE_NAME[mode]}`;
  renderSelected();
  renderHUD();
  loadTiles();
}

// The header tabs are hidden in camera modes, so eyes and blink users switch mode with one button
// that steps through the three. It only works with no words picked: a misfire mid-sentence would
// quietly change what Say it builds out of words that took him minutes.
const MODE_NEXT = { answer: 'ask', ask: 'tell', tell: 'answer' };
const MODE_NAME = { answer: 'Answer', ask: 'Ask', tell: 'Say' };
function cycleMode() {
  if (state.selected.length || state.urgent) return;
  setMode(MODE_NEXT[state.mode] ?? 'answer');
}

/** "Wait, I'm talking." keep:true: the button that buys him time must not delete his sentence. */
function sayWait() {
  // Its own one-pick row in the log, not the half-built sentence's count and clock.
  say("Wait — I'm saying something.",
    { instant: true, keep: true, startedAt: performance.now(), selections: 1 });
}

/* ---------- speaking ---------- */

// He always chooses. The model proposes; it never speaks for him.
//
// Each option is NUMBERED, so he can pick it three ways, whichever his body allows today:
//   - touch it,
//   - look at it and hold his gaze (dwell),
//   - or blink: a quick blink steps to the next option, a held blink says the highlighted one.
function showConfirm(candidates) {
  const box = $('#candidates');
  box.innerHTML = '';
  candidates.forEach((c, i) => {
    const b = document.createElement('button');
    b.className = 'candidate';
    b.innerHTML = `<span class="num">${i + 1}</span><span class="say">${escapeHtml(c)}</span>`;
    b.onclick = () => say(c);
    b.dataset.text = c;
    box.appendChild(b);
  });
  openConfirm();
}

// Every path that shows the sheet goes through here, so its selection state can never be stale.
// A reopened sheet with a leftover sheetDwellStart used to fire an option with zero dwell.
function openConfirm() {
  sheetIdx = -1;                 // NOTHING pre-armed: a lone long-blink must not speak option 1
  sheetDwellIdx = -1;
  sheetDwellStart = 0;
  sheetArmed = false;
  sheetGen++;                    // new options: a blink that started before this says nothing
  renderSheetHint();
  $('#confirm').hidden = false;
  highlightSheet();
}

/** Say the real gesture. In Eyes mode his gaze picks the option; in Blinks mode a quick blink does. */
function renderSheetHint() {
  $('#sheet-hint').hidden = state.driver !== 'gaze' && state.driver !== 'blink';
  $('#sheet-hint').textContent = state.driver === 'blink'
    ? 'Blink firmly to move to the next one. Close your eyes until the beep to say it.'
    : state.confirmBy === 'dwell'
      ? `Look at one and keep looking for ${(sheetDwellMs() / 1000).toFixed(1)} seconds to say it, or close your eyes until the beep.`
      : 'Look at one, then close your eyes until the beep to say it.';
}

const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

let sheetArmed = false;   // has he moved to an option yet? a long blink before this does nothing.
/** Move the highlight across the numbered options + Back, and show which one is current. */
function highlightSheet() {
  const opts = sheetOptions();
  // The .cursor class is the highlight. Moving DOM focus here put it on a button, where the scan
  // driver's keys stopped working, and did it again on every gaze frame over the sheet.
  opts.forEach((o, i) => o.classList.toggle('cursor', i === sheetIdx && sheetIdx >= 0));
}
// Steps visit Urgent first, then option 1, 2, ... and Back: the same order as the grid, where the
// first step from a fresh grid lands on Urgent. An emergency costs one step; a sentence one more.
function sheetNext() {
  const n = sheetOptions().length;
  sheetIdx = nextZone(sheetIdx, n, n - 1);   // Urgent is the last option on the sheet
  sheetArmed = true;
  highlightSheet();
}
function sheetConfirm() {
  // Never fire an option he hasn't landed on. A single long blink on a just-opened sheet must
  // not speak candidate 1 — he has to move to it first (a short blink, or a gaze dwell).
  if (!sheetArmed || sheetIdx < 0) return;
  sheetOptions()[sheetIdx]?.click();
}

// Urgent from the sheet: close it and open the emergency grid. Never a toggle here, or a sheet
// that somehow opened over the urgent grid would take him OUT of it.
function sheetUrgent() {
  closeConfirm();
  if (!state.urgent) toggleUrgent();
}

function closeConfirm() {
  $('#confirm').hidden = true;
  // Focus back on Say it helps a touch user on a keyboard. For the aiming drivers the cursor is
  // the only place he is, and a focused Say it left a switch user's Space reopening the sheet.
  if (state.driver === 'touch') $('#compose').focus();
  else document.activeElement?.blur?.();
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
let speakWatchdog = null;
// Each say() is one generation. An Urgent pick can interrupt a say() that is still waiting for its
// audio, where there is nothing yet to pause: when that audio arrived it used to play over "can't
// breathe", take over the shared player and watchdog, and clear state.speaking under it. Now a
// superseded call checks its generation after every await and quietly stops. Its fetch is aborted.
let sayGen = 0, sayAbort = null;
// "Turn the mic back on when this ends" belongs to whatever is speaking LAST. Kept per call, it was
// dropped by every interrupt (the interrupting call saw the mic already off), and Listen stayed
// off for the rest of the session. App prompts (speakPrompt) share it for the same reason.
let resumeListening = false;

async function say(text, { instant = false, keep = false, interrupt = false,
  startedAt: t0 = null, selections = null } = {}) {
  // Urgent interrupts; everything else waits its turn. Without interrupt, a stuck utterance could
  // block the emergency grid — with it, "can't breathe" always speaks.
  if (state.speaking && !interrupt) return;
  const gen = ++sayGen;
  const stale = () => gen !== sayGen;
  sayAbort?.abort();
  const abort = (sayAbort = new AbortController());
  if (state.listening) { resumeListening = true; stopListening(); }   // or the mic hears his voice
  clearSpeaking();                     // kill any current audio and reset the flag first
  stopPrompt();                        // his words cut off an app instruction, never talk over it
  state.speaking = true;
  renderControls();
  closeConfirm();

  const startedAt = t0 ?? state.startedAt ?? performance.now();

  let url = null;
  const drop = () => { if (url) { URL.revokeObjectURL(url); url = null; } };
  const done = () => {
    drop();
    if (stale()) return;               // a newer say() owns the flag, the watchdog and the mic
    clearTimeout(speakWatchdog);
    state.speaking = false;
    renderControls();                  // Say it was greyed while he spoke; give it back
    if (resumeListening) { resumeListening = false; startListening(); }
  };

  try {
    const res = await fetch('/api/speak', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text, voice: state.profile?.voice ?? 'placeholder' }),
      signal: abort.signal,
    });
    if (stale()) return;
    if (!res.ok) throw new Error(`speak ${res.status}`);

    const blob = await res.blob();
    if (stale()) return;
    url = URL.createObjectURL(blob);
    player = new Audio(url);
    player.onended = done;
    player.onerror = done;
    // WATCHDOG: if neither onended nor onerror ever fires (a stalled element), state.speaking
    // would stick true and silence ALL later speech, including Urgent. Force it clear after a
    // generous ceiling.
    clearTimeout(speakWatchdog);
    speakWatchdog = setTimeout(done, Math.max(4000, text.length * 140));

    await player.play();               // a rejected promise here is the browser blocking sound
    if (stale()) return drop();        // interrupted before it got going: not said, not logged
    speaking(text);
    logUtterance(text, startedAt, instant, selections ?? state.selections);
    if (!keep) reset();
  } catch (e) {
    // Superseded: the newer call owns the screen. No toast, and no sheet reopened over it.
    if (stale() || e.name === 'AbortError') return drop();
    done();
    toast(e.name === 'NotAllowedError'
      ? 'The browser blocked the sound. Tap anywhere on the page, then try again.'
      : `Could not speak: ${e.message}`);
    // NOT logged, NOT reset. His words stay on screen so he can try again without rebuilding.
    // Reopen through openConfirm() so the sheet's dwell/arm state is reset — reopening it raw used
    // to fire an option with zero dwell from leftover state.
    if (!instant && state.selected.length && $$('.candidate').length) openConfirm();
  }
}

/** Stop any current audio and clear the speaking flag — the single place that owns that state. */
function clearSpeaking() {
  clearTimeout(speakWatchdog);
  if (player) { player.onended = null; player.onerror = null; try { player.pause(); } catch {} player = null; }
  state.speaking = false;
}

/** selections_per_sentence and seconds_to_sentence are the numbers the research is about. */
function logUtterance(text, startedAt, instant, selections) {
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
      selections_per_sentence: selections,
      seconds_to_sentence: +elapsed.toFixed(1),
      effective_wpm: +(words / (elapsed / 60)).toFixed(1),
      predictive: state.predictive,
      literal: state.literal,
      driver: state.driver,
    }),
  }).catch(() => {});
}

function reset() {
  changeSelected(() => { state.selected = []; });
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
let dwellTrace = [];       // where his gaze actually sat during the dwell — free ground truth
let pursuitCued = false;   // "now follow the moving dot" said once per calibration
// The accuracy test shows the live grid (no #calib overlay), so the #calib guards do not cover it.
// While this is set his gaze arms nothing and his blinks and switches pick nothing: a blink used to
// say a word mid-test and rebuild the grid under the measurement.
let gazeTesting = false;
// Escape bumps this. The signal check, recenter and accuracy test stop at their next step.
let overlayGen = 0;
/** A check for an overlay flow: throws once Escape was pressed or this camera g has gone. */
function overlayGuard(g) {
  const gen = overlayGen;
  return () => {
    if (gen !== overlayGen) throw new Error('cancelled');
    if (gaze !== g || !g.running) throw new Error('the camera stopped');
  };
}

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

// The action buttons are gaze targets too. Which one is his gaze on?
//
// Not the exact button box. The row sits at the bottom edge, where the eye is least accurate and
// an overshoot clamps the dot to the screen edge (gaze.js), right under a short button. So the
// WHOLE band below the grid counts, down to the edge, split at the midpoints between buttons.
// The bottom tiles' own hit box already stops short of their edge (MARGIN), which is the buffer
// that keeps the band from stealing a look at a bottom-row word.
let armedControl = null;   // 'say' | 'urgent' | 'undo' | 'hold' | 'mode' | null
function controlUnder(x, y) {
  if (y < $('#grid').getBoundingClientRect().bottom) return null;
  const btns = $$('.actions > button').filter((b) => !b.hidden && b.getBoundingClientRect().width);
  const rs = btns.map((b) => b.getBoundingClientRect());
  for (let k = 0; k < btns.length; k++) {
    const lo = k === 0 ? -Infinity : (rs[k - 1].right + rs[k].left) / 2;
    const hi = k === btns.length - 1 ? Infinity : (rs[k].right + rs[k + 1].left) / 2;
    // A disabled button's slice is dead, not handed to its neighbour: looking at a greyed-out
    // Say it must not arm Urgent.
    if (x >= lo && x < hi) return btns[k].disabled ? null : btns[k].dataset.ctrl ?? null;
  }
  return null;
}
const controlEl = (c) => (c ? $(`.actions > button[data-ctrl="${c}"]`) : null);
const CONTROL_ACT = { say: () => compose(), urgent: () => toggleUrgent(), undo: () => undo(),
  hold: () => sayWait(), mode: () => cycleMode() };
/** Fire an action button from the eyes: a confirming blink or a completed dwell. */
function fireControl(c) {
  lastCommitAt = performance.now();
  resetDwell();
  CONTROL_ACT[c]?.();
}

function tileAt(x, y) {
  const tiles = $$('.tile');
  for (let i = 0; i < tiles.length; i++) {
    const r = tiles[i].getBoundingClientRect();
    if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) return i;
  }
  return -1;
}

let armedTile = -1;
let armedGen = -1;          // the grid render armedTile belongs to
let armedLockedAt = 0;      // last frame the eye was LOCKED on armedTile
let armedSince = 0;         // when this word became armed
const ARM_GRACE_MS = 300;
const ARM_SEEN_MS = 250;    // armed this long before a blink may say it: new words appearing under
                            // his eye must be SEEN armed first, not said by a blink already closing
let lastCommitAt = 0;
let lastBlinkActAt = 0;            // Blinks driver: the last blink that stepped or selected
const BLINK_REFRACTORY_MS = 400;
const clearArmed = () => [...$$('.tile'), ...$$('.actions > button')]
  .forEach((t) => t?.classList.remove('armed', 'ready'));

// DWELL LATCH. After a dwell picks a word, the predictive grid often puts the next word in the
// same slot, under the same steady eye, and a plain refractory of a few frames let the dwell
// start again and pick that new word too. Now the slot (or button) he just used stays quiet until
// his eye actually goes somewhere else.
let dwellLatch = -1;          // tile index
let ctrlLatch = null;         // control name
let ctrlMiss = 0, ctrlDwellStart = 0;

function resetDwell() {
  dwellTile = -1;
  dwellStart = 0;
  dwellTrace = [];
  armedTile = -1;
  armedControl = null;
  ctrlMiss = 0;
  ctrlDwellStart = 0;
  [...$$('.tile'), ...$$('.actions > button')].forEach((t) => { t.style.setProperty('--dwell', '0'); });
  clearArmed();
}

/**
 * The action row under his gaze. Returns true when the row has this frame, so the tiles skip it.
 *
 * Same rules as a tile: it arms only on a LOCKED eye, and one stray frame off a big button is
 * jitter, not a look away, so it stays armed until SWITCH_FRAMES frames in a row disagree.
 * In dwell mode a ring fills on the button and fires it, exactly as on a tile.
 */
function trackControl(ctrl, locked) {
  const now = performance.now();
  if (ctrlLatch && locked && ctrl !== ctrlLatch) ctrlLatch = null;   // he moved on: re-enable it
  if (armedControl && ctrl !== armedControl) {
    if (++ctrlMiss < SWITCH_FRAMES) return true;
    controlEl(armedControl)?.style.setProperty('--dwell', '0');
    armedControl = null;
    ctrlMiss = 0;
    clearArmed();
  } else ctrlMiss = 0;
  if (!ctrl) return false;
  if (!locked) { ctrlDwellStart = now; return !!armedControl; }

  const el = controlEl(ctrl);
  if (armedControl !== ctrl) {
    clearArmed();
    armedControl = ctrl; dwellTile = -1; armedTile = -1;
    ctrlDwellStart = now;
    el?.classList.add('armed');
  }
  if (state.confirmBy !== 'dwell' || ctrl === ctrlLatch) { ctrlDwellStart = now; return true; }
  const frac = Math.min(1, (now - ctrlDwellStart) / state.dwellMs);
  el?.style.setProperty('--dwell', String(frac));
  if (frac >= 1) {
    ctrlLatch = ctrl;
    flash(el, 'fired', 450);
    fireControl(ctrl);
  }
  return true;
}

// Gaze-dwell over the numbered sentence options. Bigger, forgiving targets — there are only a
// few, stacked vertically, so vertical (the weak gaze axis) does the least work.
let sheetDwellIdx = -1, sheetDwellStart = 0;
// A sentence takes longer to read than a one-word tile. With the tile dwell (0.9 s) just reading
// option 1 said it, so the sheet needs at least twice as long and never less than 2 s.
const sheetDwellMs = () => Math.max(2 * state.dwellMs, 2000);
function dwellOverSheet(x, y, locked) {
  const opts = sheetOptions();
  let hit = -1;
  for (let i = 0; i < opts.length; i++) {
    const r = opts[i].getBoundingClientRect();
    if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) { hit = i; break; }
  }
  if (hit < 0) { sheetDwellIdx = -1; opts.forEach((o) => o.style.setProperty('--dwell', '0')); return; }

  // Looking at an option ARMS it (so a blink can say it), and highlights it, so the highlighted
  // option and the one a blink fires can never diverge. Only once the eye has LANDED: an eye in
  // flight across the options must not move the highlight on the way.
  if (locked && sheetIdx !== hit) { sheetIdx = hit; highlightSheet(); }
  if (locked) sheetArmed = true;

  if (hit !== sheetDwellIdx) {
    opts.forEach((o) => o.style.setProperty('--dwell', '0'));
    sheetDwellIdx = hit;
    sheetDwellStart = performance.now();
    return;
  }
  // Only DWELL mode auto-fires by holding a gaze. In blink mode (the default) looking at an option
  // just arms it — he still has to blink — so merely READING the options never speaks one.
  if (state.confirmBy !== 'dwell') return;
  if (!locked) { sheetDwellStart = performance.now(); return; }
  const frac = Math.min(1, (performance.now() - sheetDwellStart) / sheetDwellMs());
  opts[hit].style.setProperty('--dwell', String(frac));
  if (frac >= 1) {
    sheetDwellIdx = -1;
    opts.forEach((o) => o.style.setProperty('--dwell', '0'));
    opts[hit].click();
  }
}

function onGazePoint(x, y, locked) {
  const dot = $('#gaze-dot');

  // In BLINK mode the cursor is driven by blinks, not by where he looks — so hide the gaze dot
  // and never let gaze move the cursor, or the two would fight each other.
  if (state.driver === 'blink') { dot.hidden = true; resetDwell(); return; }

  // The accuracy test measures where his eye goes on its own. A visible dot lets him steer it onto
  // the target, which scores the steering, not the tracker. Nothing arms or re-renders either.
  if (gazeTesting) { dot.hidden = true; return resetDwell(); }

  dot.hidden = false;
  dot.style.transform = `translate(${x}px, ${y}px)`;
  dot.classList.toggle('locked', !!locked);   // he can see when the eye has actually landed

  if (!$('#calib').hidden) return resetDwell();

  // The sentence sheet is its own dwell surface: he can look at a numbered option and hold his
  // gaze to say it, exactly as he selects a tile. (Blinks work here too — see onBlink.)
  if (!$('#confirm').hidden) return dwellOverSheet(x, y, locked);

  // The action row is a gaze target too. Without it an eyes-only user can build a sentence and
  // never speak it, never undo a misfire, and never reach the emergency grid.
  const ctrl = controlUnder(x, y);

  // On the grid, dwell pauses while speaking, EXCEPT for Urgent and the urgent grid. He must be
  // able to reach "can't breathe" while a long sentence is still playing (an urgent pick
  // interrupts the audio), so that one button stays live.
  // Same while Say it is building the sentence: the grid is paused until the sheet opens.
  if ((state.speaking || state.composing) && !state.urgent && ctrl !== 'urgent') return resetDwell();
  if (trackControl(ctrl, locked)) return;

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
    if (i !== dwellLatch) dwellLatch = -1;   // his eye left the slot he just picked
    if (i >= 0) { state.cursor = i; renderGrid(); }
    return;
  }
  candidateTile = i;
  candidateFrames = 0;
  if (i < 0) return;

  // He is fixated on tile i — it is now ARMED. How he commits it depends on the setting:
  //   'blink' (default): he blinks to confirm. No dwell timer runs, so nothing fires while he
  //           just reads the grid. A blink is faster and far less fatiguing than holding a stare.
  //   'dwell': he holds his gaze and a ring fills. For someone who cannot blink deliberately.
  //
  // A tile is only armed when the eye is LOCKED on it. armedTile is what a blink confirms — a
  // blink while the eye is mid-flight, or resting between tiles, must fire NOTHING.
  //
  // The lock flickers for a frame or two even while he holds still (measured: the right tile was
  // armed only 77% of settled time). A flicker is not a look away, so the arm survives it for
  // ARM_GRACE_MS. A real move to another tile still clears it at once, via resetDwell above.
  if (!locked) {
    dwellStart = performance.now();
    if (armedTile >= 0 && performance.now() - armedLockedAt > ARM_GRACE_MS) { armedTile = -1; clearArmed(); }
    return;
  }

  // Armed in BOTH modes: in dwell mode a long blink is a second way to say the filling tile.
  // armedGen: a re-render (new words) replaces the tile under his eye, so re-arm the new one.
  if (armedTile !== i || armedGen !== gridGen) {
    // New words under his eye restart the dwell too: time spent on the old word is not a choice
    // of the new one.
    if (armedGen !== gridGen) { dwellStart = performance.now(); dwellTrace = []; }
    clearArmed(); armedTile = i; armedGen = gridGen; armedSince = performance.now();
    if (state.confirmBy === 'blink') $$('.tile')[i]?.classList.add('armed');
  }
  armedLockedAt = performance.now();
  if (state.confirmBy === 'blink') return;
  if (i === dwellLatch) { dwellStart = performance.now(); return; }   // just picked: look away first

  dwellTrace.push([x, y]);
  const frac = Math.min(1, (performance.now() - dwellStart) / state.dwellMs);
  $$('.tile')[i]?.style.setProperty('--dwell', String(frac));
  if (frac >= 1) commitGazeTile(i);
}

/**
 * Commit the armed tile — from a completed dwell OR a confirming blink. Both are the same event:
 * "his gaze was on this tile and he chose it." So both re-anchor the bias, learn the sample into
 * the terrain map, and fire the selection. Sharing this is what keeps blink-confirm as smart as
 * dwell — it improves the calibration exactly the same way.
 */
function commitGazeTile(i, pre = null) {
  // pre: from a blink, what the tracker saw just BEFORE the lids moved (gaze.preBlink()). By the
  // time the blink is over the live features are the reopening eye, and learning from those would
  // teach the map that a half-closed lid means "this tile".
  const r = $$('.tile')[i]?.getBoundingClientRect();
  if (r) {
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    if (pre?.locked) gaze?.nudge(cx - pre.x, cy - pre.y);
    else if (dwellTrace.length > 3) {
      const mx = dwellTrace.reduce((s2, p) => s2 + p[0], 0) / dwellTrace.length;
      const my = dwellTrace.reduce((s2, p) => s2 + p[1], 0) / dwellTrace.length;
      gaze?.nudge(cx - mx, cy - my);
    }
    const id = gaze?.learn(pre?.f ?? null, cx, cy);
    if (id != null) {
      lastLearnId = id;
      lastLearnAt = performance.now();
      scheduleGazeSave();
    }
  }
  lastCommitAt = performance.now();
  resetDwell();
  dwellTile = -2;                      // refractory: don't instantly re-fire on the same tile
  dwellLatch = i;                      // and no new dwell in this slot until his eye leaves it
  state.cursor = i;                    // SELECT picks the cursor's tile: make sure it is this one
  bus.emit('SELECT');
}

// WHAT A BLINK CONFIRMS is frozen the moment his lids START to move (gaze.js onLidsClosing),
// not read when they reopen. By the reopen the closing lids have already dragged the dot and could
// un-arm the word. Each part carries the render it belongs to: if the words changed while his eyes
// were shut, the blink says nothing, so it can never say a word he did not see armed.
let blinkTarget = null;
function snapshotBlinkTarget() {
  const sheetOpen = !$('#confirm').hidden;
  blinkTarget = {
    sheet: sheetOpen && sheetArmed && sheetIdx >= 0 ? { idx: sheetIdx, gen: sheetGen } : null,
    tile: !sheetOpen && armedTile >= 0 && armedGen === gridGen && performance.now() - armedSince >= ARM_SEEN_MS
      ? { i: armedTile, word: state.tiles[armedTile], gen: gridGen } : null,
    control: sheetOpen ? null : armedControl,
    pre: gaze?.preBlink() ?? null,     // the open-eye point and features, for re-anchoring
  };
}
/** The element a snapshot points at, if it is still the same thing he saw armed. */
function blinkTargetEl(b) {
  if (!b) return null;
  if (b.sheet) return !$('#confirm').hidden && b.sheet.gen === sheetGen ? sheetOptions()[b.sheet.idx] : null;
  if (b.control) { const el = controlEl(b.control); return el && !el.disabled ? el : null; }
  if (b.tile) return b.tile.gen === gridGen && state.tiles[b.tile.i] === b.tile.word ? $$('.tile')[b.tile.i] : null;
  return null;
}

// Feedback he can perceive, given AFTER the decision: a gold flash means it was said, a grey ring
// means the blink was seen but did nothing. A blink that was nearly long enough on an armed word
// also says so on the word, so "not seen" and "too short" never feel the same.
function flash(el, cls, ms) {
  if (!el) return;
  el.classList.add(cls);
  setTimeout(() => el.classList.remove(cls), ms);
}
function blinkFeedback(result, el = null, held = 0) {
  const d = $('#gaze-dot');
  const dot = d && !d.hidden ? d : null;
  // The status chip is never hidden in camera modes, so every blink the app heard shows there too,
  // even in Blinks mode (no dot) and on a blink that had nothing to act on.
  flash($('#eye-status'), result === 'fired' || result === 'seen' ? 'pip-ok' : 'pip-no', 350);
  if (result === 'fired' || result === 'seen') {
    flash(dot, 'pip-ok', 350);
    if (result === 'fired') flash(el, 'fired', 450);
    return;
  }
  flash(dot ?? el, 'pip-no', 350);
  // Only for a closure longer than any natural blink: a 280 ms blink is him blinking, not a try.
  if (el && held >= Math.max(NATURAL_BLINK_MAX_MS, state.blinkMs * 0.75) && held < state.blinkMs) {
    flash(el, 'too-short', 1200);
  }
}

// THE EYE STATUS CHIP (camera modes). faceSeen: null until the camera's first frame, then whether
// it can see his face. The bar fills while his eyes are shut and turns gold at his Blink length,
// and holds where it got to for a moment after he opens them, so "not long enough" can be read.
let faceSeen = null;
let shutSince = 0, shutClearTimer = null;
function renderEyeStatus() {
  const el = $('#eye-status');
  el.classList.toggle('lost', faceSeen === false);
  $('#eye-status-text').textContent = faceSeen === null ? 'Starting'
    : !faceSeen ? 'No face' : state.driver === 'blink' ? 'Blinks' : 'Eyes';
}
function showLids(shut, heldMs) {
  const bar = $('#eye-shut');
  if (!shut) {
    if (!shutSince) return;
    shutSince = 0;
    clearTimeout(shutClearTimer);
    shutClearTimer = setTimeout(() => { bar.style.width = '0%'; bar.classList.remove('long'); }, 700);
    return;
  }
  if (!shutSince) { shutSince = performance.now(); clearTimeout(shutClearTimer); }
  // gaze.js passes how long the gate has seen them shut, the clock the beep uses. Fall back to ours.
  const ms = heldMs ?? performance.now() - shutSince;
  const frac = Math.min(1, ms / state.blinkMs);
  bar.style.width = `${frac * 100}%`;
  bar.classList.toggle('long', frac >= 1);
}

// If he undoes within a few seconds of a gaze pick, that selection was probably WRONG — retract
// what we learned from it, or the map learns his mistakes.
let lastLearnId = null, lastLearnAt = 0;
function retractIfRecent() {
  if (lastLearnId != null && performance.now() - lastLearnAt < 5000) {
    gaze?.retract(lastLearnId);
    lastLearnId = null;
    scheduleGazeSave();   // the poison may already be on disk (save debounces 4s) — overwrite it
  }
}

// Natural blinks measured 83-292 ms (2026-10-08 recording). In Eyes mode a 'short' blink under
// this is just him blinking: it gets no grey pip and no "longer" label, or the app would flicker
// and scold every ordinary blink on camera.
const NATURAL_BLINK_MAX_MS = 300;

// Blinks driver: would SELECT on the highlighted zone actually do something? A long blink that
// does nothing must not flash gold, or he is told it was said when it was not.
function zoneActs(z, el) {
  if (z === undefined || !el || el.disabled) return false;
  if (z === URGENT) return true;                                  // never blocked
  if (z === HOLD) return !state.speaking;                         // say() waits its turn
  if (z === UNDO) return !state.composing;                        // see undo()
  if (z === MODE) return !state.selected.length && !state.urgent; // see cycleMode()
  if (z === SAY) return true;                                     // disabled covers it
  return state.urgent || (!state.speaking && !state.composing);   // a word: see pick()
}

/**
 * One finished blink. Returns what it did, for the blink log: the outcome names the branch, so
 * "blink to confirm does nothing" can be read off data/gaze.jsonl.
 * kind (blinkgate.js): 'reflex' < 120 ms, 'short', 'long' >= blink length, 'rest' > 3 s.
 */
function onBlink(kind, held) {
  const snap = blinkTarget;
  // A quick blink can come just before the long one, inside the same lid movement (the gaze is
  // still frozen). Keep the snapshot for that; a long blink or a rest uses it up.
  if (kind === 'long' || kind === 'rest') blinkTarget = null;
  $$('.ready').forEach((e) => e.classList.remove('ready'));
  if (!$('#calib').hidden || gazeTesting) return 'calibrating';   // never during calibration or the test
  if (kind === 'reflex') return 'reflex';                          // a natural blink: nothing to say
  const long = kind === 'long';

  // BLINKS DRIVER: every blink is a command, so a natural one must do nothing, in the sheet as
  // on the grid. Only a blink clearly longer than an ordinary one steps; only one as long as his
  // Blink length selects (blinkscan.js). Natural blinks often come in pairs, so the second
  // blink inside BLINK_REFRACTORY_MS of an acted one is dropped too.
  if (state.driver === 'blink') {
    const act = blinkAction(kind, held, state.blinkMs);
    if (!act) return 'natural';
    if (performance.now() - lastBlinkActAt < BLINK_REFRACTORY_MS) { blinkFeedback('ignored'); return 'ignored-refractory'; }
    lastBlinkActAt = performance.now();
  } else if (kind === 'short' && held < NATURAL_BLINK_MAX_MS) {
    return 'natural';
  }

  // The sentence sheet. In Blinks mode a quick blink steps to the next numbered option; in Eyes
  // mode his gaze does that, and a stray natural blink must not move the highlight under him.
  // A long blink says the option that was highlighted when his lids started to close.
  if (!$('#confirm').hidden) {
    if (kind === 'short' && state.driver === 'blink') { sheetNext(); blinkFeedback('seen'); return 'sheet-step'; }
    // Blinks mode: only blinks move the highlight, so the live one is the one he saw.
    const t = state.driver === 'blink'
      ? { sheet: sheetArmed && sheetIdx >= 0 ? { idx: sheetIdx, gen: sheetGen } : null } : snap;
    const el = blinkTargetEl(t);
    if (long && t?.sheet && el) {
      sheetIdx = t.sheet.idx;
      sheetArmed = true;
      blinkFeedback('fired', el);
      sheetConfirm();
      return 'fired-sheet';
    }
    blinkFeedback('ignored', el ?? sheetOptions()[sheetIdx], held);
    return long ? 'ignored-sheet-no-target' : 'too-short';
  }

  // BLINK DRIVER: he selects entirely by blinking, no gaze pointing. Quick blink steps the
  // cursor to the next tile; held blink selects it (single-switch scanning, eyelid as switch).
  // Blinks are NOT blocked during speech: he must be able to reach and fire URGENT mid-sentence.
  // A new grid has nothing highlighted, so a long blink before his first step selects nothing.
  if (state.driver === 'blink') {
    if (kind === 'short') { blinkStep(); blinkFeedback('seen'); return 'step'; }
    const z = zones()[state.cursor];
    const el = z === undefined ? null : ZONE_EL[z] ? $(ZONE_EL[z]) : $$('.tile')[state.cursor];
    if (!long) { blinkFeedback('ignored', el, held); return 'ignored-rest'; }
    if (!zoneActs(z, el)) { blinkFeedback('ignored', el, held); return z === undefined ? 'ignored-no-target' : 'ignored-cannot-act'; }
    blinkFeedback('fired', el);
    bus.emit('SELECT');
    return ZONE_EL[z] ? 'fired-control' : 'fired-tile';
  }

  // EYES DRIVER: he LOOKS at a tile (it arms while his eye is LOCKED on it), then closes his
  // eyes until the beep to say it. Only the snapshot fires: never a tile his eye merely passed
  // over, never one the closing lids dragged the dot onto. Same in dwell mode, where a long
  // blink is a second way to say the tile that is filling.
  const el = blinkTargetEl(snap);
  // Just committed: ignore blinks for a beat, so one deliberate blink can't fire twice.
  if (!long || !el || performance.now() - lastCommitAt < 700) {
    blinkFeedback('ignored', el, held);
    return !long ? (kind === 'rest' ? 'ignored-rest' : 'too-short')
      : !el ? 'ignored-no-target' : 'ignored-cooldown';
  }

  // An armed action button (Say it, Urgent, Undo, Wait, Mode): an eyes-only user can reach
  // every zone. While a sentence plays only Urgent can be armed (onGazePoint), so this is safe.
  if (snap.control) {
    blinkFeedback('fired', el);
    fireControl(snap.control);
    return 'fired-control';
  }
  // In the urgent grid a blink fires even mid-sentence: pick() interrupts the audio for it.
  if (snap.tile && ((!state.speaking && !state.composing) || state.urgent)) {
    blinkFeedback('fired', el);
    commitGazeTile(snap.tile.i, snap.pre);
    return 'fired-tile';
  }
  blinkFeedback('ignored', el, held);
  return state.composing ? 'ignored-composing' : 'ignored-speaking';
}

async function startGaze() {
  if (gaze) return;
  faceSeen = null; showLids(false); renderEyeStatus();   // "Starting" until the first camera frame
  const g = createGaze({
    onGaze: onGazePoint,
    confirmMs: state.blinkMs,
    // The lids just started to move: freeze what this blink would confirm.
    onLidsClosing: snapshotBlinkTarget,
    // Eyes still shut, and shut long enough: a soft beep so he knows he can open them. Only when
    // opening them will actually DO something, or every rest of the eyes would beep at him.
    onBlinkHeld: () => {
      if (!$('#calib').hidden || gazeTesting) return;
      const sheetOpen = !$('#confirm').hidden;
      const el = state.driver !== 'blink' ? blinkTargetEl(blinkTarget)
        : sheetOpen ? (sheetArmed && sheetIdx >= 0 ? sheetOptions()[sheetIdx] : null)
        : $('.tile.cursor, .actions > button.cursor');
      if (!el) return;
      // Blinks mode: no beep on a zone that cannot act right now (a word while speaking, say).
      if (state.driver === 'blink' && !sheetOpen && !zoneActs(zones()[state.cursor], el)) return;
      beep(990, 0.07, 0.05);
      el.classList.add('ready');
    },
    // kind (blinkgate.js): 'reflex' < 120 ms, 'short', 'long' >= blink length, 'rest' > 3 s.
    // Every blink is logged with the branch that handled it (logBlink), so a blink that "did
    // nothing" in the field leaves a reason on disk.
    onBlink: (kind, held) => logBlink(kind, held, onBlink(kind, held)),
    onLids: showLids,
    onFace: (found) => {
      $('#gaze-dot').classList.toggle('lost', !found);
      if (found !== faceSeen) { faceSeen = found; renderEyeStatus(); }
      if (!found) showLids(false);
      if (!found) { resetDwell(); blinkTarget = null; }
    },
    // Only a fatal error stops tracking (gaze.js recovers from stalls and dropouts by itself).
    onError: (msg, { fatal = true } = {}) => { toast(msg); if (fatal && gaze === g) gazeStopped(); },
    onRecovered: () => toast('Camera back. Eye tracking is on again.', 'ok'),
    onCalibrationProgress: (p) => {
      if (p.state === 'done') {
        $('#calib').hidden = true;
        // The leave-one-out error, against half a tile. Anything worse and the wrong word gets
        // spoken in his voice — so say so, instead of congratulating him on a broken fit.
        // Which model won, and what each scored. If "head" wins, he is aiming with his head and
        // not his eyes — and that is a fact about him, not a bug to hide.
        $('#calib-bar').hidden = true;
        $('#gaze-state').textContent =
          `X ±${p.errX}px / Y ±${p.errY}px · tile ${p.tile.w}x${p.tile.h} · ${p.samples} samples · ${p.variant}`
          + ` · ${p.usable ? 'usable' : 'TOO LOOSE'}${p.kept ? ', kept the previous map' : ''}`;
        // Persist a good fresh terrain immediately: he should never re-teach the app his own eyes.
        // Only a good one. A loose fit is not installed over a working map (gaze.js), and one
        // installed for lack of anything better is never written over the map on disk.
        if (p.usable) scheduleGazeSave();
        fetch('/api/gazelog', {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ kind: 'calibration', ...p,
            screen: { w: window.innerWidth, h: window.innerHeight }, probe: gaze?.probe() }),
        }).catch(() => {});
        // Judge each axis against its OWN tile dimension. Horizontal was already fine while
        // vertical was failing, and a single blended number hid that completely.
        const okX = p.errX < p.tile.w * 0.45, okY = p.errY < p.tile.h * 0.45;
        const loose = (okX && okY)
          ? `Calibrated. Now run the test.`
          : !okX && !okY ? `Calibration is too loose in both directions. Sit still and try again.`
          : okX ? `Left and right is good, but up and down is too loose. Keep your head level and try again.`
          : `Up and down is good, but left and right is too loose. Try again.`;
        const msg = p.kept ? `${loose} Your previous calibration is still in use.` : loose;
        toast(msg, okX && okY ? 'ok' : 'error');
        speakPrompt(msg);
        return;
      }
      $('#calib').hidden = false;
      const d = $('#calib-dot');
      d.style.left = `${p.x * 100}%`;
      d.style.top = `${p.y * 100}%`;
      d.classList.toggle('sampling', p.state === 'sampling');

      if (p.state === 'pursuit') {
        // The dot MOVES and he follows it. Following a slowly moving target is a reflex, not a
        // skill — which is why this collects hundreds of clean samples where nine dots gave nine.
        d.classList.add('pursuit');
        // The switch from still dots to a moving one needs a cue, or he is still waiting for a beep
        // while the dot sets off without him.
        if (p.start && !pursuitCued) {
          pursuitCued = true;
          beep(660, 0.12);
          speakPrompt('Now follow the moving dot.');
        }
        $('#calib-msg').textContent = 'Follow the dot with your eyes. Head still.';
        // At the top, above the path (the bounds start at 10% or lower). At 86% the words sat on
        // the bottom of the loop, the dot passed under them and he lost it.
        $('#calib-msg').style.left = '50%';
        $('#calib-msg').style.top = '3%';
        $('#calib-bar').style.width = `${Math.round(p.progress * 100)}%`;
        $('#calib-bar').hidden = false;
        return;
      }
      d.classList.remove('pursuit');
      $('#calib-bar').hidden = true;

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

  // Restore the terrain map from last time — he should never re-teach the app his own eyes just
  // because he reopened it. If it loads, he can select immediately; if not, he calibrates once.
  const saved = loadGazeMap();
  if (saved && g.import(saved)) {
    $('#gaze-state').textContent = `Camera on (${g.backend}) — remembered your eyes (${g.terrainSize} samples). Recenter if the dot is off.`;
    toast('Welcome back — your eye calibration was remembered.', 'ok');
  } else if (saved && g.importProblem === 'camera' && state.driver !== 'blink') {
    // Not loaded, and not deleted: until he calibrates this camera, plugging the old one back in
    // brings its map back.
    $('#gaze-state').textContent = `Camera on (${g.backend}). Different camera from last time: not calibrated yet.`;
    toast('This is a different camera from last time. Calibrate once for it.', 'info');
  } else if (state.driver === 'blink') {
    $('#gaze-state').textContent = `Camera on (${g.backend}) — blink to select.`;
    toast('Camera on. Blink to move. Close your eyes until the beep to pick.', 'info');
  } else {
    $('#gaze-state').textContent = `Camera on (${g.backend}) — not calibrated yet.`;
    toast('Camera on. Calibrate, then look at a word and close your eyes until the beep to say it.', 'info');
  }
}

/* ---------- the terrain map, persisted ---------- */

const GAZE_KEY = 'stillme.gaze.v2';
let gazeSaveTimer = null;
function scheduleGazeSave() {
  clearTimeout(gazeSaveTimer);
  gazeSaveTimer = setTimeout(() => {
    try {
      const m = gaze?.export();
      if (m) localStorage.setItem(GAZE_KEY, JSON.stringify(m));
    } catch { /* storage full or blocked — the in-memory map still works this session */ }
  }, 4000);
}
// Flush any pending gaze save when the page is hidden or closed — the 4s debounce would otherwise
// lose his most recent learning every time he closes the app.
window.addEventListener('pagehide', flushGazeSave);
document.addEventListener('visibilitychange', () => { if (document.hidden) flushGazeSave(); });
function flushGazeSave() {
  clearTimeout(gazeSaveTimer);
  try { const m = gaze?.export(); if (m) localStorage.setItem(GAZE_KEY, JSON.stringify(m)); } catch {}
}
function loadGazeMap() {
  try { const s = localStorage.getItem(GAZE_KEY); return s ? JSON.parse(s) : null; } catch { return null; }
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
  if (gaze.calibrating) return toast('Calibration is running. Press Escape to stop it.');
  // Never on top of the accuracy test or a recenter: two flows would share the one overlay.
  if (gazeTesting || !$('#calib').hidden) return;
  // This camera, held for the whole check: if it stops, the check stops, instead of reading a
  // module-level gaze that is now null and leaving the overlay up until a reload.
  const g = gaze;
  const live = overlayGuard(g);
  try {
    await runSignalCheck(g, live);
  } catch (e) {
    if (String(e.message) !== 'cancelled') toast(`Signal check stopped: ${e.message}`);
  } finally {
    endCalibrationUI();
  }
}

async function runSignalCheck(g, live) {
  // New, open-eye frames only: raw() is null with no face or mid-blink, and seq skips a frame
  // already counted (the poll is faster than the camera).
  const grab = async (ms) => {
    const out = [];
    let seen = -1;
    const until = performance.now() + ms;
    while (performance.now() < until) {
      live();
      const s = g.raw();
      if (s && s.seq !== seen) { seen = s.seq; out.push(s.v); }
      await new Promise((r) => setTimeout(r, 40));
    }
    return out;
  };
  // k: 0 = left/right (iris x), 1 = up/down (iris y).
  const mean = (a, k) => a.reduce((s, v) => s + v[k], 0) / (a.length || 1);
  const sd = (a, k) => {
    const m = mean(a, k);
    return Math.sqrt(a.reduce((s, v) => s + (v[k] - m) ** 2, 0) / (a.length || 1));
  };

  $('#calib').hidden = false;
  $('#calib-dot').style.display = '';

  // You cannot read an instruction in the middle of the screen while looking at the far edge of
  // it. That is the whole point of the test. So the app SAYS it out loud, and the target moves
  // to where he should be looking — he never has to look away to find out what to do next.
  const step = async (msg, x, y, settleMs, sampleMs) => {
    live();
    await prompt(msg, x, y);
    await new Promise((r) => setTimeout(r, settleMs));
    live();
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
  // Up and down too. That is the axis that fails (the calibration verdict says so most often),
  // and a check that only looked left and right said "good" while it was failing.
  const top    = await step('Now the dot at the top.',                   0.5, 0.05, 1200, 1300);
  // 0.88, not the very edge: looking far down droops the lids, those frames read as blinks and are
  // dropped, and a good camera came out as "could not see your face". Calibration stops at 0.86.
  const bottom = await step('Now the dot at the bottom.',                0.5, 0.88, 1200, 1300);

  $('#calib').hidden = true;

  if (still.length < 10 || left.length < 6 || right.length < 6 || top.length < 6 || bottom.length < 6) {
    return toast('Could not see your face well enough. More light, and sit closer.');
  }

  // The same iris reading the old check used; a ratio with no units, so it stands in fine for the
  // model's own features. Each axis judged on its own, and the verdict is the weaker one.
  const noiseX = sd(still, 0), noiseY = sd(still, 1);
  const travelX = Math.abs(mean(right, 0) - mean(left, 0));
  const travelY = Math.abs(mean(bottom, 1) - mean(top, 1));
  const snrX = travelX / (noiseX || 1e-6), snrY = travelY / (noiseY || 1e-6);
  const snr = Math.min(snrX, snrY);
  const verdict = snr > 8 ? 'good' : snr > 4 ? 'usable' : 'too noisy';
  const p = g.probe();
  const cam = p.camera ? `${p.camera.w}x${p.camera.h}` : '?';

  $('#gaze-state').textContent =
    `${snrX.toFixed(1)}x left/right, ${snrY.toFixed(1)}x up/down (${verdict})`
    + ` · noise ${noiseX.toFixed(4)}/${noiseY.toFixed(4)} · ${cam}, iris ${p.irisPx}px`;

  const verdictMsg = p.irisPx && p.irisPx < 12
    ? `Your iris is only ${p.irisPx} pixels wide. Sit closer to the screen.`
    : snr > 4
      ? `Signal is ${verdict}. Now calibrate.`
      : `${snrY < snrX ? 'Up and down' : 'Left and right'} is too noisy. Sit closer, put more light on your face, and raise the camera to eye level.`;
  toast(`${snr.toFixed(1)}x: ${verdictMsg}`, snr > 4 && !(p.irisPx && p.irisPx < 12) ? 'ok' : 'error');
  speakPrompt(verdictMsg);   // he does not have to read it either

  fetch('/api/gazelog', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ kind: 'signal', noise: +noiseX.toFixed(5), travel: +travelX.toFixed(4),
      snr: +snr.toFixed(2), verdict, probe: p,
      x: { noise: +noiseX.toFixed(5), travel: +travelX.toFixed(4), snr: +snrX.toFixed(2) },
      y: { noise: +noiseY.toFixed(5), travel: +travelY.toFixed(4), snr: +snrY.toFixed(2) },
      samples: { still: still.length, left: left.length, right: right.length, top: top.length, bottom: bottom.length } }),
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

/**
 * Instructions, spoken. This is an app that talks; it should talk to HIM too.
 *
 * The app's own voice, never his. Calibration verdicts, test results and "look at the dot" used to
 * go through say(): spoken in his cloned voice, shown in the banner as his words, logged as his
 * utterances (a third of sessions.jsonl, at 750-2,700 wpm), and they shut the sentence sheet under
 * him. Here: the placeholder voice, no banner, no log, the sheet left alone, and state.speaking
 * untouched, so his gaze and blinks stay live while it talks. It never talks over his own speech,
 * and say() cuts it off. Resolves when it has finished (or could not play).
 */
let promptPlayer = null, promptGen = 0;
async function speakPrompt(text) {
  if (state.speaking) return;
  const gen = ++promptGen;
  stopPrompt(false);
  // The mic must not hear the app either. Same hand-off as say().
  if (state.listening) { resumeListening = true; stopListening(); }
  let url = null;
  const finish = () => {
    if (url) { URL.revokeObjectURL(url); url = null; }
    if (gen !== promptGen) return;
    promptPlayer = null;
    if (!state.speaking && resumeListening) { resumeListening = false; startListening(); }
  };
  try {
    const res = await fetch('/api/speak', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text, voice: 'placeholder' }),
    });
    if (!res.ok || gen !== promptGen || state.speaking) return finish();
    const blob = await res.blob();
    if (gen !== promptGen || state.speaking) return finish();
    url = URL.createObjectURL(blob);
    const p = (promptPlayer = new Audio(url));
    await new Promise((resolve) => {
      p.onended = p.onerror = resolve;
      p._stop = resolve;               // stopPrompt() settles it too
      p.play().catch(resolve);
    });
  } catch {}
  finish();
}
/** Silence an app instruction. bump: also void one still waiting for its audio. */
function stopPrompt(bump = true) {
  if (bump) promptGen++;
  const p = promptPlayer;
  promptPlayer = null;
  if (p) { try { p.pause(); } catch {} p._stop?.(); }
}

/** A tone when sampling starts, a lower one when it ends. He needs to know when to hold still. */
let actx = null;
function beep(hz = 880, len = 0.12, gain = 0.15) {
  try {
    actx = actx ?? new AudioContext();
    const o = actx.createOscillator(), g = actx.createGain();
    o.frequency.value = hz;
    o.type = 'sine';
    g.gain.setValueAtTime(0.0001, actx.currentTime);
    g.gain.exponentialRampToValueAtTime(gain, actx.currentTime + 0.01);
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
  if (gaze.calibrating || gazeTesting) return;
  const g = gaze;
  const live = overlayGuard(g);
  try {
    gazeTesting = true;
    await runGazeAccuracy(g, live);
  } catch (e) {
    const stopped = String(e.message) === 'cancelled';
    toast(stopped ? 'Test stopped.' : `Test failed: ${e.message}`, stopped ? 'info' : 'error');
  } finally {
    // No throw may leave selection switched off, or the overlay up.
    gazeTesting = false;
    endCalibrationUI();
    $('#test-chip').hidden = true;
    $$('.tile').forEach((t) => t.classList.remove('target'));
  }
}

async function runGazeAccuracy(g, live) {
  // The settings drawer sits over the right-hand tiles. Testing with it open asked him to look at
  // words he could not see — which is exactly what happened, and it made the numbers meaningless.
  clearScreen();
  // No gold cursor or armed tile left over from before: the target is the only thing lit.
  resetDwell();
  state.cursor = -1;
  renderGrid();
  await new Promise((r) => setTimeout(r, 400));

  speakPrompt('Look at each highlighted word.');
  await new Promise((r) => setTimeout(r, 1400));

  // He sees the live grid the whole time, so the progress goes in a chip over it. Inside #calib
  // (hidden for the test) it was never seen.
  const chip = $('#test-chip');
  // The toast sits in the same top strip ("Calibrated. Now run the test." is often still up).
  clearTimeout(toastTimer);
  $('#toast').hidden = true;
  chip.hidden = false;
  const n = $$('.tile').length;
  const results = [];

  for (let i = 0; i < n; i++) {
    live();
    // Look the tiles up fresh every time. A re-render (new suggestions, a partner's words) swaps
    // the buttons, and a stale one measures as a zero box at (0,0).
    const cur = $$('.tile');
    if (!cur[i]) break;
    cur.forEach((t, j) => t.classList.toggle('target', i === j));
    chip.textContent = `Look at the blue word · ${i + 1} of ${n} · Escape to stop`;
    await new Promise((r) => setTimeout(r, 1300));   // settle
    live();

    // Twelve readings, each from a new frame: sample() is null with no face, and seq skips a
    // frame already counted.
    const pts = [];
    let seen = -1;
    for (let k = 0; k < 12; k++) {
      const s = g.sample();
      if (s && s.seq !== seen) { seen = s.seq; pts.push(s); }
      await new Promise((r) => setTimeout(r, 45));
    }
    if (!pts.length) { results.push({ tile: i, hit: false, reason: 'no face' }); continue; }

    const mx = pts.reduce((a, p) => a + p.x, 0) / pts.length;
    const my = pts.reduce((a, p) => a + p.y, 0) / pts.length;
    const el = $$('.tile')[i];
    const r = el?.getBoundingClientRect();
    if (!r?.width) { results.push({ tile: i, hit: false, reason: 'tile gone' }); continue; }
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;

    results.push({
      tile: i,
      label: state.tiles[i],
      hit: tileAt(mx, my) === i,
      landedOn: tileAt(mx, my),
      offsetPx: Math.round(Math.hypot(mx - cx, my - cy)),
      mx: Math.round(mx), my: Math.round(my),
      cx: Math.round(cx), cy: Math.round(cy),
      raw: pts[pts.length - 1].raw,
    });
  }

  // THE PROOF: is the miss a constant bias? Subtract the mean residual and re-score. If this
  // number is high while the raw score is low, the tracker is fine and only the anchor is off —
  // which the per-selection nudge now fixes during real use.
  const withPos = results.filter((r) => r.mx !== undefined);
  const bdx = withPos.reduce((s2, r) => s2 + (r.cx - r.mx), 0) / (withPos.length || 1);
  const bdy = withPos.reduce((s2, r) => s2 + (r.cy - r.my), 0) / (withPos.length || 1);
  const hitsDebiased = withPos.filter((r) => tileAt(r.mx + bdx, r.my + bdy) === r.tile).length;

  $$('.tile').forEach((t) => t.classList.remove('target'));
  chip.hidden = true;

  const hits = results.filter((r) => r.hit).length;
  const pct = Math.round((hits / results.length) * 100);
  // Plain words out loud. The raw and bias-corrected numbers stay in #gaze-state and the log.
  const msg = `You hit ${hits} of ${results.length} words.`;
  toast(msg, 'info');
  speakPrompt(msg);
  $('#gaze-state').textContent =
    `Accuracy ${hits}/${results.length} raw · ${hitsDebiased}/${withPos.length} debiased · bias (${Math.round(bdx)}, ${Math.round(bdy)})px`;

  // Send it to the server so it can actually be read, instead of living in a toast.
  fetch('/api/gazelog', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      kind: 'accuracy',
      hits, total: results.length, pct,
      hitsDebiased, bias: { x: Math.round(bdx), y: Math.round(bdy) },
      screen: { w: window.innerWidth, h: window.innerHeight },
      probe: g.probe(),
      results,
    }),
  }).catch(() => {});
}

function stopGaze() {
  // A run in progress stops with the camera, and its overlay comes down with it: a dead camera
  // under a black screen with a dot still moving is the one place he cannot get out of.
  gaze?.cancelCalibration();
  gaze?.stop();
  gaze = null;
  endCalibrationUI();
  resetDwell();
  $('#gaze-dot').hidden = true;
  $('#gaze-row').hidden = true;
}

// Eye tracking could not go on (camera blocked, no camera, no face model). Do not leave the
// setting saying Eyes over a dead camera: drop back to touch so the screen tells the truth, and
// choosing Eyes again in Settings is a real change that restarts it.
function gazeStopped() {
  stopGaze();
  $('#driver').value = 'touch';
  $('#driver').onchange({ target: $('#driver') });
}

/* ---------- feedback ---------- */

let toastTimer = null;
// kind: 'error' (red, the default: never fail quietly), 'ok' (green, it worked), 'info' (neutral).
// A success in red taught the room that red means nothing, so the real failures stopped standing out.
function toast(msg, kind = 'error') {
  const el = $('#toast');
  el.textContent = msg;
  el.className = `toast ${kind}`;
  el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
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
  gridGen++;   // every tile element is new: a blink snapshot from before this is void
  const aiming = state.driver === 'scan' || state.driver === 'gaze' || state.driver === 'blink';

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

  // The cursor can rest on the action buttons too, or a switch user could never speak.
  const z = zones()[state.cursor];
  for (const [zone, sel] of Object.entries(ZONE_EL)) $(sel)?.classList.toggle('cursor', aiming && z === zone);
}

function renderSelected() {
  $('#composing').hidden = state.selected.length === 0;
  $('#composing-words').textContent = state.selected.join(' ');
  renderControls();
}

/**
 * The big Undo and Mode buttons, shown for every aiming input (eyes, blinks, switch). They stay in
 * place when they cannot act, just greyed, so the buttons never move under his eye.
 */
function renderControls() {
  const aiming = state.driver === 'scan' || state.driver === 'gaze' || state.driver === 'blink';
  for (const b of $$('.eyes-extra')) b.hidden = !aiming;
  // Every change to these flags re-renders here (say() start and end, compose, urgent), so Say it
  // can never stay greyed after the reason has gone. It used to stick after an Undo mid-speech.
  $('#compose').disabled = !state.selected.length || state.speaking || state.composing || state.urgent;
  $('#undo-eye').disabled = state.urgent || !state.selected.length;
  $('#mode-eye').disabled = state.urgent || state.selected.length > 0;
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
$('#sheet-urgent').onclick = sheetUrgent;
$('#listen').onclick = () => (state.listening ? stopListening() : startListening());
$('#hold').onclick = sayWait;
$('#undo-eye').onclick = undo;
$('#mode-eye').onclick = cycleMode;
$('#confirm').onclick = (e) => { if (e.target.id === 'confirm') closeConfirm(); };
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !$('#confirm').hidden) closeConfirm();
});
$('#partner-said').onchange = () => { if (!state.selected.length) loadTiles(); };

$('#gear').onclick = () => {
  const open = $('#settings').hidden;
  $('#settings').hidden = !open;
  document.body.classList.toggle('settings-open', open);   // the grid makes room; nothing hides
  $('#gear').setAttribute('aria-expanded', String(open));
};
$('#close-settings').onclick = () => {
  $('#settings').hidden = true;
  document.body.classList.remove('settings-open');
  $('#gear').setAttribute('aria-expanded', 'false');
};
$('#literal').onchange = (e) => { state.literal = e.target.checked; };
$('#predictive').onchange = (e) => { state.predictive = e.target.checked; loadTiles(); };
$('#driver').onchange = (e) => {
  state.driver = e.target.value;
  const cam = state.driver === 'gaze' || state.driver === 'blink';   // both need the camera
  $('#scan-help').hidden = state.driver !== 'scan';
  $('#blink-help').hidden = state.driver !== 'blink';
  syncSettingsRows();
  // Full-bleed for both camera drivers: the vertical half-tile is the error budget, and at the
  // default layout it is 144px (~1.6°) — too tight. Shrinking the chrome raises it ~44%.
  document.body.classList.toggle('gaze-mode', cam);
  if (cam) startGaze(); else stopGaze();
  // Blinks start with nothing highlighted: his first step lands on Urgent, then the tiles. Starting
  // on tile 0 ("yes") let one eye-rest right after the switch say "Yes." for him.
  if (state.driver === 'blink') { state.cursor = -1; lastBlinkActAt = performance.now(); }
  renderControls();
  renderGrid();
};

/** Show only the settings that do something for the current input. */
function syncSettingsRows() {
  const eyes = state.driver === 'gaze';
  // Look-then-blink: the plain cursor turns into a thin blue edge, so gold means only "armed, a
  // blink says it now" (style.css). Other drivers keep the gold cursor: there it IS the selection.
  document.body.classList.toggle('look-blink', eyes && state.confirmBy === 'blink');
  renderEyeStatus();
  $('#confirm-row').hidden = !eyes;   // look-then-blink vs dwell is an Eyes-mode choice
  // The dwell slider only drives Eyes mode with Confirm by Dwell. Shown anywhere else it promised a
  // hold that never happened.
  $('#dwell-row').hidden = !(eyes && state.confirmBy === 'dwell');
  $('#blink-row').hidden = !(eyes || state.driver === 'blink');
  showBlinkMs();   // the label says more in Blinks mode (the step length too)
}

// Confirm by: look then blink, or look and hold. Switching clears anything half-armed or half-filled,
// so changing the setting can never fire a word by itself.
$('#confirm-by').value = state.confirmBy;
$('#confirm-by').onchange = (e) => {
  state.confirmBy = e.target.value === 'dwell' ? 'dwell' : 'blink';
  try { localStorage.setItem(CONFIRM_KEY, state.confirmBy); } catch { /* still set this session */ }
  resetDwell();
  dwellLatch = -1; ctrlLatch = null;
  sheetDwellIdx = -1; sheetDwellStart = 0;
  sheetOptions().forEach((o) => o.style.setProperty('--dwell', '0'));
  renderSheetHint();
  syncSettingsRows();
};
syncSettingsRows();
$('#gear-float').onclick = () => $('#gear').onclick();
// Never let him drive with the camera while a panel is over a third of the tiles.
bus.on(() => { if (state.driver === 'gaze' || state.driver === 'blink') clearScreen(); });
/** Get the settings panel off the screen. It covers the right column of tiles. */
function clearScreen() {
  $('#settings').hidden = true;
  document.body.classList.remove('settings-open');
  $('#gear').setAttribute('aria-expanded', 'false');
}

$('#calibrate').onclick = async () => {
  if (!gaze?.running) return toast('Turn the camera on first.');
  if (gaze.calibrating) return toast('Calibration is already running. Press Escape to stop it.');
  if (gazeTesting) return;
  clearScreen();
  await new Promise((r) => setTimeout(r, 350));   // let the panel finish getting out of the way

  // Calibrate across the area he will actually USE — the tiles — not the corners of the glass,
  // where the eyelid swallows the iris and the readings are lies.
  // The action row (Say it, Urgent, Undo...) is a gaze target too, so the bottom dots reach into
  // its upper half. Not to its bottom edge: that is the glass corner the comment above warns about.
  const tiles = $$('.tile').map((el) => el.getBoundingClientRect());
  const bar = $('.actions').getBoundingClientRect();
  const bottom = Math.max(Math.max(...tiles.map((r) => r.bottom)) - 30, bar.height ? bar.top + bar.height * 0.4 : 0);
  const bounds = tiles.length
    ? {
        x0: Math.max(0.06, (Math.min(...tiles.map((r) => r.left)) + 40) / window.innerWidth),
        x1: Math.min(0.94, (Math.max(...tiles.map((r) => r.right)) - 40) / window.innerWidth),
        y0: Math.max(0.10, (Math.min(...tiles.map((r) => r.top)) + 30) / window.innerHeight),
        y1: Math.min(0.93, bottom / window.innerHeight),
      }
    : undefined;

  pursuitCued = false;
  speakPrompt('Look at each dot, then follow the moving one with your eyes.');
  try {
    // The bar shows the progress. A frame count ("412 good frames") meant nothing to him or a viewer.
    await gaze.calibrate({ bounds, onSample: () => { $('#calib-count').textContent = 'Keep following the dot'; } });
  } catch (e) {
    // A throw used to leave him stranded on a full-screen black overlay with no way out.
    // Too few good frames lands here too, with the camera still on, so he can press Calibrate
    // again straight away. It is said aloud, as a good result is.
    if (String(e.message) !== 'cancelled') {
      toast(`Calibration failed: ${e.message}`);
      speakPrompt(`Calibration failed: ${e.message}`);
    }
  } finally {
    endCalibrationUI();
  }
};

/** Always get him off the calibration screen. No failure mode leaves him staring at black. */
function endCalibrationUI() {
  $('#calib').hidden = true;
  $('#calib-bar').hidden = true;
  $('#calib-dot').classList.remove('pursuit', 'sampling');
  $('#calib-dot').style.display = '';
  $('#calib-count').textContent = '';
}

// Escape always aborts. A full-screen overlay with no exit is unacceptable anywhere, and this is
// a device for someone who cannot ask for help.
// Every overlay flow, not just calibration: the signal check, recenter and accuracy test stop at
// their next step, and an overlay left behind by anything else is taken down.
window.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  overlayGen++;
  if (gaze?.calibrating) { gaze.cancelCalibration(); toast('Calibration stopped.', 'info'); }
  else if (!$('#calib').hidden) endCalibrationUI();
});
$('#test-gaze').onclick = testGazeAccuracy;
$('#signal-check').onclick = signalCheck;
$('#recenter').onclick = async () => {
  if (!gaze?.calibrated) return toast('Calibrate first.');
  if (gaze.calibrating || gazeTesting || !$('#calib').hidden) return;   // not over the signal check
  const g = gaze;
  const live = overlayGuard(g);
  clearScreen();
  try {
    await new Promise((r) => setTimeout(r, 350));
    live();
    $('#calib').hidden = false;
    $('#calib-dot').style.display = '';
    $('#calib-dot').style.left = '50%';
    $('#calib-dot').style.top = '50%';
    $('#calib-dot').classList.add('sampling');
    $('#calib-msg').textContent = 'Look at the dot.';
    $('#calib-msg').style.left = '50%';
    $('#calib-msg').style.top = '64%';
    speakPrompt('Look at the dot.');
    await new Promise((r) => setTimeout(r, 1100));
    live();
    const r = await g.recenter(0.5, 0.5);
    live();
    if (r) scheduleGazeSave();   // the new offset is worth keeping now, not at the next pick
    toast(r ? `Recentered (${r.dx > 0 ? '+' : ''}${r.dx}, ${r.dy > 0 ? '+' : ''}${r.dy} px).`
            : 'Could not see your eyes well enough. The old setting is kept.', r ? 'ok' : 'error');
  } catch (e) {
    if (String(e.message) !== 'cancelled') toast(`Recenter stopped: ${e.message}`);
  } finally {
    endCalibrationUI();
  }
};
function showDwellMs() {
  $('#dwell').value = String(state.dwellMs);
  $('#dwell-label').textContent = `Hold a tile or button for ${(state.dwellMs / 1000).toFixed(1)}s to pick it.`;
}
$('#dwell').oninput = (e) => {
  state.dwellMs = +e.target.value;
  try { localStorage.setItem(DWELL_MS_KEY, String(state.dwellMs)); } catch { /* still set this session */ }
  showDwellMs();
};
showDwellMs();
// Blink length: how long his eyes stay shut to say a word. Remembered on this device, because the
// right value is a fact about him (how fast he can close and hold), not about one session.
// In Blinks mode the same slider also sets how long a blink must be to move (blinkscan.js), and
// the caregiver's only fix for natural blinks moving the highlight is to slide it up, so say so.
function showBlinkMs() {
  $('#blink-ms').value = String(state.blinkMs);
  const say = `Close your eyes for ${(state.blinkMs / 1000).toFixed(2)}s (until the beep) to say a word.`;
  $('#blink-label').textContent = state.driver !== 'blink' ? say
    : `${say} A blink of ${(stepMinMs(state.blinkMs) / 1000).toFixed(2)}s or more moves. ` +
      'If the highlight moves by itself, slide this up (0.45s stops ordinary blinks).';
}
$('#blink-ms').oninput = (e) => {
  state.blinkMs = +e.target.value;
  if (gaze) gaze.confirmMs = state.blinkMs;
  try { localStorage.setItem(BLINK_MS_KEY, String(state.blinkMs)); } catch { /* still set this session */ }
  showBlinkMs();
};
showBlinkMs();

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

/* ---------- blink telemetry ---------- */

// "Blink to confirm doesn't work" left no trace: nothing on disk said whether his blinks were
// seen, too short, or dropped by a guard (cooldown, no armed tile, speaking). onBlink calls this
// once per blink with the branch that ran, and the row lands in data/gaze.jsonl next to the
// calibration rows. `extra` carries anything the caller has, such as the lid peak.
function logBlink(blinkKind, held, outcome, extra = {}) {
  fetch('/api/gazelog', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      kind: 'blink', blink: blinkKind, held: Math.round(held), outcome,
      driver: state.driver, confirmBy: state.confirmBy, armedTile, armedControl,
      speaking: state.speaking, sheetOpen: !$('#confirm').hidden,
      sinceCommit: Math.round(performance.now() - lastCommitAt), ...extra,
    }),
  }).catch(() => {});
}
