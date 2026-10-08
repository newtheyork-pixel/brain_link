// The blink gate. Pure logic, no DOM, so the browser and the offline replay
// (eval/gaze/blink.mjs) run exactly the same code over the same lid numbers.
//
// WHY THIS EXISTS. A blink is not a moment, it is ~100-300 ms of lids closing and reopening, and
// every one of those frames moves the iris landmarks and the lid-shape features. The old tracker
// only stopped reading gaze once a lid passed 0.6, so the closing frames dragged the dot down a
// row, broke the fixation lock and un-armed the tile he was looking at, just before the blink that
// was meant to say it landed. Measured on the 2026-10-08 recording: 0 of the armed tiles survived
// their own blink.
//
// So: notice the lids the instant they START to move (a fast rise is a lid, a slow one is a glance
// down), hold the gaze still from then until they are back where they were, and decide shut/open
// on BOTH eyes together, measured against HIS open-eye level, not a fixed number.
//
// step(t, L, R) is fed every camera frame with the two eyeBlink scores and returns:
//   gated    - true: this frame's gaze must not be used for anything
//   entered  - the gate just closed (the lids started moving). Snapshot what was armed NOW.
//   released - the gate just opened again. Restore the pre-blink smoothing.
//   events   - blink events: { type: 'held', held } once when a closure passes confirmMs while the
//              eyes are still shut; { type: 'blink', kind, held } when the eyes reopen.
//              kind: 'reflex' (< shortMin), 'short', 'long' (>= confirmMs), 'rest' (> restMs).
//              Contiguous, so every closure gets a kind: no length falls into a silent gap.

export const BLINK_DEFAULTS = {
  riseWin: 50,      // ms. The rise is measured against the lowest lid in this window...
  rise: 0.12,       // ...and a jump this big that fast is a lid closing, not the eye moving down.
  level: 0.20,      // Or both lids sit this far above his open level: freeze for a slow close too.
  release: 0.08,    // Let go once the lids are back within this of where they were before...
  settle: 100,      // ...and have stayed there this long (the reopen tail still moves the iris).
  shutOn: 0.25,     // Shut: both-eye mean this far above his open level.
  shutOff: 0.20,    // Open again below this (hysteresis: a dip mid-blink must not split it).
  shutRiseMs: 250,  // Shut must follow a FAST rise this recently. Lids creeping up as he leans in
                    //   or looks down are never a blink, however high they get.
  baseWin: 1500,    // ms of recent open-eye frames his open level is taken from...
  basePct: 0.2,     // ...as a low percentile, so it drops at once when he reopens, rises slowly.
  steadyMs: 250,    // A lid that stops moving at a new height for this long is a new posture
  steadyRange: 0.04, //   (looking down, slumping), not a blink. Let go and adopt it.
  shortMin: 120,    // Under this a closure is a reflex: seen, never acted on.
  confirmMs: 350,   // A closure this long or longer is a deliberate "say it".
  restMs: 3000,     // Longer than this he is resting his eyes, not asking for anything.
};

export function makeBlinkGate(opts = {}) {
  const o = { ...BLINK_DEFAULTS, ...opts };
  const recent = [];      // [t, max lid, mean lid] for the rise test, every frame
  const openBuf = [];     // [t, mean lid] of ungated frames, for his open level
  let gated = false, gateAt = 0, preLid = 0, base = null;
  let shut = false, shutAt = 0, heldFired = false, shutBase = 0;
  let backSince = -1;     // when the lids got back near the pre-blink level (for settle)
  let lastRiseAt = -Infinity;

  const openLevel = () => {
    if (!openBuf.length) return null;
    const s = openBuf.map((p) => p[1]).sort((a, b) => a - b);
    return s[Math.floor(o.basePct * (s.length - 1))];
  };
  // Has the lid held one height for steadyMs? (range of max lid over that window)
  const steady = (t) => {
    let lo = Infinity, hi = -Infinity, n = 0;
    for (let i = recent.length - 1; i >= 0 && recent[i][0] >= t - o.steadyMs; i--) {
      lo = Math.min(lo, recent[i][1]); hi = Math.max(hi, recent[i][1]); n++;
    }
    return n >= 3 && t - gateAt >= o.steadyMs && hi - lo < o.steadyRange;
  };
  const kindOf = (held) => held > o.restMs ? 'rest' : held >= o.confirmMs ? 'long'
    : held >= o.shortMin ? 'short' : 'reflex';

  function step(t, L, R) {
    const m = Math.max(L, R), c = (L + R) / 2;
    const out = { gated: false, entered: false, released: false, events: [] };

    // Lowest lid in the rise window, BEFORE this frame.
    while (recent.length && recent[0][0] < t - Math.max(o.riseWin, o.steadyMs)) recent.shift();
    let pre = null;
    for (let i = recent.length - 1; i >= 0 && recent[i][0] >= t - o.riseWin; i--) if (!pre || recent[i][1] < pre[1]) pre = recent[i];
    if (!pre && recent.length) pre = recent.at(-1);   // slow camera: compare with the last frame
    recent.push([t, m, c]);

    const b = gated ? base : openLevel() ?? c;
    const rising = !!pre && m - pre[1] > o.rise;
    const high = c - b > o.level;

    if (!gated && (rising || high)) {
      gated = true; gateAt = t; base = b; backSince = -1;
      preLid = pre ? Math.min(pre[1], m) : m;
      out.entered = true;
    }
    // Shut is judged against where the lids were just BEFORE this closure started, or his open
    // level if that is higher. Looking at the bottom row lifts the lid score; measured against an
    // older, lower level, an ordinary blink there read as a 400-500 ms "deliberate" one.
    if (rising) {
      if (!shut && t - lastRiseAt > o.riseWin) {
        let before = Infinity;   // lowest both-eye lid in the 150 ms before this rise began
        for (const r of recent) if (r[0] >= t - 150 && r[0] < t) before = Math.min(before, r[2]);
        shutBase = Math.max(base, before === Infinity ? base : before);
      }
      lastRiseAt = t;
    }

    if (gated) {
      // Shut / open on the mean of both eyes against his own open level. One weak or drooping eye
      // can no longer veto a blink the other eye made plainly.
      if (!shut && c - shutBase > o.shutOn && t - lastRiseAt <= o.shutRiseMs) { shut = true; shutAt = t; heldFired = false; }
      else if (shut && c - shutBase < o.shutOff) {
        shut = false;
        const held = t - shutAt;
        out.events.push({ type: 'blink', kind: kindOf(held), held });
      }
      if (shut && !heldFired && t - shutAt >= o.confirmMs && t - shutAt <= o.restMs) {
        heldFired = true;
        out.events.push({ type: 'held', held: t - shutAt });
      }

      if (!shut) {
        if (m <= preLid + o.release) { if (backSince < 0) backSince = t; }
        else backSince = -1;
        const settled = backSince >= 0 && t - backSince >= o.settle;
        // The lids stopped at a new height and stayed there: a posture change, not a blink.
        // Take it as his new open level, or the gaze would stay frozen for as long as he looks down.
        const moved = steady(t);
        if (settled || moved) {
          gated = false; out.released = true;
          // Back, but not down to the open level we had: his resting lid has moved. Start his open
          // level again from here, or the level test would re-trip on the very next frame.
          if (moved || c - base > o.level / 2) openBuf.length = 0;
        }
      }
    }

    if (!gated && !out.released) {
      openBuf.push([t, c]);
      while (openBuf.length && openBuf[0][0] < t - o.baseWin) openBuf.shift();
    }
    out.gated = gated || out.released;   // the release frame itself still carries the reopen tail
    return out;
  }

  return {
    step,
    get gated() { return gated; },
    get shut() { return shut; },
    /** When the current shut began, on the frame clock step() was given. */
    get shutAt() { return shutAt; },
    get confirmMs() { return o.confirmMs; },
    set confirmMs(ms) { o.confirmMs = ms; },
    /** Face lost: forget the blink in progress and his open level; both are stale now. */
    reset() {
      recent.length = 0; openBuf.length = 0;
      gated = false; shut = false; heldFired = false; backSince = -1; base = null;
      lastRiseAt = -Infinity;
    },
  };
}
