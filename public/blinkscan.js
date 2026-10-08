// The Blinks driver's rules. Pure, no DOM, so node can check them (eval/blinkscan-smoke.mjs).
//
// In the Blinks driver the eyelid is the only switch: every blink he makes is a command. So a
// natural blink must do NOTHING. The blink gate (blinkgate.js) already sorts each closure against
// his own Blink length setting (state.blinkMs, the per-user "say it" threshold): 'long' is at least
// that long, 'short' is anything from 120 ms up to it. 120 ms is still inside the natural range
// (83-292 ms on the 2026-10-08 recording), so here a short blink only steps when it is clearly
// longer than an ordinary one. The floor sits 150 ms under his Blink length with no upper cap, so
// one slider moves both. At the default 350 ms the floor is 200 ms, which still lets about 1 in 5
// natural blinks step (the long ones measure 250-292 ms). A caregiver who sees the highlight move
// by itself slides Blink length up: at 450 ms the floor is 300 ms, above every natural blink on
// the recording. Shortening it still leaves room between "step" and "say".

/** Shortest closure that steps the cursor, for his Blink length. Always >= 130 ms below it. */
export const stepMinMs = (blinkMs) => Math.max(120, blinkMs - 150);

/**
 * What a blink does in the Blinks driver: 'step', 'select', or null (a natural blink, or his eyes
 * resting for longer than 3 s). kind and held come straight from blinkgate.js.
 */
export function blinkAction(kind, held, blinkMs) {
  if (kind === 'long') return 'select';
  if (kind === 'short' && held >= stepMinMs(blinkMs)) return 'step';
  return null;
}

/**
 * The next zone a step lands on. cursor -1 means nothing is highlighted, which is where every new
 * grid starts, so a lone long blink selects nothing. first is the zone the scan visits before the
 * tiles (Urgent, so an emergency is one step away, not nine), or -1 for the plain order.
 */
export function nextZone(cursor, n, first = -1) {
  if (n <= 0) return -1;
  const order = first >= 0 && first < n
    ? [first, ...Array.from({ length: n }, (_, i) => i).filter((i) => i !== first)]
    : Array.from({ length: n }, (_, i) => i);
  return order[(order.indexOf(cursor) + 1) % n];   // indexOf -1 (nothing yet) lands on order[0]
}
