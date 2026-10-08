// Guards the Blinks driver's rules (public/blinkscan.js). Run: node eval/blinkscan-smoke.mjs
//   1. no natural blink (83-292 ms shut on the 2026-10-08 recording) ever selects, at any Blink
//      length above that range; an ordinary one (under 200 ms) never even steps at the default
//      350 ms or longer; and from 450 ms up no natural blink steps at all (the caregiver's fix
//      when the long natural blinks move the highlight),
//   2. there is always room to step: the step floor sits at least 100 ms under the Blink length,
//   3. a fresh grid has nothing highlighted, and the first step lands on Urgent,
//   4. a step visits every zone once per lap, in the urgent grid in plain order.
import { blinkAction, stepMinMs, nextZone } from '../public/blinkscan.js';
import { BLINK_DEFAULTS } from '../public/blinkgate.js';

let fails = 0;
const check = (ok, msg) => { if (!ok) { fails++; console.log(`FAIL ${msg}`); } };
// What blinkgate.js calls a closure of this length, for this Blink length (its own thresholds,
// so this cannot drift from the gate).
const { shortMin, restMs } = BLINK_DEFAULTS;
const kindOf = (held, blinkMs) =>
  held > restMs ? 'rest' : held >= blinkMs ? 'long' : held >= shortMin ? 'short' : 'reflex';

for (let blinkMs = 250; blinkMs <= 1200; blinkMs += 50) {
  for (let held = 83; held <= 292; held += 1) {
    const a = blinkAction(kindOf(held, blinkMs), held, blinkMs);
    if (blinkMs > 292) check(a !== 'select', `natural ${held} ms selects at Blink length ${blinkMs}`);
    if (blinkMs >= 350 && held < 200) check(a === null, `natural ${held} ms acts (${a}) at Blink length ${blinkMs}`);
    if (blinkMs >= 450) check(a === null, `natural ${held} ms acts (${a}) at Blink length ${blinkMs}`);
  }
  check(blinkMs - stepMinMs(blinkMs) >= 100, `step window under 100 ms at Blink length ${blinkMs}`);
  check(blinkAction('short', blinkMs - 1, blinkMs) === 'step', `a firm short blink cannot step at ${blinkMs}`);
  check(blinkAction('long', blinkMs, blinkMs) === 'select', `a long blink cannot select at ${blinkMs}`);
  check(blinkAction('rest', 3500, blinkMs) === null, `resting eyes act at ${blinkMs}`);
}

// Main grid: 8 tiles, Say it (8), Urgent (9), Hold (10), Mode (11).
check(nextZone(-1, 12, 9) === 9, 'first step from a fresh grid is not Urgent');
check(nextZone(9, 12, 9) === 0, 'the step after Urgent is not tile 0');
check(nextZone(8, 12, 9) === 10, 'the step after Say it does not skip Urgent');
check(nextZone(11, 12, 9) === 9, 'the lap does not wrap back to Urgent');
const lap = []; let c = -1;
for (let i = 0; i < 12; i++) lap.push(c = nextZone(c, 12, 9));
check(new Set(lap).size === 12, `a lap misses a zone: ${lap}`);
// Urgent grid: plain order, so the first step is the first urgent tile, not Back.
check(nextZone(-1, 11, -1) === 0, 'urgent grid first step is not tile 0');
check(nextZone(10, 11, -1) === 0, 'urgent grid does not wrap');
check(nextZone(-1, 0, 3) === -1, 'empty grid');

console.log(fails ? `${fails} failed` : 'blinkscan: all checks passed');
process.exit(fails ? 1 : 0);
