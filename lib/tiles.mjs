// Grid construction. Shared by the app and the eval, so the thing we measure is the
// thing we ship.

export const norm = (s) => String(s).toLowerCase().trim().replace(/[^a-z' ]/g, '');

// "no" / "nope" / "nah" / "none" is ONE tile, not four. On a channel where a pick costs
// 40 seconds, spending four of his eight slots on synonyms is stealing from him.
const SAME = [
  ['no', 'nope', 'nah', 'none', 'not', 'negative', 'no thanks', 'nothing', 'not really'],
  ['yes', 'yeah', 'yep', 'yup', 'sure', 'ok', 'okay'],
  ['stop', 'quit', 'enough', 'done', 'finished'],
  ['tired', 'exhausted', 'worn out', 'weary'],
  ['bad', 'not good', 'awful', 'terrible', 'rough'],
  ['hurts', 'hurt', 'pain', 'painful', 'sore', 'aches'],
  ['thank you', 'thanks', 'thank'],
  ['wait', 'hold on', 'later', 'not now'],
];

export const clusterOf = (t) => SAME.findIndex((g) => g.includes(norm(t)));

/** Drop exact repeats and synonyms of tiles already on the grid. */
export function dedupe(tiles, taken = []) {
  const words = new Set(taken.map(norm));
  const groups = new Set(taken.map(clusterOf).filter((c) => c >= 0));
  return tiles.filter((t) => {
    const w = norm(t);
    const c = clusterOf(t);
    if (!w || words.has(w)) return false;
    if (c >= 0 && groups.has(c)) return false;
    words.add(w);
    if (c >= 0) groups.add(c);
    return true;
  });
}

/** 0-10. How many genuinely different things can he say from this grid? */
export function distinctness(tiles) {
  if (!tiles.length) return 0;
  const meanings = new Set(
    tiles.map((t) => (clusterOf(t) >= 0 ? `c${clusterOf(t)}` : norm(t))),
  );
  return Math.round((meanings.size / tiles.length) * 100) / 10;
}

/**
 * Build the grid he actually sees.
 *
 * The first `coreSlots` positions NEVER change. AAC systems (LAMP, Words for Life) are
 * built on motor learning: a tile that stays put gets memorized, and a user gets faster
 * at it for years. A grid that reshuffles on every pick destroys that — each selection
 * becomes a fresh visual search, and that cost is invisible in a naive WPM number.
 *
 * So coreSlots is a knob, not a guess: 0 = fully predictive, 8 = a fixed board.
 * The eval measures the tradeoff. That measurement is the honest version of the
 * research claim.
 */
export function buildGrid({ core = [], predicted = [], coreSlots = 2, size = 8 }) {
  const pinned = core.slice(0, coreSlots);
  const rest = dedupe(predicted, pinned).slice(0, size - pinned.length);
  return [...pinned, ...rest];
}
