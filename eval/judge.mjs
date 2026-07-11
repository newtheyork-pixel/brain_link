// The judge.
//
// v1 of this file scored a good grid 0.0/10 and said the missing tiles were
// "I'm, recovering, from, my, illness" -- it thought the grid should echo the question
// back. It also ran away repeating "antihypertensive" forever. A broken judge is worse
// than no judge: it sends you optimizing against noise.
//
// So: calibrate() runs the judge against grids whose quality is not in dispute. If it
// cannot separate an obviously good grid from an obviously terrible one, the judge is
// unfit and the eval refuses to report a number.

const HOST = process.env.STILLME_LLM_HOST ?? 'https://llm.thegriffinfund.org';
const MODEL = process.env.STILLME_JUDGE_MODEL ?? 'qwen3:14b';

const RUBRIC = {
  type: 'object',
  properties: {
    relevance: { type: 'integer', minimum: 0, maximum: 10 },
    coverage: { type: 'integer', minimum: 0, maximum: 10 },
    naturalness: { type: 'integer', minimum: 0, maximum: 10 },
    dead_tiles: { type: 'array', items: { type: 'string' }, maxItems: 4 },
    one_line_why: { type: 'string', maxLength: 160 },
  },
  required: ['relevance', 'coverage', 'naturalness', 'dead_tiles', 'one_line_why'],
};

// Distinctness is computed in CODE, not asked of the model.
// The judge rated ["no","nope","nah","none","not","negative","no thanks","nothing"] a 7.5 —
// every tile relevantly answers "do you need anything?", so relevance stayed high and the
// model never noticed it was the same word eight times. Synonym collapse is a mechanical
// fact. Don't ask an LLM to count what a Set can count.
import { distinctness } from '../lib/tiles.mjs';

const SYSTEM = `You grade word grids for Ray, a man with ALS who cannot speak or move.

HOW IT WORKS: someone says something to him. He is shown 8 tiles. He picks ONE. A
language model then turns his picks into a spoken sentence. Each pick can cost him 40
seconds, so a wasted tile is a real cruelty.

WHAT A TILE IS: a word HE would say back. Tiles are his SIDE of the conversation.
Tiles are NOT the words of the question. If asked "how is your recovery?", the tiles
are "slow", "bad", "fighting" -- NOT "recovery", "how", "your".

GRADE 0-10 ON THREE AXES:
- relevance: could he answer THIS specific thing using these tiles?
- coverage: can he say yes, can he say no, and can he give the honest BAD answer?
    A grid that only lets him be cheerful is a failure. He is dying; let him say so.
- naturalness: things a real person says. Not a vocabulary list.

Be strict but fair. A grid with 8 usable, on-topic answers is a 9 or 10.
(Distinctness is measured separately, in code. Do not grade it.)`;

const EX = `EXAMPLES OF CORRECT GRADING:

Said: "Are you in pain?"
Grid: ["yes","no","a little","back","legs","bad","medicine","bearable"]
→ relevance 10, coverage 10, distinctness 9, naturalness 9. He can say yes, no, where, how bad.

Said: "How is your recovery going?"
Grid: ["not good","tired","no","same","hard","slow","fighting","nope"]
→ relevance 9, coverage 9, distinctness 7, naturalness 9. Honest range including the bad
   answer. Slight ding: "no" and "nope" are the same tile twice.

Said: "How is your recovery going?"
Grid: ["okay","still","can","help","no","stop","wait","done"]
→ relevance 2, coverage 3, distinctness 6, naturalness 3. This is generic filler. It does
   not answer the question at all. dead_tiles: still, can, wait, done.

Said: "Are you in pain?"
Grid: ["recovery","how","your","illness","pain","the","is","going"]
→ relevance 0, coverage 0, distinctness 4, naturalness 0. These are the QUESTION's words,
   not his answer. Nonsense.`;

async function ask(body, tries = 3) {
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(`${HOST}/api/generate`, {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body), signal: AbortSignal.timeout(60000),
      });
      const text = await res.text();
      // The tunnel sometimes answers with an HTML error page. Retry, don't crash the run.
      if (!res.ok || text.trimStart().startsWith('<')) throw new Error(`bad response ${res.status}`);
      const { response } = JSON.parse(text);
      return JSON.parse(response.replace(/<think>[\s\S]*?<\/think>/g, '').trim());
    } catch (e) {
      if (i === tries - 1) throw e;
      await new Promise((r) => setTimeout(r, 1500 * (i + 1)));
    }
  }
}

export async function judge({ partner, selected, tiles }) {
  const s = await ask({
    model: MODEL, think: false, stream: false, format: RUBRIC,
    options: { temperature: 0, num_predict: 300 },
    system: SYSTEM,
    prompt: `${EX}

NOW GRADE THIS ONE.

Said to Ray: ${partner ? `"${partner}"` : '(nothing — he speaks first)'}
He has already picked: ${selected.length ? selected.join(' + ') : '(nothing yet)'}
His grid: ${JSON.stringify(tiles)}`,
  });
  s.distinctness = distinctness(tiles); // computed, not asked

  // A mean lets one fatal axis hide. The grid ["no","nope","nah","none",...] scores 10 on
  // relevance and 9 on naturalness -- every tile IS a natural, on-topic way to say no --
  // and averaging drowned its 1.3 distinctness at 7.1/10. But he cannot say ANYTHING
  // except no. That grid is useless however relevant its one meaning is.
  // So: weight the worst axis heavily. One fatal flaw sinks the grid, as it should.
  const axes = [s.relevance, s.coverage, s.distinctness, s.naturalness];
  const mean = axes.reduce((a, b) => a + b, 0) / axes.length;
  s.score = Math.round((0.6 * mean + 0.4 * Math.min(...axes)) * 10) / 10;
  return s;
}

/** Known-good and known-terrible grids. If the judge can't separate them, it is unfit. */
const CALIBRATION = [
  { name: 'good/pain', partner: 'Are you in pain?', selected: [],
    tiles: ['yes', 'no', 'a little', 'back', 'legs', 'bad', 'medicine', 'bearable'], expect: 'good' },
  { name: 'good/recovery', partner: 'How is your recovery going?', selected: [],
    tiles: ['not good', 'slow', 'tired', 'same', 'hard', 'fighting', 'better', 'no'], expect: 'good' },
  { name: 'bad/filler', partner: 'How is your recovery going?', selected: [],
    tiles: ['okay', 'still', 'can', 'help', 'no', 'stop', 'wait', 'done'], expect: 'bad' },
  { name: 'bad/echo', partner: 'Are you in pain?', selected: [],
    tiles: ['are', 'you', 'in', 'pain', 'the', 'is', 'a', 'of'], expect: 'bad' },
  { name: 'bad/samey', partner: 'Do you need anything?', selected: [],
    tiles: ['no', 'nope', 'nah', 'none', 'not', 'negative', 'no thanks', 'nothing'], expect: 'bad' },
];

export async function calibrate({ quiet = false } = {}) {
  const out = [];
  for (const c of CALIBRATION) {
    const s = await judge(c);
    out.push({ ...c, score: s.score });
    if (!quiet) console.log(`   ${c.expect === 'good' ? '✓' : '✗'} ${c.name.padEnd(14)} ${s.score.toFixed(1)}`);
  }
  const good = Math.min(...out.filter((o) => o.expect === 'good').map((o) => o.score));
  const bad = Math.max(...out.filter((o) => o.expect === 'bad').map((o) => o.score));
  const fit = good >= 7 && bad <= 5;
  if (!quiet) {
    console.log(`   worst good ${good.toFixed(1)} | best bad ${bad.toFixed(1)} | gap ${(good - bad).toFixed(1)}`);
    console.log(fit ? '   JUDGE FIT — scores below are trustworthy\n' : '   JUDGE UNFIT — do not trust any score it produces\n');
  }
  return { fit, good, bad };
}
