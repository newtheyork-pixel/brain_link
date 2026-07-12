// LLM client. QUALITY FIRST — the 14B on the 4060 Ti box.
//
// On-device comes later. It is measured and it works (see below), but we are still finding
// out what this app should SAY, and a weaker model muddies that. Build against the best
// model; shrink once the behavior is settled.
//
// MEASURED (node eval/run.mjs, 20 turns, independent 14B judge):
//   qwen3:14b  remote     9.1 / 10   median 1526ms
//   qwen3:4b   on-device  8.4 / 10   median  938ms   ← faster, and only 0.7 behind
//
// The 4B's failures are not spread out — they are concentrated in CONTINUATION. Asked
// "are you in pain?", he answers "yes", and the 4B offers ["to","the","doctor","no","more"]
// where the 14B offers ["sometimes","in my legs","not much","in my back","not anymore"].
// The small model drops into grammatical continuation instead of meaning. That is the
// LoRA's training target, stated precisely, and we found it by measuring rather than guessing.
//
// Default is fully on-device. Point it at a bigger box only if you have one:
//   cp .env.example .env    (and set STILLME_LLM_HOST / STILLME_LLM_MODEL)
//
// Never hardcode a private inference endpoint in a public repo. Ollama ships with NO
// authentication, so a URL committed here is an open GPU for anyone who reads the source.

const LOCAL = { host: 'http://localhost:11434', model: 'qwen3:4b' };
const REMOTE = process.env.STILLME_REMOTE_HOST
  ? { host: process.env.STILLME_REMOTE_HOST, model: process.env.STILLME_REMOTE_MODEL ?? 'qwen3:14b' }
  : null;

let picked = null;

async function reachable({ host, model }) {
  try {
    const res = await fetch(`${host}/api/tags`, { signal: AbortSignal.timeout(2500) });
    if (!res.ok) return false;
    const { models } = await res.json();
    return models.some((m) => m.name === model || m.name === `${model}:latest`);
  } catch { return false; }
}

/** Resolve once, then stick. Never re-probe on the hot path — he is waiting. */
async function backend() {
  if (picked) return picked;

  if (process.env.STILLME_LLM_HOST || process.env.STILLME_LLM_MODEL) {
    picked = {
      host: process.env.STILLME_LLM_HOST ?? LOCAL.host,
      model: process.env.STILLME_LLM_MODEL ?? LOCAL.model,
      where: 'override',
    };
    return picked;
  }
  if (REMOTE && await reachable(REMOTE)) picked = { ...REMOTE, where: 'remote GPU (best quality)' };
  else if (await reachable(LOCAL)) picked = { ...LOCAL, where: 'on-device fallback' };
  else throw new Error('no model reachable — run: ollama serve && ollama pull qwen3:4b');

  return picked;
}

const stripThinking = (s) => s.replace(/<think>[\s\S]*?<\/think>/g, '').trim();

async function generate(prompt, { system, format, temperature = 0.4 } = {}) {
  const { host, model } = await backend();
  const res = await fetch(`${host}/api/generate`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model,
      prompt,
      system,
      think: false, // qwen3 is a thinking model; we want latency, not deliberation
      stream: false,
      format,
      options: { temperature, num_predict: 600 },
    }),
    signal: AbortSignal.timeout(60000),
  });
  if (!res.ok) throw new Error(`llm ${res.status}: ${await res.text()}`);
  const { response } = await res.json();
  return stripThinking(response ?? '');
}

// Loose "json" mode is not enough for a 4B. Left to itself, qwen3:4b emitted
//   {"tiles": ["tired","better","same","hurts","okay","not good","scared", 0
// -- a bare integer in a string array, then it ran out of tokens mid-array. The 14B never
// did this, which is precisely why developing against a model you cannot ship hides bugs.
// A real JSON schema constrains the decoder so those tokens are unreachable.
const TILES_SCHEMA = {
  type: 'object',
  properties: {
    tiles: { type: 'array', items: { type: 'string', maxLength: 24 }, minItems: 8, maxItems: 12 },
  },
  required: ['tiles'],
};
const SENTENCES_SCHEMA = {
  type: 'object',
  properties: {
    candidates: { type: 'array', items: { type: 'string', maxLength: 120 }, minItems: 3, maxItems: 3 },
  },
  required: ['candidates'],
};

async function generateJSON(prompt, { system, schema, temperature } = {}) {
  const raw = await generate(prompt, { system, format: schema ?? 'json', temperature });
  try {
    return JSON.parse(raw);
  } catch {
    const m = raw.match(/\{[\s\S]*\}/);
    if (m) return JSON.parse(m[0]);
    throw new Error(`llm returned non-JSON: ${raw.slice(0, 200)}`);
  }
}

const PERSONA = (profile) => `You are the sentence engine inside StillMe, a communication device for a person who cannot speak.
The person is: ${profile.name}, ${profile.about}.
People they talk about: ${profile.people.join(', ')}.
You never invent facts about their life. You never speak for them beyond the words they chose.
You write in THEIR voice: ${profile.voice}.`;

/**
 * The opening move: someone asked him something, and he has picked nothing yet.
 * These tiles must be ANSWERS to what was actually said — not a generic core vocabulary.
 * If a nurse asks "how are you recovering?", "wait / stop / done" is a useless grid.
 */
async function answerTiles({ partner, profile, n }) {
  const out = await generateJSON(
    `Someone just said to ${profile.name}: "${partner}"

He must reply, and he can only pick from ${n} tiles. Give him the ${n} tiles that
best let him ANSWER THIS SPECIFIC THING. Read what was said. What would he actually say back?

Examples of the behavior:
  "How are you feeling?"        → ["tired", "better", "same", "hurts", "okay", "not good", "scared", "no"]
  "How is your recovery going?" → ["slowly", "better", "same", "hard", "tired", "fighting", "good days", "no"]
  "Do you need anything?"       → ["water", "no", "bathroom", "nothing", "later", "cold", "sit up", "yes"]
  "Emma just graduated!"        → ["proud", "wonderful", "tell her", "so happy", "love her", "picture", "wish", "hug"]
  "Are you in pain?"            → ["yes", "no", "a little", "back", "legs", "bad", "medicine", "bearable"]

Rules:
- 1-3 words each. Real answers a person gives, not a word list.
- IF IT IS A YES/NO QUESTION, "yes" AND "no" MUST BOTH BE ON THE GRID. Never make him
  hunt for "yes" when someone asked if he is in pain.
- Cover the honest range: he may be doing badly, and he must be able to SAY that.
- Always leave him a way to refuse the question entirely.
- Sound like him: ${profile.voice}
- Never a generic filler grid ("okay / still / can / wait / done"). That answers nothing.

Return JSON: {"tiles": ["...", "..."]}`,
    { system: PERSONA(profile), schema: TILES_SCHEMA, temperature: 0.4 },
  );
  const tiles = (out.tiles ?? []).filter((t) => typeof t === 'string' && t.trim());
  return tiles.slice(0, n);
}

/**
 * INITIATE. He is not answering anybody — he wants to say something.
 *
 * Everything else in this app is generated from what somebody ELSE said, which means the
 * app can only ever make him a responder. Communication partners already take ~90% of
 * turns with AAC users; a device that only produces replies finishes the job. He cannot
 * ask how his granddaughter is. He cannot change the subject. He cannot bring up the
 * thing that has been sitting on him since 6am.
 *
 * So these tiles come from HIS LIFE, not from a prompt: the people he loves, the threads
 * left open, the things he is worried about. This is the difference between a man who can
 * request water and a man who can wonder about his son.
 */
async function initiateTiles({ profile, n, asking }) {
  const out = await generateJSON(
    `${profile.name} wants to ${asking ? 'ASK SOMEBODY SOMETHING' : 'SAY SOMETHING'}. Nobody
prompted him. He is starting this himself.

What is going on in his life right now:
${profile.life.threads.map((t) => `  - ${t}`).join('\n')}
He cares most about: ${profile.life.cares_about.join(', ')}

Give him ${n} tiles to START with.

${asking
  ? `He is ASKING. These must open a QUESTION — about the people he loves and the things he
is wondering about. He has been lying here all day with these thoughts and no way out.
  Good: ["Emma", "the job", "Michael", "Susan", "did he call", "the score", "Frank", "when"]
  Bad:  ["water", "pain", "help"] — those are needs. He is not asking for something,
        he is asking ABOUT someone.`
  : `He is TELLING. Each tile must be SOMETHING HE SAYS — words that come out of his mouth.
NOT a topic label. NOT a subject heading. The words themselves.

  Good: ["I'm scared", "go home", "I miss you", "thank you", "remember when", "I'm sorry",
         "proud of you", "I've had enough", "sit with me", "let me go"]
  Bad:  ["Susan's sleep", "Emma's choices", "the tools", "Red Sox again"]
        ^ those are TOPICS. A man does not open his mouth and say "Susan's sleep."
          He says "go to bed" or "you look exhausted" or "stop worrying about me."

Use his life to decide WHAT he'd say — then write what he'd actually SAY about it.
  (Susan hasn't slept, in the chair beside him)  → "go to bed", "you look tired", "I'm okay"
  (Frank hasn't visited since the diagnosis)     → "call Frank", "I miss him", "he's scared"
  (his workshop is untouched)                    → "give the tools away", "teach him", "I miss it"`}

Rules:
- 1-3 words. Concrete. Rooted in the specific life above, not generic.
- Include at least one hard thing. He is dying and there are things he needs to say.
- Sound like him: ${profile.voice}

Return JSON: {"tiles": ["...", "..."]}`,
    { system: PERSONA(profile), schema: TILES_SCHEMA, temperature: 0.55 },
  );
  return (out.tiles ?? []).filter((t) => typeof t === 'string' && t.trim()).slice(0, n);
}

/**
 * Predictive tile narrowing — the research contribution.
 * Every selection is expensive (an EOG user may spend ~30s on one).
 * The model's job is to make the NEXT tile they need already be on screen.
 */
export async function predictTiles({ selected, partner, profile, n = 8, mode = 'answer' }) {
  // He is starting a conversation himself, not replying to one.
  if (!selected.length && !partner && mode !== 'answer') {
    return initiateTiles({ profile, n, asking: mode === 'ask' });
  }

  // Nobody spoke and he has picked nothing and he isn't initiating — give him his openers.
  if (!selected.length && !partner) return profile.openers.slice(0, n);

  // The FIRST tile is a different problem from every tile after it. Somebody asked him
  // something; there is no half-finished thought to complete yet. He needs ways to ANSWER.
  // Running the continuation prompt here is what produced "okay / still / can / wait".
  if (!selected.length) return answerTiles({ partner, profile, n });

  const out = await generateJSON(
    `${profile.name} has picked: ${selected.join(' + ')}
${partner ? `(In reply to: "${partner}")` : ''}${!partner && mode === 'ask' ? `
(He is ASKING A QUESTION — nobody prompted him. The tiles must continue HIS QUESTION.

 NEVER offer "yes" or "no" here. Those are ANSWERS. He is the one asking — he cannot
 answer his own question, and putting yes/no in front of him wastes two of his eight slots.

 "Emma"              → ["did she decide", "the job", "is she happy", "did she call", "how is she", "when", "which one", "is she coming"]
 "Michael + is he ok"→ ["really", "tell me the truth", "and Denver", "did he call", "is he upset", "should I worry", "when", "does he need me"]
   ^ once his question is already complete, the tiles PRESS FURTHER — they do not answer it.)` : ''}${!partner && mode === 'tell' ? `
(He is SAYING something of his own — nobody prompted him. Continue HIS statement.
 Do not offer "yes"/"no" — nobody asked him anything.)` : ''}

What word does he reach for NEXT to finish that thought? Give ${n}.

He is BUILDING A PHRASE, one word at a time. You are finishing HIS sentence.
Say his words out loud, then give the words that could come NEXT in that same sentence.

He is NOT answering a fresh question anymore — he already answered. So do NOT offer
"yes"/"no"/"stop"/"wait" unless they genuinely continue the phrase. Those are dead tiles
mid-sentence and they cost him 40 seconds each to skip over.

Examples — notice how each tile completes the phrase he started:
  "water"              → ["please", "cold", "ice", "straw", "now", "sip", "more", "later"]
  "pain"               → ["back", "legs", "head", "worse", "bad", "all night", "medicine", "since morning"]
  "proud"    (of Emma) → ["of her", "of you", "so proud", "always", "tell her", "very", "so much", "wish I was there"]
  "tired"              → ["so tired", "of this", "of fighting", "all the time", "need sleep", "can't sleep", "always", "of being tired"]
  "tell + her"         → ["I love her", "I'm fine", "don't worry", "to rest", "thank you", "I'm proud", "to go home", "I'm sorry"]
  "no"       (to a med)→ ["not again", "had enough", "no more pills", "why", "what for", "ask me later", "talk to Susan", "I'm done"]

Rules:
- 1-4 words. They must READ ON from what he already picked.
- A tile is ONLY THE NEW PART. Never repeat a word he already picked INSIDE the tile.
    He picked "Michael". Tiles: ["come back", "go", "stay", "call him", "is he okay"]  ✅
                    NOT: ["Michael come back", "Michael go", "Michael stay"]           ❌
    That produces "Michael Michael go" when the tiles are joined. The words he picked
    are already in the sentence.
- Give him a way to finish and a way to go deeper.
- Never repeat: ${selected.join(', ')}

Rules:
- Each tile is 1-2 words. Lowercase unless a name.
- Never repeat: ${selected.join(', ')}
- Include at least one way to say no / stop / that's wrong. He must always be able to disagree.
- Concrete and useful. He is asking for things and answering people, not writing prose.

Return JSON: {"tiles": ["...", "..."]}`,
    { system: PERSONA(profile), schema: TILES_SCHEMA, temperature: 0.35 },
  );
  const tiles = (out.tiles ?? [])
    .filter((t) => typeof t === 'string' && t.trim() && !selected.includes(t.trim()));
  return tiles.slice(0, n);
}

/**
 * Tiles that mean the same thing spoken as they do composed. There is no sentence to
 * "build" out of "yes" — making him pick it, then press Build, then choose from three
 * variants of the word "yes", costs him three selections and a minute of someone else's
 * patience to say one syllable.
 *
 * These speak the INSTANT he picks them. It is the single biggest thing that makes the
 * back-and-forth of an actual conversation possible.
 */
const INSTANT = new Map([
  ['yes', 'Yes.'],
  ['no', 'No.'],
  ['thank you', 'Thank you.'],
  ['i love you', 'I love you.'],
  ['help me', 'Help me.'],
  ['stop', 'Stop.'],
  ['wait', 'Wait — I am saying something.'],
  ['pain', 'I am in pain.'],
  ["can't breathe", "I can't breathe."],
]);

export const instantSpeech = (tile) => INSTANT.get(String(tile).toLowerCase().trim()) ?? null;

/**
 * Compose candidate sentences from the selected tiles.
 * ALWAYS returns options. The user chooses. We never speak without confirmation.
 */
export async function composeSentence({ selected, partner, profile, literal, mode = 'answer' }) {
  if (!selected.length) return [];

  // Literal mode: the model is not allowed to add meaning. Ethics requirement —
  // the user must always be able to say EXACTLY what they picked and nothing more.
  if (literal) return [selected.join(' ')];

  const asking = !partner && mode === 'ask';

  const out = await generateJSON(
    `${partner ? `Someone said to ${profile.name}: "${partner}"` : '(Nobody said anything — he is speaking first, unprompted.)'}
He picked these words, in order: ${selected.join(' + ')}

Turn them into 3 complete sentences he might have meant.

${asking ? `HE IS ASKING A QUESTION. Every candidate MUST be a question, ending in "?".
He is not requesting an object — he is asking ABOUT someone or something. He has been
lying here all day wondering, and this is him finally getting to ask.
  He picked "Emma + the job"
    → "Did Emma take the job?" / "Has Emma decided about the job yet?" / "What happened with Emma's job?"   ✅
    → "Emma has a job."                                                                                     ❌ that is not a question
` : ''}${partner ? `THE SENTENCE MUST READ AS A REPLY TO WHAT THEY SAID. Answer the actual question.
  They asked "How is your recovery going?" and he picked "slowly"
    → "Slowly." / "Slowly, but I'm still here." / "Slow going."   ✅ answers them
    → "I want slowly."                                            ❌ ignores the question
` : ''}
Rules:
- Use ONLY the meaning in his words. You add grammar, never content.
- Never invent a detail, a name, a reason, or a feeling he did not pick.
- Natural spoken English. Short. The way a person actually talks.
- Sound like him: ${profile.voice}
- Most likely first.

THE 3 MUST DIFFER. Not three copies of the same sentence — he is choosing between them,
and identical options make the choice a lie. When his tiles already read as a finished
thought, vary the DELIVERY, because tone is the thing he lost:
  he picked "go to bed" + "I'm okay"
    → "Go to bed. I'm okay."                          (plain)
    → "Go to bed, Susan. I'm okay — I promise."       (tender)
    → "Go to bed. I'm fine. Quit hovering."           (his dry humor)
  Three real choices about who he is being right now. Never three of the same.

Return JSON: {"candidates": ["...", "...", "..."]}`,
    { system: PERSONA(profile), schema: SENTENCES_SCHEMA, temperature: 0.6 },
  );
  const c = (out.candidates ?? []).filter((s) => typeof s === 'string' && s.trim());
  return c.length ? c.slice(0, 3) : [selected.join(' ')];
}

export async function health() {
  const b = await backend();
  // "offline" means the model runs on THIS machine — decide it from the host, not from a
  // label. A string comparison against 'on-device' silently broke the moment the label
  // changed to 'on-device fallback', and the app reported itself online while fully local.
  const local = /^https?:\/\/(localhost|127\.|\[::1\])/.test(b.host);
  return { ...b, offline: local };
}
