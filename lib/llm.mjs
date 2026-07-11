// LLM client. Dev: qwen3:14b on the 4060 Ti box.
//
// 14b over 8b is not a luxury — it is the difference between tiles that answer the
// question and tiles that don't. Asked "Michael got the job in Denver," 8b offered
// "huh / really / wait"; 14b offered "tell him / proud / good job / son". Same prompt.
//
// This matters for the ship target (a 3-4B model, fully on-device): a small model
// cannot do this from prompting alone. It will need a LoRA fine-tuned on
// telegraphic-input -> next-tile pairs. That gap is a finding, not a surprise.

const HOST = process.env.STILLME_LLM_HOST ?? 'https://llm.thegriffinfund.org';
const MODEL = process.env.STILLME_LLM_MODEL ?? 'qwen3:14b';

const stripThinking = (s) => s.replace(/<think>[\s\S]*?<\/think>/g, '').trim();

async function generate(prompt, { system, format, temperature = 0.4 } = {}) {
  const res = await fetch(`${HOST}/api/generate`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      prompt,
      system,
      think: false, // qwen3 is a thinking model; we want latency, not deliberation
      stream: false,
      format,
      options: { temperature, num_predict: 400 },
    }),
  });
  if (!res.ok) throw new Error(`llm ${res.status}: ${await res.text()}`);
  const { response } = await res.json();
  return stripThinking(response ?? '');
}

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
    { system: PERSONA(profile), temperature: 0.4 },
  );
  const tiles = (out.tiles ?? []).filter((t) => typeof t === 'string' && t.trim());
  return tiles.slice(0, n);
}

/**
 * Predictive tile narrowing — the research contribution.
 * Every selection is expensive (an EOG user may spend ~30s on one).
 * The model's job is to make the NEXT tile they need already be on screen.
 */
export async function predictTiles({ selected, partner, profile, n = 8 }) {
  // Nobody spoke and he has picked nothing — he is starting the conversation himself.
  // There is nothing to answer and nothing to continue. Give him his openers, instantly:
  // asking the model here produced "tool / stupid / hammer / crap" (it leaned on his
  // carpenter history for want of anything better to do).
  if (!selected.length && !partner) return profile.openers.slice(0, n);

  // The FIRST tile is a different problem from every tile after it. Somebody asked him
  // something; there is no half-finished thought to complete yet. He needs ways to ANSWER.
  // Running the continuation prompt here is what produced "okay / still / can / wait".
  if (!selected.length) return answerTiles({ partner, profile, n });

  const out = await generateJSON(
    `${profile.name} has picked: ${selected.join(' + ')}
${partner ? `(In reply to: "${partner}")` : ''}

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
- Give him a way to finish and a way to go deeper.
- Never repeat: ${selected.join(', ')}

Rules:
- Each tile is 1-2 words. Lowercase unless a name.
- Never repeat: ${selected.join(', ')}
- Include at least one way to say no / stop / that's wrong. He must always be able to disagree.
- Concrete and useful. He is asking for things and answering people, not writing prose.

Return JSON: {"tiles": ["...", "..."]}`,
    { system: PERSONA(profile), temperature: 0.35 },
  );
  const tiles = (out.tiles ?? [])
    .filter((t) => typeof t === 'string' && t.trim() && !selected.includes(t.trim()));
  return tiles.slice(0, n);
}

/**
 * Compose candidate sentences from the selected tiles.
 * ALWAYS returns options. The user chooses. We never speak without confirmation.
 */
export async function composeSentence({ selected, partner, profile, literal }) {
  if (!selected.length) return [];

  // Literal mode: the model is not allowed to add meaning. Ethics requirement —
  // the user must always be able to say EXACTLY what they picked and nothing more.
  if (literal) return [selected.join(' ')];

  const out = await generateJSON(
    `${partner ? `Someone said to ${profile.name}: "${partner}"` : '(Nobody said anything — he is speaking first.)'}
He picked these words, in order: ${selected.join(' + ')}

Turn them into 3 complete sentences he might have meant.

${partner ? `THE SENTENCE MUST READ AS A REPLY TO WHAT THEY SAID. Answer the actual question.
  They asked "How is your recovery going?" and he picked "slowly"
    → "Slowly." / "Slowly, but I'm still here." / "Slow going."   ✅ answers them
    → "I want slowly."                                            ❌ ignores the question
` : ''}
Rules:
- Use ONLY the meaning in his words. You add grammar, never content.
- Never invent a detail, a name, a reason, or a feeling he did not pick.
- Natural spoken English. Short. The way a person actually talks.
- Vary the 3 meaningfully — different plausible readings, not three rewordings.
- Sound like him: ${profile.voice}
- Most likely first.

Return JSON: {"candidates": ["...", "...", "..."]}`,
    { system: PERSONA(profile), temperature: 0.6 },
  );
  const c = (out.candidates ?? []).filter((s) => typeof s === 'string' && s.trim());
  return c.length ? c.slice(0, 3) : [selected.join(' ')];
}

export async function health() {
  const res = await fetch(`${HOST}/api/tags`, { signal: AbortSignal.timeout(5000) });
  if (!res.ok) throw new Error(`llm unreachable: ${res.status}`);
  const { models } = await res.json();
  return { host: HOST, model: MODEL, available: models.some((m) => m.name === MODEL) };
}
