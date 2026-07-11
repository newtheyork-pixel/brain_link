// LLM client. Dev: qwen3:8b on the 4060 Ti box.
// Ships as: 3-4B quantized, fully on-device (MLX). Same prompts, same contract.

const HOST = process.env.STILLME_LLM_HOST ?? 'https://llm.thegriffinfund.org';
const MODEL = process.env.STILLME_LLM_MODEL ?? 'qwen3:8b';

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
 * Predictive tile narrowing — the research contribution.
 * Every selection is expensive (an EOG user may spend ~30s on one).
 * The model's job is to make the NEXT tile they need already be on screen.
 */
export async function predictTiles({ selected, partner, profile, n = 8 }) {
  const out = await generateJSON(
    `${profile.name} has picked: ${selected.join(' + ')}
${partner ? `(In reply to: "${partner}")` : ''}

What word does he reach for NEXT to finish that thought? Give ${n}.

These are NOT a generic vocabulary list. They must CONTINUE the specific words he
already picked — the next word in the SENTENCE HE IS BUILDING. Read what he has
picked, imagine the sentence he is reaching for, and offer the missing pieces.
If he picked a person and a feeling, give him the words that aim that feeling AT them.

Examples of the behavior:
  picked "water"          → ["please", "cold", "ice", "straw", "now", "sip", "no", "more"]
  picked "pain"           → ["back", "legs", "head", "worse", "bad", "medicine", "now", "stop"]
  picked "Michael"        → ["here", "call", "tell", "come", "thank", "love", "proud", "ask"]
  picked "Susan + tired"  → ["rest", "go", "sleep", "please", "sorry", "worry", "love", "no"]
  picked "Emma + here"    → ["stay", "sit", "close", "talk", "good", "happy", "now", "please"]
  picked "yes + but"      → ["wait", "later", "first", "slow", "careful", "help", "ask", "no"]

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
    `Conversation partner just said: ${partner ? `"${partner}"` : '(nothing)'}
${profile.name} selected these words, in order: ${selected.join(' + ')}

Turn them into 3 different complete sentences they might have meant.
Rules:
- Use ONLY the meaning in their words. Add grammar, not content.
- Do not invent details, names, reasons, or feelings they did not select.
- Natural spoken English. Short. How a person actually talks.
- Vary the 3 options meaningfully — different plausible readings, not 3 rewordings.
- Order them most-likely first.

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
