# StillMe

Give a person back their own voice — for as long as they have one signal left to give.

An offline AAC app. A person who cannot speak selects word-tiles; an on-device language model
composes the sentence; the iPad says it aloud **in their own voice**, rebuilt from old family home
videos. Nothing ever leaves the device.

```bash
node server.mjs     # → http://localhost:8000
```

No dependencies. Node 22+. Dev LLM is `qwen3:8b` on the 4060 Ti box; ships as a 3–4B quantized model
running fully on-device.

## The ladder

One app. The same tiles, the same sentence engine, the same voice — all the way down.

| | Signal | For whom |
|---|---|---|
| **T0** | Touch | While they have hands |
| **T1** | iPad camera gaze (built into iPadOS) | While the camera can see them |
| **T2** | **EOG** — the eye's own charge, read through a closed lid | **When the camera has quit but the eyes still move** |
| **T3** | **EMG** — a muscle too weak to visibly move | When the eyes are gone |
| **T4** | Stop. Say so out loud. | Completely locked-in. No non-invasive system serves them. |

## The architecture that matters

Every input driver emits the **same six events**:

```
LEFT · RIGHT · UP · DOWN · SELECT · UNDO
```

Touch emits them. Arrow keys emit them. And when the EOG board arrives, its Bluetooth peripheral
emits them too — a left flick of the eye is `LEFT`, a blink is `SELECT`.

**Nothing in the app changes.** Which is why you can build and test the entire eye-controlled
interface today, with a keyboard, before a single electrode ships. Switch the input dropdown to
**Scanning (EOG)** and drive it with the arrow keys — that *is* the eye interface.

## Ethics, enforced in code

An input channel this slow makes it punishingly expensive to correct a wrong sentence — so a user
**will** accept words they didn't mean rather than spend forty seconds rejecting them. Being spoken
*for* is the exact indignity these users are fighting.

- **The model proposes; it never speaks.** Three candidates, he always picks. `showConfirm()`.
- **Literal mode** — say exactly my words, add nothing. Always one tap away.
- **Every grid includes a way to say no.** He must always be able to disagree.
- **`false_completion_rate`** — how often he accepts a sentence that wasn't what he meant.
  Nobody in the LLM-AAC literature reports this. We will.

## The research contribution

Every published language-model speller sits on a *fast* channel, where the model buys a nice 15–60%.
**Nobody has measured how the value of language-model assistance scales as the channel collapses.**

> Language-model assistance is a convenience at 60 bits per minute and load-bearing at 5.
> Its value grows as the channel gets worse — and the worst channel is the one that needs it most.

`data/sessions.jsonl` logs `selections_per_sentence` and `seconds_to_sentence` on every utterance.
Toggle **Predictive tiles** off and on, run the same conversations, and that file is the paper.

You can prove it **with zero hardware**: simulate the channel across accuracy and latency, replay real
AAC utterances through it, and compare a static grid, an n-gram baseline (mandatory — beating "no
prediction" proves nothing), and the on-device model.

## Layout

```
server.mjs           HTTP + API. Zero deps.
lib/llm.mjs          Tile prediction + sentence composition.
lib/voice.mjs        The voice engine. Swap ONE function when the clone is trained.
voices/train_clone.md  The clone pipeline. Go/no-go: a warm banked voice by Aug 15.
data/profile.json    Who he is. The model never invents facts about his life.
data/sessions.jsonl  The research data.
SHOPPING-LIST.md     What to order Monday.
```

## Known gaps

- **Tile prediction is weak on expressive phrases.** Needs and symptoms predict well ("water" →
  please/cold/straw/sip); feelings collapse toward generic filler. This is a fine-tuning job, not a
  prompt job — LoRA on telegraphic-input → next-tile pairs.
- The voice is a **stranger's** until the clone is trained. The app says so, on purpose.
