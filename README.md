# StillMe

**Give a person back their own voice — for as long as they have one signal left to give.**

An offline communication device for people who have lost the ability to speak (ALS,
laryngectomy, brainstem stroke). They select word-tiles; an on-device language model composes
the sentence; it is spoken aloud **in their own voice**, rebuilt from old family home videos.

Nothing ever leaves the device. No cloud, no account, no subscription.

```bash
./asr/setup          # once — builds whisper.cpp for local speech recognition
ollama pull qwen3:4b # once — the on-device language model
node server.mjs      # → http://localhost:8000
```

Zero npm dependencies. Node 22+.

---

## Why

Existing AAC devices cost $8,000–15,000, run at 8–15 words per minute, and speak in a robot
stranger's voice. Meanwhile almost every family already has hours of the person's real voice
sitting in old phone videos.

And the devices fail early. In a 2020 study, a commercial eye-tracker scored **0.0% on one
late-stage ALS patient and 29% on another** — both awake, alert, and *trying*. Their eyes still
moved. The camera simply couldn't read them: a drooping eyelid hides the pupil, and a dry cornea
destroys the infrared reflection every tracker depends on.

**A camera needs light bouncing off the eye. An electrode reads charge from inside it.**

## The ladder

One app. The same tiles, the same sentence engine, the same voice — all the way down.

| | Signal | For whom |
|---|---|---|
| **T0** | Touch | While they have hands |
| **T1** | Camera eye-tracking (built into iPadOS) | While the camera can see them |
| **T2** | **EOG** — the eye's own standing charge, read *through a closed lid* | When the camera has quit but the eyes still move |
| **T3** | **EMG** — a muscle too weak to visibly move | When the eyes are gone |
| **T4** | Stop, and say so. | Completely locked-in. No non-invasive system serves them, and we don't claim to. |

## He can start, not just answer

Every tile used to be generated from what somebody *else* said — which can only ever make him a
responder. Communication partners already take ~90% of turns with AAC users.

- **Answer them** — reply to what was just said (the mic listens; nobody types)
- **Ask something** — *"Did Emma take the job?"* Generated from **his life**, not a prompt
- **Say something** — *"Go to bed. I'm fine. Stop fussing."*

## The architecture that matters

Every input driver emits the **same six events**:

```
LEFT · RIGHT · UP · DOWN · SELECT · UNDO
```

Touch emits them. Arrow keys emit them. And when the EOG board arrives, its Bluetooth peripheral
emits them too — a flick of the eye is `LEFT`, a blink is `SELECT`. **Nothing above that line
changes.** Which is why the whole eye-controlled interface is testable today, with a keyboard,
before a single electrode ships. (Settings → Input → Scanning.)

## Ethics, enforced in code

A channel this slow makes correcting a wrong sentence punishingly expensive — so a user **will**
accept words he didn't mean rather than spend forty seconds rejecting them. Being spoken *for* is
the exact indignity these users are fighting.

- **The model proposes; it never speaks.** Three candidates. He always picks.
- **Literal mode** — *say exactly my words, add nothing.* Always one tap away.
- **Every grid contains a way to say no.**
- **Urgent** is one selection from any screen, and it interrupts — a predicted grid about a
  graduation must never trap a man who cannot breathe.

**Planned, not yet built:** **`false_completion_rate`** — how often he accepts a sentence that
wasn't what he meant. Nobody in the LLM-AAC literature reports it; we intend to measure it. It is
*not* enforced in code today.

## The research contribution

Every published language-model speller sits on a *fast* channel, where the model buys a nice
15–60%. **Nobody has measured how the value of language-model assistance scales as the channel
collapses.**

> Language-model assistance is a convenience at 60 bits per minute and load-bearing at 5.
> Its value grows as the channel gets worse — and the worst channel is the one that needs it most.

`data/sessions.jsonl` records `selections_per_sentence` and `seconds_to_sentence` for every
utterance. Toggle predictive tiles off, run the same conversations, and that file is the result.

### Tile quality is measured, not eyeballed

```bash
node eval/run.mjs      # 20 conversational turns, judged by an independent model
```

The judge must **prove it can separate a good grid from a bad one** before it is allowed to
report a number (`eval/judge.mjs`). It failed that gate twice during development and had to be
fixed — an early version scored a good grid 0.0/10 because it wanted the tiles to echo the
question back.

| | score | median latency |
|---|---|---|
| `qwen3:14b` | **9.1** / 10 | 1526 ms |
| `qwen3:4b` (on-device) | **8.4** / 10 | **938 ms** |

The 4B's failures are concentrated in *continuation*: asked "are you in pain?" he answers "yes",
and it offers `to · the · doctor` where the 14B offers `sometimes · in my legs · not much`. It
drops into grammatical continuation instead of meaning. **That is the fine-tune's training
target** — found by measuring, not by guessing.

## Layout

```
server.mjs             HTTP + API. Zero dependencies.
lib/llm.mjs            Tile prediction + sentence composition.
lib/tiles.mjs          Grid construction: echo-stripping, synonym dedupe, pinned core.
lib/listen.mjs         whisper.cpp — speech recognition, on this machine.
lib/voice.mjs          The voice engine. Swap ONE function when the clone is trained.
public/mic.js          Mic → 16kHz WAV, in the page. Live transcription.
voices/train_clone.md  The clone pipeline. Go/no-go: a warm banked voice by Aug 15.
data/profile.json      Who he is, and what is going on in his life.
eval/                  The measuring instrument.
```

## Status

Working: tiles → LLM → speech, live local transcription, ask/tell/answer, urgent grid,
scanning input, the eval harness.

Not yet: the voice clone (the whole point — that's the August gate), and the EOG hardware.

---

*Built for the 2026 Congressional App Challenge.*
