# Voice clone — the pipeline

The hero of the whole project. It runs on the 4060 Ti box, not the Mac.

**Go/no-go: one banked voice speaking a warm, full sentence by Aug 15.** If the clone sounds
uncanny — 80%-right, dead-relative-adjacent — the project pivots. Test this FIRST, before you
build anything else. It is the single riskiest component and the only one that can't be worked around.

## 1. Harvest (the family's phone, not a studio)

Old home videos. Birthday parties, holidays, someone talking at a table. You need
**30–60 seconds of the cleanest speech you can find** — but harvest everything and pick later.

```bash
# pull audio out of every video in the folder
for f in raw/*.{mov,mp4,m4v}; do
  ffmpeg -i "$f" -vn -ac 1 -ar 22050 "wav/$(basename "${f%.*}").wav"
done
```

## 2. Clean

```bash
# strip music, TV, other voices — Demucs separates the speaker from everything else
python -m demucs --two-stems=vocals wav/*.wav

# then hand-cut the segments where ONLY he is talking. Do this by ear. It matters more
# than any hyperparameter. 60 curated seconds beats 20 noisy minutes.
```

## 3. Bake-off — run BOTH, pick by ear, week one

| Approach | What it is | When it wins |
|---|---|---|
| **F5-TTS / XTTS-v2** (zero-shot) | Feed 10–30s reference audio, no training. Minutes to try. | Try this FIRST. If it sounds like him, you may be done. |
| **Piper / VITS fine-tune** | Actually train on his audio. Hours on the 4060 Ti. | More control, usually warmer, exports clean ONNX for on-device. |

Ship target is **ONNX → on-device**. Nothing about his voice ever leaves the iPad.

## 4. Judge it honestly — the blind family test

Do not grade this yourself. Play three clips to relatives who have not heard the plan:
one real recording of him, two synthesized. Ask: *"Is this him?"*

**Put the number on a title card in the video.** *"9 of 10 relatives identified the voice as his."*
That single sentence is worth more than any architecture diagram.

## 5. Pre-render before you film

Synthesize his ~200 most frequent utterances to disk ahead of time. It's a legitimate product
feature — instant speech on a channel where a sentence can cost minutes — **and it is demo
insurance.** Never let the filmed take depend on live cold-path inference.

Wire it up: `lib/voice.mjs` → `synthCloned()`. Set `voiceModel` in `data/profile.json` and the
whole app switches over. Nothing else changes.

## Say this out loud, on camera

Apple's Personal Voice only works if you record **before** you lose your speech. ElevenLabs and
Bridging Voice already give ALS patients free clones — **name them, don't pretend they don't exist.**

> "Personal Voice only works if you bank it before you lose your speech. StillMe rebuilds a voice
> *after* — from the home videos every family already has. Offline, free, and for the people those
> programs leave out."

Beating known prior art impresses. Being caught unaware of it is fatal.

## Consent — settle this before the first conversation, not after

Written consent for the voice clone. The model stays device-bound. **No export button.** Say that
on camera: *"the safeguards Congress is debating — built in."*
