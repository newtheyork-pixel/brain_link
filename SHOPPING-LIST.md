# StillMe — purchase order

**We swept the market: no buyable EOG-to-iPad device exists.** Not on Amazon, not for $2,000, not
for $20,000. JINS MEME (EOG glasses — the *right* product) was discontinued in 2024 and its SDK is
dead. OpenBCI Ganglion is sold out permanently (EOL radio). BITalino has no iOS API. FRENZ gates raw
data behind a research contract. **That empty shelf is the thesis.** Build.

**Total: ~$210.** Order the Upside Down Labs items TODAY — they ship from India, 1–3 weeks.

## Order 1 — Upside Down Labs · ⚠️ LONG LEAD, ORDER MONDAY

| Part | What it does | Qty | Price |
|---|---|---|---|
| **NPG Lite "Explorer Pack"** | **Take this over loose parts.** ESP32-C6 (BLE) + 3 BioAmp channels + battery connector + electrodes, **on one board**, open firmware. It kills your most demo-hostile failure mode — 60 Hz mains hum and drift on a hand-wired rig in September. This is still **100% "build"**: you write every line of firmware, the decoder, the BLE service, the app. You just don't breadboard an analog front end six weeks before filming. *(Low stock — order now.)* | 1 | ~$105 |
| **BioAmp EXG Pill** | Spares and the EMG rung. **Buy 4, not 2 — you will destroy one, and a fried board in week 14 kills the submission.** Bonus: the *same board* covers both rungs of the ladder — it ships configured for EOG, and you bridge two solder pads to reconfigure it for EMG. One part number, both tiers. Say that on camera; it's a design decision, not a coincidence. | 4 | ~$25 ea |

## Order 2 — Seeed Studio · $22.47 · in stock

| Part | What it does | Qty | Price |
|---|---|---|---|
| **XIAO ESP32-S3** | The brain of the box. Reads the Pills, runs the saccade decoder, speaks Bluetooth LE to the iPad. Onboard LiPo charger, thumbnail-sized. **Buy 3 — you will kill one.** | 3 | $7.49 ea |

## Order 3 — bio-medical.com · $37.94 · in stock

| Part | What it does | Qty | Price |
|---|---|---|---|
| **Kendall H124SG 24mm Ag/AgCl electrodes, 50-pack** | The electrodes. Disposable, pre-gelled, the research standard for facial EOG. 5 per session. Recurring cost — budget $25/50. | 1 | $24.99 |
| **Nuprep skin-prep gel, 4oz** | Drops skin impedance from ~50 kΩ to under 5 kΩ. **Without it your signal is mud.** | 1 | $12.95 |

## Order 4 — Adafruit / Seeed · $8.90

| Part | Qty | Price |
|---|---|---|
| **3.7V 500mAh LiPo, JST-PH 1.25** (one in the box, one charging) | 2 | $4.45 ea |

## From the mentor's bench (~$5 if not)
47 kΩ 1% resistors ×10 (series current limit in **every** electrode lead — non-negotiable),
alcohol prep pads, perfboard, hookup wire, a small plastic project box.

---

## What NOT to buy

- **OpenBCI Cyton — $1,249, sold out, and architecturally fatal.** Its radio speaks Nordic's
  proprietary GZLL, not Bluetooth LE, and requires a USB dongle iPadOS will never expose to a
  third-party app. OpenBCI's own staff tell people to bolt on a Raspberry Pi as a bridge — which
  destroys "one iPad, fully offline."
- **OpenBCI Ganglion — $624.99, sold out.** Its BLE *does* reach iOS, so credit where due. But it's
  6× the budget, on an end-of-life radio, and the firmware is sealed — which kills the one thing the
  competition actually scores: *you built it and can explain it.*
- **Any ADS1299 / ADS1292R board.** They're 24-bit and DC-coupled. A skin electrode's ±300 mV
  half-cell offset times a gain of 24 is **7.2 V — it rails, you see a flat line, and you conclude the
  whole idea is dead.** The EXG Pill is AC-coupled *ahead* of the gain stage, which blocks the offset
  before amplification. **This is exactly why the $35 board is right and the "better" chip is wrong.**
- **Muse, Emotiv.** Wrong signal (brainwaves), wrong electrode locations, closed firmware. Emotiv
  paywalls your own raw data.

---

## Two facts that save you a week

**A custom Bluetooth LE peripheral does NOT need Apple MFi certification.** Apple's own QA1657:
BLE accessories use CoreBluetooth and are not required to be MFi compliant. MFi applies to Lightning
and Classic Bluetooth. You advertise a service UUID; the iPad connects. No license, no approval.

**The #1 demo killer is iOS's stale GATT cache.** Change the characteristic layout in firmware and
the iPad keeps serving the old one — your characteristic "vanishes" and nothing in your app fixes it.
**Bump the BLE device name every time you change the GATT layout during development.**

Also: use **NimBLE**, not Bluedroid. **Disable Wi-Fi in firmware** (shared antenna — a real source of
BLE dropouts). Read the Pills on **ADC1 only** — ADC2 fights the radio.

---

## Three engineering calls that decide whether the demo works

**1. Ship TWO Bluetooth profiles from the same board.**
- A standard **BLE HID keyboard** → drives iPadOS Switch Control system-wide, no SDK, no MFi. You can
  validate the hardware against Notes.app on **day one**, and it's a bulletproof backup demo.
- A **custom GATT service** → StillMe reads the six events directly.

**The trap:** if the device *only* speaks Switch-Control-ese, **iOS eats the events before your app
ever sees them** — you get generic rectangle-scanning over your own tile grid instead of your own
logic. Keep Switch Control OFF while StillMe is foregrounded. Ship both profiles.

**2. Freeze the demo vocabulary at three events: `LEFT · RIGHT · BLINK`.** That already drives a full
row-column scanner over the whole grid. UP/DOWN and DOUBLE_BLINK are stretch goals —
**blink-vs-upward-saccade is the hardest classifier in the stack** (Bell's phenomenon rolls the eyes
up on every lid closure). Do not stake your video on it.

**3. Report false activations per minute, not accuracy.** For an AAC device, a 97% accuracy figure is
noise; the number that matters is how often it fires when he didn't mean it. Knowing that will
impress a technical judge more than any percentage.

## The insight to say out loud on camera

Someone will object: *"EOG needs DC coupling, and your amplifier is AC-coupled."* **They're wrong —
because of a choice you already made.** Since you decode discrete *events* and not gaze *position*,
the band-limited front end turns each saccade step into a signed transient: sign gives direction,
amplitude gives size. **You inherit immunity to electrode drift for free.** The discrete-events
decision buys you the hard half of the signal processing. That's a real architectural insight — say it.

---

## Day the parts arrive: the bench test

Stick electrodes at the outer corners of your eyes. Look left. Look right. Blink.

**If you see three distinct, unmistakable shapes on the trace within an hour, the project is real.**
No training, no model, no calibration — the eye's charge is ~100× bigger than a brainwave and you
will see it with your naked eye on the first try.

That is the go/no-go, and it costs one afternoon.
