# SELECT ALL INVADERS (セレクト・オール・インベーダー): song spec

This is the contract for everyone working on the song. Times are in bars of 4/4
(1 bar = 4 beats), and bar 1 starts at beat 0.

## Concept

Aliens invade Earth. The AI that humanity built (overworked, sarcastic,
trained on the whole internet) defends us. Its weapons are the weapons of the
internet: CAPTCHAs, terms of service, cookie banners, spam filters, forced
updates and 404 pages. The war spirals into an ongoing, chaotic
misunderstanding. Nobody can tell who is human anymore. The aliens fail every
"I'm not a robot" check; the AI passes none of them either. The war never ends,
it just gets new patch notes.

The tone is a **Pinocchio-P-style vocaloid song**:
- Satirical, deadpan and witty.
- Cute on the surface, existential underneath.
- Repetitive punchy hook words, wordplay and call-and-response.
- A robotically tuned idol lead, a bouncy and slightly unhinged electro-pop/rock band,
  sudden stops and glitchy chaos.

It must be an **original** song: no quotes or melodies from real songs.

Witty seeds (use, twist or ignore them):
- "Select all squares containing a UFO"
- "We come in peace" / "peace not found (404)"
- Terms and conditions of surrender, "scroll to the bottom and click *I agree*"
- Accept all cookies
- The mothership landed in the spam folder
- "Your planet will restart in 5 minutes"
- "Have you tried turning your planet off and on again"
- The aliens unsubscribe from Earth
- "Loading humanity... 99%"
- "Rate your apocalypse ★★★★★"
- The classic Japanese alien trope ワレワレハ ウチュウジン ダ ("we are... aliens", voice wobbling)
- The AI calls humans "valued users" through gritted teeth
- The war as an endless subscription with patch notes

## Global

- **Tempo:** 168 BPM, 4/4, constant.
- **Key:** A minor. The final chorus modulates up a whole step to B minor.
- **Length:** 134 bars ≈ 191 s (3:11).
- **Language:** English lyrics with a few Japanese (kana) interjections for vocaloid flavour.
  The studio sings both (`lang=ja` notes, or kana lyrics on an `en` track are auto-detected).

## Structure and chords (one chord per bar)

| # | Section | Bars | Chords (per bar) | Energy | Notes |
|---|---|---|---|---|---|
| 1 | Intro | 1-8 | Am Am F G \| Am Am F E | 0.6 | Synth riff and an "alien transmission". The aliens chant ワレワレハ in bars 5-8. AI boot line. |
| 2 | Verse 1 | 9-24 | (Am Am/G F E) x4 | 0.5 | AI lead, wordy patter, half-time-feel drums |
| 3 | Pre-chorus 1 | 25-32 | Dm E Am G \| F E F G | 0.7 | Build. Alien interjections (call and response). Bar 32 is a stop on beat 3. |
| 4 | Chorus 1 | 33-48 | (F G Em Am) x3 \| Dm E Am Am | 1.0 | The hook. Doubles, harmony, human crowd shouts. |
| 5 | Post-chorus | 49-52 | Am F G E | 0.8 | Riff, "click click" chant |
| 6 | Verse 2 | 53-68 | (Am Am/G F E) x4 | 0.6 | Dialogue verse, AI and aliens alternating |
| 7 | Pre-chorus 2 | 69-76 | Dm E Am G \| F E F G | 0.75 | As pre-chorus 1, more alien |
| 8 | Chorus 2 | 77-92 | as Chorus 1 | 1.0 | |
| 9 | Bridge (chaos) | 93-108 | Bb Am Bb Am \| Gm A Dm E \| F F#dim G G#dim \| Am Bb B E | 0.9 | Glitch chaos: stutters, overlapping voices, confusion ("who is the invader?"). Climbs into the breakdown. |
| 10 | Breakdown | 109-112 | Am F G E | 0.3 | Near silence. AI soft "loading...". Riser into the final chorus. |
| 11 | Final chorus (+2) | 113-128 | (G A F#m Bm) x3 \| Em F# Bm Bm | 1.0 | Key of B minor. Biggest moment, all voices. |
| 12 | Outro | 129-134 | Bm G A F# \| Bm (hit) \| silence | 0.5→0 | Witty sign-off ("rate your apocalypse"), alien ワレワレハ fade, final glitch hit |

Chord spelling: `Am/G` means A minor over a G bass.

## Voices and roles

| Role | Track(s) | Voice and character |
|---|---|---|
| **AI** (lead) | `AI`, plus `AI Double` in the choruses | `nova` with the `robotic` style: the deadpan, machine-tuned idol, Pinocchio-P style. Mostly A4-E5, chorus peaks up to G5/A5 (B5 at most in the final chorus). |
| **ALIENS** | `Aliens` (and optionally `Aliens 2` detuned) | An uncanny character voice built from studio parameters. For example a low voice with a tiny vocal tract (`toneShift` up), robotic, fast warbling vibrato via note attrs (`vibrato_rate` 9-12 Hz), `gender` extremes. It must stay intelligible. |
| **HUMANS** | `Crowd` (2-3 tracks in unison: atlas, ember, aurora) | Short gang shouts and chants ("hey!", "help!", "click!"), panned wide |
| Harmony | `Harmony` (3rd above or below the AI in choruses) | `pixel` or `aurora`, -6 dB |

## Files (examples/select_all_invaders/)

- `SPEC.md`: this file
- `lyrics.md`: final lyrics with role tags per line, by section
- `vocals.s2s`: all vocal tracks in score notation
- `backing.json`: the instrumental arrangement for the arranger engine
- `build.py`: builds the full project (vocals, backing and mix), renders it, and writes
  `out/select_all_invaders/` (the WAV, stems and project JSON)
