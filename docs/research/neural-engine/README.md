# Neural engine research notes

Working notes from the studies that shaped the neural singing engine (`docs/NEURAL.md`
is the contract; this folder is the evidence behind it). They are lab notebooks, not
documentation: paths point into a scratch folder that is not part of the repository, and
later sections sometimes correct earlier ones. Numbers are word-error rates of three
speech recognisers (tiny.en / base.en / small.en) and voice-quality measures from
`prototype/voiceq.py`.

| file | question | short answer |
|---|---|---|
| `q7-ceiling-and-ladder.md` | Is the first voicebank (Qixuan) driven badly, or is it the voice? | The voice: driving it natively changes nothing. It is airy, and the showcase song sits 8 semitones above its range at twice the syllable rate. |
| `q6-out-of-breath.md` | Why does it sound out of breath? | One vowel in nine of the fast verse is whispered (the variance model predicts collapsed voicing); a floor on vowel voicing fixes it. The song's lead EQ, tuned for the formant engine, thins a neural voice. The `cpp` measure falls with pitch. |
| `q1-fast-verse-clarity.md` | Why are fast verses hard to understand? | The bank cannot articulate at 5.6 syllables per second. Slow-motion rendering (sing fast notes up to 2x slower, squeeze the mel back) recovers a third of the errors. |
| `q5-pitch-predictor.md` | Should the bank's own pitch predictor replace Synth2's curves? | Yes where the bank has one, scaled by style (`expr` 0.5 robotic, 1.0 otherwise); it is sung more clearly than flat or hand-built pitch. |
| `q4-voicebanks-and-licences.md` | Are there clearer voicebanks, and what do their terms allow? | Nishiren v2.1 is much clearer, but its terms need the author's approval for use in another program, for commercial use, and restrict use with AI-generated lyrics and music. Qixuan is the only bank found whose terms plainly allow commercial works. |
| `v1-combined-engine.md`, `v1-config.json` | Do the fixes hold together, on both banks? | Yes. Qixuan verse 0.64 / 0.57 / 0.42 to 0.44 / 0.42 / 0.26; the register is the strongest control. |
| `demo3-song-demo.md`, `demo3/` | How far does Qixuan go on the showcase song with every fix? | Verse 0.64 / 0.57 / 0.42 to 0.42 / 0.27 / 0.24 with the song 5 semitones lower; vowel-only breathiness is a trade between tone and words; low voices need the acoustic model an octave above the rendered pitch; the lead EQ and balance for a neural voice. `demo3/` has the backend, cast and song-build copies the demo was made with. |
| `flow-continuous-line.md`, `flow/` | Why does the line stop and start, and what makes it continuous? | Holes sit on stop consonants, unvoiced fricatives, whispered voiced consonants and dead rests between lines. Capping consonant length (never under half of what the duration model gave), a voicing floor on voiced consonants and a breath in every gap between lines take the verse from 3.2 to 2.4 holes a second and join the chorus lines into 11 s phrases. Costs some words on the strictest recogniser. `prototype/flow.py` is the measure. |
| `production-chain-and-relaxed-verse.md`, `production/` | What makes the vocal sound finished? | A chain that reads only the sung (periodic) level: rider, two compressors, de-esser, consonant leveller; a short room; the double made from the lead take instead of a second take; and a rewrite of the fast verse at 3.7 syllables a second. Level spread of the lead 8.7 to 4.4 dB. |
| `separation-adlibs.md`, `separation/` | Why do laughs, shouts and sound words come out as a slur? | They were written on the same tracks as the sung lines, so the engine sang each as one legato phrase. 149 of the song's 860 notes are ad-libs; they move to tracks of their own (nine written tracks, one job each), are sung as detached hits with a designed pitch and no breath, and each distinct sound is rendered once and stamped at every repetition. `separation/repo.patch` is the arrangement change for the song. |

`prototype/` holds the research engine the production code in `synth2/neural/` is ported
from (`dsinfer.py`, `dspitch.py`) and the voice-quality measure (`voiceq.py`). It is not
imported by Synth2 and has hard-coded scratch paths.

Voicebank and vocoder weights are not part of this repository. Their licence terms are
quoted in `q4-voicebanks-and-licences.md`; read the publishers' own text before using any
of them.
