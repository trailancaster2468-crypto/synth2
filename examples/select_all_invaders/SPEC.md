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
- **Key:** E minor. The final chorus modulates up a whole step to F# minor. (The song was written in
  A minor; it sits five semitones lower because that is where the neural voice sings it best. The key is
  written into the score and the band, not applied by the build.)
- **Length:** 134 bars ≈ 191 s (3:11).
- **Language:** English lyrics with a few Japanese (kana) interjections for vocaloid flavour.
  The studio sings both (`lang=ja` notes, or kana lyrics on an `en` track are auto-detected).

## Structure and chords (one chord per bar)

| # | Section | Bars | Chords (per bar) | Energy | Notes |
|---|---|---|---|---|---|
| 1 | Intro | 1-8 | Em Em C D \| Em Em C B | 0.6 | Synth riff and an "alien transmission". The aliens chant ワレワレハ in bars 5-8. AI boot line. |
| 2 | Verse 1 | 9-24 | (Em Em/D C B) x4 | 0.5 | AI lead, wordy patter, half-time-feel drums |
| 3 | Pre-chorus 1 | 25-32 | Am B Em D \| C B C D | 0.7 | Build. Alien interjections (call and response). Bar 32 is a stop on beat 3. |
| 4 | Chorus 1 | 33-48 | (C D Bm Em) x3 \| Am B Em Em | 1.0 | The hook. Doubles, harmony, human crowd shouts. |
| 5 | Post-chorus | 49-52 | Em C D B | 0.8 | Riff, "click click" chant |
| 6 | Verse 2 | 53-68 | (Em Em/D C B) x4 | 0.6 | Dialogue verse, AI and aliens alternating |
| 7 | Pre-chorus 2 | 69-76 | Am B Em D \| C B C D | 0.75 | As pre-chorus 1, more alien |
| 8 | Chorus 2 | 77-92 | as Chorus 1 | 1.0 | |
| 9 | Bridge (chaos) | 93-108 | F Em F Em \| Dm E Am B \| C C#dim D D#dim \| Em F F# B | 0.9 | Glitch chaos: stutters, overlapping voices, confusion ("who is the invader?"). Climbs into the breakdown. |
| 10 | Breakdown | 109-112 | Em C D B | 0.3 | Near silence. AI soft "loading...". Riser into the final chorus. |
| 11 | Final chorus (+2) | 113-128 | (D E C#m F#m) x3 \| Bm C# F#m F#m | 1.0 | Key of F# minor. Biggest moment, all voices. |
| 12 | Outro | 129-134 | F#m D E C# \| F#m (hit) \| silence | 0.5→0 | Witty sign-off ("rate your apocalypse"), alien ワレワレハ fade, final glitch hit |

Chord spelling: `Em/D` means E minor over a D bass.

## Voices, tracks and roles

Three characters sing. `vocals.s2s` holds what was written by hand, one job per track (9 tracks);
`vocals.py` derives the doubles, the harmony and the crowd's other voices and then gives **every kind of
sound a track of its own**: 65 vocal tracks, each with one job, a place in the mix (`build.py`) and a pinned
seed. "Role" is the track's `role` for the neural engine and the mixer (docs/NEURAL.md); the aliens and the
crowd have none and keep the mixer's channel strip.

| Kind of sound | Tracks | Role | Voice and how it is sung | Place in the mix |
|---|---|---|---|---|
| **AI**, the lead | `AI` (choruses 1 and 2, intro), `AI Verses`, `AI Pre`, `AI Bridge`, `AI Breakdown`, `AI Final` | `lead` | `nova`, robotic style: the deadpan, machine-tuned idol. Mostly E4-B4, chorus peaks C5 (D5 / F#5 in the final chorus). The crisp dial, breaths between the lines. | Centre, alone. A channel per section so that it can ride lower in the verses and higher in the bridge and the final chorus. |
| The lead's double | `AI Double`, `AI Double R` | `double` | Not sung: the lead's own take over the chorus notes, delayed and detuned by the mixer. | A pair close around the lead (±0.28), about 10 dB under it. |
| Harmony | `Harmony`, `Harmony L`, `Harmony Final`, `Harmony Final L` | `harmony`, `double:Harmony` | `pixel`, a chord-aware 3rd or 4th under the lead on the hook lines; the left side is a copy of the right side's take. | Both sides (±0.70), darker, more room. |
| The lead's calls | `AI Answer` | `answer` | `nova`: "blocked", "accept all", "try again" in other voices' sections. | A step right of the centre. |
| The lead's click | `AI Click` | `adlib` | `nova`, four detached hits (bar 52). | Left, band-limited, dry. |
| ハイ ハイ | `AI Hai` | `adlib` | `nova`, two hits (bar 71). | Left, band-limited, one repeat, thrown into the delay. |
| The lead's stutter | `AI Stutter` | `adlib` | `nova`: "re- re- re-" into "reboot" (bar 93). | Centre, the lead's tone. |
| **ALIENS**, their lines | `Aliens`, `Aliens Verse`, `Aliens Bridge` | - | `basso`, robotic, the acoustic model an octave above the sung pitch: a low voice that keeps its words. | Right of the centre. |
| The aliens' answers | `Alien Answer`, `Alien Answer Bridge` | - | The same voice: "cookies? delicious", "thank you", "reply all" ... | Further right, 1.5 dB down while the lead sings. |
| The aliens' backing | `Aliens Hook`, `Alien Unison` | - | The final-chorus hook two octaves under the lead; the unison "who is who". | Darker, further back, 3 dB down under the lead. |
| ouch, ナンデ | `Alien Ouch`, `Alien Nande` | - | Detached hits, a falling "ouch". | Far right, band-limited, one repeat. |
| The aliens' click | `Alien Click` | - | Hits (bars 52, 118); the take of "click" was chosen from several. | Far right, band-limited, dry. |
| The laugh | `Alien Laugh` | - | "ha ha ha" (bar 66), each "ha" falling. | Far right, thrown into the delay. |
| The aliens' stutter | `Alien Stutter` | - | "re- re- re-" into "reply all" (bar 93). | Right. |
| The chant | `Alien Chant`, `Alien Chant 2` (+ `Alien Chant 2 B`, an octave down) | - | ワレワレハ ウチュウジン ダ, sung per syllable with the "voice into a fan" chop on each. | Two voices left and right of everything the aliens sing. |
| **HUMANS**, the crowd's line | `Crowd`, `Crowd 2`, `Crowd 3` | - | `atlas`, `ember`, `aurora` (an octave up): "who is the invader". Every crowd track has a second take (`... B`), which is what makes three singers a gang. | Wide (left, right, half left), far, 3 dB down under the lead. |
| The crowd's answers | `Crowd Answer`, `Crowd 2 Answer`, `Crowd 3 Answer` | - | "four oh four", "five stars". | As the line. |
| The whisper | `Crowd Whisper`, `Crowd 2 Whisper` | - | "hurry" (bar 112). | As the line, its fader up: a whisper is quiet. |
| The crowd's hook | `Crowd Hook`, `Crowd 2 Hook`, `Crowd 3 Hook` | - | The final chorus' hook an octave under the lead. | As the line, lower. |
| The shouts | `Crowd Shout`, `Crowd Shout 2`, `Crowd Shout 3` | - | Hits: "help", "hey", "what", "huh", "here". | Wide, less far. |
| The crowd's click | `Crowd Click`, `Crowd Click 2`, `Crowd Click 3` | - | The sound word of the choruses and the post-chorus. | As the shouts. |

## Engines and sounds

- The score says `engine: auto`. With the neural engine's models installed (`synth2 models install`) the
  voicebank sings; without them the formant engine sings the same tracks (roles and the cast are then
  ignored, a double sings its own notes) and the mixer's channel strip stands in for the roles' chains.
- The band (`backing.json`, 22 instruments): the drums (one stem per kind of drum), the pick and slap
  basses, both guitars, the palm mutes, the verse plucks, the bell, the music box and the pad play sampled
  sounds from two sets, `generaluser-gs` (GeneralUser GS License v2.0) and `karoryfer-growlybass` (CC0 1.0):
  `synth2 sounds install generaluser-gs karoryfer-growlybass`. An instrument whose set is not installed plays
  its synthesised preset. The chip leads and their hook tags, the arpeggio, the supersaws, the growl and sub
  basses, the chip kit (tick and glitch percussion) and the effects (transitions and gags) are synthesised
  on purpose: they are the song's chiptune side.

## Credits a render carries

The render report lists them (`report["credits"]`), and a published version should too:

- Vocals: AI-synthesised singing voice, voicebank 绮萱 (Qixuan) v2.7.0 (voice provider 颜绮萱, DiffSinger
  Community), synthesised with DiffSinger (OpenVPI). The voicebank's terms ask for this statement.
- Vocoder: PC-NSF-HiFiGAN (DiffSinger Community Vocoder Project, OpenVPI Team), CC BY-NC-SA 4.0: a render
  sung with it is for non-commercial use (docs/NEURAL.md, "Licences").
- Instruments: GeneralUser GS v2.0.3 by S. Christian Collins (a courtesy; the licence allows any use).
- Electric bass: Growlybass by Karoryfer Samples (CC0; a courtesy).

A render with the formant engine and the synthesised band carries no credits.

## Files (examples/select_all_invaders/)

- `SPEC.md`: this file
- `lyrics.md`: final lyrics with role tags per line, by section
- `vocals.s2s`: the hand-written vocal tracks in score notation (E minor)
- `vocals.py`: the derived tracks and the one-kind-per-track arrangement, with roles, cast and seeds
- `backing.json`: the instrumental arrangement for the arranger engine, with its sampled sounds
- `build.py`: builds the full project (vocals, backing and mix), renders it with `render_project`, and
  writes `out/select_all_invaders/` (the WAV, stems, project JSON and the report with engines and credits)
