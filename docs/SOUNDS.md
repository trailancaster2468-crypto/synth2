# Sound sets

Sampled instruments, drum kits and impulse responses are not part of Synth2. They are
downloaded from their publishers when you ask for them, checked, and kept in a folder of
your own. Each set keeps its own licence: read it before you publish music made with it.

## Commands

```
synth2 sounds list                      what is installed and what the registry offers
synth2 sounds list --kind soundfont     only one kind (soundfont, sfz, drumkit, samples, impulse)
synth2 sounds list --json               the same as JSON
synth2 sounds install generaluser-gs    download, verify and install one or more sets
synth2 sounds install --all             every set whose publisher states that commercial use
                                        is allowed and that is not share-alike
synth2 sounds install ID --from-file X  use a local copy of the set's archive (or a folder
                                        holding its source files) instead of downloading
synth2 sounds add PATH                  register a soundfont, an SFZ instrument or a folder
                                        of samples you already have
synth2 sounds remove ID                 delete an installed set
synth2 sounds path                      print the sounds directory
```

`install` shows the licence, the publisher's own sentence on commercial use, what the
licence asks and the credit line, and asks for a yes. `--accept-license` answers yes for a
script; without a terminal and without that flag nothing is downloaded.

A good first install is one General MIDI soundfont (every band instrument gets a sampled
version) and one drum kit:

```
synth2 sounds install generaluser-gs avl-drumkits
```

From Python:

```python
from synth2 import sounds
sounds.install("generaluser-gs", accept_license=True)   # -> InstalledSound
sounds.installed()                                       # every installed set
sounds.find("generaluser-gs").files                      # [Path(".../GeneralUser-GS.sf2")]
sounds.credits()                                         # credit lines of the installed sets
sounds.remove("generaluser-gs")
```

## Where sets live

```
<sounds_dir>/<id>/synth2-sound.json     manifest
<sounds_dir>/<id>/files/...             the unmodified contents of the download
```

`sounds_dir` is `$SYNTH2_SOUNDS`, else `%LOCALAPPDATA%\synth2\sounds` on Windows, else
`${XDG_DATA_HOME:-~/.local/share}/synth2/sounds`: the folder next to the models directory
(`docs/NEURAL.md`).

### Manifest (`synth2-sound.json`)

```json
{
  "format": "synth2-sound",
  "version": 1,
  "id": "generaluser-gs",
  "kind": "soundfont",
  "name": "GeneralUser GS",
  "sound_version": "2.0.3",
  "publisher": "S. Christian Collins",
  "root": "files",
  "files": ["GeneralUser-GS.sf2"],
  "presets": [],
  "tags": ["gm", "gs", "drums", "piano"],
  "license": "GeneralUser GS License v2.0",
  "license_file": "files/LICENSE.txt",
  "license_url": "https://github.com/mrbumpy409/GeneralUser-GS/blob/main/documentation/LICENSE.txt",
  "license_source": "documentation/LICENSE.txt in the publisher's repository ...",
  "commercial_use": true,
  "commercial_use_statement": "You may use GeneralUser GS without restriction for your own music creation, private or commercial.",
  "share_alike": false,
  "terms": ["..."],
  "credits": ["Instruments: GeneralUser GS v2.0.3 by S. Christian Collins."],
  "unclear": "...",
  "homepage": "https://www.schristiancollins.com/generaluser",
  "source": "https://raw.githubusercontent.com/...",
  "size": 32321713,
  "sha256": "...",
  "external": false,
  "installed_size": 32321713,
  "installed_at": "2026-01-01T00:00:00Z"
}
```

- `kind`: `soundfont` (`.sf2` files), `sfz` (`.sfz` mappings with their samples), `drumkit`
  (folders of one-shot drum samples), `samples` (a folder of audio files), `impulse`
  (impulse responses for reverb and cabinets).
- `root` is the folder the names in `files` and `presets` are relative to, itself relative
  to the set's folder. For a set registered in place with `synth2 sounds add` it is an
  absolute path and `external` is true; removing such a set deletes only the manifest.
- `files` are the files that matter: the `.sf2` files, the `.sfz` files worth loading, or
  the sample folders. `presets` is a guide for instrument pickers: `name`, `file` and
  `category`. An SFZ preset may also carry `defines` (values for `$variables` the `.sfz`
  file uses without defining them) and `cc` (controller number and starting value, to
  pick a variant of the program). A soundfont's full preset list is in the file itself.
- `tags` say what the set covers (`gm` = a complete General MIDI set; `flac` = the samples
  are FLAC, which the sampler reads only when the optional `soundfile` package is
  installed).
- `license` is the licence the publisher names and `license_source` where it was read.
  `commercial_use` is `true` or `false` when the publisher (or the licence the publisher
  names) says so, absent when nothing is stated. `commercial_use_statement` quotes the
  publisher. `unclear` records what the publisher leaves open. `terms` is what the licence
  asks; `credits` are the lines a published work should carry.
- `size` and `sha256` describe the download. A set made of several downloads has one
  digest over the list of its files' hashes.

## The registry

Every set below was downloaded from the URL the registry names and unpacked by the
installer before it was listed. "Commercial use" repeats what the publisher states; it is
not legal advice. Sizes are download sizes.

| Id | Kind | Size | What it is | Licence | Commercial use stated |
|---|---|---|---|---|---|
| `generaluser-gs` | soundfont | 32 MB | General MIDI and GS: 261 presets, 13 drum kits | GeneralUser GS License v2.0 | yes |
| `musescore-general-hq` | soundfont | 274 MB (490 MB installed) | large General MIDI set | MIT | yes |
| `fluidr3-gm` | soundfont | 135 MB | the classic General MIDI set, with a GS bank | MIT | yes |
| `avl-drumkits` | soundfont | 152 MB | five acoustic drum and percussion kits | CC BY-SA 3.0 with an exception for produced music | yes; share-alike for sample libraries |
| `drum-machines-oramics` | drumkit | 2.5 MB | CR-78, LinnDrum, TR-505, Rhythm King one-shots | "Public Domain" as stated by the collection | not stated by the recordists |
| `karoryfer-growlybass` | sfz | 167 MB | electric bass | CC0 1.0 | yes |
| `karoryfer-meatbass` | sfz | 255 MB | upright bass, plucked and bowed | CC0 1.0 | yes |
| `karoryfer-black-and-green-guitars` | sfz | 482 MB | two clean electric guitars | CC0 1.0 | yes |
| `karoryfer-shinyguitar` | sfz | 368 MB | archtop guitar, microphone and pickup | CC0 1.0 | yes |
| `jrhodes3d` | sfz | 236 MB | Rhodes Mark I electric piano | free for music; samples CC BY-NC 4.0 | yes (music); redistribution of the samples non-commercial |
| `greg-sullivan-e-pianos` | sfz | 21 MB | Wurlitzer EP200, CP80, Pianet T (FLAC) | CC BY 3.0 | yes |
| `upright-piano-kw` | sfz | 33 MB | Kawai upright piano (FLAC) | CC0 1.0 | yes |
| `splendid-grand-piano` | sfz | 77 MB | Steinway grand (FLAC) | "Public Domain" as stated by the packager | not stated by AKAI |
| `vsco2-strings` | sfz | 755 MB | violin, viola, cello sections, solo violin, contrabass, harp | CC0 1.0 | yes |
| `vsco2-brass` | sfz | 330 MB | trumpet, French horn, trombone, tuba | CC0 1.0 | yes |
| `vsco2-woodwinds` | sfz | 334 MB | flute, piccolo, oboe, clarinet, bassoon | CC0 1.0 | yes |
| `vsco2-percussion` | sfz | 269 MB | orchestral percussion, timpani, glockenspiel, xylophone, marimba, tubular bells | CC0 1.0 | yes |
| `vsco2-keys` | sfz | 547 MB | pipe organ and two upright pianos | CC0 1.0 | yes |
| `voxengo-ir` | impulse | 14 MB | 38 impulse responses: halls, rooms, a church, four speaker cabinets | Voxengo impulse files licence | yes |

What the publishers say, in their own words, and where it was read:

- **GeneralUser GS** (`documentation/LICENSE.txt` in github.com/mrbumpy409/GeneralUser-GS):
  "You may use GeneralUser GS without restriction for your own music creation, private or
  commercial." Left open by the publisher: some samples came from other free banks and he
  "cannot be 100% sure where all of the samples originated".
- **MuseScore General HQ** (`MuseScore_General_HQ_License.md` in the source archive):
  "MuseScore_General_HQ.sf2 is shared under the MIT license as described in COPYING, as was
  FluidR3Mono and FluidR3 before it." and "The acknowledgements and copyright notices above
  must be included in any derivative work."
- **Fluid R3** (`README` in the source archive): "I hereby release Fluid under the MIT
  license, as described in COPYING."
- **AVL Drumkits** (`sf2/README` in github.com/x42/avldrums.lv2): "CC-BY-SA with the
  following exception: The by-sa condition is imperative if you modify the samples
  themselves or create new sample libraries with this licensed product. However, produced
  music and other non-sample-library works can be licensed freely." The README does not
  use the word "commercial".
- **Karoryfer sets** (`readme.txt` and `LICENSE` in each release archive): "Royalty-free for
  all commercial and non-commercial use." with the CC0 1.0 text as the licence.
- **VSCO 2 Community Edition** (`Readme.txt` and `LICENSE` on the SFZ branch of
  github.com/sgossner/VSCO-2-CE): "You are permitted to use these samples for ANY purpose.
  We ask that you do not sell the samples directly" and "Please provide credit to Versilian
  Studios/Sam Gossner, and/or Ivy Audio/Simon Dalzell where applicable, and link to the
  VSCO: CE homepage." The licence file is CC0 1.0.
- **jRhodes3d** (`LICENSE` in github.com/jlearman/jRhodes3d-wav): "This sample set is
  provided free for use as a musical instrument. You can use the sounds in any way in your
  works as an artist, just as you could with a real instrument. To distribute the samples
  themselves [...] the jRhodes samples are licensed under CC BY-NC 4.0." Left open: the
  header of the `.sfz` files names "CC-SA-BY" instead.
- **Greg Sullivan's E-Pianos** (`README.md` in github.com/sfzinstruments/GregSullivan.E-Pianos):
  "made by Greg Sullivan, with the author permission with the request for attribution" and
  "licensed under a Creative Commons Attribution 3.0 Unported License". The author's own
  page was not read.
- **Upright Piano KW** (`README.md` in github.com/freepats/upright-piano-kw): "Published
  under the terms of Creative Commons CC0 public domain dedication".
- **Splendid Grand Piano** (`README.md` in github.com/sfzinstruments/SplendidGrandPiano):
  "This samples set was released as public domain in early 2000 by Akai company." This is
  the packager's statement; no document from AKAI comes with the files.
- **Drum machines** (`README.md` of each machine in github.com/oramics/sampled): "license:
  Public Domain", with the page each recording came from. This is the collector's
  statement; the pages of the people who made the recordings were not read.
- **Voxengo impulse responses** (`LICENSE.txt` of the pack): "Permission is granted to
  anyone to use these impulse files royalty-free for any purpose, including commercial
  usage." and "You may not sell these impulse files or earn any direct or indirect profit
  from their distribution." The files come from the copy in github.com/oramics/sampled.

## Credits to show when you publish

`synth2 sounds list` prints the credit line of every installed set and
`synth2.sounds.credits()` returns them. Put the lines of the sets a song uses into its
description or liner notes:

| Set | Credit line | Required by the licence? |
|---|---|---|
| `generaluser-gs` | Instruments: GeneralUser GS v2.0.3 by S. Christian Collins. | no, a courtesy |
| `musescore-general-hq` | Instruments: MuseScore General HQ soundfont 0.2.1 (MIT): Fluid R3 by Frank Wen, mono conversion by Michael Cowgill, adaptation by S. Christian Collins, temple blocks by Ethan Winer, drumline cymbals by Michael Schorsch. | for copies of the soundfont; a courtesy for music |
| `fluidr3-gm` | Instruments: Fluid (R3) SoundFont, Copyright (c) 2000-2002, 2008 Frank Wen (MIT). | for copies of the soundfont; a courtesy for music |
| `avl-drumkits` | Drums: AVL Drumkits by Glen MacArthur (bandshed.net/avldrumkits), CC BY-SA 3.0. | yes (attribution) |
| `drum-machines-oramics` | Drum machine samples: oramics/sampled collection (CR-78 from boxedear.com, LinnDrum from machines.hyperreal.org, MRK-2 from submodern.com, TR-505 from progsounds.com). | not stated |
| `karoryfer-*` | Electric bass: Growlybass / Upright bass: Meatbass / Electric guitars: Black And Green Guitars / Guitar: Shinyguitar, by Karoryfer Samples (CC0). | no, a courtesy |
| `jrhodes3d` | Electric piano: jRhodes3d by Jeff Learman. | no for music |
| `greg-sullivan-e-pianos` | Electric pianos: Greg Sullivan's E-Pianos (sullivang.net), SFZ mapping by kinwie, CC BY 3.0. | yes (attribution) |
| `upright-piano-kw` | Piano: Upright Piano KW, FreePats project (CC0). | no, a courtesy |
| `splendid-grand-piano` | Piano: Splendid Grand Piano, samples by AKAI (public domain), SFZ mapping by kinwie. | not stated |
| `vsco2-*` | Orchestral samples: VSCO 2 Community Edition by Versilian Studios / Sam Gossner and Ivy Audio / Simon Dalzell (CC0), vis.versilstudios.net/vsco-community.html | asked for by the publisher ("Please provide credit") |
| `voxengo-ir` | Reverb impulse responses: Voxengo (Aleksey Vaneev), voxengo.com/impulses. | no, a courtesy |

## Your own samples

```
synth2 sounds add ~/Music/MyKit                      a folder of WAV files, registered in place
synth2 sounds add ~/sf2/NicePiano.sf2 --id nice-piano
synth2 sounds add ~/sfz/Strings --kind sfz --name "My strings" --license "CC BY 4.0" \
       --credit "Strings: recorded by A. Player"
synth2 sounds add ~/Downloads/kit.zip                an archive is unpacked into the sounds folder
synth2 sounds add ~/IR/rooms --kind impulse
synth2 sounds add ~/Music/MyKit --copy               copy instead of pointing at the folder
```

- The kind is detected: a `.sf2` file or a folder of them is a `soundfont`, a folder with
  `.sfz` files is `sfz`, a folder of audio files is `samples`, or `drumkit` when the file
  names look like drums (`kick`, `snare`, `hat`, `tom`, `crash`, `ride`, `clap` ...). Say
  `--kind impulse` for impulse responses.
- Nothing is copied unless you pass `--copy`: the manifest points at your folder, and
  `synth2 sounds remove` deletes only the manifest.
- Synth2 knows nothing about the licence of your own sounds. `--license` and `--credit`
  record what you know; without them the listing says the licence is your responsibility.

## How downloads are checked

- The licence is shown and accepted before anything is fetched.
- Every download has a pinned size and SHA-256 in the registry and is rejected when either
  differs. A set that is fetched file by file from a pinned commit has a size and hash for
  each file (`synth2/sounds/lists/`).
- Archives (zip, tar, Debian packages) are unpacked without trusting them: only plain
  files and folders are created, with fixed permissions; an entry with an absolute path,
  `..`, a drive letter, a link or a special file, a repeated name, or an archive that
  unpacks beyond the size limits is rejected. Nothing in a download is executed.
- A set is built in a hidden staging folder and moved into place at the end, so a failed
  install leaves nothing behind and never damages an installed set.
- The same code installs the neural models (`synth2 models`, `docs/NEURAL.md`).

## Leads: sets that are not in the registry

Found while building the registry and not verified. Check the publisher's terms yourself
before using any of them; `synth2 sounds add` registers a copy you download.

| Set | Where | What is known | Why it is not in the registry |
|---|---|---|---|
| Virtuosity Drums (Versilian Studios) | github.com/sfzinstruments/virtuosity_drums | multi-microphone acoustic kit, SFZ, 1.4 GB, mostly FLAC; reachable | size; licence file not read |
| Karoryfer Big Rusty Drums | github.com/sfzinstruments/karoryfer.big-rusty-drums (release v1.100) | acoustic kit, SFZ, 0.7 GB, mostly FLAC; the repository's LICENSE is CC0 1.0 | size and FLAC; release archive not downloaded |
| Karoryfer Swirly Drums | github.com/sfzinstruments/karoryfer.swirly-drums | drum kit, SFZ, 1.8 GB of WAV; reachable | size; licence file not read |
| Karoryfer Frankensnare, Unruly Drums, Caveman Cosmonaut, Weresax | github.com/sfzinstruments/karoryfer.* | repositories exist | not examined |
| D. Smolken double bass | github.com/sfzinstruments/dsmolken.double-bass | SFZ, 277 MB of WAV, LICENSE is CC0 1.0, readme: "Royalty-free for all commercial and non-commercial use." | the same instrument as Meatbass; not installed through the installer |
| SM Drums, Salamander Grand Piano, jRhodes3c | github.com/sfzinstruments/SMDrums, /SalamanderGrandPiano, /jlearman.jRhodes3c | repositories exist | not examined |
| Versilian Community Sample Library (VCSL) | github.com/sgossner/VCSL | repository exists | not examined, licence not read |
| Sonatina Symphonic Orchestra | github.com/peastman/sso | repository exists | not examined, licence not read |
| The Open Source Drum Kit (Real Music Media) | github.com/crabacus/the-open-source-drumkit | 794 WAV files, 630 MB; reachable | the repository states no licence, only a link to a forum thread |
| TR-808 and TR-909 samples, EMT 140 plate impulses, Pearl Master Studio kit | github.com/oramics/sampled | small WAV sets; reachable | no licence stated for the 808 and 909; the plate says "Creative Commons Attibution-Any", the Pearl kit "Creative Commons Attribution 3.0", both the collector's words |
| TR-808 samples (fluid-music/open-drums) | github.com/fluid-music/open-drums | repository exists | not examined |
| Hydrogen drum kits | archive.ubuntu.com, package `hydrogen-drumkits` (source archive 316 MB) | the source archive is reachable | not examined, licences not read |
| TimGM6mb, FreePats (2006), OPL3 soundfonts | archive.ubuntu.com, packages `timgm6mb-soundfont`, `freepats`, `opl3-soundfont` | small General MIDI sets; the package files are reachable | not examined, licences not read |
| FreePats instruments, OpenAIR impulse responses, DrumGizmo kits, Musical Artifacts, Freesound, the publishers' own sites of several registry sets | freepats.zenvoid.org, openair.hosted.york.ac.uk, drumgizmo.org, musical-artifacts.com, freesound.org, voxengo.com, schristiancollins.com, bandshed.net, karoryfer.com, versilian-studios.com | | not reachable from the machine the registry was built on |

Not covered by any registry set with its own samples: driven electric guitars, synth
leads and pads, and sound effects (the General MIDI soundfonts have presets for all
three), plate reverbs, and a multi-microphone acoustic kit in WAV.
