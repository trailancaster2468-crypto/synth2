# Q4 findings: more models

Status (third run, 2026-11-10 11:50): all five priorities have results on disk. Sections are in the order they were
written, not in reading order. Where to look:
- Nishiren's terms of use: section 6 (6 full text, 6a the five questions, 6b its vocoder, 6c bottom line, 6d
  questions for the author). Alemon's and Aurelian's terms: section 7.
- Nishiren mode map for casting: section 8 (8a table, 8b modes by the numbers, 8c registers and casting sheet,
  8d a male cast by lowering the formants, 8e Screamo used the way its author says).
- Bank comparison: section 3 (Qixuan, Nishiren), 3b / 3c (Alemon, Aurelian), 3d (one table for all).
- Vocoder licence statements: sections 1 and 1b; vocoder swap test: section 9.
- Candidates: section 4. Loader requirements: section 5. Registry entries: section 10 and `q4/registry/*.json`.
- What is not verified: section 11.
All paths are under `$S/q4/` (S = the team scratch folder). Clips are in `q4/out/`.

## 1. Vocoder licence (first result)

Source: git clone of https://github.com/openvpi/vocoders (branch main, the source of
https://openvpi.github.io/vocoders/), file `index.md`, section "License", as of commit
97fe6ee7854ce81cee31d6c72c552fb500bdfd70 "Add license clarification", author yqzhishen,
2026-11-06 00:32:47 +0800. Local copy: q4/ref/vocoders.

> The model weights are licensed under the CC BY-NC-SA 4.0. Anyone who distributes the
> model weights should include a copy of the license, a notice informing that the models
> are provided by **OpenVPI** (or **DiffSinger Community**), and a link referring this
> page (or a complete contribution list).
>
> - Redistribution and integration into applications are permitted under the terms of
>   this license
> - If you want to ship this vocoder with commercial products, or automatically download
>   it from commercial products, do the following **before your users purchase or
>   subscribe**:
>   - Let your users know that this vocoder is a non-commercial component distributed
>     under CC BY-NC-SA 4.0
>   - Provide a link to the download page of this vocoder or its fine-tuned version
>   - Add proper attributions to OpenVPI or DiffSinger Community
> - Synthesizing or producing commercial work with this vocoder is not constrained by
>   the license

## 2. Downloads on disk (archives verified 01:35, second run)

All five archives pass `unzip -tq` (no errors). Total 1.56 GB of the 3 GB limit.

| folder | file | bytes | sha256 |
|---|---|---|---|
| dl/alemon | Alemon_CN.EN.JP_v3.5.zip | 526312536 | 47acc0112c236b51a230de71f0a6cf2c167d346b1eeed2853cd68a7a64958e19 |
| dl/aurelian | Aurelian_DS_v108.zip | 419136941 | fd2192a7c21f387ebfecfe471752476a347ad701916f7b2e12b31fa6b5ecf116 |
| dl/nishiren | Nishiren.Diffsinger.v2.1.zip | 623822790 | 138baf89820fc9084fc8b2313cb80b3f6ab0974009e2b3638646d7036f73a5c7 |
| dl/potluck-dsvocoder | dsvocoder.zip | 52705814 | 0bf6c58ba572225db8dbe773075fae21159435c9c6253c0ad01d8a76ced5e912 |
| dl/potluck-oudep | potluck_pc-nsf_lite.oudep | 52705734 | 3ad0ccb66f7453dbc8e128572b587e24d6ba883cca98f8d44eec528127c55ecf |

## 3. Bank comparison against Qixuan (appended as each bank is measured)

Method: `q4/run_case.sh NAME BANK SPEAKER TRANSPOSE CASES` = `q4/bank_spike.py` (copy of the spike's
`ds_spike.py` on `q4/dsinfer4.py`, a generic-bank copy of `dsinfer.py`; Synth2 planner and Synth2 pitch, bank
duration + variance models, steps 10, depth 0.6, the bank's OWN bundled vocoder, 2 threads, onnxruntime seed 1),
then `$S/tools/score.py --models tiny.en,base.en` and `$S/tools/voiceq.py`. Same material as the Qixuan demo
(`$S/ds1`): verse1 (88 words, MIDI 69-76), chorus1 (35 words, MIDI 69-77), plain (24 words, MIDI 61-69).
"t0" = our song's pitch; "t-N" = N semitones lower. Noise floor from q7: cpp +-0.2, verse1 base WER +-0.05,
chorus1 WER moves by +-0.2 between runs (35 words), tiny.en loops on verse1.

| bank / mode / pitch | case | cpp | hnr | voiced | WER tiny.en | WER base.en |
|---|---|---|---|---|---|---|
| Qixuan (the demo, `$S/ds1`) t0 | verse1 | 20.78 | 18.20 | 0.88 | 0.56 | 0.56 |
| Qixuan t0 | chorus1 | 20.99 | 21.40 | 0.93 | 0.29 | 0.26 |
| Qixuan t0 | plain | 24.96 | 20.20 | 0.90 | 0.08 | 0.04 |
| Nishiren v2.1 Standard t0 | verse1 | 25.72 | 18.71 | 0.94 | 0.51 | 0.48 |
| Nishiren v2.1 Standard t0 | chorus1 | 25.72 | 25.64 | 0.95 | 0.20 | 0.06 |
| Nishiren v2.1 Standard t0 | plain | 29.67 | 24.37 | 0.92 | 0.00 | 0.00 |

First result (01:38): Nishiren Standard is +4.7 to +4.9 dB cpp over Qixuan on all three cases at our song's
pitch, i.e. in the range of the "standard" modes of published banks (27-29) on plain and above every Qixuan
number measured so far (ceiling 22.4-24.0). Render cost under load 9-12: 52 s for 22.6 s sung (verse1).

Official release URLs of the three banks on disk, verified 01:42 with a ranged GET (HTTP 206, total size equal to
the file on disk): 
- https://github.com/Gardanana/Nishiren-AI-Diffsinger/releases/download/v2.1/Nishiren.Diffsinger.v2.1.zip (623822790)
- https://github.com/tuskinekinase/alemon_ds/releases/download/v3.5/Alemon_CN.EN.JP_v3.5.zip (526312536)
- https://github.com/AlexFloareaVT/Aurelian_DS/releases/download/v108/Aurelian_DS_v108.zip (419136941)

## 1b. Vocoder: what the publishers state (verified 01:45 against the local files)

1. OpenVPI PC-NSF-HiFiGAN 2025.02 (our default). Three sources, all OpenVPI's own:
   - `$S/models/installed/pc-nsf-hifigan/files/NOTICE.txt`: "All model weights in the DiffSinger Community
     Vocoder Project [...] including model weights in this directory, are provided by the OpenVPI Team, under the
     Attribution-NonCommercial-ShareAlike 4.0 International license."
   - `STATEMENTS.txt` (same folder): "No technical or legal restrictions are currently imposed on listed parties,
     as the vocoder is still licensed under CC BY-NC-SA 4.0"; "Reserve the right to apply further restrictions in
     case of persistent malicious acts".
   - openvpi/vocoders `index.md`, section "License", commit 97fe6ee "Add license clarification" (quoted in
     section 1 above; re-read 01:44, lines 104-113 of q4/ref/vocoders/index.md). The lines that answer our
     question, in OpenVPI's words: "Synthesizing or producing commercial work with this vocoder is not
     constrained by the license"; "Redistribution and integration into applications are permitted under the terms
     of this license"; a commercial product that ships or auto-downloads it must, before purchase, tell users it
     is "a non-commercial component distributed under CC BY-NC-SA 4.0", link the download page and attribute
     OpenVPI / DiffSinger Community.
   What stays unclear: the clarification is on the project page, not inside the release archive we installed
   (its NOTICE predates it); it is a statement by the licensor, not a change of licence text; whether a
   non-commercial open-source app that auto-downloads needs the same three notices is not said (they cost
   nothing, so do them anyway).
2. Potluck Vocoder "lite" (tigermeat), release v100, PC-NSF-HiFiGAN, same mel settings as ours (vocoder.yaml:
   44100 / hop 512 / win 2048 / 128 bins / fmin 40 / fmax 16000 / mel_base e / slaney / pitch_controllable), ONNX
   inputs `mel float[1,F,128]`, `f0 float[1,F]` -> `waveform` (same size as ours, 56616583 bytes).
   - https://github.com/mrtigermeat/potluck-vocoder/releases/download/v100/dsvocoder.zip (52705814 bytes,
     sha256 0bf6c58b...e912) and .../v100/potluck_pc-nsf_lite.oudep (52705734 bytes, sha256 3ad0ccb6...5ecf);
     both hold the same onnx (sha256 fb369505d9d42b68211c00a3939ef33af658c8ceee2287e4308705f06027f4d8). Both
     URLs answer 206 from this machine.
   - LICENSE (in the archive and identical in the repo, commit 92bebb1): "(Adapted from CC-BY 4.0 License)",
     "The product is free for any individual to use, distribute, fine-tune, or build upon. Commercial usage is
     allowed ONLY FOR DIFFSINGER APPLICATIONS."; "Attribution must be provided. You must give appropriate credit,
     provide a link to this license and indicate if changes were made."; "You may not use the output or weights
     of this model to train or assist training any type of Generative AI/ML model for any reason."; "Any attempts
     to distill, extract or reverse-engineer the weights of the model is strictly prohibited."; "All usage of the
     model must fall in line with United States law."
   - README: "The goal of this project was to create a vocoder whos weights are not bound to a non-commercial
     license, thus allowing unlimited commercial usage of DiffSinger models"; readme.txt in the archive: the lite
     version "does not perform well on vocals outside of it's dataset, but It does allow for full commercial
     usage. I highly recommend fine-tuning this model".
   - Sound test against our default and against Nishiren's bundled vocoder: section 9.
3. Vocoders bundled with the three banks (all 44.1k / hop 512 / 128 bins, `mel` + `f0` in, `waveform` out):
   Nishiren `gda_pc-hifigan.onnx` (sha256 dba2bbab...29a1; dsvocoder/credit.txt: "Finetuned vocoder trained on
   Potluck Vocoder from tigermeat"); Aurelian `tgm_hifigan_v107.onnx` (sha256 93449227...30f6; readme.txt
   "fine-tuned by tigermeat", no licence text in the bank); Alemon `nsf_hifigan.onnx` (sha256 2b938485...e8ac,
   NOT the 2025.02 OpenVPI file b297c0c7...155d although dsconfig.yaml names
   `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02`; origin not stated in the bank).

More rows (same method; 01:41-01:45):

| bank / mode / pitch | case | cpp | hnr | voiced | WER tiny.en | WER base.en |
|---|---|---|---|---|---|---|
| Nishiren v2.1 Power t0 | verse1 | 26.06 | 17.60 | 0.94 | 0.52 | 0.47 |
| Nishiren v2.1 Power t0 | chorus1 | 25.97 | 24.61 | 0.94 | 0.34 | 0.29 |
| Nishiren v2.1 Power t0 | plain | 30.08 | 22.94 | 0.92 | 0.04 | 0.00 |
| Nishiren v2.1 Standard t-5 (comfortable: README range G3-D5, verse then spans MIDI 64-71) | verse1 | 27.23 | 18.73 | 0.92 | 0.50 | 0.33 |
| Nishiren v2.1 Standard t-5 | chorus1 | 28.41 | 25.46 | 0.95 | 0.17 | 0.11 |

Transcripts read (Standard t0): tiny.en and base.en both loop on "pop up on a pop up" ("Papa Papa Papa", "Pup,
pup, pup, ..."), the rest of the verse is followed line by line; chorus1 base.en is word for word except
"the robot" for "a robot".

| bank / mode / pitch | case | cpp | hnr | voiced | WER tiny.en | WER base.en |
|---|---|---|---|---|---|---|
| Qixuan t-5 (same harness, control for the row above) | verse1 | 21.89 | 18.05 | 0.83 | 0.55 | 0.56 |
| Qixuan t-5 | chorus1 | 22.28 | 20.56 | 0.90 | 0.40 | 0.29 |

=> At the same pitch 5 semitones below our song, Nishiren Standard is +5.3 dB cpp (verse1) and +6.1 dB (chorus1)
over Qixuan, and base.en WER is 0.33 against 0.56 (verse1) and 0.11 against 0.29 (chorus1). Lowering the song
helps Nishiren (+1.5 / +2.7 cpp) more than Qixuan (+1.1 / +1.3).

## 5. What a generic loader must handle that Qixuan did not show

Seen in the three banks on disk (`q4/inspect_bank.py ROOT` prints phoneme sets and every ONNX signature) and
handled in `q4/dsinfer4.py` (all three banks load and sing with it):

1. Speaker embeddings. Every bank here needs `spk_embed`, also the single-speaker one (Aurelian).
   - `speakers:` in EACH dsconfig.yaml (root, dsdur, dsvariance, dspitch), entries are paths without extension
     relative to that dsconfig's folder and may contain sub-folders: root `dsmain/embeds/Standard`, dsdur
     `embeds/Standard` (Nishiren); `normal`, `falsetto` (Alemon); `dsmain/Aurelian` / `Aurelian` (Aurelian).
   - `<name>.emb` = raw little-endian float32, length = that model's `hidden_size`: Nishiren acoustic 384 but
     dur/variance/pitch 256; Alemon 384 everywhere; Aurelian 256. Take the length from the ONNX input, not from
     the root config.
   - Input shapes: acoustic `spk_embed float[1, n_frames, H]`, variance and pitch `[1, n_frames, H]`, dur
     `[1, n_tokens, H]`.
   - `character.yaml` `subbanks:` maps the voice-colour name to a suffix (`{color: Power, suffix: embeds/Power}`,
     Alemon: colour "1-Amber" -> suffix `normal`, "2-Airy" -> `falsetto`). OpenUtau resolves the suffix against
     `speakers` by exact match, then by trailing path segments, then falls back to speaker 0
     (DiffSingerSpeakerEmbedManager.getSpeakerIndexBySuffix). Aurelian has no subbanks.
   - Mixing modes = per-frame weighted sum of embeddings (OpenUtau voice-colour curves); not tested here.
2. Variance inputs differ per bank; build the feed from the session's own input list.
   - `tension` (Nishiren, Alemon): acoustic input, variance input, `tension_pred` output, variance `retake` is
     `bool[1, F, 3]` (one column per predicted variance). OpenUtau clamps tension to [-10, 10].
   - No variance at all (Aurelian): acoustic takes only tokens, languages, durations, f0, gender, velocity,
     spk_embed, depth, steps; the bank has no `dsvariance/` and no `dspitch/`.
   - `energy` (`use_energy_embed`, `predict_energy`): none of the three use it; OpenUtau feeds it like
     breathiness when configured.
3. `depth` is optional: Nishiren has `use_variable_depth: false` and its acoustic model has `steps` but no
   `depth` input. Older exports take `speedup` instead of `steps` (OpenUtau: largest factor of 1000 not above
   1000/steps); none of the three need it, `dsinfer4.py` passes it when asked.
4. Duration model inputs: Nishiren `dur.onnx` also takes `word_div` and `word_dur` (plus `spk_embed`); Qixuan
   and Alemon take only `encoder_out`, `x_masks`, `ph_midi`.
5. Paths in dsconfig.yaml are relative to the config's folder and may point into sub-folders
   (`acoustic: dsmain/acoustic.onnx`, `phonemes: dsmain/phonemes.json`). There is no fixed file naming.
6. Vocoder. All three banks bundle `dsvocoder/vocoder.yaml` + onnx. OpenUtau uses a bundled `dsvocoder/` first
   and only otherwise looks up `dsconfig.vocoder` among installed vocoders (DiffSingerSinger.getVocoder), so the
   `vocoder:` value can be a name that is never installed (`gda_pc-hifigan`, `dsvocoder`). Alemon names the
   OpenVPI 2025.02 vocoder and still bundles a different file. Aurelian's vocoder.yaml uses `fmin`/`fmax`
   instead of `mel_fmin`/`mel_fmax` and has no `win_size`/`fft_size`.
7. Language ids differ per bank: `en` is 1 in Qixuan, Alemon and Aurelian, 3 in Nishiren (15 languages). Always
   read `languages.json` of the sub-model in use.
8. Phoneme names. All three use `lang/phoneme` with global `SP AP`; every Synth2 English phoneme exists in all
   three as `en/<same name>`. Differences: globals `cl vf gs axh exh` (Nishiren), `cl vf trash ctrash vtrash
   trasj` (Aurelian), `CL` upper case (Alemon); English extras `q ll mm nn` (Nishiren, no `_r`), `q dr tr rr nn
   in oh M N ex exh` (Aurelian), `_r` (Alemon, same set as Qixuan). Japanese sets differ in the palatalised
   and devoiced symbols (`F I U` in Alemon, `I O U dz fy q` in Aurelian, no `by gy ky ...` in Nishiren).
9. Dictionaries: `dsdur/dsdict-en.yaml` has `symbols:` then `entries:` (Nishiren 38815 lines, Alemon 271632,
   Aurelian 225 lines = symbols only); `dictionary-en.txt` is a two-column tab file phoneme -> bank phoneme
   (Alemon, Aurelian: used by OpenUtau's G2P remap, e.g. `tr_t ch`, `dr_d zh`). Synth2 brings its own G2P, so
   only the `symbols` types (vowel / stop / ...) matter. YAML with a BOM, JSON may have one too (`utf-8-sig`).
10. Archive hygiene: root folder names with spaces and `&` ("Nishiren Diffsinger v2.1", "Alemon_CN&EN&JP_v3.5");
    Nishiren ships two phonemizer DLLs (18 MB) and 28 MB of art: never load or execute them; name comes from
    character.yaml or character.txt (`name=`).

## 6. Nishiren's terms of use (third run, 11:10; FOUND, quoted from the author's own repo)

Where: the archive has no licence file (README.txt, character.txt, dsvocoder/credit.txt only), but the author's
release repo has `terms-of-service.md`, and its README says, right under the download link: "By downloading this
voicebank you agree to their [Terms of Service](/terms-of-service.md)". Source read: full `git clone` of
https://github.com/Gardanana/Nishiren-AI-Diffsinger, branch main, HEAD 52479e2d5537ba7fed385189b944b75ac2f54415
(2026-11-06 00:44:21 -0500, author Gardanana, "Updated terms related to other AI usages"); local copy
`q4/ref/Gardanana_Nishiren-AI-Diffsinger/terms-of-service.md` (2491 bytes, sha256 39ec256b...5760).
Raw URL: https://raw.githubusercontent.com/Gardanana/Nishiren-AI-Diffsinger/main/terms-of-service.md
The file's first line: "BY DOWNLOADING AND USING CONTENT DISTRIBUTED OF THIS CHARACTER AND VOICE, YOU AGREE TO
THESE TERMS". It ends "These terms of use were made with the help of TUBS (https://tools.tubs.wtf/vbtougen/)".

Full text of the clauses (verbatim, bullets as in the file):

- ATTRIBUTION: "When publishing work using this voicebank, you are required to state the name of this
  voicebank." / "It is not necessary to state the name of this voicebank's author, but it would be greatly
  appreciated." / "You cannot call the voicebank by any other name, nor can you attribute the voicebank to any
  other author."
- DISCLAIMER: "Gardanana holds no responsibility to any incidents, damage, or loss by the user from downloading
  or using the voicebank or character." (and the same for third parties).
- COMMERCIAL USE: "Contact Gardanana for individual approval for commercial use of any voicebanks and voicebank
  characters." / "Non-commericial uses do not require approval." / "You cannot use this voicebank or any included
  assets in projects involving non-fungible tokens (NFTs) or cryptocurrency."
- USAGE CONTENT: "Seek author permission to create sexual content." / "Seek author permission to create violent
  content." / "Do not create political content." / "Do not create religious content." / "Do not use with
  unethically AI generated lyrics, music, art, or videos." / "The following types of content are unconditionally
  prohibited: - Criminal or illegal content - Explicit underage sexual content - Bigotry and hateful content" /
  "Gardanana reserves the right to request takedown of published content at any time, and users are expected to
  comply."
- REDISTRIBUTION: "Any current or discontinued voicebank may only be redistributed as the entire unedited
  voicebank. You must correctly attribute the voicebank's name and author."
- EDITING: "You can only use this voicebank in the program it was intended for or similar programs that do not
  require any edits to the voicebank. Porting to any other program requires author approval." / "Do not use
  source or generated audio files as data for machine learning without author approval."
- CHARACTER: "Published work using the voicebank may or may not include visual depictions of the voicebank
  character." / "Existing artwork of the voicebank character can be used with proper attribution to the
  individual artists."

History of the file (`git log -- terms-of-service.md`): created 2024-05-16 (5651d64); 2025-10-15 (88775aa)
spelling of the author's name and removal of "Violations of UTAU's Usage Policy" from the prohibited list;
2026-11-06 (1aa02e3, 94c3fdf, 52479e2: three commits, FOUR DAYS AGO) added "Do not use with unethically AI
generated lyrics or music." and then widened it to "lyrics, music, art, or videos". The tag `v2.1` points at
bd2a211 (2026-06-02), i.e. the terms as tagged with the release did NOT yet contain the AI clause; the README's
link goes to the file on main, so the current text is the one a downloader is told to accept.

### 6a. What Gardanana's terms say on the five questions (the publisher's words; no legal reading added)

| question | what the terms state | what stays unclear |
|---|---|---|
| Commercial use of the audio | "Contact Gardanana for individual approval for commercial use of any voicebanks and voicebank characters." "Non-commericial uses do not require approval." No NFT / cryptocurrency projects. | "Commercial use" is not defined (monetised video? paid product? an app that offers the voice?). Approval is per case ("individual approval"); nothing is granted in advance. |
| Credit | "you are required to state the name of this voicebank"; author's name "not necessary ... but ... greatly appreciated"; "You cannot call the voicebank by any other name". | Exact form of the name: character.txt says "Nishiren DS v2.1", README "Nishiren Gard (西蓮ガルド)". Whether a Synth2 voice preset with its own name ("Lead", "Alien") counts as calling the voicebank by another name is not addressed; the credit line has to state the bank's name either way. |
| Forbidden content | Never: "Criminal or illegal content", "Explicit underage sexual content", "Bigotry and hateful content". "Do not create political content." "Do not create religious content." Sexual and violent content: "Seek author permission". "Do not use with unethically AI generated lyrics, music, art, or videos." The author may "request takedown of published content at any time, and users are expected to comply". | "unethically AI generated" is not defined, and the clause was added on 2026-11-06. Synth2 is an AI studio whose lyrics and scores are routinely written by a language model (the demo song is), so this clause meets Synth2's main use head on; only the author can say whether that use is meant. |
| Automatic download at install time | Not addressed. The terms bind "BY DOWNLOADING AND USING"; the repo README: "By downloading this voicebank you agree to their Terms of Service". | Whether a program may fetch the release zip for the user. What is stated implies the person must be shown the terms before the download starts (the archive itself carries no copy of them). |
| Redistribution | "Any current or discontinued voicebank may only be redistributed as the entire unedited voicebank. You must correctly attribute the voicebank's name and author." | Fetching the author's own release asset from GitHub is not addressed as such. "Entire unedited" is the one stated condition for passing the bank on. |
| Use in another program (not asked, but it applies to us) | "You can only use this voicebank in the program it was intended for or similar programs that do not require any edits to the voicebank. Porting to any other program requires author approval." | Synth2 reads the unmodified OpenUtau files (no conversion, no edits), but it is not OpenUtau. Whether it counts as a "similar program" is the author's call. |
| Machine learning | "Do not use source or generated audio files as data for machine learning without author approval." | Scoring clips with a speech recogniser is inference, not training; the terms do not distinguish. |

Other sources checked:
- Release notes of v2.1 (https://github.com/Gardanana/Nishiren-AI-Diffsinger/releases/tag/v2.1, read through the
  web-fetch tool, summarised by it, not verbatim): no licence statement. Assets: `Nishiren.Diffsinger.v2.1.zip`
  (sha256 shown by GitHub 138baf89...a5c7 = our file) and `Nishiren.Diffsinger.v2.1.Legacy.zip` (sha256
  4a78a1b8...052d, "for editors NOT updated since October 15, 2026": the main zip needs the editor's new duration
  model support, i.e. the `word_div`/`word_dur`/`spk_embed` inputs of `dur.onnx` that `dsinfer4.py` feeds).
- The author's site: NOT REACHABLE from this machine (https://gardanana.nekoweb.org/ and
  https://gardanana.neocities.org/: proxy answers 403 to CONNECT; the web-fetch tool cannot resolve them). A web
  search says the full terms are also at gardanana.neocities.org/terms and that the UTAU wiki page lists
  "commercial use of voicebank: permission required", "R-18: permission required" (secondary, not read: the wiki
  host is blocked too). So: the site's copy of the terms is UNVERIFIED; it may differ from the repo's.
- `$S/ref/svs-index` (openutau/svs-index, commit ac0126e, 2026-10-22), `data/singers/n.json`, id `nishiren`:
  owner and author Gardanana, homepage the nekoweb site, download page the GitHub repo; the index carries no
  licence field.
- Archive README.txt: no terms. It lists "Private datasets trained with: Nishiren - Gardanana, Takeo Kasai -
  enlilgen, Yamada Riku - henby, Link Akasaka - chiru6, Kao Nashi - TaliaTheWitch, Kihwan - K" and the repo README
  lists "Multispeaker public datasets: PJS, NIT070, OfutonP, TIGER, Hitsune Kumi, TREi, M4Singer, CSD, Mechol,
  Printto TH Dataset, Tiny Tagalog Dataset, Gianloop's datasets, Namine Ryoku, BRAPINHA, Petit Millefeuille,
  SSPCD, Daichi Yoshida, beatboxset1". The author does not restate those datasets' own licences; not checked here.

### 6b. The vocoder Nishiren bundles (`dsvocoder/gda_pc-hifigan.onnx`)

- What the bank says: `dsvocoder/credit.txt`, whole file: "Finetuned vocoder trained on Potluck Vocoder from
  tigermeat" + "https://github.com/mrtigermeat/potluck-vocoder". No licence text of its own. The repo changelog:
  "New vocoder finetuned on Potluck Vocoder".
- What Potluck's publisher states (LICENSE in https://github.com/mrtigermeat/potluck-vocoder, HEAD 92bebb1,
  2026-03-19, unchanged at 11:15 today; licensor "Tyler K.M. / tigermeat", "(Adapted from CC-BY 4.0 License)"):
  "The product is free for any individual to use, distribute, fine-tune, or build upon. Commercial usage is
  allowed ONLY FOR DIFFSINGER APPLICATIONS. The vocoder can be used with other systems, but only for personal use
  if not for DiffSinger." Allowances: "You may share, copy, and/or redistribute the material in any medium or
  format." "You may adapt, remix, transform and build upon the material." Terms: "Attribution must be provided.
  You must give appropriate credit, provide a link to this license and indicate if changes were made."
  "Commercial use is only allowed for use with DiffSinger models." "You may not use the output or weights of this
  model to train or assist training any type of Generative AI/ML model for any reason." "Any attempts to distill,
  extract or reverse-engineer the weights of the model is strictly prohibited." "All usage of the model must fall
  in line with United States law." "There are no additional restrictions."
  README: the goal was "a vocoder whos weights are not bound to a non-commercial license, thus allowing unlimited
  commercial usage of DiffSinger models (running under the assumption that DS Model weights are also commercial
  friendly)"; "Please ensure you read the license before using, training and/or distributing your own Potluck FT
  model."
- Unclear: Gardanana publishes no separate terms for the fine-tuned file, so it is either under the bank's Terms
  of Service (as "content distributed of this character and voice": commercial use by individual approval) or
  under Potluck's licence as a derivative, or both; neither publisher says. Potluck's licence has no share-alike
  clause and does not say what licence a fine-tune carries. Whether Synth2 is a "DiffSinger application" in
  tigermeat's sense (it runs DiffSinger models, but is not the DiffSinger project or OpenUtau) is not defined.
  The bank's terms are the stricter of the two on commercial use in any case.

### 6c. Bottom line for "can Nishiren be the default" (facts only)

- Terms FOUND and quoted; they are the author's own, current as of 2026-11-06.
- Stated plainly by the author: non-commercial use needs no approval; credit = the voicebank's name; commercial
  use needs the author's individual approval; redistribution only whole and unedited.
- Three points only the author can settle, each touching a default role in Synth2: (1) the new clause against
  "unethically AI generated lyrics, music, art, or videos" versus an AI studio that writes lyrics and scores with
  a language model; (2) "the program it was intended for or similar programs ... Porting to any other program
  requires author approval" versus loading the files in Synth2; (3) automatic download by an installer. Contact
  given in the README: see the README of the release repository.
- Compared with Qixuan (current default): its installed manifest summarises its terms as "Works made with this
  voicebank may be used commercially or non-commercially"; Nishiren's terms require approval for commercial use.

## 3b. Alemon and Aurelian, first rows (rendered 01:50-01:55 by the interrupted run, harvested 11:12 from work/)

Same method as section 3 (each bank's own bundled vocoder, steps 10, seed 1). `python3 -I q4/harvest.py NAME...`
prints these rows from `work/NAME/score.json` + `voiceq.json`. hf = energy above 5 kHz against below, dB (higher
= airier / brighter top); lvl_sd = level spread of voiced frames, dB.

| run | case | cpp | hnr | voiced | hf | lvl_sd | WER tiny.en | WER base.en |
|---|---|---|---|---|---|---|---|---|
| alemon_normal_t0 | verse1 | 23.67 | 17.64 | 0.90 | -30.61 | 5.25 | 0.75 | 1.00 |
| alemon_normal_t0 | chorus1 | 23.76 | 22.05 | 0.93 | -31.22 | 4.42 | 0.71 | 0.77 |
| alemon_normal_t0 | plain | 28.73 | 21.17 | 0.91 | -30.47 | 4.99 | 0.17 | 0.21 |
| alemon_normal_t-12 | verse1 | 27.79 | 17.23 | 0.92 | -33.46 | 5.40 | 0.43 | 0.35 |
| alemon_normal_t-12 | chorus1 | 29.26 | 21.46 | 0.93 | -32.71 | 5.39 | 0.17 | 0.14 |
| alemon_normal_t-12 | plain | 29.70 | 19.13 | 0.91 | -34.53 | 4.45 | 0.08 | 0.08 |
| aurelian_t0 | verse1 | 24.97 | 17.57 | 0.93 | -29.10 | 6.06 | 1.00 | 1.00 |
| aurelian_t0 | chorus1 | 24.59 | 21.72 | 0.96 | -29.68 | 5.82 | 0.94 | 0.54 |
| aurelian_t0 | plain | 27.55 | 18.70 | 0.92 | -30.23 | 4.89 | 0.33 | 0.21 |
| aurelian_t-12 | verse1 | 26.60 | 14.76 | 0.87 | -29.16 | 5.19 | 0.74 | 1.00 |
| aurelian_t-12 | chorus1 | 27.49 | 18.68 | 0.91 | -30.09 | 4.80 | 0.60 | 0.66 |
| aurelian_t-12 | plain | 27.87 | 17.95 | 0.89 | -32.10 | 4.42 | 0.29 | 0.25 |

First reading: at our song's pitch (MIDI 69-77) both are far worse than Nishiren on the song (Alemon base.en
1.00 / 0.77, Aurelian 1.00 / 0.54 on verse1 / chorus1); an octave lower Alemon recovers (0.35 / 0.14, cpp 27.8 /
29.3) while Aurelian does not (1.00 / 0.66). Falsetto, the -5 rows and the transcripts: section 3c; all banks in
one table: section 3d.

## 7. Terms of the other two banks on disk (11:16; publishers' words, Chinese translated by me)

### Alemon v3.5 (tuskinekinase/alemon_ds, HEAD b57bb90, 2026-08-18)
- The archive carries no licence file. The repo root has `LICENSE` (20851 bytes: the unmodified legal code of
  "Attribution-NonCommercial-ShareAlike 4.0 International") and `LICENSE_zh` (7294 bytes, "声库使用须知", the
  bank's own terms, Chinese only; "使用本声库即代表您理解并同意下述所有条款" = using the bank means you accept
  all clauses). README and release notes do not mention either file. Which of the two governs when they differ is
  not stated.
- LICENSE_zh, the clauses that matter to us:
  - Credit when publishing: "需在显著位置注明该作品使用 AI（人工智能）技术合成" (state prominently that it is AI
    synthesised) and "需在显著位置标示声库名称“Alemon”（或“アレモン”、“柠檬君”…）与合成技术名称“DiffSinger”"
    (show the bank name "Alemon" and the technology name "DiffSinger").
  - Free without contacting the managers, among others: "使用既有的编辑器与推理引擎对模型进行推理操作，并保存
    与分发其输出" (run inference with existing editors and engines, keep and distribute the output) and
    "编写与开发新的推理代码和程序，以支持模型所包含的功能" (write new inference code and programs that support
    the model's functions). This is the only one of the three banks that states a third-party program is welcome.
  - Needs case-by-case permission: "所有的线下同人活动、商业活动（无论有偿无偿）" (all offline fan events and
    COMMERCIAL activities, paid or not); "用于给Alemon之外的角色配音等，将音源作为其他人的声音的使用行为" (using
    the voice as the voice of a character other than Alemon); "其他本协议未明确规定的使用行为" (anything not
    listed).
  - Forbidden: passing the output off as the voice provider; crediting under another name; illegal / offensive
    content; "具有争议或误导性成分的政治宣传、宗教布教" (contentious or misleading political propaganda, religious
    proselytising); output as training data for voice synthesis / conversion; "逆向、转换、修改、微调、拆分、合并
    模型权重文件" (reverse, convert, modify, fine-tune, split or merge the weight files).
  - Redistribution ("二次分发", which the terms define to include "向他人提供本声库的使用接口或网络服务" =
    offering an interface or network service to the bank): allowed if complete ("不得单独分发…组成部件"),
    unmodified terms / weights / art, free ("不得以直接或间接的方式售卖本声库或以付费订阅等形式提供其网络合成服务"),
    and the recipient is told it is free and to read the terms first; inside mainland China the deep-synthesis
    regulations apply.
  - Automatic download: not addressed; the recipient "must be clearly reminded to read these terms before use".
- Unclear: the terms file is not in the archive (so a redistributed archive cannot "keep the terms unmodified"
  unless LICENSE_zh is added from the repo); there is no English text of the bank's own terms.

### Aurelian Silva v108 (AlexFloareaVT/Aurelian_DS, HEAD ed48491, 2025-01-05)
- NO TERMS FOUND. Repo: README.md only (character sheet, phoneme notes, data credits), no LICENSE file; archive:
  character.txt is empty, `dsvocoder/readme.txt` = "tgm_hifigan_v107 | fine-tuned by tigermeat" + a list of data
  credits; release notes of v108 (read through the web-fetch tool): changes only, no licence.
- README credits: "trained alongside" Millefeuille (French), Namine Criss Spanish Dataset, PJS Corpus, Thai
  datasets, "Various datasets by TigerMeat", "Project AI❤dol Public English Dataset"; vocoder "tgm_hifigan v107
  ... trained by TigerMeat" (tigermeat publishes no licence for that file in this bank; it predates the Potluck
  vocoder and its licence).
- With no statement from the publisher, nothing is granted in writing: not commercial use, not redistribution,
  not automatic download. It can be a user-supplied bank (`add_local`), not a registry entry with a download URL,
  until the author answers.

## 8. Nishiren mode map (done 11:45)

Method: `q4/modes.sh "0 -5 -12" "Standard Power Soft Sweet Emotional Screamo 2P"` renders plain + chorus1 for each
mode and transposition into `work/nishiren_<mode>_t<N>/` (`q4/bank_spike.py` on `q4/dsinfer4.py`: Synth2 planner and
pitch, the bank's duration and variance models, steps 10, depth 0.6, the bank's own vocoder, 2 threads, onnxruntime
seed 1), scores tiny.en + base.en (`$S/tools/score.py`), runs `$S/tools/voiceq.py`; rows in `q4/modes.tsv`.
`q4/queue3.sh` renders chorus1 a second time with seed 2 (`work/nishiren_<mode>_t<N>b/`, rows in `q4/modes2.tsv`).
`python3 -I q4/timbre.py DIR` gives the colour numbers (centroid Hz; press = 2-5 kHz re 0.1-1 kHz, dB),
`q4/var_stats.py` what the bank's own variance model predicts on sung frames (tension; breathiness and voicing in dB),
`python3 -I q4/modemap.py` prints this table. Load 1-4 while rendering (another engineer on the machine);
render about 10 s for plain (11.8 s sung) and 18 s for chorus1 (20.7 s sung).

Clips for every row: `q4/out/nishiren-<mode>_<case>.mp3` (the song's pitch) and `..._t-5.mp3`, `..._t-12.mp3`: take 1,
160k mp3, level-matched to -16 LUFS (the Standard / Power mp3 of the first run were re-encoded the same way; their
wav and verse1 files are untouched).

### 8a. Every mode, case and register (WER = tiny.en / base.en; take 2 = chorus1 with seed 2)

| mode | case | sung range | cpp | hnr | hf | centroid Hz | press dB | WER tiny / base (take 1) | WER take 2 | tension | breath | voicing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Standard | chorus1 t0 | A4-F5 | 25.72 | 25.64 | -30.24 | 779 | -21.3 | 0.20 / 0.06 | 0.37 / 0.14 | -0.48 | -45.7 | -22.47 |
| Standard | chorus1 t-5 | E4-C5 | 28.06 | 25.16 | -28.91 | 816 | -18.0 | 0.23 / 0.11 | 0.23 / 0.11 | 0.41 | -46.25 | -22.98 |
| Standard | chorus1 t-12 | A3-F4 | 29.98 | 23.83 | -29.13 | 716 | -17.5 | 0.06 / 0.09 | 0.17 / 0.09 | 1.22 | -46.72 | -23.41 |
| Standard | plain t0 | C#4-A4 | 29.67 | 24.37 | -28.41 | 661 | -17.8 | 0.00 / 0.00 | - | -0.04 | -49.08 | -22.64 |
| Standard | plain t-5 | G#3-E4 | 29.93 | 23.29 | -30.92 | 505 | -20.1 | 0.04 / 0.04 | - | 0.34 | -49.34 | -22.9 |
| Standard | plain t-12 | C#3-A3 | 29.69 | 23.19 | -34.48 | 361 | -23.2 | 0.04 / 0.04 | - | 0.91 | -50.7 | -26.35 |
| Power | chorus1 t0 | A4-F5 | 25.97 | 24.61 | -29.35 | 890 | -18.0 | 0.34 / 0.29 | 0.43 / 0.29 | 0.03 | -46.7 | -23.87 |
| Power | chorus1 t-5 | E4-C5 | 28.95 | 23.56 | -28.21 | 915 | -15.6 | 0.09 / 0.14 | 0.23 / 0.17 | 1.5 | -47.73 | -24.5 |
| Power | chorus1 t-12 | A3-F4 | 30.22 | 22.98 | -28.82 | 804 | -16.5 | 0.17 / 0.14 | 0.14 / 0.14 | 2.47 | -47.36 | -25.78 |
| Power | plain t0 | C#4-A4 | 30.08 | 22.94 | -29.05 | 779 | -15.7 | 0.04 / 0.00 | - | 0.75 | -50.44 | -25.06 |
| Power | plain t-5 | G#3-E4 | 30.38 | 21.79 | -31.20 | 640 | -18.0 | 0.08 / 0.00 | - | 1.28 | -50.97 | -26.11 |
| Power | plain t-12 | C#3-A3 | 29.48 | 19.11 | -31.45 | 569 | -19.1 | 0.17 / 0.08 | - | 2.53 | -49.76 | -30.64 |
| Soft | chorus1 t0 | A4-F5 | 24.27 | 25.04 | -30.45 | 743 | -23.0 | 0.03 / 0.11 | 0.17 / 0.14 | -0.62 | -46.19 | -25.04 |
| Soft | chorus1 t-5 | E4-C5 | 26.22 | 25.30 | -30.71 | 684 | -22.0 | 0.14 / 0.06 | 0.11 / 0.11 | -0.12 | -46.83 | -25.15 |
| Soft | chorus1 t-12 | A3-F4 | 27.29 | 24.87 | -30.96 | 530 | -21.1 | 0.20 / 0.09 | 0.20 / 0.20 | 0.22 | -47.21 | -26.02 |
| Soft | plain t0 | C#4-A4 | 27.29 | 26.05 | -32.09 | 481 | -23.1 | 0.08 / 0.08 | - | -0.84 | -49.62 | -24.82 |
| Soft | plain t-5 | G#3-E4 | 27.28 | 26.11 | -33.40 | 352 | -24.9 | 0.08 / 0.00 | - | -0.72 | -49.79 | -25.01 |
| Soft | plain t-12 | C#3-A3 | 27.18 | 25.98 | -34.94 | 246 | -27.0 | 0.04 / 0.00 | - | -0.22 | -51.28 | -28.18 |
| Sweet | chorus1 t0 | A4-F5 | 25.79 | 24.80 | -30.11 | 843 | -17.4 | 0.40 / 0.11 | 0.11 / 0.09 | -0.25 | -45.99 | -22.84 |
| Sweet | chorus1 t-5 | E4-C5 | 28.36 | 24.08 | -29.35 | 935 | -14.8 | 0.17 / 0.11 | 0.23 / 0.14 | 0.74 | -46.68 | -23.54 |
| Sweet | chorus1 t-12 | A3-F4 | 30.59 | 23.14 | -29.64 | 818 | -14.4 | 0.09 / 0.09 | 0.14 / 0.11 | 1.61 | -47.44 | -23.96 |
| Sweet | plain t0 | C#4-A4 | 30.32 | 23.16 | -28.86 | 792 | -15.3 | 0.00 / 0.00 | - | 0.38 | -49.68 | -23.68 |
| Sweet | plain t-5 | G#3-E4 | 30.48 | 22.47 | -30.69 | 609 | -17.4 | 0.04 / 0.00 | - | 0.58 | -49.97 | -24.1 |
| Sweet | plain t-12 | C#3-A3 | 30.41 | 22.45 | -34.74 | 407 | -21.6 | 0.00 / 0.00 | - | 0.86 | -50.85 | -27.69 |
| Emotional | chorus1 t0 | A4-F5 | 25.78 | 25.34 | -29.44 | 764 | -20.5 | 0.37 / 0.11 | 0.43 / 0.17 | -0.28 | -46.57 | -22.27 |
| Emotional | chorus1 t-5 | E4-C5 | 28.58 | 24.49 | -28.64 | 823 | -17.5 | 0.11 / 0.14 | 0.17 / 0.14 | 0.99 | -47.55 | -23.11 |
| Emotional | chorus1 t-12 | A3-F4 | 30.56 | 22.73 | -28.69 | 748 | -17.5 | 0.11 / 0.09 | 0.14 / 0.09 | 2.34 | -48.28 | -23.92 |
| Emotional | plain t0 | C#4-A4 | 30.04 | 23.53 | -28.93 | 700 | -17.1 | 0.08 / 0.04 | - | 0.55 | -50.34 | -24.07 |
| Emotional | plain t-5 | G#3-E4 | 30.38 | 22.17 | -30.31 | 572 | -18.2 | 0.04 / 0.00 | - | 1.22 | -50.47 | -24.6 |
| Emotional | plain t-12 | C#3-A3 | 30.63 | 20.72 | -32.32 | 484 | -19.1 | 0.04 / 0.04 | - | 2.35 | -51.5 | -28.69 |
| Screamo | chorus1 t0 | A4-F5 | 23.62 | 25.40 | -27.12 | 697 | -25.4 | 0.66 / 0.23 | 0.71 / 0.37 | -0.8 | -41.44 | -20.0 |
| Screamo | chorus1 t-5 | E4-C5 | 25.50 | 24.70 | -26.88 | 629 | -21.6 | 0.37 / 0.20 | 0.34 / 0.34 | 0.04 | -42.07 | -20.98 |
| Screamo | chorus1 t-12 | A3-F4 | 24.66 | 21.79 | -26.60 | 556 | -22.4 | 0.34 / 0.14 | 0.40 / 0.29 | 1.34 | -41.31 | -22.29 |
| Screamo | plain t0 | C#4-A4 | 28.56 | 25.43 | -29.38 | 523 | -21.1 | 0.08 / 0.00 | - | -0.03 | -45.7 | -20.19 |
| Screamo | plain t-5 | G#3-E4 | 26.48 | 23.42 | -29.13 | 434 | -21.9 | 0.04 / 0.04 | - | 0.39 | -44.5 | -20.7 |
| Screamo | plain t-12 | C#3-A3 | 24.21 | 21.98 | -29.16 | 346 | -26.7 | 0.04 / 0.00 | - | 1.61 | -43.19 | -21.85 |
| 2P | chorus1 t0 | A4-F5 | 26.13 | 24.10 | -30.58 | 936 | -14.6 | 0.34 / 0.06 | 0.29 / 0.17 | 0.21 | -45.95 | -23.18 |
| 2P | chorus1 t-5 | E4-C5 | 28.67 | 23.04 | -28.63 | 1001 | -12.5 | 0.17 / 0.14 | 0.17 / 0.11 | 1.46 | -46.88 | -24.38 |
| 2P | chorus1 t-12 | A3-F4 | 30.32 | 21.42 | -30.09 | 977 | -12.0 | 0.11 / 0.09 | 0.14 / 0.11 | 2.63 | -46.8 | -25.18 |
| 2P | plain t0 | C#4-A4 | 30.33 | 21.94 | -28.73 | 911 | -13.8 | 0.04 / 0.04 | - | 1.18 | -50.02 | -25.45 |
| 2P | plain t-5 | G#3-E4 | 30.74 | 20.67 | -31.13 | 784 | -15.1 | 0.00 / 0.00 | - | 1.45 | -49.95 | -25.96 |
| 2P | plain t-12 | C#3-A3 | 31.08 | 19.16 | -33.63 | 634 | -15.4 | 0.00 / 0.00 | - | 1.89 | -48.88 | -29.39 |

### 8b. The modes by the numbers (take 1)

Each mode against Standard at the same case and pitch, mean over the six register points
(`python3 -I q4/modedelta.py`): cpp dB, hnr dB, hf dB, centroid %, press dB, air dB, the bank's predicted tension,
breathiness dB, voicing dB, level spread dB.

| mode | cpp | hnr | hf | centroid | press | air | tension | breath | voicing | lvl_sd |
|---|---|---|---|---|---|---|---|---|---|---|
| Standard | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Power | +0.34 | -1.75 | +0.67 | +23% | +2.5 | +0.5 | +1.03 | -0.86 | -2.54 | +0.80 |
| Soft | -2.25 | +1.31 | -1.74 | -23% | -3.9 | -1.7 | -0.78 | -0.52 | -2.24 | -0.08 |
| Sweet | +0.48 | -0.90 | -0.22 | +15% | +2.8 | -0.1 | +0.26 | -0.47 | -0.84 | -0.02 |
| Emotional | +0.49 | -1.08 | +0.63 | +9% | +1.4 | +0.6 | +0.80 | -1.15 | -0.99 | +0.52 |
| Screamo | -3.34 | -0.46 | +2.30 | -16% | -3.5 | +2.2 | +0.03 | +4.93 | +2.46 | +0.14 |
| 2P | +0.70 | -2.52 | -0.12 | +41% | +5.8 | -1.0 | +1.08 | -0.12 | -2.13 | +0.23 |

One line each (numbers first, the author's README words in quotes):
- Standard: the neutral reference, the cleanest harmonics of the bright modes (hnr 25.6 on the chorus). "Standard tone".
- Power: bright and pressed (+23 % centroid, +2.5 dB at 2-5 kHz, tension +1.0), rougher (hnr -1.8), wider
  dynamics. "Stronger and aggressive tone".
- Soft: dark and covered (-23 % centroid, -3.9 dB at 2-5 kHz, tension -0.8), least ring (cpp -2.3) but the cleanest
  (hnr +1.3). Not breathy by the numbers (breath -0.5 dB, air -1.7 dB): gentle, not whispery. "Darker and gentler".
- Sweet: bright and forward (+15 %, +2.8 dB), otherwise Standard. "Upbeat and nasal tone".
- Emotional: the closest to Standard (+9 %, +1.4 dB), a little more tension (+0.8). "Passionate and dramatic".
- Screamo: the only noisy mode: breathiness +4.9 dB, air +2.2 dB, cpp -3.3 (the least clear), darker (-16 %).
  "Breathy and growly ... usually requires variance parameter editing".
- 2P: by far the brightest and most forward (+41 % centroid, +5.8 dB at 2-5 kHz), the roughest harmonics (hnr -2.5):
  a nasal character voice. "Nasal and strongly voice acted" (a second character, Kayama Gard).

### 8c. Where each mode is clearest, and a casting sheet (11:46)

How far the numbers carry (21 pairs of takes): cpp differs by 0.20 dB on average between the two takes (largest
0.66); chorus1 WER differs by 0.07 (tiny.en, largest 0.29) and 0.05 (base.en, largest 0.15), e.g. Standard at the
song's pitch 0.20 / 0.06 then 0.37 / 0.14, Sweet 0.40 / 0.11 then 0.11 / 0.09. So a WER gap under about 0.15
between two modes is noise. cpp also falls with pitch for every mode (about 30 at or below A4, 28 around E4-C5, 26
at A4-F5), so compare modes at the same pitch. plain is slow and easy: every mode is at 0.00-0.17 / 0.00-0.08 there.
Nobody has listened to these clips: the descriptions are the numbers plus the author's README words.

Mean chorus1 WER over both takes and both recognisers (4 numbers), and cpp of take 1:

| mode | A4-F5 (song's pitch) | E4-C5 (5 lower) | A3-F4 (12 lower) | clearest register | avoid |
|---|---|---|---|---|---|
| Standard | 0.19, cpp 25.7 | 0.17, cpp 28.1 | 0.10, cpp 30.0 | A3-C5; usable to F5 | - |
| Power | 0.34, cpp 26.0 | 0.16, cpp 28.9 | 0.15, cpp 30.2 | G#3-C5 | above C5 (base.en 0.29 in both takes); below A3 (hnr 19.1, plain WER 0.17 / 0.08) |
| Soft | 0.11, cpp 24.3 | 0.11, cpp 26.2 | 0.17, cpp 27.3 | C#4-F5, the one mode that does not gain from lowering | low and fast: centroid drops to 246-530 Hz |
| Sweet | 0.18, cpp 25.8 | 0.16, cpp 28.4 | 0.11, cpp 30.6 | A3-C5; usable to F5 | - |
| Emotional | 0.27, cpp 25.8 | 0.14, cpp 28.6 | 0.11, cpp 30.6 | A3-C5 | above C5 on fast lines |
| Screamo | 0.49, cpp 23.6 | 0.31, cpp 25.5 | 0.29, cpp 24.7 | slow lines around C#4-A4 only (plain: cpp 28.6, WER 0.08 / 0.00) | fast lines anywhere; low notes (plain cpp 26.5 at G#3-E4, 24.2 at C#3-A3) |
| 2P | 0.21, cpp 26.1 | 0.15, cpp 28.7 | 0.11, cpp 30.3 | C#3-C5, the strongest low end (plain at C#3-A3: cpp 31.1, WER 0.00 / 0.00) | - |

The author's stated range is G3-D5. Our lead (verse A4-E5, chorus A4-F5) sits at and above its top. Five semitones
down every mode gains 1.9-3.0 dB cpp (part of that is the measure itself, see above); in words the gain is larger
than the noise only for Power (0.34 -> 0.16), Emotional (0.27 -> 0.14) and Screamo (0.49 -> 0.31). Standard, Sweet
and 2P are understood about as well at the song's pitch as below it; Soft does not gain at all.

Casting by the numbers (mode, pitch):
- Lead that must be understood at the song's pitch: Standard or Sweet (0.19 / 0.18); Soft for a gentle lead (0.11,
  but 1.5 dB less ring and 23 % darker).
- Belted chorus: Power only if the part is at or below C5; at A4-F5 it is the second least intelligible mode.
- Quiet verse, backing pad, doubles: Soft (cleanest harmonics, hnr +1.3 dB over Standard; 3.9 dB less energy at
  2-5 kHz, so it sits behind a lead).
- Second character / duet partner / low harmony: 2P (41 % brighter, +5.8 dB at 2-5 kHz: it will not blend with
  Standard, it stands apart), also the best choice for a part an octave below the lead.
- Ballad colour: Emotional, a shade brighter and tenser than Standard; keep it at or below C5.
- Effect only: Screamo (breathiness +4.9 dB, cpp -3.3); the README says it needs tension and breathiness raised by
  hand; the map rows have all offsets at 0, section 8e has it with them raised.
- Male role: none of the seven is male. An octave down Standard, Sweet, Emotional and 2P all stay clear (chorus
  WER 0.10-0.11, cpp 30.0-30.6); section 8d adds the gender input (formants down) on top.

## 3c. Alemon (normal, falsetto) and Aurelian: full rows (11:27-11:34; same method as section 3)

Sung ranges: t0 = verse1 A4-E5, chorus1 A4-F5, plain C#4-A4; t-5 = E4-B4 / E4-C5 / G#3-E4; t-12 = A3-E4 / A3-F4 /
C#3-A3. Clips: `q4/out/alemon-normal_<case>[_t-5|_t-12].mp3`, `alemon-falsetto_...`, `aurelian_...` (-16 LUFS).

| run | case | cpp | hnr | voiced | hf | lvl_sd | WER tiny.en | WER base.en |
|---|---|---|---|---|---|---|---|---|
| alemon_normal_t0 | verse1 | 23.67 | 17.64 | 0.90 | -30.61 | 5.25 | 0.75 | 1.00 |
| alemon_normal_t0 | chorus1 | 23.76 | 22.05 | 0.93 | -31.22 | 4.42 | 0.71 | 0.77 |
| alemon_normal_t0 | plain | 28.73 | 21.17 | 0.91 | -30.47 | 4.99 | 0.17 | 0.21 |
| alemon_normal_t-5 | verse1 | 26.35 | 18.73 | 0.90 | -29.96 | 5.14 | 0.84 | 0.43 |
| alemon_normal_t-5 | chorus1 | 26.94 | 22.44 | 0.93 | -29.11 | 4.75 | 0.57 | 0.34 |
| alemon_normal_t-5 | plain | 29.46 | 19.38 | 0.91 | -31.61 | 4.80 | 0.04 | 0.08 |
| alemon_normal_t-12 | verse1 | 27.79 | 17.23 | 0.92 | -33.46 | 5.40 | 0.43 | 0.35 |
| alemon_normal_t-12 | chorus1 | 29.26 | 21.46 | 0.93 | -32.71 | 5.39 | 0.17 | 0.14 |
| alemon_normal_t-12 | plain | 29.70 | 19.13 | 0.91 | -34.53 | 4.45 | 0.08 | 0.08 |
| alemon_falsetto_t0 | verse1 | 22.92 | 18.66 | 0.89 | -30.70 | 5.10 | 0.81 | 1.00 |
| alemon_falsetto_t0 | chorus1 | 23.04 | 22.39 | 0.93 | -31.03 | 4.39 | 1.17 | 0.83 |
| alemon_falsetto_t0 | plain | 26.62 | 23.36 | 0.91 | -29.97 | 4.43 | 0.21 | 0.21 |
| alemon_falsetto_t-5 | verse1 | 24.92 | 20.22 | 0.90 | -30.84 | 4.98 | 0.59 | 0.49 |
| alemon_falsetto_t-5 | chorus1 | 25.29 | 23.60 | 0.93 | -30.48 | 4.73 | 0.63 | 0.43 |
| alemon_falsetto_t-5 | plain | 27.00 | 21.28 | 0.90 | -29.09 | 3.91 | 0.08 | 0.04 |
| alemon_falsetto_t-12 | verse1 | 26.01 | 19.17 | 0.92 | -31.15 | 5.23 | 0.44 | 0.40 |
| alemon_falsetto_t-12 | chorus1 | 26.76 | 22.82 | 0.91 | -31.74 | 5.27 | 0.51 | 0.23 |
| alemon_falsetto_t-12 | plain | 27.56 | 20.56 | 0.91 | -30.86 | 4.51 | 0.08 | 0.04 |
| aurelian_t0 | verse1 | 24.97 | 17.57 | 0.93 | -29.10 | 6.06 | 1.00 | 1.00 |
| aurelian_t0 | chorus1 | 24.59 | 21.72 | 0.96 | -29.68 | 5.82 | 0.94 | 0.54 |
| aurelian_t0 | plain | 27.55 | 18.70 | 0.92 | -30.23 | 4.89 | 0.33 | 0.21 |
| aurelian_t-5 | verse1 | 25.89 | 15.92 | 0.91 | -28.73 | 5.96 | 1.00 | 1.00 |
| aurelian_t-5 | chorus1 | 25.86 | 19.90 | 0.92 | -29.98 | 5.49 | 0.74 | 0.51 |
| aurelian_t-5 | plain | 27.55 | 18.25 | 0.89 | -31.00 | 4.16 | 0.33 | 0.21 |
| aurelian_t-12 | verse1 | 26.60 | 14.76 | 0.87 | -29.16 | 5.19 | 0.74 | 1.00 |
| aurelian_t-12 | chorus1 | 27.49 | 18.68 | 0.91 | -30.09 | 4.80 | 0.60 | 0.66 |
| aurelian_t-12 | plain | 27.87 | 17.95 | 0.89 | -32.10 | 4.42 | 0.29 | 0.25 |

Reading (one take each; chorus1 WER moves by about 0.2 between takes, so only the large gaps count):
- Alemon normal ("Amber"): a male-register voice. At our song's pitch it fails (verse1 0.75 / 1.00, chorus1 0.71 /
  0.77; base.en answers "[Music]" on the verse), 5 semitones down it is half way (0.84 / 0.43, 0.57 / 0.34), an
  octave down it works: verse1 0.43 / 0.35, chorus1 0.17 / 0.14, plain 0.08 / 0.08, cpp 27.8 / 29.3 / 29.7. Bright
  and forward there (centroid 895 Hz, press -11.6 dB on chorus1 t-12 against Nishiren Standard 716 Hz / -17.5 dB).
- Alemon falsetto ("Airy"): softer and darker than normal at every pitch (cpp -0.7 to -2.5, centroid 611 against
  895 Hz at t-12), slightly cleaner (hnr +1 to +2). It does NOT rescue the high register: at t0 chorus1 is 1.17 /
  0.83, verse1 0.81 / 1.00. Best an octave down too (verse1 0.44 / 0.40, chorus1 0.51 / 0.23, plain 0.08 / 0.04),
  and there normal is the clearer of the two. Use: a gentle colour for the same low register, not a head voice.
- Aurelian (male, no variance model): the slow plain line is understood at every pitch (0.33-0.29 / 0.21-0.25),
  the 168 BPM song is not: verse1 1.00 / 1.00 (t0), 1.00 / 1.00 (t-5), 0.74 / 1.00 (t-12; both recognisers answer
  "[Music]" or notes), chorus1 0.94 / 0.54, 0.74 / 0.51, 0.60 / 0.66. hnr is the lowest of all banks (14.8-21.7)
  and falls as the pitch drops; cpp 24.6-27.9. Lowering does not help it. With no breathiness / voicing / tension
  inputs there is nothing to steer. Not a candidate for fast English in this harness; and it has no published terms.
- Against Nishiren Standard on the same notes: at t-12 Nishiren chorus1 is cpp 29.98, WER 0.06 / 0.09 (Alemon
  normal 29.26, 0.17 / 0.14); at t0 Nishiren verse1 0.51 / 0.48 and chorus1 0.20 / 0.06 where both other banks
  fail. So the male role is best cast as Alemon normal an octave below the lead (or Nishiren an octave down, which
  the numbers rate at least as clear), never Aurelian.
- Not tried: Alemon's and Nishiren's own pitch predictors, voice-colour mixing (normal + falsetto weights), other
  seeds for these two banks, Japanese / Chinese.

## 4. Candidate voicebanks and vocoders (11:36; "verified" = downloaded, loaded and sung here)

| name | kind / voice | official release | size | languages | commercial use (publisher's words) | redistribution | credit | host reachable | status |
|---|---|---|---|---|---|---|---|---|---|
| Nishiren Gard v2.1 (Gardanana) | bank, contralto, 7 modes | github.com/Gardanana/Nishiren-AI-Diffsinger v2.1 | 623822790 | en, ja recorded; 11 more cross-lingual | "Contact Gardanana for individual approval" | "only ... as the entire unedited voicebank" | voicebank name required | yes | VERIFIED, sha256 138baf89...a5c7; terms in section 6 |
| Alemon v3.5 (tuskinekinase) | bank, male register, 2 modes | github.com/tuskinekinase/alemon_ds v3.5 | 526312536 | zh, en, ja | case-by-case permission for "商业活动（无论有偿无偿）" | complete, unmodified, free | AI-synthesised + "Alemon" + "DiffSinger" | yes | VERIFIED, sha256 47acc011...8e19; terms in section 7 |
| Aurelian Silva v108 (AlexFloareaVT) | bank, male | github.com/AlexFloareaVT/Aurelian_DS v108 | 419136941 | en, ja, zh, ko, fr, es, th | no terms published | no terms published | none stated | yes | VERIFIED to load and sing, sha256 fd2192a7...f116; unintelligible on the fast song; no licence |
| Shisakune Demo v4.5.1 (tuskinekinase) | bank, the Alemon team's other singer | github.com/tuskinekinase/shisakunedemo_ds v4.5.1 | 3 assets, names not read | zh, en, ja | same terms text as Alemon (LICENSE_zh differs only in names and staff) | same | "试作音Demo" + "DiffSinger" | repo yes, assets not tried | NOT downloaded |
| LUNAI Project singers v170 (19 singers, e.g. Keiro Revenant "first male", Allen Crow, Nero, Albus Korde) | banks, mostly Russian or Japanese main language | github.com/lunaiproject/lunai_singers release 170, `<Name>_v170.zip` | 432-433 MB each (206 answers) | ru / ja / de main, others cross-lingual | "Written permission from LUNAI Project team is required" (terms of 2026-01-16) | prohibited: "Distributing models or their copies" | '"Character Name" from LUNAI Project' | yes | NOT downloaded. Terms also prohibit "Porting models to other engines (e.g., RVC, Character.AI, UTAU) or modifying them" |
| Canary v106 (JAE VOCAL PROJECT / tigermeat) | bank, male | github.com/spicytigermeat/canary_diffsinger v106 | 3 assets, names not read | en (+ Korean cross-lingual per notes) | "may not be used ... for commercial purposes without the express written permission ... and/or purchase of a commercial usage license" | CC BY-NC-ND 4.0 | per CC BY | repo yes, assets not tried | NOT downloaded; ships OpenUtau plugin DLLs |
| Lingyuan Yousa v1.5.2 | bank, female | github.com/yousa-ling-official-production/yousa-ling-diffsinger-v1 | not read | zh only | "Non-commercial use ... is free of charge"; CC BY-NC-SA 4.0 | CC BY-NC-SA 4.0 | per licence | repo yes | NOT downloaded (no English) |
| TeamBRAPA Kyoumishii, Reo (male), Umidaji | banks | teambrapa.com.br/beta | ? | pt, en, ja, es | not read | not read | not read | host not tried (not GitHub) | lead only, from svs-index |
| Potluck Vocoder lite v100 (tigermeat) | vocoder 44.1k / 512 / 128 | github.com/mrtigermeat/potluck-vocoder v100 | 52705814 | - | "Commercial usage is allowed ONLY FOR DIFFSINGER APPLICATIONS" | allowed ("share, copy, and/or redistribute") | required, with link to the licence | yes | VERIFIED file and signature, sha256 0bf6c58b...e912; sound test in section 9 |
| OpenVPI PC-NSF-HiFiGAN 2025.02 | vocoder (our default) | github.com/openvpi/vocoders | installed | - | weights CC BY-NC-SA 4.0; "Synthesizing or producing commercial work with this vocoder is not constrained by the license" | allowed under the licence, with notices | OpenVPI / DiffSinger Community | yes | installed; statements in sections 1, 1b |
| Kouon vocoder v2.0 | vocoder (NSF-HiFiGAN) | github.com/Kouon-Project/Kouon_Vocoder | not read | - | "intended for research and non-commercial use only", CC BY-NC-SA 4.0 | CC BY-NC-SA 4.0 | per licence | repo yes | NOT downloaded; no gain over the default on licence |
| HiFiPLN (Scarfmonster) | vocoder training code | github.com/Scarfmonster/HiFiPLN | - | - | code MIT; licence of any released weights not checked | - | - | repo yes | NOT checked further |

No bank found so far states plain "commercial use allowed" except Qixuan (our current default). Every other bank with
terms asks for the author's permission case by case; Aurelian states nothing. Multi-speaker banks verified: Nishiren (7
embeddings), Alemon (2). A male voice with usable English at speed: Alemon normal an octave below the lead.

## 10. Registry entries (11:36)

Files: `q4/registry/nishiren.json`, `alemon.json`, `aurelian.json`, `potluck-vocoder-lite.json`, in the shape of
`$S/models/installed/*/synth2-model.json` (format, version, id, kind, name, model_version, engine, root, languages,
license, license_file, license_url, commercial_use, terms, credits, homepage, source, sha256, default, external) plus
`size` (archive bytes), `installed_size`, `speakers` / `default_speaker`, `bundled_vocoder`, `license_checked`,
`notes`. Checked: copied as `synth2-model.json` into `q4/models_test/<id>/` (with `files` -> the extracted
archive), `SYNTH2_MODELS=$S/q4/models_test` and the repo's `synth2.neural.models.installed()` reads all four:
root holds `dsconfig.yaml` / `vocoder.yaml`, the three banks have `dsvocoder/vocoder.yaml`, `commercial_use` reads
False (nishiren, alemon), None (aurelian: left out, unknown), True (potluck lite, "only for DiffSinger
applications" in terms).
- `commercial_use: false` for Nishiren and Alemon means "not without the author's individual approval"; the terms
  lines say so.
- `license_file` is null for the three banks: none of the archives contains its terms. An installer should fetch
  `license_url` and store it next to the manifest, and show it before download.
- `aurelian.json` is for adding the bank in place only (no terms published): keep it out of a built-in registry.
- The manifest has no field for "which speaker embedding"; `speakers` and `default_speaker` are proposals.
- Re-verified 11:53: for all four entries the archive on disk has the entry's `sha256` and `size`, and `source`
  answers a ranged GET with that total size. The three `license_url` values (raw.githubusercontent.com) answer 200
  and are byte-identical to the cloned files quoted in sections 6, 6b and 7 (sha256 39ec256b..., 57752a67...,
  5b3da863...); the Potluck LICENSE inside the archive equals the repo's.

### 6d. Questions only Gardanana can answer (draft for whoever writes; nothing was sent)

1. Synth2 is a free, open-source singing studio that loads the unmodified OpenUtau DiffSinger files of a voicebank
   (no conversion, no edits, weights never bundled). Does that count as a "similar program" under EDITING, or is
   approval needed?
2. May Synth2's installer download `Nishiren.Diffsinger.v2.1.zip` from the GitHub release on the user's command,
   after showing the Terms of Service and asking the user to accept them?
3. Synth2's lyrics and scores are often written with a language model by the person using it. Is that the
   "unethically AI generated lyrics, music" the terms forbid, or is that clause about something else (for example
   generative-audio services trained on scraped music)?
4. What does a user have to do to get approval for a monetised video or a released track?
Contacts: see the README of the release repository.

## 9. Vocoder swap on identical mels (11:45-11:48): Potluck lite and OpenVPI 2025.02 against the bundled ones

Method: `q4/queue4.sh` = the same bank, mode, seed and case order as the reference run, only `--vocoder` changed,
so the acoustic model's output (the mel) is the same and only the vocoder differs. Rows in `q4/vocoders.tsv`; clips
`q4/out/voc-nishiren-potluck_<case>.mp3`, `voc-nishiren-openvpi_<case>.mp3`, `voc-qixuan-potluck_<case>_t-5.mp3`,
`voc-qixuan-openvpi_<case>_t-5.mp3` (-16 LUFS).

| bank, pitch | vocoder | verse1 cpp / hnr | verse1 WER | chorus1 cpp / hnr | chorus1 WER | plain cpp / hnr | plain WER |
|---|---|---|---|---|---|---|---|
| Nishiren Standard t0 | bundled gda_pc-hifigan (Potluck fine-tune) | 25.72 / 18.71 | 0.51 / 0.48 | 25.72 / 25.64 | 0.20 / 0.06 | 29.67 / 24.37 | 0.00 / 0.00 |
| Nishiren Standard t0 | Potluck lite v100 | 25.47 / 19.14 | 0.60 / 0.41 | 25.25 / 26.40 | 0.17 / 0.06 | 29.66 / 25.38 | 0.00 / 0.04 |
| Nishiren Standard t0 | OpenVPI PC-NSF-HiFiGAN 2025.02 | 26.22 / 18.57 | 0.56 / 0.41 | 25.51 / 26.10 | 0.14 / 0.03 | 30.39 / 24.43 | 0.00 / 0.00 |
| Qixuan t-5 | OpenVPI 2025.02 (its intended one) | 21.89 / 18.05 | 0.55 / 0.56 | 22.28 / 20.56 | 0.40 / 0.29 | - | - |
| Qixuan t-5 | Potluck lite v100 | 22.37 / 18.53 | 0.58 / 0.50 | 23.00 / 20.73 | 0.51 / 0.40 | - | - |

Spectral difference on the same mel (`python3 -I q4/lsd.py A.wav B.wav`: rms difference of third-octave band levels
over active frames; for scale, two takes of the same bank and vocoder with different noise seeds differ by 5.8 /
4.3 / 4.3 dB in the three bands):
- Nishiren chorus1, bundled -> Potluck lite: 4.0 dB (0.1-5 kHz), 2.5 dB (5-10 kHz), 1.0 dB (10-16 kHz); mean level
  +1.7 / -0.8 / -0.3 dB. Bundled -> OpenVPI: 3.5 / 2.8 / 1.7 dB; level +0.4 / +0.2 / +0.5 dB.
- Qixuan chorus1, OpenVPI -> Potluck lite: 3.8 / 1.5 / 1.2 dB; level +0.6 / 0.0 / -0.8 dB (verse1: 3.9 / 1.4 / 1.2).

Reading: on these three cases the three vocoders are interchangeable by every number here: cpp within 0.9 dB, hnr
within 1.0 dB, WER differences within the take-to-take spread, and the spectra closer to each other than two takes
of one vocoder are. Potluck lite has slightly less energy above 10 kHz (-0.3 to -0.8 dB). Its author's own caveat
stands ("does not perform well on vocals outside of it's dataset ... I highly recommend fine-tuning"): two voices,
one pitch range each, no listening test, no sustained high notes or growl were checked here. Vocoder time is the
same (same network size: 8.8 s against 7.1-8.6 s for 20.7 s sung at 2 threads, load 3.5-4).

## 5b. Loader notes added in the third run

11. One release can carry two builds: Nishiren v2.1 has `Nishiren.Diffsinger.v2.1.zip` (duration model with
    `word_div` / `word_dur` / `spk_embed` inputs) and `...v2.1.Legacy.zip` "for editors NOT updated since October
    15, 2026". Feeding each session from its own input list (as `dsinfer4.py` does) covers the main build; the
    Legacy build was not downloaded.
12. A speaker name is needed per render, not per bank: all seven Nishiren embeddings and both Alemon ones load
    through the same sessions (`DS(root, speaker="Power")` matches the trailing path segment of `speakers:` in each
    sub-model's dsconfig). Nishiren's four sub-models list the same seven names; Alemon's dsdur and dspitch list none
    and take no `spk_embed`. A loader should fail loudly if a sub-model that takes `spk_embed` lacks
    the chosen name rather than fall back to speaker 0 silently, as OpenUtau does.
13. The variance model's output differs by speaker: mean predicted tension over sung frames runs from -0.8 (Soft)
    to +2.6 (2P, Power an octave down), breathiness -51 to -41 dB (Screamo). Fixed offsets tuned on one mode do not
    carry to another.
14. Any 44.1 kHz / hop 512 / 128-bin PC-NSF-HiFiGAN works with any of these banks' mels (section 9), so the
    loader's vocoder choice is a licence and taste decision, not a compatibility one.

## 11. What remains unverified (plain list)

Terms:
- The copy of Nishiren's terms on the author's site (gardanana.nekoweb.org, gardanana.neocities.org/terms): both
  hosts are blocked from this machine and from the web-fetch tool. Only the GitHub repo's `terms-of-service.md` was
  read. The UTAU wiki summary was not read either (host blocked).
- Whether Gardanana accepts Synth2 as a "similar program", allows an installer to download the release, and what
  "unethically AI generated" covers: needs the author's answer (section 6d). Nobody was contacted.
- Release notes of Nishiren v2.1, Aurelian v108 and Alemon v3.5 were read through a summarising fetch tool, not
  verbatim (github.com HTML is not reachable directly). Asset names and sizes of Shisakune Demo and Canary not read.
- Licence and origin of Alemon's bundled `dsvocoder/nsf_hifigan.onnx` and of Aurelian's `tgm_hifigan_v107.onnx`.
- The licences of the datasets Nishiren and Aurelian list as training data (M4Singer, PJS, CSD, ...) were not
  looked up; the authors do not restate them.
- What tigermeat counts as a "DiffSinger application", and which licence a Potluck fine-tune carries.
- Alemon's English `LICENSE` (CC BY-NC-SA 4.0 legal code) against its Chinese terms: which governs is not stated.
  The Chinese clauses in section 7 are my translation.
- Aurelian: no terms exist to verify.
Sound:
- Nobody listened. Every description of a voice is from voiceq / timbre numbers, recogniser scores and the
  authors' own words. "Nasal", "growly" and the like are the authors' words, not measurements.
- Mode map: plain has one take per point; chorus1 two takes (seeds 1 and 2); verse1 only for Standard (t0, t-5,
  t-12) and Power (t0). Recogniser scores on chorus1 move by 0.05-0.07 on average between takes, up to 0.29.
- Not tried on any bank here: the banks' own pitch predictors (a colleague does that for Nishiren in `$S/v67`),
  mixing two modes, the velocity input, variance offsets on any mode but Screamo (8e), the gender input beyond the
  three rows of 8d, Japanese or any language but English, more than 10 diffusion steps.
- Alemon and Aurelian: one take per row. Aurelian's failure on the fast song was not investigated beyond checking
  that its English phoneme names match ours one to one.
- Vocoder swap: two voices (Nishiren Standard at the song's pitch, Qixuan 5 semitones down), three cases, numbers
  only.
- Timings were taken at load 1-4 with another engineer rendering on the same 4 CPUs: noisy.
Downloads:
- Not downloaded: Nishiren Legacy zip, Shisakune Demo, the 19 LUNAI singers (reachable, 432-433 MB each), Canary,
  Lingyuan Yousa, Kouon vocoder, TeamBRAPA banks (host not tried). Total on disk 1.56 GB of the 3 GB allowed.

## 3d. One table for every bank (11:52; all rows from this harness: Synth2 pitch, steps 10, seed 1, one take)

`python3 -I q4/banktable.py` prints it from `work/*/score.json` + `voiceq.json`. WER = tiny.en / base.en. Pitch: t0 =
verse1 A4-E5, chorus1 A4-F5, plain C#4-A4; t-5 and t-12 = 5 and 12 semitones lower. Last column: WER x words summed
over the three cases (88 + 35 + 24 words). Each bank with its own bundled vocoder, Qixuan with OpenVPI 2025.02.
Nishiren t-5 and t-12 rows are put together from two processes each (verse1 separately). Clips for every cell:
`q4/out/<bank>[-<mode>]_<case>[_t-5|_t-12].mp3` (qixuan, nishiren-standard, nishiren-power, alemon-normal,
alemon-falsetto, aurelian), all -16 LUFS.

| bank / mode | pitch | verse1 cpp / hnr | verse1 WER | chorus1 cpp / hnr | chorus1 WER | plain cpp / hnr | plain WER | words wrong of 147 (tiny / base) |
|---|---|---|---|---|---|---|---|---|
| Qixuan v2.7.0 (OpenVPI vocoder) | t0 | 20.98 / 18.23 | 1.41 / 0.47 | 21.10 / 21.76 | 0.26 / 0.17 | 24.68 / 20.53 | 0.08 / 0.04 | 135 / 48 |
| Qixuan v2.7.0 (OpenVPI vocoder) | t-5 | 21.89 / 18.05 | 0.55 / 0.56 | 22.28 / 20.56 | 0.40 / 0.29 | - | - | - |
| Qixuan v2.7.0 (OpenVPI vocoder) | t-12 | 20.96 / 16.07 | 0.52 / 0.50 | 23.25 / 19.52 | 0.31 / 0.14 | 18.47 / 14.57 | 0.21 / 0.08 | 62 / 51 |
| Nishiren v2.1 Standard | t0 | 25.72 / 18.71 | 0.51 / 0.48 | 25.72 / 25.64 | 0.20 / 0.06 | 29.67 / 24.37 | 0.00 / 0.00 | 52 / 44 |
| Nishiren v2.1 Standard | t-5 | 27.23 / 18.73 | 0.50 / 0.33 | 28.41 / 25.46 | 0.17 / 0.11 | 29.93 / 23.29 | 0.04 / 0.04 | 51 / 34 |
| Nishiren v2.1 Standard | t-12 | 27.32 / 17.23 | 0.31 / 0.29 | 29.98 / 23.83 | 0.06 / 0.09 | 29.69 / 23.19 | 0.04 / 0.04 | 30 / 30 |
| Nishiren v2.1 Power | t0 | 26.06 / 17.60 | 0.52 / 0.47 | 25.97 / 24.61 | 0.34 / 0.29 | 30.08 / 22.94 | 0.04 / 0.00 | 59 / 51 |
| Alemon v3.5 normal | t0 | 23.67 / 17.64 | 0.75 / 1.00 | 23.76 / 22.05 | 0.71 / 0.77 | 28.73 / 21.17 | 0.17 / 0.21 | 95 / 120 |
| Alemon v3.5 normal | t-5 | 26.35 / 18.73 | 0.84 / 0.43 | 26.94 / 22.44 | 0.57 / 0.34 | 29.46 / 19.38 | 0.04 / 0.08 | 95 / 52 |
| Alemon v3.5 normal | t-12 | 27.79 / 17.23 | 0.43 / 0.35 | 29.26 / 21.46 | 0.17 / 0.14 | 29.70 / 19.13 | 0.08 / 0.08 | 46 / 38 |
| Alemon v3.5 falsetto | t0 | 22.92 / 18.66 | 0.81 / 1.00 | 23.04 / 22.39 | 1.17 / 0.83 | 26.62 / 23.36 | 0.21 / 0.21 | 117 / 122 |
| Alemon v3.5 falsetto | t-5 | 24.92 / 20.22 | 0.59 / 0.49 | 25.29 / 23.60 | 0.63 / 0.43 | 27.00 / 21.28 | 0.08 / 0.04 | 76 / 59 |
| Alemon v3.5 falsetto | t-12 | 26.01 / 19.17 | 0.44 / 0.40 | 26.76 / 22.82 | 0.51 / 0.23 | 27.56 / 20.56 | 0.08 / 0.04 | 59 / 44 |
| Aurelian v108 | t0 | 24.97 / 17.57 | 1.00 / 1.00 | 24.59 / 21.72 | 0.94 / 0.54 | 27.55 / 18.70 | 0.33 / 0.21 | 129 / 112 |
| Aurelian v108 | t-5 | 25.89 / 15.92 | 1.00 / 1.00 | 25.86 / 19.90 | 0.74 / 0.51 | 27.55 / 18.25 | 0.33 / 0.21 | 122 / 111 |
| Aurelian v108 | t-12 | 26.60 / 14.76 | 0.74 / 1.00 | 27.49 / 18.68 | 0.60 / 0.66 | 27.87 / 17.95 | 0.29 / 0.25 | 93 / 117 |

Reading:
- Nishiren Standard is ahead of every other bank at every pitch: cpp +4.6 to +6.7 dB over Qixuan on the same notes
  (+11.2 dB on the low plain line, where Qixuan breaks), and the fewest
  wrong words (52 / 44 at the song's pitch, 30 / 30 an octave down; Qixuan 135 / 48 and 62 / 51, where 135 is
  tiny.en looping on the verse).
- Qixuan in this harness repeats the demo (`$S/ds1`: cpp 20.78 / 20.99 / 24.96): the gap to Nishiren is the bank,
  not the harness. An octave down Qixuan breaks on the low line (plain cpp 18.5, hnr 14.6).
- Alemon is a low voice: unusable at the song's pitch, level with Nishiren-at-t0 once it is an octave down
  (46 / 38). Falsetto is the weaker colour at every pitch.
- Aurelian is last or next to last at every pitch (93-129 / 111-117 wrong words) and the only bank that lowering
  does not rescue on base.en (112, 111, 117).

## 8d. A male cast from Nishiren: an octave down with the formants lowered (11:54-11:56, one take)

`q4/queue6.sh`: `--gender G` is the model's own `gender` input (key-shift embedding; negative = formants down; the
bank's range is +-5 semitones, G = -1.0 is the bottom of it). Rows in `q4/gender.tsv`, clips
`q4/out/nishiren-standard_<case>_t-12_g-0.6.mp3`, `..._g-1.0.mp3`, `nishiren-2p_<case>_t-12_g-0.6.mp3`.

| mode, pitch | gender | chorus1 cpp / hnr | chorus1 WER | chorus1 centroid Hz / press dB | plain cpp / hnr | plain WER |
|---|---|---|---|---|---|---|
| Standard t-12 | 0 | 29.98 / 23.83 | 0.06 / 0.09 (take 2: 0.17 / 0.09) | 716 / -17.5 | 29.69 / 23.19 | 0.04 / 0.04 |
| Standard t-12 | -0.6 | 30.39 / 24.21 | 0.26 / 0.09 | 691 / -17.3 | 30.03 / 23.54 | 0.00 / 0.00 |
| Standard t-12 | -1.0 | 30.29 / 24.56 | 0.26 / 0.09 | 665 / -18.4 | 31.03 / 23.59 | 0.00 / 0.04 |
| 2P t-12 | 0 | 30.32 / 21.42 | 0.11 / 0.09 (take 2: 0.11 / 0.11) | 977 / -12.0 | 31.08 / 19.16 | 0.00 / 0.00 |
| 2P t-12 | -0.6 | 30.75 / 21.94 | 0.06 / 0.11 | 879 / -14.0 | 30.64 / 19.49 | 0.04 / 0.04 |
| Alemon normal t-12 (for scale) | 0 | 29.26 / 21.46 | 0.17 / 0.14 | 895 / -11.6 | 29.70 / 19.13 | 0.08 / 0.08 |

Reading: lowering the formants costs nothing measurable (cpp and hnr unchanged or up, base.en unchanged, the
tiny.en change on Standard is inside the take-to-take spread). The colour moves the right way but not far by these
numbers (centroid -3 % and -7 % for Standard, -10 % for 2P). Whether it reads as a man is a listening question;
the clips are there for it. If it does, Nishiren can play the male part too and the cast needs one bank and one
set of terms.

## 8e. Screamo the way its README says to use it: tension and breathiness raised (11:57-11:59, one take)

`q4/queue7.sh`: `--tension 4 --breath 8` (added to what the variance model predicts; in OpenUtau's units that is
TENC +80 and BREC +67, from `DiffSingerUtils.VarianceDeltaFunctions`: tension + y / 20, breathiness + y * 12 / 100).
Rows in `q4/screamo.tsv`, clips `q4/out/nishiren-screamo_<case>_x.mp3` and `nishiren-screamo_<case>_t-5_x.mp3`.

| Screamo | case | cpp | hnr | voiced | hf | centroid Hz | air dB | WER tiny / base |
|---|---|---|---|---|---|---|---|---|
| offsets 0, t0 | chorus1 | 23.62 | 25.40 | 0.96 | -27.12 | 697 | -26.9 | 0.66 / 0.23 |
| tension +4, breath +8 dB, t0 | chorus1 | 19.03 | 14.75 | 0.84 | -22.58 | 1158 | -21.3 | 0.69 / 0.34 |
| offsets 0, t0 | plain | 28.56 | 25.43 | 0.95 | -29.38 | 523 | -29.2 | 0.08 / 0.00 |
| tension +4, breath +8 dB, t0 | plain | 17.37 | 11.19 | 0.81 | -20.79 | 969 | -20.1 | 0.17 / 0.08 |
| offsets 0, t-5 | chorus1 | 25.50 | 24.70 | 0.96 | -26.88 | 629 | -26.8 | 0.37 / 0.20 |
| tension +4, breath +8 dB, t-5 | chorus1 | 16.62 | 9.20 | 0.79 | -21.00 | 998 | -20.1 | 0.60 / 0.34 |
| offsets 0, t-5 | plain | 26.48 | 23.42 | 0.94 | -29.13 | 434 | -29.1 | 0.04 / 0.04 |
| tension +4, breath +8 dB, t-5 | plain | 16.28 | 10.84 | 0.80 | -22.02 | 680 | -21.1 | 0.17 / 0.04 |

Reading: with the offsets the mode becomes what its name says: harmonics-to-noise falls by 11-16 dB, cpp by 5-11 dB,
the top above 6 kHz rises by 6-9 dB. The slow line is still understood (0.17 / 0.04-0.08), the fast chorus is not
(0.60-0.69 / 0.34). With offsets 0 (the rows of the mode map) it is only a slightly noisy dark voice. An effect for
single words or slow shouted lines, not a singing colour.
