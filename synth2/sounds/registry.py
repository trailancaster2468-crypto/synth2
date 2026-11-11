"""The sound sets ``synth2 sounds install`` knows.

Every entry was downloaded from the URL it names and checked. ``license`` and
``commercial_use_statement`` quote the publisher; ``license_source`` says where the text was
read. ``commercial_use`` is True or False only where the publisher says so in plain words.
"""
from __future__ import annotations

from ..assets import Source
from .installer import SoundSpec

_RAW = "https://raw.githubusercontent.com"


_UBUNTU = "https://archive.ubuntu.com/ubuntu/pool/universe"
_GM_TAGS = ("gm", "gs", "drums", "piano", "bass", "guitar", "strings", "brass", "woodwind",
            "mallets", "organ", "synth", "percussion", "fx")
_CC0_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
_KARORYFER_SAYS = "Royalty-free for all commercial and non-commercial use."


# Shinyguitar's program names its sample folder through a variable that only its Sforzando
# bank file defines.
_SHINY_DEFINES = {"$sample_dir": "../Samples"}


def _presets(category: str, *rows: tuple[str, str]) -> tuple[dict, ...]:
    return tuple({"name": name, "file": file, "category": category} for name, file in rows)


def _karoryfer(sid: str, name: str, version: str, repo: str, asset: str, sha256: str, size: int,
               *, root: str, files: tuple[str, ...], presets: tuple[dict, ...],
               tags: tuple[str, ...], description: str, credit: str,
               said_in: str = "readme.txt", statement: str = _KARORYFER_SAYS,
               license_file: str = "") -> SoundSpec:
    """A Karoryfer Samples library from the release page of its repository."""
    return SoundSpec(
        id=sid, kind="sfz", name=name, version=version, publisher="Karoryfer Samples",
        homepage=f"https://github.com/sfzinstruments/{repo}",
        description=description, tags=tags, root=root, files=files, presets=presets,
        sources=(Source(url=f"https://github.com/sfzinstruments/{repo}/releases/download/"
                            f"v{version}/{asset}", sha256=sha256, size=size, unpack="zip"),),
        size=size, license="CC0 1.0 Universal",
        license_file=license_file or f"{root}/LICENSE", license_url=_CC0_URL,
        license_source=f"LICENSE (CC0 1.0 Universal) and {said_in} of "
                       f"github.com/sfzinstruments/{repo}",
        commercial_use=True, commercial_use_statement=statement,
        terms=("No credit is required.",), credits=(credit,))


def _vsco(part: str, title: str, size: int, files: tuple[str, ...], tags: tuple[str, ...],
          description: str, category: str) -> SoundSpec:
    """One instrument family of VSCO 2 Community Edition, from the SFZ branch of its repository."""
    presets = tuple({"name": f[:-4], "file": f, "category": category} for f in files)
    return SoundSpec(
        id=f"vsco2-{part}", kind="sfz", name=f"VSCO 2 Community Edition: {title}",
        version="1.1.0", publisher="Versilian Studios (Sam Gossner)",
        homepage="https://github.com/sgossner/VSCO-2-CE",
        description=description, tags=tags + ("orchestral",), files=files, presets=presets,
        listing=f"vsco2_{part}", size=size, license="CC0 1.0 Universal",
        license_file="files/LICENSE", license_url=_CC0_URL,
        license_source="LICENSE and Readme.txt on the SFZ branch of github.com/sgossner/"
                       "VSCO-2-CE (commit 6dd651d)",
        commercial_use=True,
        commercial_use_statement="This set is provided and designed for the creation of "
                                 "personal sample libraries for all purposes, both educational "
                                 "and commercial. [...] You are permitted to use these samples "
                                 "for ANY purpose. We ask that you do not sell the samples "
                                 "directly",
        terms=("\"Please provide credit to Versilian Studios/Sam Gossner, and/or Ivy "
               "Audio/Simon Dalzell where applicable, and link to the VSCO: CE homepage.\"",),
        credits=("Orchestral samples: VSCO 2 Community Edition by Versilian Studios / Sam "
                 "Gossner and Ivy Audio / Simon Dalzell (CC0), "
                 "vis.versilstudios.net/vsco-community.html",))


def _specs() -> list[SoundSpec]:
    return [
        SoundSpec(
            id="generaluser-gs", kind="soundfont", name="GeneralUser GS", version="2.0.3",
            publisher="S. Christian Collins",
            homepage="https://www.schristiancollins.com/generaluser",
            description="General MIDI / Roland GS soundfont: 261 instrument presets and 13 drum "
                        "kits in 31 MB.",
            tags=_GM_TAGS, files=("GeneralUser-GS.sf2",),
            sources=(
                Source(url=f"{_RAW}/mrbumpy409/GeneralUser-GS/"
                           "684543d5e5efaef08d02be50dcda8d552478fa60/GeneralUser-GS.sf2",
                       sha256="9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe",
                       size=32_319_396),
                Source(url=f"{_RAW}/mrbumpy409/GeneralUser-GS/"
                           "684543d5e5efaef08d02be50dcda8d552478fa60/documentation/LICENSE.txt",
                       sha256="7b32efefdf95ce38a043799f0659853ddc00fbaa14d8c50f0aca16b9b8b405be",
                       size=2_317),
            ),
            size=32_321_713,
            license="GeneralUser GS License v2.0", license_file="files/LICENSE.txt",
            license_url="https://github.com/mrbumpy409/GeneralUser-GS/blob/main/documentation/"
                        "LICENSE.txt",
            license_source="documentation/LICENSE.txt in the publisher's repository "
                           "github.com/mrbumpy409/GeneralUser-GS (commit 684543d)",
            commercial_use=True,
            commercial_use_statement="You may use GeneralUser GS without restriction for your "
                                     "own music creation, private or commercial.",
            unclear="The publisher writes that some samples came from other free banks and "
                    "that he \"cannot be 100% sure where all of the samples originated\".",
            terms=(
                "The licence asks for no credit.",
                "Do not link directly to the publisher's download files from your own website.",
            ),
            credits=("Instruments: GeneralUser GS v2.0.3 by S. Christian Collins.",),
        ),
        SoundSpec(
            id="avl-drumkits", kind="soundfont", name="AVL Drumkits", version="LV2 edition",
            publisher="Glen MacArthur (AVLinux), SF2 edition by Robin Gareus (x42)",
            homepage="http://www.bandshed.net/avldrumkits/",
            description="Five acoustic kits as soundfonts with velocity layers: Black Pearl, "
                        "Red Zeppelin, Blonde Bop (sticks and hot rods) and the Buskman's "
                        "Holiday percussion set.",
            tags=("drums", "acoustic", "percussion"),
            root="files/sf2",
            files=("Black_Pearl_4_LV2.sf2", "Red_Zeppelin_4_LV2.sf2", "Blonde_Bop_LV2.sf2",
                   "Blonde_Bop_HR_LV2.sf2", "Buskmans_Holiday_LV2.sf2"),
            presets=(
                {"name": "Black Pearl", "file": "Black_Pearl_4_LV2.sf2", "category": "drums"},
                {"name": "Red Zeppelin", "file": "Red_Zeppelin_4_LV2.sf2", "category": "drums"},
                {"name": "Blonde Bop", "file": "Blonde_Bop_LV2.sf2", "category": "drums"},
                {"name": "Blonde Bop Hot Rods", "file": "Blonde_Bop_HR_LV2.sf2",
                 "category": "drums"},
                {"name": "Buskman's Holiday", "file": "Buskmans_Holiday_LV2.sf2",
                 "category": "percussion"},
            ),
            listing="avl_drumkits", size=152_290_338,
            license="CC BY-SA 3.0 with an exception for produced music",
            license_file="files/sf2/README",
            license_url="http://creativecommons.org/licenses/by-sa/3.0/",
            license_source="sf2/README in github.com/x42/avldrums.lv2 (commit 9389f16)",
            commercial_use=True, share_alike=True,
            commercial_use_statement="The by-sa condition is imperative if you modify the "
                                     "samples themselves or create new sample libraries with "
                                     "this licensed product. However, produced music and other "
                                     "non-sample-library works can be licensed freely.",
            unclear="The README does not use the word \"commercial\"; the licence it names, "
                    "CC BY-SA 3.0, permits commercial use in its own text.",
            terms=(
                "Attribution: \"keeping all authors information and this readme-file or its "
                "content in the release\".",
                "A modified sample library must be shared under the same licence and under a "
                "different name.",
            ),
            credits=("Drums: AVL Drumkits by Glen MacArthur (bandshed.net/avldrumkits), "
                     "CC BY-SA 3.0.",),
        ),
        SoundSpec(
            id="musescore-general-hq", kind="soundfont", name="MuseScore General HQ",
            version="0.2.1", publisher="MuseScore (S. Christian Collins and contributors)",
            homepage="https://musescore.org/en/handbook/3/soundfonts-and-sfz-files",
            description="Large General MIDI soundfont (490 MB), a fork of Fluid R3 with many "
                        "samples replaced or reprogrammed (a Steinway grand among them). Uses "
                        "SoundFont 2.01 modulators.",
            tags=_GM_TAGS, root="files/musescore-general-soundfont-0.2.1.orig",
            files=("MuseScore_General_HQ.sf2",),
            sources=(Source(url=f"{_UBUNTU}/m/musescore-general-soundfont/"
                                "musescore-general-soundfont_0.2.1.orig.tar.xz",
                            sha256="0e481d40ac7114d6aef5bdcf7d024682"
                                   "9b8f8873de02049896cd120d8490bcae",
                            size=274_424_736, unpack="tar"),),
            size=274_424_736, license="MIT",
            license_file="files/musescore-general-soundfont-0.2.1.orig/documentation/"
                         "MuseScore_General_HQ_License.md",
            license_url="https://opensource.org/license/mit",
            license_source="documentation/MuseScore_General_HQ_License.md in the source archive "
                           "of the Debian/Ubuntu package musescore-general-soundfont 0.2.1 "
                           "(the publisher's server ftp.osuosl.org was not reachable when this "
                           "entry was made)",
            commercial_use=True,
            commercial_use_statement="MuseScore_General_HQ.sf2 is shared under the MIT license "
                                     "as described in COPYING [...] to deal in the Software "
                                     "without restriction, including without limitation the "
                                     "rights to use, copy, modify, merge, publish, distribute, "
                                     "sublicense, and/or sell copies of the Software",
            terms=("\"The acknowledgements and copyright notices above must be included in any "
                   "derivative work\" (of the soundfont; see the licence file).",),
            credits=("Instruments: MuseScore General HQ soundfont 0.2.1 (MIT): Fluid R3 by Frank "
                     "Wen, mono conversion by Michael Cowgill, adaptation by S. Christian "
                     "Collins, temple blocks by Ethan Winer, drumline cymbals by Michael "
                     "Schorsch.",),
        ),
        SoundSpec(
            id="fluidr3-gm", kind="soundfont", name="Fluid R3 GM", version="3.1",
            publisher="Frank Wen",
            homepage="https://packages.ubuntu.com/source/fluid-soundfont",
            description="The classic 148 MB General MIDI soundfont, with its small GS "
                        "companion bank.",
            tags=_GM_TAGS, root="files/fluid-soundfont-3.1",
            files=("FluidR3_GM.sf2", "FluidR3_GS.sf2"),
            sources=(Source(url=f"{_UBUNTU}/f/fluid-soundfont/fluid-soundfont_3.1.orig.tar.gz",
                            sha256="2621acaa1c78e4abdb24bdd163230cc5"
                                   "77e61276936d6aa6e3180582142f0343",
                            size=134_835_922, unpack="tar"),),
            size=134_835_922, license="MIT", license_file="files/fluid-soundfont-3.1/COPYING",
            license_url="https://opensource.org/license/mit",
            license_source="README and COPYING in the source archive of the Debian/Ubuntu "
                           "package fluid-soundfont 3.1 (the author's original site is gone)",
            commercial_use=True,
            commercial_use_statement="I hereby release Fluid under the MIT license, as described "
                                     "in COPYING. [...] to deal in the Software without "
                                     "restriction, including without limitation the rights to "
                                     "use, copy, modify, merge, publish, distribute, sublicense, "
                                     "and/or sell copies of the Software",
            terms=("Copies of the soundfont must carry its copyright and permission notice.",),
            credits=("Instruments: Fluid (R3) SoundFont, Copyright (c) 2000-2002, 2008 Frank Wen "
                     "(MIT).",),
        ),
        _karoryfer(
            "karoryfer-growlybass", "Karoryfer Growlybass", "1.002", "karoryfer.growlybass",
            "Karoryfer.Growlybass.v1.002.zip",
            "b92dca75494fa39d6e8d8b8addaabfe70c5aaa59b5f9a0f624be6de51a2374bc", 167_454_256,
            root="files/Growlybass",
            files=("growlybass_clean.sfz", "growlybass_dirty.sfz", "growlybass_vicious.sfz",
                   "growlybass_angry.sfz", "growlybass_excessive.sfz", "growlybass_shifty.sfz",
                   "growlybass_strange.sfz"),
            presets=_presets(
                "bass",
                ("Clean (sustained, no release samples)", "growlybass_clean.sfz"),
                ("Dirty (sustained with release samples)", "growlybass_dirty.sfz"),
                ("Vicious (staccato)", "growlybass_vicious.sfz"),
                ("Angry (sustained and staccato)", "growlybass_angry.sfz"),
                ("Excessive (nine voices)", "growlybass_excessive.sfz"),
                ("Shifty (three voices)", "growlybass_shifty.sfz"),
                ("Strange (release noises)", "growlybass_strange.sfz")),
            tags=("bass", "electric"),
            description="Electric bass (Squier Jazz Bass, recorded direct): four "
                        "velocity layers and four round robins, staccato and release samples. "
                        "24-bit WAV.",
            credit="Electric bass: Growlybass by Karoryfer Samples (CC0)."),
        _karoryfer(
            "karoryfer-meatbass", "Karoryfer Meatbass", "1.001", "karoryfer.meatbass",
            "Karoryfer.Meatbass.v1.001.zip",
            "bc053061d4f39fb76ba56bc1f323efa228d1f63757c331738812479bcf04fe96", 255_425_351,
            root="files/Meatbass",
            files=("Programs/04_pizz.sfz", "Programs/02_arco_3vel.sfz",
                   "Programs/03_arco_5vel.sfz", "Programs/01_arco_modwheel.sfz"),
            presets=_presets(
                "bass",
                ("Pizzicato", "Programs/04_pizz.sfz"),
                ("Arco (3 velocity layers)", "Programs/02_arco_3vel.sfz"),
                ("Arco (5 velocity layers)", "Programs/03_arco_5vel.sfz"),
                ("Arco (mod wheel)", "Programs/01_arco_modwheel.sfz")),
            tags=("bass", "upright", "strings"),
            description="Upright bass (1958 Otto Rubner double bass, played by D. Smolken): "
                        "pizzicato with four velocity layers and four round robins, and bowed "
                        "sustains. 24-bit WAV; the programs use #include.",
            credit="Upright bass: Meatbass by Karoryfer Samples, played by Drogomir Smolken "
                   "(CC0)."),
        _karoryfer(
            "karoryfer-black-and-green-guitars", "Karoryfer Black And Green Guitars", "1.000",
            "karoryfer.black-and-green-guitars", "Karoryfer_Black_And_Green_Guitars_1000.zip",
            "ba95221d32fb6386c5d0a0799b93904fd55520aca017c5f121da290075c140c0", 481_838_512,
            root="files",
            files=("Programs/04-green_twang.sfz", "Programs/05-green_staccato.sfz",
                   "Programs/06-green_hammer-on.sfz", "Programs/07-black_twang.sfz",
                   "Programs/08-black_staccato.sfz", "Programs/09-black_behind_the_bridge.sfz",
                   "Programs/10-combo_twang.sfz", "Programs/11-combo_staccato.sfz",
                   "Programs/01-green_keyswitch.sfz", "Programs/02-black_keyswitch.sfz",
                   "Programs/03-combo_keyswitch.sfz"),
            presets=_presets(
                "guitar",
                ("Green (Gretsch) twang", "Programs/04-green_twang.sfz"),
                ("Green staccato", "Programs/05-green_staccato.sfz"),
                ("Green hammer-on", "Programs/06-green_hammer-on.sfz"),
                ("Black (Hofner) twang", "Programs/07-black_twang.sfz"),
                ("Black staccato", "Programs/08-black_staccato.sfz"),
                ("Black behind the bridge", "Programs/09-black_behind_the_bridge.sfz"),
                ("Both guitars twang", "Programs/10-combo_twang.sfz"),
                ("Both guitars staccato", "Programs/11-combo_staccato.sfz"),
                ("Green keyswitched", "Programs/01-green_keyswitch.sfz"),
                ("Black keyswitched", "Programs/02-black_keyswitch.sfz"),
                ("Both keyswitched", "Programs/03-combo_keyswitch.sfz")),
            tags=("guitar", "electric"),
            description="Two clean electric guitars (a Gretsch Anniversary and a Hofner Club): "
                        "sustained, staccato, hammer-on and noises. 24-bit WAV; the programs "
                        "use #include and #define.",
            credit="Electric guitars: Black And Green Guitars by Karoryfer Samples, recorded "
                   "by Brian Wood (CC0)."),
        _karoryfer(
            "karoryfer-shinyguitar", "Karoryfer Shinyguitar", "1.002", "karoryfer.shinyguitar",
            "Karoryfer.Shinyguitar.v1.002.zip",
            "23cf4030cbf9ce9e2c84d7cbb1c022fbca124bfff0ad118a14eda7cf6c921d7b", 368_411_844,
            root="files/Shinyguitar",
            files=("Programs/main.sfz",),
            presets=(
                {"name": "Acoustic (microphone)", "file": "Programs/main.sfz",
                 "category": "guitar", "defines": _SHINY_DEFINES, "cc": {"100": 127}},
                {"name": "Electric (magnetic pickup)", "file": "Programs/main.sfz",
                 "category": "guitar", "defines": _SHINY_DEFINES, "cc": {"100": 0}},
                {"name": "Blend of pickup and microphone", "file": "Programs/main.sfz",
                 "category": "guitar", "defines": _SHINY_DEFINES, "cc": {"100": 64}},
            ),
            tags=("guitar", "acoustic", "electric"),
            description="Archtop guitar recorded through a microphone and its magnetic pickup "
                        "at once, with release noises. 24-bit WAV. One program: controller 100 "
                        "blends pickup (0) and microphone (127); its sample folder is the "
                        "variable $sample_dir, which the presets define.",
            statement="This sample library is royalty-free for all commercial and "
                      "non-commercial use.",
            credit="Guitar: Shinyguitar by Karoryfer Samples, played by D. Smolken (CC0)."),
        _vsco("strings", "strings", 755_436_264,
              ("ViolinEnsSusVib.sfz", "ViolinEnsSusVib-Quiet.sfz", "ViolinEnsPizz.sfz",
               "ViolinEnsSpic.sfz", "ViolinEnsTrem.sfz", "ViolinEns-KS.sfz",
               "ViolaEnsSusVib.sfz", "ViolaEnsSusVib-Quiet.sfz", "ViolaEnsPizz.sfz",
               "ViolaEnsSpic.sfz", "ViolaEnsTrem.sfz", "ViolaEns-KS.sfz",
               "CelloEnsSusVib.sfz", "CelloEnsSusVib-Quiet.sfz", "CelloEnsPizz.sfz",
               "CelloEnsSpic.sfz", "CelloEnsTrem.sfz", "CelloEns-KS.sfz",
               "SViolinVib.sfz", "SViolinVib-Quiet.sfz", "SViolinPizz.sfz", "SViolinSpic.sfz",
               "SViolinTrem.sfz", "SViolin-KS.sfz",
               "ContrabassSusVB.sfz", "ContrabassSusVB-Quiet.sfz", "ContrabassSusNV.sfz",
               "ContrabassPizz.sfz", "ContrabassSpic.sfz", "ContrabassTrem.sfz",
               "Contrabass-KS.sfz", "Harp.sfz"),
              ("strings",),
              "Violin, viola and cello sections, solo violin, solo contrabass and harp: "
              "sustained with vibrato, pizzicato, spiccato and tremolo. WAV.", "strings"),
        _vsco("brass", "brass", 329_761_943,
              ("TrumpetSus.sfz", "TrumpetSusVib.sfz", "TrumpetStac.sfz",
               "TrumpetStraightMuteSus.sfz", "TrumpetHarmonMuteSus.sfz", "FHornSus.sfz",
               "FHornStac.sfz", "FHornMute.sfz", "TromboneSus.sfz", "TromboneVib.sfz",
               "TromboneStac.sfz", "TubaSus.sfz", "TubaStac.sfz", "Tuba-KS.sfz"),
              ("brass",),
              "Trumpet (open and muted), French horn, tenor trombone and tuba: sustained and "
              "staccato. WAV.", "brass"),
        _vsco("woodwinds", "woodwinds", 334_397_113,
              ("FluteSusVib.sfz", "FluteSusNV.sfz", "FluteExpVib.sfz", "FluteStac.sfz",
               "Flute-KS.sfz", "PiccoloSus.sfz", "PiccoloStac.sfz", "OboeSusVib.sfz",
               "OboeSusNV.sfz", "OboeStac.sfz", "ClarinetSus.sfz", "ClarinetStac.sfz",
               "Clarinet-KS.sfz", "BassoonSus.sfz", "BassoonVib.sfz", "BassoonStac.sfz"),
              ("woodwind",),
              "Flute, piccolo, oboe, clarinet and bassoon: sustained and staccato. WAV.",
              "woodwind"),
        _vsco("percussion", "percussion and mallets", 268_664_416,
              ("GM-StylePerc.sfz", "Timpani.sfz", "TimpaniRolls.sfz", "Glockenspiel.sfz",
               "Xylophone.sfz", "Marimba.sfz", "TubularBells.sfz"),
              ("percussion", "mallets", "drums"),
              "Orchestral percussion on a General MIDI style key map, timpani hits and rolls, "
              "glockenspiel, xylophone, marimba and tubular bells. WAV.", "percussion"),
        _vsco("keys", "organ and upright pianos", 547_261_728,
              ("UprightPiano.sfz", "VSUpright1.sfz", "OrganLoud.sfz", "OrganQuiet.sfz",
               "OrganLoudPedal.sfz", "OrganQuietPedal.sfz"),
              ("piano", "organ"),
              "Two upright pianos and a pipe organ (loud and quiet registrations, manuals "
              "and pedal). WAV.", "keys"),
        SoundSpec(
            id="jrhodes3d", kind="sfz", name="jRhodes3d", version="2022-10-02",
            publisher="Jeff Learman", homepage="https://github.com/jlearman/jRhodes3d-wav",
            description="1977 Rhodes Mark I Stage 73 electric piano, full-length samples in "
                        "five velocity layers, mono and stereo mappings. WAV.",
            tags=("piano", "electric-piano", "keys"),
            files=("jRhodes3d-mono-wav/_jRhodes3d-mono-wav.sfz",
                   "jRhodes3d-st-wav/_jRhodes3d-st-wav.sfz",
                   "jRhodes3d-mono-wav/_jRhodes3d-mono-no-xfade-wav.sfz",
                   "jRhodes3d-st-wav/_jRhodes3d-st-no-xfade-wav.sfz"),
            presets=_presets(
                "electric piano",
                ("Rhodes (mono)", "jRhodes3d-mono-wav/_jRhodes3d-mono-wav.sfz"),
                ("Rhodes (stereo)", "jRhodes3d-st-wav/_jRhodes3d-st-wav.sfz"),
                ("Rhodes (mono, no velocity crossfade)",
                 "jRhodes3d-mono-wav/_jRhodes3d-mono-no-xfade-wav.sfz"),
                ("Rhodes (stereo, no velocity crossfade)",
                 "jRhodes3d-st-wav/_jRhodes3d-st-no-xfade-wav.sfz")),
            listing="jrhodes3d", size=235_987_495,
            license="Free for use as a musical instrument (the author's own terms); the "
                    "samples themselves CC BY-NC 4.0",
            license_file="files/LICENSE",
            license_url="https://github.com/jlearman/jRhodes3d-wav/blob/master/LICENSE",
            license_source="LICENSE in github.com/jlearman/jRhodes3d-wav (commit a886e6c)",
            commercial_use=True,
            commercial_use_statement="License for musicians using this to make music: CC0 [...] "
                                     "This sample set is provided free for use as a musical "
                                     "instrument. You can use the sounds in any way in your "
                                     "works as an artist, just as you could with a real "
                                     "instrument.",
            unclear="The header of the .sfz files names another licence for the set (\"CC-SA-BY"
                    "\", with the comment \"Free to use in your music without share-alike or "
                    "attribution\"); the LICENSE file says CC BY-NC 4.0 for distributing the "
                    "samples.",
            terms=("Distributing the samples themselves (in an application, an instrument or a "
                   "sample set) is licensed CC BY-NC 4.0: credit required, non-commercial only.",
                   "No credit is asked for music made with it."),
            credits=("Electric piano: jRhodes3d by Jeff Learman.",),
        ),
        SoundSpec(
            id="upright-piano-kw", kind="sfz", name="Upright Piano KW", version="2022-02-21",
            publisher="FreePats project (Gonzalo and Roberto)",
            homepage="https://github.com/freepats/upright-piano-kw",
            description="Kawai upright piano recorded in a living room. FLAC samples: the "
                        "sampler needs the optional soundfile package.",
            tags=("piano", "flac"), files=("UprightPianoKW-20220221.sfz",),
            presets=_presets("piano", ("Upright Piano KW", "UprightPianoKW-20220221.sfz")),
            listing="upright_piano_kw", size=32_967_852,
            license="CC0 1.0 Universal", license_file="files/LICENSE", license_url=_CC0_URL,
            license_source="README.md and LICENSE in github.com/freepats/upright-piano-kw "
                           "(commit 570f6c6)",
            commercial_use=True,
            commercial_use_statement="Published under the terms of Creative Commons CC0 public "
                                     "domain dedication",
            unclear="The README does not use the word \"commercial\"; CC0 waives all rights "
                    "in its own text.",
            terms=("No credit is required.",),
            credits=("Piano: Upright Piano KW, FreePats project (CC0).",),
        ),
        SoundSpec(
            id="greg-sullivan-e-pianos", kind="sfz", name="Greg Sullivan's E-Pianos",
            version="2020-08-23", publisher="Greg Sullivan, SFZ mapping by kinwie",
            homepage="https://github.com/sfzinstruments/GregSullivan.E-Pianos",
            description="Yamaha CP80 electric grand, Hohner Pianet T and Wurlitzer EP200. FLAC "
                        "samples: the sampler needs the optional soundfile package; the "
                        "mappings use ARIA extensions.",
            tags=("piano", "electric-piano", "keys", "flac"),
            files=("Wurlitzer EP200/Wurlitzer EP200.sfz", "CP80/CP80.sfz",
                   "Pianet T/Pianet T.sfz"),
            presets=_presets(
                "electric piano",
                ("Wurlitzer EP200", "Wurlitzer EP200/Wurlitzer EP200.sfz"),
                ("Yamaha CP80 electric grand", "CP80/CP80.sfz"),
                ("Hohner Pianet T", "Pianet T/Pianet T.sfz")),
            listing="greg_sullivan_e_pianos", size=21_484_144,
            license="CC BY 3.0 Unported", license_file="files/LICENSE",
            license_url="http://creativecommons.org/licenses/by/3.0/",
            license_source="README.md and LICENSE in github.com/sfzinstruments/"
                           "GregSullivan.E-Pianos (commit 8c3e581)",
            commercial_use=True,
            commercial_use_statement="This repository contains 3 piano instruments made by Greg "
                                     "Sullivan, with the author permission with the request for "
                                     "attribution. [...] This work is licensed under a Creative "
                                     "Commons Attribution 3.0 Unported License.",
            unclear="The README does not use the word \"commercial\"; the licence it names, "
                    "CC BY 3.0, permits commercial use in its own text. The author's own page "
                    "(sullivang.net) was not read.",
            terms=("Attribution is required.",),
            credits=("Electric pianos: Greg Sullivan's E-Pianos (sullivang.net), SFZ mapping by "
                     "kinwie, CC BY 3.0.",),
        ),
        SoundSpec(
            id="splendid-grand-piano", kind="sfz", name="Splendid Grand Piano",
            version="2024-05-06", publisher="AKAI (samples), SFZ mapping by kinwie",
            homepage="https://github.com/sfzinstruments/SplendidGrandPiano",
            description="Steinway grand in four velocity layers. FLAC samples: the sampler needs "
                        "the optional soundfile package; the mapping uses #include, #define and "
                        "ARIA extensions.",
            tags=("piano", "flac"), files=("Splendid Grand Piano.sfz",),
            presets=_presets("piano", ("Splendid Grand Piano", "Splendid Grand Piano.sfz")),
            listing="splendid_grand_piano", size=76_861_250,
            license="Public domain (as stated by the packager)",
            license_file="files/README.md",
            license_source="README.md in github.com/sfzinstruments/SplendidGrandPiano "
                           "(commit 1c595d8)",
            commercial_use=None,
            commercial_use_statement="Public Domain samples by AKAI. This samples set was "
                                     "released as public domain in early 2000 by Akai company.",
            unclear="The statement is the packager's; no document from AKAI comes with the "
                    "files. The repository has no licence file for the SFZ mapping; its README "
                    "asks that a redistributed derivative says it is derived from this "
                    "instrument.",
            credits=("Piano: Splendid Grand Piano, samples by AKAI (public domain), SFZ mapping "
                     "by kinwie.",),
        ),
        SoundSpec(
            id="drum-machines-oramics", kind="drumkit",
            name="Drum machines (CR-78, LinnDrum, TR-505, Rhythm King)", version="2017-03-05",
            publisher="oramics/sampled collection (danigb); recordings by others",
            homepage="https://github.com/oramics/sampled",
            description="One-shot samples of four vintage drum machines: Roland CR-78, LinnDrum "
                        "LM-2, Roland TR-505 and Maestro Rhythm King MRK-2. One folder of WAV "
                        "files per machine, named kick, snare, hihat and so on.",
            tags=("drums", "electronic", "percussion"), root="files/DM",
            files=("LM-2/samples", "TR-505/samples", "CR-78/samples", "MRK-2/samples"),
            presets=(
                {"name": "LinnDrum LM-2", "file": "LM-2/samples", "category": "drums"},
                {"name": "Roland TR-505", "file": "TR-505/samples", "category": "drums"},
                {"name": "Roland CR-78", "file": "CR-78/samples", "category": "drums"},
                {"name": "Maestro Rhythm King MRK-2", "file": "MRK-2/samples",
                 "category": "drums"},
            ),
            listing="oramics_drum_machines", size=2_452_025,
            license="Public Domain (as stated by the collection)",
            license_file="files/DM/LM-2/README.md",
            license_source="the README.md of each machine in github.com/oramics/sampled (commit "
                           "84d3405), which reads \"license: Public Domain\" and names the "
                           "source of the recordings",
            commercial_use=None,
            commercial_use_statement="license: Public Domain",
            unclear="The statement is the collector's, not the recordists'. The READMEs name "
                    "the sources (boxedear.com, machines.hyperreal.org, submodern.com, "
                    "progsounds.com); those pages were not reachable and their terms were not "
                    "read.",
            credits=("Drum machine samples: oramics/sampled collection (CR-78 from boxedear.com, "
                     "LinnDrum from machines.hyperreal.org, MRK-2 from submodern.com, TR-505 "
                     "from progsounds.com).",),
        ),
        SoundSpec(
            id="voxengo-ir", kind="impulse", name="Voxengo free reverb impulse responses",
            version="IMreverbs", publisher="Voxengo (Aleksey Vaneev)",
            homepage="https://www.voxengo.com/impulses/",
            description="38 modelled stereo impulse responses: halls, rooms, a church, drum "
                        "rooms, a parking garage, an opera hall and four speaker cabinets. "
                        "16-bit WAV, 44.1 kHz.",
            tags=("ir", "reverb", "room", "hall", "cabinet"), root="files/IR/Voxengo",
            files=("samples",),
            presets=(
                {"name": "Nice drum room", "file": "samples/nice_drum_room.wav",
                 "category": "room"},
                {"name": "Small drum room", "file": "samples/small_drum_room.wav",
                 "category": "room"},
                {"name": "Ruby room", "file": "samples/ruby_room.wav", "category": "room"},
                {"name": "Vocal duo", "file": "samples/vocal_duo.wav", "category": "room"},
                {"name": "French 18th century salon",
                 "file": "samples/french_18th_century_salon.wav", "category": "room"},
                {"name": "Large long echo hall", "file": "samples/large_long_echo_hall.wav",
                 "category": "hall"},
                {"name": "Musikvereinsaal", "file": "samples/musikvereinsaal.wav",
                 "category": "hall"},
                {"name": "Scala Milan opera hall", "file": "samples/scala_milan_opera_hall.wav",
                 "category": "hall"},
                {"name": "St Nicolaes church", "file": "samples/st_nicolaes_church.wav",
                 "category": "church"},
                {"name": "Direct cabinet 1", "file": "samples/direct_cabinet_n1.wav",
                 "category": "cabinet"},
                {"name": "Direct cabinet 2", "file": "samples/direct_cabinet_n2.wav",
                 "category": "cabinet"},
                {"name": "Direct cabinet 3", "file": "samples/direct_cabinet_n3.wav",
                 "category": "cabinet"},
                {"name": "Direct cabinet 4", "file": "samples/direct_cabinet_n4.wav",
                 "category": "cabinet"},
            ),
            listing="voxengo_ir", size=14_454_813,
            license="Voxengo impulse files licence", license_file="files/IR/Voxengo/LICENSE.txt",
            license_url="https://www.voxengo.com/impulses/",
            license_source="IR/Voxengo/LICENSE.txt in github.com/oramics/sampled (commit "
                           "84d3405), a copy of the pack: voxengo.com was not reachable when "
                           "this entry was made, so the files were not compared with the "
                           "publisher's own download",
            commercial_use=True,
            commercial_use_statement="Permission is granted to anyone to use these impulse "
                                     "files royalty-free for any purpose, including commercial "
                                     "usage.",
            terms=("\"You may not sell these impulse files or earn any direct or indirect "
                   "profit from their distribution.\"",
                   "Copies must be complete, unaltered, free of charge and carry the licence "
                   "text.",),
            credits=("Reverb impulse responses: Voxengo (Aleksey Vaneev), voxengo.com/impulses.",),
        ),
    ]


REGISTRY: dict[str, SoundSpec] = {s.id: s for s in _specs()}
