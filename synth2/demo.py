"""Bundled demo productions, built through the same tool layer an AI agent uses."""
from __future__ import annotations

from pathlib import Path

# The demo songs are embedded so `synth2 demo` works in any installation (the repository's examples/ directory
# is not part of the package). examples/<name>.s2s holds the same text for users; a test keeps them identical.
SCORES = {
    "neon_hearts": """\
# Neon Hearts - an original pop demo for Synth2 Vocal Studio
# Lead (Aurora) sings verse + chorus; the demo script adds an AI-generated harmony.
title: Neon Hearts
tempo: 112
time: 4/4
key: C major

track: Lead voice=aurora style=pop lang=en pan=0
r w
# --- verse
E4 e city    E4 e +    G4 q lights   G4 e are   A4 e calling  G4 e +   E4 e out  | D4 e my  E4 h. name  r e
G4 e all     G4 e the  A4 q heart    G4 e beats E4 e burning   F4 e +   E4 e like | D4 e a   C4 h. flame r e
E4 e we      E4 e keep G4 e dancing  G4 e +     A4 e through   C5 e the B4 e silver A4 e + | G4 h. rain  r q
F4 e hold    E4 e me   D4 q close    E4 e and   F4 e never     E4 e +   D4 e let  | C4 e me  D4 h. go    r e
r w
# --- chorus
C5 q neon{tension=0.3}  B4 e +   G4 q. hearts   r e   A4 e we  | C5 q light  D5 e up  C5 e the  E5 h sky{vibrato_depth=0.45}
D5 e burning C5 e +  A4 q bright  C5 e until  D5 e +  C5 e the  A4 e morning | G4 e +  A4 h. light  r e
C5 q neon    B4 e +   G4 q. hearts   r e   A4 e we'll | C5 e never D5 e + C5 q say D5 q goodbye E5 q + | E5 h - D5 h -
A4 e we  C5 e are  C5 q shining  B4 e +  A4 e we  G4 e are  A4 e shining | G4 q +  A4 q tonight  C5 h + | C5 w -
""",
    "amazing_grace": """\
# Amazing Grace (traditional hymn) - soft ballad lead; the demo adds a low harmony with a second voice
title: Amazing Grace
tempo: 72
time: 3/4
key: G major

track: Lead voice=seraph style=ballad lang=en reverb=0.4
r h
D4 q amazing | G4 h + B4 e + G4 e - | B4 h grace A4 q how | G4 h sweet E4 q the | D4 h sound D4 q that
G4 h saved B4 e a G4 e - | B4 h wretch A4 q like | D5 h. me{vibrato_depth=0.4} | r h B4 q i
D5 h once B4 e was G4 e - | B4 h lost A4 q but | G4 h now E4 q am | D4 h found D4 q was
G4 h blind B4 e but G4 e - | B4 h now A4 q i | G4 h. see | G4 h. -
""",
    "sakura": """\
# Sakura Sakura (traditional Japanese folk song) - Japanese lyrics, classic machine-tuned voice
title: Sakura
tempo: 76
time: 4/4
key: A minor

track: Lead voice=nova style=robotic lang=ja
r w
A4 q さ  A4 q く  B4 h ら   | A4 q さ  A4 q く  B4 h ら
A4 q や  B4 q よ  C5 q い  B4 q の | A4 q そ  B4 e ら  A4 e -  F4 h は
E4 q み  C4 q わ  E4 q た  F4 q す | E4 q か  E4 e ぎ  C4 e -  B3 h り
""",
}

# (demo name, extra tool calls applied after loading the score)
DEMOS = {
    "neon_hearts": [
        ("add_harmony", {"source_track": "Lead", "steps": -2, "voice": "atlas", "name": "Harmony",
                         "start_beat": 40, "gain_db": -5, "pan": 0.3}),
        ("add_harmony", {"source_track": "Lead", "steps": 2, "voice": "seraph", "name": "High",
                         "start_beat": 40, "gain_db": -8, "pan": -0.3}),
    ],
    "amazing_grace": [
        ("add_harmony", {"source_track": "Lead", "steps": -2, "voice": "ember", "name": "Low",
                         "style": "ballad", "gain_db": -4, "pan": 0.25}),
    ],
    "sakura": [],
}


def build_demo(name: str):
    """Return a tools Session holding the finished demo project."""
    from .ai.tools import Session, call_tool

    session = Session()
    call_tool(session, "new_project", {"score": SCORES[name]})
    for tool, args in DEMOS[name]:
        call_tool(session, tool, args)
    return session


def render_demos(out_dir: Path, names=None) -> list[Path]:
    from .ai.tools import call_tool

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names or DEMOS:
        session = build_demo(name)
        res = call_tool(session, "render", {"path": str(out_dir / f"{name}.wav")})
        paths.append(Path(res.get("path") or out_dir / f"{name}.wav"))
        call_tool(session, "save_project", {"path": str(out_dir / f"{name}.s2p.json")})
    return paths
