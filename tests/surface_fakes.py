"""Stand-ins for the surface tests (command line, AI tools, web studio): a neural engine without
models or onnxruntime, and a small installed sound set made of generated WAV files."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

BANK_CREDIT = "Vocals: Fake Bank v1 (a test voicebank)."
VOCODER_CREDIT = "Vocoder: Fake Vocoder (test), CC BY-NC 4.0."
SOUND_CREDIT = "Bass: Tiny Bass by Nobody (test)."

# what the fake backend was asked to render: (track name, quality)
CALLS: list[tuple[str, str]] = []


def _manifest(base: Path, mid: str, kind: str, **extra) -> None:
    folder = base / mid
    (folder / "files").mkdir(parents=True, exist_ok=True)
    data = {"format": "synth2-model", "version": 1, "id": mid, "kind": kind, "name": mid.title(),
            "model_version": "1.0", "root": "files", "default": True, **extra}
    (folder / "synth2-model.json").write_text(json.dumps(data), encoding="utf-8")


class FakeNeuralBackend:
    """Sings with the formant engine and reports the way the neural backend does."""

    name = "neural"

    def __init__(self, bank=None):
        self.bank = bank or "fakebank"

    def voices(self):
        from synth2.voices import VOICES

        return list(VOICES)

    def render_track(self, project, track, sample_rate, t0, t1, note_range=None):
        from synth2.engine.backend import current_quality
        from synth2.engine.renderer import FormantBackend

        audio, report = FormantBackend().render_track(project, track, sample_rate, t0, t1, note_range)
        CALLS.append((track.name, current_quality()))
        report.update(engine="neural", bank=self.bank, role=track.role, credits=[BANK_CREDIT, VOCODER_CREDIT],
                      neural={"quality": current_quality(), "alignment": [{"frames": [1, 2, 3]}]})
        return audio, report


def install_fake_neural(monkeypatch, tmp_path) -> Path:
    """Make the neural engine look installed: a models folder with a voicebank that allows commercial use
    and a vocoder that does not, and a backend that needs neither. Returns the models folder."""
    import synth2.neural as neural

    base = Path(tmp_path) / "fake-models"
    _manifest(base, "fakebank", "voicebank", license="Fake Bank Terms", commercial_use=True,
              languages=["en", "ja"], credits=[BANK_CREDIT], terms=["Credit the voicebank."])
    _manifest(base, "fakevoc", "vocoder", license="CC BY-NC 4.0", commercial_use=False,
              credits=[VOCODER_CREDIT], terms=["Non-commercial use only."])
    monkeypatch.setenv("SYNTH2_MODELS", str(base))
    monkeypatch.delenv("SYNTH2_NEURAL_BANK", raising=False)

    def available(bank=None):
        if bank in (None, "fakebank"):
            return True, ""
        return False, f"voicebank {bank!r} is not installed in {base}"

    monkeypatch.setattr(neural, "available", available)
    monkeypatch.setattr(neural, "backend", lambda bank=None: FakeNeuralBackend(bank))
    CALLS.clear()
    return base


def empty_models(monkeypatch, tmp_path) -> Path:
    """Point the neural engine at an empty models folder (nothing installed)."""
    import synth2.neural as neural

    base = Path(tmp_path) / "no-models"
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SYNTH2_MODELS", str(base))
    monkeypatch.delenv("SYNTH2_NEURAL_BANK", raising=False)
    neural.refresh()
    return base


def _write_wav(path: Path, hz: float, sr: int = 22050, seconds: float = 0.4) -> None:
    t = np.arange(int(sr * seconds)) / sr
    x = (0.5 * np.sin(2 * np.pi * hz * t) * np.exp(-3.0 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())


def install_fake_sound(monkeypatch, tmp_path, sid: str = "tiny-bass", **extra) -> Path:
    """An installed sound set ``sid``: a two-note multisample in a folder of WAV files, with a credit
    line. Returns the sounds folder (``SYNTH2_SOUNDS`` points at it)."""
    base = Path(tmp_path) / "fake-sounds"
    folder = base / sid / "files" / "bass"
    folder.mkdir(parents=True, exist_ok=True)
    _write_wav(folder / "bass_A1.wav", 55.0)
    _write_wav(folder / "bass_A2.wav", 110.0)
    data = {"format": "synth2-sound", "version": 1, "id": sid, "kind": "samples", "name": "Tiny Bass",
            "files": ["bass"], "tags": ["bass"], "license": "CC BY 4.0", "commercial_use": True,
            "credits": [SOUND_CREDIT], **extra}
    (base / sid / "synth2-sound.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("SYNTH2_SOUNDS", str(base))
    try:
        from synth2.arranger import soundbank

        soundbank.clear_cache()
    except Exception:
        pass
    return base


def backing(sound="tiny-bass") -> dict:
    """A two-bar arrangement with one bass that plays ``sound``."""
    return {"instruments": {"bass": {"preset": "bass_pick", "sound": sound}},
            "patterns": {"line": {"kind": "line", "steps": "R...R...R...R..."}},
            "sections": [{"name": "A", "start_bar": 1, "bars": 2, "chords": ["Am"], "parts": {"bass": "line"}}]}
