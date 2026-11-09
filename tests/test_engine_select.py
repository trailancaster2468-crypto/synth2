"""Engine selection, the project's engine fields and the neural package's read side."""
import json

import numpy as np
import pytest

from synth2 import Project, render_project
from synth2.engine import backend as eb
from synth2.engine.backend import (EngineUnavailable, current_quality, engine_signature,
                                   normalize_engine, parse_engine, parse_quality, render_options,
                                   resolve_engine)
from synth2.neural import models


def _song(voice="aurora"):
    p = Project.new("t", bpm=120, voice=voice)
    p.tracks[0].add_note(0, 1, "C4", "la")
    return p


@pytest.fixture
def no_env(monkeypatch, tmp_path):
    """Auto mode with an empty models directory."""
    import synth2.neural as neural

    monkeypatch.delenv("SYNTH2_ENGINE", raising=False)
    monkeypatch.delenv("SYNTH2_QUALITY", raising=False)
    monkeypatch.delenv("SYNTH2_NEURAL_BANK", raising=False)
    monkeypatch.setenv("SYNTH2_MODELS", str(tmp_path / "models"))
    neural.refresh()
    yield tmp_path / "models"
    neural.refresh()


def _write_model(base, mid, kind="voicebank", **extra):
    folder = base / mid
    (folder / "files").mkdir(parents=True)
    manifest = {"format": "synth2-model", "version": 1, "id": mid, "kind": kind, "name": mid.title(),
                "model_version": "1.0", "root": "files", **extra}
    (folder / "synth2-model.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


# ---- specs ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("spec,want", [
    (None, ("auto", None)), ("", ("auto", None)), ("  ", ("auto", None)), ("auto", ("auto", None)),
    ("AUTO", ("auto", None)), ("formant", ("formant", None)), (" Neural ", ("neural", None)),
    ("neural:qixuan", ("neural", "qixuan")), ("neural: My Bank ", ("neural", "My Bank")),
])
def test_parse_engine(spec, want):
    assert parse_engine(spec) == want


@pytest.mark.parametrize("spec", ["neural:", "formant:x", "auto:x", "vocaloid", ":x", 3, ["neural"]])
def test_parse_engine_rejects(spec):
    with pytest.raises(ValueError):
        parse_engine(spec)


def test_normalize_engine():
    assert normalize_engine(None) == ""
    assert normalize_engine("auto") == ""
    assert normalize_engine(" FORMANT ") == "formant"
    assert normalize_engine("Neural: qixuan") == "neural:qixuan"


def test_parse_quality():
    assert parse_quality(None) is None and parse_quality("") is None
    assert parse_quality(" High ") == "high"
    with pytest.raises(ValueError):
        parse_quality("ultra")


def test_render_options_nest_and_reset(monkeypatch):
    monkeypatch.delenv("SYNTH2_QUALITY", raising=False)
    assert current_quality() == "normal"
    with render_options(quality="draft"):
        assert current_quality() == "draft"
        with render_options(quality=None):          # None does not override
            assert current_quality() == "draft"
        with render_options(quality="high"):
            assert current_quality() == "high"
        assert current_quality() == "draft"
    assert current_quality() == "normal"
    monkeypatch.setenv("SYNTH2_QUALITY", "high")
    assert current_quality() == "high"
    monkeypatch.setenv("SYNTH2_QUALITY", "nonsense")
    assert current_quality() == "normal"


# ---- resolution -----------------------------------------------------------------------------------

def test_auto_falls_back_to_formant(no_env):
    p = _song()
    assert resolve_engine(p, p.tracks[0]) == ("formant", None)
    assert engine_signature(p) == ["formant"]
    r = render_project(p)
    assert r.report["engines"] == ["formant"]
    assert r.report["tracks"][0]["engine"] == "formant"
    assert "credits" not in r.report


def test_explicit_neural_raises_when_unavailable(no_env):
    p = _song()
    with pytest.raises(EngineUnavailable) as e:
        resolve_engine(p, p.tracks[0], "neural")
    assert "synth2 models install" in str(e.value)
    p.engine = "neural:nobank"
    with pytest.raises(EngineUnavailable):
        render_project(p)
    assert engine_signature(p) == ["unavailable"]     # cache keys never raise
    assert render_project(p, engine="formant").report["engines"] == ["formant"]


def test_resolution_order(no_env, monkeypatch):
    monkeypatch.setattr(eb, "_neural_available", lambda bank=None: (True, ""))
    p = _song()
    tr = p.tracks[0]
    assert resolve_engine(p, tr) == ("neural", None)             # auto prefers neural
    monkeypatch.setenv("SYNTH2_ENGINE", "formant")
    assert resolve_engine(p, tr) == ("formant", None)            # env
    p.engine = "neural:a"
    assert resolve_engine(p, tr) == ("neural", "a")              # project beats env
    tr.engine = "formant"
    assert resolve_engine(p, tr) == ("formant", None)            # track beats project
    assert resolve_engine(p, tr, "neural:b") == ("neural", "b")  # render argument beats all
    assert resolve_engine(p, tr, "auto") == ("formant", None)    # auto defers to the next spec
    tr.engine = "auto"
    assert resolve_engine(p, tr) == ("neural", "a")
    assert engine_signature(p) == ["neural:a"]


def test_malformed_spec_raises_value_error(no_env):
    p = _song()
    p.tracks[0].engine = "vocaloid"
    with pytest.raises(ValueError):
        render_project(p)
    with pytest.raises(ValueError):
        render_project(_song(), quality="ultra")


class _Beep:
    name = "beep"

    def voices(self):
        return ["beeper"]

    def render_track(self, project, track, sample_rate, t0, t1, note_range=None):
        n = max(0, int(round((t1 - t0) * sample_rate)))
        return np.full(n, 0.1, np.float32), {"id": track.id, "name": track.name, "voice": track.voice,
                                             "notes": len(track.notes), "warnings": [], "credits": ["beep by test"]}


def test_plugin_backend_keeps_its_voice_in_auto(no_env, monkeypatch):
    monkeypatch.setattr(eb, "_BACKENDS", dict(eb._BACKENDS))
    monkeypatch.setattr(eb, "_VOICE_BACKEND", dict(eb._VOICE_BACKEND))
    eb.register_backend(_Beep())
    p = _song("beeper")
    assert resolve_engine(p, p.tracks[0]) == ("beep", None)
    r = render_project(p)
    assert r.report["engines"] == ["beep"]
    assert r.report["tracks"][0]["engine"] == "beep"       # filled in for backends that omit it
    assert r.report["credits"] == ["beep by test"]
    assert resolve_engine(p, p.tracks[0], "formant") == ("formant", None)   # an explicit spec wins


# ---- project fields -------------------------------------------------------------------------------

def test_engine_fields_roundtrip():
    p = _song()
    d = p.to_dict()
    assert "engine" not in d and "engine" not in d["tracks"][0]       # unset is not stored
    assert "engine" not in p.summary() and "engine" not in p.summary()["tracks"][0]
    p.engine = "neural"
    p.tracks[0].engine = "neural:qixuan"
    q = Project.from_json(p.to_json())
    assert q.engine == "neural" and q.tracks[0].engine == "neural:qixuan"
    assert q.summary()["engine"] == "neural" and q.summary()["tracks"][0]["engine"] == "neural:qixuan"
    assert q.copy().tracks[0].engine == "neural:qixuan"
    old = Project.from_dict({k: v for k, v in d.items()})              # files from before the field
    assert old.engine == "" and old.tracks[0].engine == ""


# ---- installed models -----------------------------------------------------------------------------

def test_models_dir_env(no_env):
    assert models.models_dir() == no_env
    assert models.installed() == [] and models.find("voicebank") is None


def test_installed_and_find(no_env, monkeypatch):
    _write_model(no_env, "beta", vocoder="voc2", credits=["Voice: Beta"], commercial_use=True,
                 languages=["en"])
    _write_model(no_env, "alpha")
    _write_model(no_env, "voc1", "vocoder", default=True, credits=["Vocoder: One"], commercial_use=False)
    _write_model(no_env, "voc2", "vocoder")
    (no_env / "junk").mkdir()                                       # no manifest
    (no_env / "broken").mkdir()
    (no_env / "broken" / "synth2-model.json").write_text("{not json")
    _write_model(no_env, "wrongkind", "sampler")
    (no_env / ".tmp-install").mkdir()
    assert [m.id for m in models.installed()] == ["alpha", "beta", "voc1", "voc2"]
    assert [m.id for m in models.installed("vocoder")] == ["voc1", "voc2"]
    assert models.find("voicebank").id == "alpha"                   # first by id without a default
    assert models.find("voicebank", "beta").credits == ["Voice: Beta"]
    assert models.find("voicebank", "nope") is None
    assert models.find("vocoder").id == "voc1"                      # "default": true
    monkeypatch.setenv("SYNTH2_NEURAL_BANK", "beta")
    assert models.find("voicebank").id == "beta"
    monkeypatch.setenv("SYNTH2_NEURAL_BANK", "missing")
    assert models.find("voicebank").id == "alpha"
    beta = models.find("voicebank", "beta")
    assert models.vocoder_for(beta).id == "voc2"                    # the manifest's vocoder
    assert models.vocoder_for(models.find("voicebank", "alpha")).id == "voc1"   # the default
    s = beta.summary()
    assert s["commercial_use"] is True and s["languages"] == ["en"] and s["vocoder"] == "voc2"
    assert beta.root == no_env / "beta" / "files" and beta.version == "1.0"


def test_manifest_needs_existing_root(no_env):
    folder = _write_model(no_env, "gone")
    (folder / "files").rmdir()
    assert models.read_manifest(folder) is None
    assert models.installed() == []


def test_neural_status_and_availability(no_env):
    import synth2.neural as neural

    ok, reason = neural.available()
    assert ok is False and reason
    st = neural.status()
    assert st["available"] is False and st["banks"] == [] and st["default_bank"] is None
    assert st["install_hint"] and st["models_dir"] == str(no_env)
    assert neural.credits() == []
    _write_model(no_env, "bank", credits=["Voice: Bank"], commercial_use=True)
    _write_model(no_env, "voc", "vocoder", credits=["Vocoder: V"], commercial_use=False)
    neural.refresh()
    st = neural.status()
    assert st["default_bank"] == "bank" and [b["id"] for b in st["banks"]] == ["bank"]
    assert st["commercial_use"] is False                            # the vocoder forbids it
    assert neural.credits() == ["Voice: Bank", "Vocoder: V"]
    ok, reason = neural.available("other")
    assert ok is False and reason
