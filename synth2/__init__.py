"""Synth2 Vocal Studio - an AI-native singing voice synthesizer."""
from .project import Project, Track, Note, Curve, CURVE_SPECS, NOTE_ATTRIBUTES
from .timing import note_to_midi, midi_to_note, parse_duration

__version__ = "0.1.0"

__all__ = [
    "Project", "Track", "Note", "Curve", "CURVE_SPECS", "NOTE_ATTRIBUTES",
    "note_to_midi", "midi_to_note", "parse_duration", "render_project", "__version__",
]


def render_project(project, **kw):
    """Render a project to audio. See :func:`synth2.engine.render_project`."""
    from .engine import render_project as _render

    return _render(project, **kw)
