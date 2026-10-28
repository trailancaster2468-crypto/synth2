"""Synthesis engine: planner -> expression -> parameter tracks -> vocoder -> mix."""


def __getattr__(name):
    if name in ("render_project", "render_track", "RenderResult"):
        from . import renderer

        return getattr(renderer, name)
    raise AttributeError(name)


__all__ = ["render_project", "render_track", "RenderResult"]
