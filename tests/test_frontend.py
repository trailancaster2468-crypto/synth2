"""Browser regression tests for the web studio front end (synth2/server/static/app.js).

A live studio server runs in a background thread and the page is driven in headless
Chromium through Playwright for Node: one browser run covers every scenario, each in a
fresh page and project. The tests are skipped when node, the ``playwright`` package or
its Chromium build is not installed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from synth2.server.app import make_server

# Scenario -> what it guards against.
SCENARIOS = {
    "lyric_brackets": "inline lyric '[s t aa r]' is one explicit-phoneme lyric, not split over the next notes",
    "concurrent_put": "a GUI edit after a tool call merges with it instead of overwriting the tool's note",
    "poll_history_race": "an edit made while poll() awaits /api/history still merges the server change",
    "poll_reload_race": "an edit made while poll() reloads the project is neither dropped nor overwrites",
    "put_retry": "a failed PUT stays pending, is retried, and export refuses to use the stale server copy",
    "drag_jitter": "clicking an off-grid note with 1px jitter neither moves, snaps nor resizes it",
    "loop_padding": "loop playback wraps at the loop end even when the render is shorter",
    "loop_toggle_playing": "moving the loop end or toggling the loop during playback takes effect",
    "inspector_click": "inspector buttons still click when the panel would re-render mid-click",
    "tracklist_click": "renaming a track then clicking its Solo button applies both",
    "lyric_track_switch": "switching track while the lyric editor is open keeps the typed lyric",
    "color_pick": "a dismissed color picker does not recolor that track on a later pick",
    "noop_commands": "Q / ArrowLeft / Shift+ArrowLeft / reset that change nothing keep the redo stack",
    "merge3_unit": "three-way project merge keeps both sides' edits; the local edit wins conflicts",
}

E2E_JS = r"""
const { chromium } = require('playwright');

const BASE = process.argv[2].replace(/\/$/, '');

async function api(method, path, body) {
  const r = await fetch(BASE + path, {
    method, headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined,
  });
  const text = await r.text();
  if (!r.ok) throw new Error(`${method} ${path}: ${r.status} ${text}`);
  return text ? JSON.parse(text) : null;
}
let nid = 0;
const note = (onset, duration, pitch, lyric) => ({ id: 'n' + (++nid), onset, duration, pitch, lyric });
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function wav(sec, sr = 22050) {
  const n = Math.round(sec * sr), ch = 2, size = n * ch * 2;
  const b = Buffer.alloc(44 + size);
  b.write('RIFF', 0); b.writeUInt32LE(36 + size, 4); b.write('WAVE', 8); b.write('fmt ', 12);
  b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(ch, 22); b.writeUInt32LE(sr, 24);
  b.writeUInt32LE(sr * ch * 2, 28); b.writeUInt16LE(ch * 2, 32); b.writeUInt16LE(16, 34);
  b.write('data', 36); b.writeUInt32LE(size, 40);
  return b;
}
/** Answer /api/render with `sec` seconds of silence (the real engine is not under test here). */
async function stubRender(page, sec) {
  await page.route('**/api/render', async (route) => {
    const body = JSON.parse(route.request().postData() || '{}');
    const startSec = (body.start_beat || 0) * 0.5; // 120 bpm
    await route.fulfill({
      status: 200, body: wav(sec),
      headers: { 'Content-Type': 'audio/wav', 'X-Render-Report': JSON.stringify({ start_seconds: startSec, duration_s: sec }) },
    });
  });
}

async function setup(page, tracks) {
  await api('POST', '/api/new', { title: 'E2E', bpm: 120 });
  const p = await api('GET', '/api/project');
  const t0 = p.tracks[0];
  p.tracks = tracks.map((t, i) => ({ ...JSON.parse(JSON.stringify(t0)), id: 'trk' + (i + 1), name: t.name, notes: t.notes }));
  await api('PUT', '/api/project', p);
  await page.goto(BASE + '/');
  await page.waitForFunction(() => {
    const A = window.Synth2App;
    return A && A.store && A.store.revision != null && A.store.project.title === 'E2E';
  });
  await frames(page, 2);
}
async function frames(page, n = 2) {
  await page.evaluate((k) => new Promise((res) => { let i = 0; const f = () => (++i >= k ? res() : requestAnimationFrame(f)); requestAnimationFrame(f); }), n);
}
/** Wait until the client has pushed its edits. */
async function settle(page, ms = 100) {
  await sleep(ms);
  await page.evaluate(async () => {
    const s = window.Synth2App.store;
    for (let i = 0; i < 5 && s.pending; i++) await s.flush();
  });
}
const serverNotes = async (ti = 0) => (await api('GET', '/api/project')).tracks[ti].notes;
const pairs = (ns) => ns.map((n) => [n.lyric, n.pitch]);
const clientPairs = (page) => page.evaluate(() => window.Synth2App.store.track.notes.map((n) => [n.lyric, n.pitch]));
async function selectNote(page, idx) {
  await page.evaluate((i) => { const A = window.Synth2App; A.store.select([A.store.track.notes[i].id]); }, idx);
  await frames(page, 2);
}
/** Client-space point inside a note ('mid') or on its resize edge ('edge'). */
async function notePoint(page, idx, where) {
  return page.evaluate(([i, w]) => {
    const A = window.Synth2App, r = A.roll, n = A.store.track.notes[i];
    const rect = r.cv.getBoundingClientRect();
    const nx = r.xOf(n.onset), nw = Math.max(3, n.duration * r.ppb);
    const x = w === 'edge' ? nx + nw - 1.5 : nx + Math.min(10, nw / 3);
    return { x: rect.left + x, y: rect.top + r.yOf(n.pitch) + r.noteH / 2 };
  }, [idx, where]);
}

const S = {};

S.lyric_brackets = async (page) => {
  await setup(page, [{ name: 'Lead', notes: ['twinkle', '+', 'little', '+', 'star'].map((l, i) => note(i, 1, 60, l)) }]);
  const typeLyric = async (text) => {
    await page.evaluate(() => { const A = window.Synth2App; A.roll.openLyric(A.store.track.notes[0]); });
    await page.keyboard.type(text);
    await page.keyboard.press('Enter');
    await settle(page, 300);
  };
  await typeLyric('[s t aa r]');
  const a = (await serverNotes()).map((n) => n.lyric);
  await typeLyric('hello [w er l d]');
  const b = (await serverNotes()).map((n) => n.lyric);
  return { ok: eq(a, ['[s t aa r]', '+', 'little', '+', 'star']) && eq(b, ['hello', '+', '[w er l d]', '+', 'star']), detail: { a, b } };
};

const AGENT = { name: 'add_notes', args: { track: 0, notes: [{ onset: 2, duration: 1, pitch: 'E4', lyric: 'agent' }] } };
const WANT = [['la', 61], ['agent', 64]];

S.concurrent_put = async (page) => {
  await setup(page, [{ name: 'Lead', notes: [note(0, 1, 60, 'la')] }]);
  await selectNote(page, 0);
  await api('POST', '/api/tool', AGENT);
  await page.keyboard.press('ArrowUp');
  await settle(page);
  const server = pairs(await serverNotes()), client = await clientPairs(page);
  return { ok: eq(server, WANT) && eq(client, WANT), detail: { server, client } };
};

S.poll_history_race = async (page) => {
  await setup(page, [{ name: 'Lead', notes: [note(0, 1, 60, 'la')] }]);
  await selectNote(page, 0);
  await api('POST', '/api/tool', AGENT);
  await page.route('**/api/history', async (route) => { await sleep(300); await route.continue(); });
  await page.evaluate(() => { window.__poll = window.Synth2App.poll(); });
  await sleep(80);
  await page.keyboard.press('ArrowUp'); // lands while poll() waits for /api/history
  await page.evaluate(() => window.__poll);
  await settle(page);
  await page.unroute('**/api/history');
  const server = pairs(await serverNotes());
  return { ok: eq(server, WANT), detail: { server } };
};

S.poll_reload_race = async (page) => {
  await setup(page, [{ name: 'Lead', notes: [note(0, 1, 60, 'la')] }]);
  await selectNote(page, 0);
  await api('POST', '/api/tool', AGENT);
  await page.route('**/api/project', async (route) => {
    if (route.request().method() === 'GET') await sleep(300);
    await route.continue();
  });
  await page.evaluate(() => { window.__poll = window.Synth2App.poll(); });
  await sleep(120);
  await page.keyboard.press('ArrowUp'); // lands while poll() reloads the project
  await page.evaluate(() => window.__poll);
  await settle(page);
  await page.unroute('**/api/project');
  const server = pairs(await serverNotes()), client = await clientPairs(page);
  return { ok: eq(server, WANT) && eq(client, WANT), detail: { server, client } };
};

S.put_retry = async (page) => {
  await setup(page, [{ name: 'Lead', notes: [note(0, 1, 60, 'la')] }]);
  await selectNote(page, 0);
  let failNext = 1, failed = 0, exports = 0;
  page.on('request', (r) => { if (r.url().includes('/api/export')) exports++; });
  await page.route('**/api/project', (route) => {
    if (route.request().method() === 'PUT' && failNext > 0) { failNext--; failed++; return route.abort(); }
    return route.continue();
  });
  await page.keyboard.press('ArrowUp');
  await sleep(600);
  const pendingAfterFail = await page.evaluate(() => window.Synth2App.store.pending);
  const before = (await serverNotes())[0].pitch;
  await page.waitForFunction(() => !window.Synth2App.store.pending, null, { timeout: 8000 }).catch(() => {});
  const after = (await serverNotes())[0].pitch; // retried automatically
  failNext = 100; // server unreachable: export must not ship the stale server copy
  await page.keyboard.press('ArrowUp');
  const synced = await page.evaluate(() => window.Synth2App.store.sync());
  await page.evaluate(() => window.Synth2App.exportAs('mid'));
  failNext = 0;
  const resynced = await page.evaluate(() => window.Synth2App.store.sync());
  const final = (await serverNotes())[0].pitch;
  await page.unroute('**/api/project');
  const ok = pendingAfterFail === true && before === 60 && after === 61 && synced === false && exports === 0 && resynced === true && final === 62;
  return { ok, detail: { failed, pendingAfterFail, before, after, synced, exports, resynced, final } };
};

S.drag_jitter = async (page) => {
  await setup(page, [{ name: 'Lead', notes: [note(1.3, 1, 60, 'a'), note(6.1, 0.7, 62, 'b'), note(9, 0.1, 64, 'c')] }]);
  await page.evaluate(() => {
    const A = window.Synth2App, r = A.roll;
    A.store.view.ppb = 60; A.store.view.scrollX = 0;
    r.scrollY = (127 - 62) * r.noteH - r.gridH / 2; r.clampScroll(); r.draw();
  });
  const rev0 = (await api('GET', '/api/history')).revision;
  const jiggle = async (pt) => {
    await page.mouse.move(pt.x, pt.y);
    await page.mouse.down();
    await page.mouse.move(pt.x + 1, pt.y);
    await page.mouse.up();
  };
  await jiggle(await notePoint(page, 0, 'mid')); // off-grid onset (snap 1/8)
  await jiggle(await notePoint(page, 1, 'edge')); // off-grid end
  await jiggle(await notePoint(page, 2, 'edge')); // shorter than the minimum drag length
  await settle(page);
  const local = await page.evaluate(() => window.Synth2App.store.track.notes.map((n) => [n.onset, n.duration]));
  const server = (await serverNotes()).map((n) => [n.onset, n.duration]);
  const rev1 = (await api('GET', '/api/history')).revision;
  const p = await notePoint(page, 0, 'mid'); // a real drag (one beat = 60 px) still moves and snaps
  await page.mouse.move(p.x, p.y); await page.mouse.down();
  for (let i = 1; i <= 6; i++) await page.mouse.move(p.x + i * 10, p.y);
  await page.mouse.up();
  await settle(page);
  const moved = (await serverNotes())[0].onset;
  const want = [[1.3, 1], [6.1, 0.7], [9, 0.1]];
  return { ok: eq(local, want) && eq(server, want) && rev1 === rev0 && moved === 2.5, detail: { local, server, rev0, rev1, moved } };
};

S.loop_padding = async (page) => {
  await setup(page, [{ name: 'Lead', notes: [note(0, 2, 60, 'la')] }]);
  await stubRender(page, 0.5);
  const run = (start, end) => page.evaluate(async ([a, b]) => {
    const A = window.Synth2App, T = A.transport;
    Object.assign(A.store.loop, { on: true, start: a, end: b });
    await T.play();
    const out = { loopLen: T.loopLen, dur: T.src && T.src.buffer.duration, loopEnd: T.src && T.src.loopEnd, loop: T.src && T.src.loop };
    T.stop();
    return out;
  }, [start, end]);
  const a = await run(0, 16); // 8 s loop, 0.5 s render
  const b = await run(8, 16); // loop entirely after the last note
  const near = (x, y) => Math.abs(x - y) < 0.01;
  const ok = near(a.loopLen, 8) && near(a.dur, 8) && near(a.loopEnd, 8) && a.loop && near(b.loopLen, 4) && near(b.dur, 4);
  return { ok, detail: { a, b } };
};

S.loop_toggle_playing = async (page) => {
  await setup(page, [{ name: 'Lead', notes: [note(0, 2, 60, 'la'), note(4, 4, 62, 'lo')] }]);
  await stubRender(page, 6);
  await page.evaluate(async () => {
    const A = window.Synth2App;
    Object.assign(A.store.loop, { on: true, start: 0, end: 2 });
    await A.transport.play();
  });
  const st0 = await page.evaluate(() => ({ loopLen: window.Synth2App.transport.loopLen }));
  const pts = await page.evaluate(() => {
    const A = window.Synth2App, r = A.roll, rect = r.cv.getBoundingClientRect();
    A.store.view.scrollX = 0; r.clampScroll(); r.draw();
    return { x0: rect.left + r.xOf(2), x1: rect.left + r.xOf(4), y: rect.top + 6 };
  });
  await page.mouse.move(pts.x0, pts.y); await page.mouse.down(); // drag the loop end in the ruler: beat 2 -> 4
  for (let i = 1; i <= 5; i++) await page.mouse.move(pts.x0 + (pts.x1 - pts.x0) * i / 5, pts.y);
  await page.mouse.up();
  await sleep(300);
  const st1 = await page.evaluate(() => ({ end: window.Synth2App.store.loop.end, loopLen: window.Synth2App.transport.loopLen }));
  await page.keyboard.press('l');
  await sleep(300);
  const st2 = await page.evaluate(() => {
    const T = window.Synth2App.transport;
    return { on: window.Synth2App.store.loop.on, loopLen: T.loopLen, srcLoop: T.src ? T.src.loop : null };
  });
  await page.evaluate(() => window.Synth2App.transport.stop());
  const ok = Math.abs(st0.loopLen - 1) < 0.01 && st1.end === 4 && Math.abs(st1.loopLen - 2) < 0.01 && st2.on === false && st2.loopLen === 0 && st2.srcLoop !== true;
  return { ok, detail: { st0, st1, st2 } };
};

S.inspector_click = async (page) => {
  await setup(page, [{ name: 'Lead', notes: [note(0, 1, 60, 'a'), note(1, 1, 62, 'b'), note(2, 1, 64, 'c')] }]);
  await page.evaluate(() => window.Synth2App.inspector.setTab('note'));
  await selectNote(page, 0);
  await page.keyboard.press('ArrowUp');
  await frames(page, 2);
  const btn = page.locator('#insp-body button', { hasText: 'Delete note' });
  await btn.scrollIntoViewIfNeeded();
  const box = await btn.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  // what arrives while the button is held: the PUT response, the phoneme refresh, a poll reload
  await page.evaluate(() => { const s = window.Synth2App.store; s.emit('history'); s.emit('phonemes'); s.emit('project', { keepView: true }); });
  await frames(page, 3);
  await page.mouse.up();
  await settle(page);
  const a = (await serverNotes()).map((n) => n.lyric);
  await selectNote(page, 0); // edit the inspector lyric, then click Delete note straight away
  const lyr = page.locator('#insp-body input[data-key="lyric"]');
  await lyr.click();
  await lyr.fill('zzz');
  await page.locator('#insp-body button', { hasText: 'Delete note' }).click({ delay: 90 });
  await settle(page);
  const b = (await serverNotes()).map((n) => n.lyric);
  return { ok: eq(a, ['b', 'c']) && eq(b, ['c']), detail: { a, b } };
};

const TWO_TRACKS = () => [{ name: 'Lead', notes: [note(0, 1, 60, 'la')] }, { name: 'Harmony', notes: [note(0, 1, 64, 'ha')] }];

S.tracklist_click = async (page) => {
  await setup(page, TWO_TRACKS());
  const name = page.locator('.track-card[data-id="trk1"] input.tc-name');
  await name.click();
  await name.fill('Lead vocal');
  await page.locator('.track-card[data-id="trk1"] .ms-btn.solo').click({ delay: 90 });
  await settle(page);
  const t = (await api('GET', '/api/project')).tracks[0];
  return { ok: t.name === 'Lead vocal' && t.solo === true, detail: { name: t.name, solo: t.solo } };
};

S.lyric_track_switch = async (page) => {
  await setup(page, TWO_TRACKS());
  await page.evaluate(() => { const A = window.Synth2App; A.roll.openLyric(A.store.track.notes[0]); });
  await page.keyboard.type('hello');
  await page.locator('.track-card[data-id="trk2"] .tc-lbl').first().click();
  await settle(page);
  const lyric = (await serverNotes(0))[0].lyric;
  const track = await page.evaluate(() => window.Synth2App.store.trackId);
  return { ok: lyric === 'hello' && track === 'trk2', detail: { lyric, track } };
};

S.color_pick = async (page) => {
  await setup(page, TWO_TRACKS());
  await page.evaluate(() => { document.getElementById('color-input').click = () => {}; }); // no native picker
  await page.locator('.track-card[data-id="trk1"] .tc-swatch').click(); // dismissed: no change event
  await page.locator('.track-card[data-id="trk2"] .tc-swatch').click();
  await page.evaluate(() => { const i = document.getElementById('color-input'); i.value = '#ff0000'; i.dispatchEvent(new Event('change')); });
  await settle(page);
  const colors = (await api('GET', '/api/project')).metadata.track_colors || {};
  return { ok: eq(colors, { trk2: '#ff0000' }), detail: { colors } };
};

S.noop_commands = async (page) => {
  await setup(page, [{ name: 'Lead', notes: [note(0, 0.25, 60, 'la')] }]);
  await selectNote(page, 0);
  await page.keyboard.press('ArrowUp');
  await settle(page);
  await page.evaluate(() => window.Synth2App.store.undo());
  const h0 = await api('GET', '/api/history');
  await page.keyboard.press('Escape');
  await page.keyboard.press('q'); // nothing selected
  await settle(page);
  const h1 = await api('GET', '/api/history');
  await selectNote(page, 0);
  await page.keyboard.press('ArrowLeft'); // already at beat 0
  await page.keyboard.press('Shift+ArrowLeft'); // already at the minimum length
  await page.evaluate(() => window.Synth2App.roll.clearAttrs()); // nothing to clear
  await settle(page);
  const h2 = await api('GET', '/api/history');
  return { ok: h0.redo === 1 && h1.redo === 1 && h2.redo === 1 && h2.revision === h0.revision, detail: { h0, h1, h2 } };
};

S.merge3_unit = async (page) => {
  await setup(page, [{ name: 'Lead', notes: [note(0, 1, 60, 'la')] }]);
  const out = await page.evaluate(() => {
    const base = { title: 'T', tracks: [{ id: 't1', gain_db: 0, notes: [
      { id: 'a', pitch: 60, lyric: 'a' }, { id: 'b', pitch: 62, lyric: 'b' }, { id: 'c', pitch: 64, lyric: 'c' }] }] };
    const mine = clone(base), theirs = clone(base);
    mine.title = 'Mine'; mine.tracks[0].notes[0].pitch = 61; mine.tracks[0].notes.splice(1, 1);
    mine.tracks[0].notes.push({ id: 'm', pitch: 70, lyric: 'mine' });
    theirs.title = 'Theirs'; theirs.tracks[0].gain_db = -3; theirs.tracks[0].notes[0].lyric = 'agent'; theirs.tracks[0].notes[2].pitch = 65;
    theirs.tracks[0].notes.push({ id: 't', pitch: 72, lyric: 'theirs' });
    return merge3(base, mine, theirs);
  });
  const want = { title: 'Mine', tracks: [{ id: 't1', gain_db: -3, notes: [
    { id: 'a', pitch: 61, lyric: 'agent' }, { id: 'c', pitch: 65, lyric: 'c' }, { id: 't', pitch: 72, lyric: 'theirs' }, { id: 'm', pitch: 70, lyric: 'mine' }] }] };
  return { ok: eq(out, want), detail: { out } };
};

(async () => {
  let browser;
  try {
    browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
  } catch (e) {
    process.stdout.write(JSON.stringify({ __launch_error__: String(e && e.message || e) }) + '\n');
    return;
  }
  const results = {};
  for (const [name, fn] of Object.entries(S)) {
    const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
    const page = await ctx.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    try {
      results[name] = await fn(page);
    } catch (e) {
      results[name] = { ok: false, detail: { exception: String(e && e.stack || e) } };
    }
    if (errors.length) { results[name].ok = false; results[name].page_errors = errors; }
    await ctx.close();
  }
  await browser.close();
  process.stdout.write(JSON.stringify(results) + '\n');
})();
"""


def _node_env(node: str) -> dict:
    env = dict(os.environ)
    global_modules = Path(node).resolve().parent.parent / "lib" / "node_modules"  # `npm root -g`
    env["NODE_PATH"] = os.pathsep.join(p for p in (env.get("NODE_PATH"), str(global_modules)) if p)
    return env


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    env = _node_env(node)
    probe = subprocess.run([node, "-e", "require.resolve('playwright')"], env=env, capture_output=True, timeout=60)
    if probe.returncode:
        pytest.skip("the playwright package for node is not installed")
    script = tmp_path_factory.mktemp("frontend") / "studio_e2e.js"
    script.write_text(E2E_JS, encoding="utf-8")
    srv = make_server("127.0.0.1", 0, quiet=True)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        proc = subprocess.run([node, str(script), srv.url], env=env, capture_output=True, text=True, timeout=300)
    finally:
        srv.shutdown()
        srv.server_close()
    lines = proc.stdout.strip().splitlines()
    assert proc.returncode == 0 and lines, f"browser run failed:\n{proc.stdout}\n{proc.stderr}"
    data = json.loads(lines[-1])
    if "__launch_error__" in data:
        pytest.skip(f"cannot launch Chromium: {data['__launch_error__'][:300]}")
    return data


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_studio_scenario(results, name):
    r = results.get(name)
    assert r is not None, f"scenario {name} did not run"
    assert r["ok"], f"{SCENARIOS[name]}\n{json.dumps(r, indent=1)}"
