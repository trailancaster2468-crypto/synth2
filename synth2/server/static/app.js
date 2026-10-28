/* Synth2 Vocal Studio - browser front end.
 *
 * Vanilla JS, no build step, no external assets. Sections:
 *   Utilities . TempoMap . Api . Store . TimeGrid . PianoRoll . ParamLane
 *   TrackList . MasterStrip . Inspector . Transport . Chat . App (boot, menus, shortcuts)
 *
 * State model: the client keeps the project JSON (same shape as Project.to_dict()).
 * Every edit mutates it locally, then Store.commit() schedules a debounced
 * PUT /api/project, which pushes one server-side undo snapshot. Undo/redo are
 * server-side (shared with the AI Director and tool calls); after them the
 * client reloads the project.
 */
'use strict';

/* ============================================================== Utilities */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const EPS = 1e-6;
const round6 = (x) => Math.round(x * 1e6) / 1e6;
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);
const TRACK_COLORS = ['#5eead4', '#fca5a5', '#93c5fd', '#fcd34d', '#c4b5fd', '#86efac', '#f9a8d4', '#fdba74'];
const CURVE_LABELS = {
  pitchDelta: 'Pitch Deviation', loudness: 'Loudness', tension: 'Tension', breathiness: 'Breathiness',
  gender: 'Gender', voicing: 'Voicing', toneShift: 'Tone Shift', vibratoEnv: 'Vibrato Env',
};
const CURVE_TABS = {
  pitchDelta: 'Pitch', loudness: 'Loudness', tension: 'Tension', breathiness: 'Breath',
  gender: 'Gender', voicing: 'Voicing', toneShift: 'Tone', vibratoEnv: 'Vibrato',
};
const CURVE_COLORS = {
  pitchDelta: '#f472b6', loudness: '#60a5fa', tension: '#f59e0b', breathiness: '#a3e635',
  gender: '#c084fc', voicing: '#2dd4bf', toneShift: '#fb7185', vibratoEnv: '#38bdf8',
};
const ATTR_LABELS = {
  vibrato_depth: 'Vibrato depth', vibrato_rate: 'Vibrato rate', vibrato_start: 'Vibrato start',
  transition: 'Transition', overshoot: 'Overshoot', consonant_length: 'Consonant len.',
  tension: 'Tension', breathiness: 'Breathiness', detune: 'Detune', velocity: 'Velocity', language: 'Language',
};
const ATTR_UNITS = {
  vibrato_depth: 'st', vibrato_rate: 'Hz', vibrato_start: 's', transition: 's', overshoot: '',
  consonant_length: 'x', tension: '', breathiness: '', detune: 'ct', velocity: 'x',
};
const STYLE_OVERRIDES = [
  ['transition', 'Transition', 's'], ['overshoot', 'Overshoot', ''], ['scoop', 'Scoop', 'st'], ['fall', 'Fall', 'st'],
  ['vibrato_depth', 'Vibrato depth', 'x'], ['vibrato_rate', 'Vibrato rate', 'x'], ['vibrato_start', 'Vibrato start', 's'],
  ['drift_cents', 'Pitch drift', 'ct'], ['phrase_arc_db', 'Phrase arc', 'dB'], ['tension', 'Tension', ''],
  ['breathiness', 'Breathiness', ''], ['end_breathiness', 'End breathiness', ''], ['timing_humanize', 'Humanize', 's'],
];
const SCALES = {
  major: [0, 2, 4, 5, 7, 9, 11], minor: [0, 2, 3, 5, 7, 8, 10], dorian: [0, 2, 3, 5, 7, 9, 10],
  phrygian: [0, 1, 3, 5, 7, 8, 10], lydian: [0, 2, 4, 6, 7, 9, 11], mixolydian: [0, 2, 4, 5, 7, 9, 10],
  locrian: [0, 1, 3, 5, 6, 8, 10], 'harmonic minor': [0, 2, 3, 5, 7, 8, 11], chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
};
const KEY_TONICS = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];

const midiName = (m) => NOTE_NAMES[((m % 12) + 12) % 12] + (Math.floor(m / 12) - 1);
function parsePitch(s) {
  s = String(s ?? '').trim();
  if (/^-?\d+$/.test(s)) return parseInt(s, 10);
  const m = /^([A-Ga-g])([#b♯♭]{0,2})(-?\d+)$/.exec(s);
  if (!m) return null;
  let v = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }[m[1].toUpperCase()];
  for (const ch of m[2]) v += (ch === '#' || ch === '♯') ? 1 : -1;
  return v + (parseInt(m[3], 10) + 1) * 12;
}
function pitchClassOf(name) {
  const p = parsePitch(name + '4');
  return p == null ? null : ((p % 12) + 12) % 12;
}
function parseKey(key) {
  const m = /^\s*([A-Ga-g][#b]?)\s*(.*)$/.exec(String(key || ''));
  if (!m) return null;
  const tonic = pitchClassOf(m[1].charAt(0).toUpperCase() + m[1].slice(1));
  let mode = (m[2] || 'major').trim().toLowerCase();
  if (mode === 'm' || mode === 'min') mode = 'minor';
  if (mode === 'maj' || mode === '') mode = 'major';
  if (mode === 'aeolian') mode = 'minor';
  if (mode === 'ionian') mode = 'major';
  const steps = SCALES[mode];
  if (tonic == null || !steps) return null;
  return new Set(steps.map((s) => (s + tonic) % 12));
}
function newId() {
  const a = new Uint8Array(6);
  crypto.getRandomValues(a);
  return Array.from(a, (b) => b.toString(16).padStart(2, '0')).join('');
}
function parseFrac(v) {
  v = String(v);
  if (v.includes('/')) { const [a, b] = v.split('/').map(Number); return b ? a / b : 0; }
  return parseFloat(v) || 0;
}
function hexToRgb(hex) {
  let h = String(hex || '#888888').replace('#', '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function rgba(hex, a) { const [r, g, b] = hexToRgb(hex); return `rgba(${r},${g},${b},${a})`; }
function shade(hex, amt) {
  const [r, g, b] = hexToRgb(hex);
  const f = (c) => Math.round(amt < 0 ? c * (1 + amt) : c + (255 - c) * amt);
  return `rgb(${f(r)},${f(g)},${f(b)})`;
}
function fmtNum(v, d = 2) {
  if (v == null || Number.isNaN(v)) return '';
  const s = Number(v).toFixed(d);
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}
function fmtDb(v) { return (v > 0 ? '+' : '') + Number(v).toFixed(1) + ' dB'; }
function fmtPan(v) {
  if (Math.abs(v) < 0.005) return 'C';
  return (v < 0 ? 'L' : 'R') + Math.round(Math.abs(v) * 100);
}
function fmtTime(sec) {
  sec = Math.max(0, sec);
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(3).padStart(6, '0')}`;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function el(tag, props = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
    else if (k === 'dataset') Object.assign(e.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else if (k === 'html') e.innerHTML = v;
    else if (k in e && typeof v !== 'string') e[k] = v;
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return e;
}
function icon(name) {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  const u = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  u.setAttribute('href', '#i-' + name);
  s.append(u);
  return s;
}
function toast(msg, kind = 'info', ms = 3800) {
  const t = el('div', { class: 'toast ' + kind }, msg);
  $('#toasts').append(t);
  setTimeout(() => { t.style.transition = 'opacity .3s'; t.style.opacity = '0'; setTimeout(() => t.remove(), 320); }, ms);
}
function status(msg, isErr = false) {
  const s = $('#sb-msg');
  s.textContent = msg;
  s.classList.toggle('err', !!isErr);
}
function isTyping(target) {
  const t = target || document.activeElement;
  if (!t) return false;
  const tag = t.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable;
}
function sortNotes(track) { track.notes.sort((a, b) => a.onset - b.onset || a.pitch - b.pitch); }
function setRangeFill(input) {
  const min = parseFloat(input.min), max = parseFloat(input.max), v = parseFloat(input.value);
  input.style.setProperty('--fill', `${((v - min) / (max - min)) * 100}%`);
}
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = el('a', { href: url, download: filename });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 1000);
}

/* =============================================================== TempoMap */
/** Piecewise-constant tempo map, identical to synth2.timing.TempoMap. */
class TempoMap {
  constructor(list) {
    let pts = (list || []).map((p) => ({ beat: +p.beat || 0, bpm: +p.bpm })).filter((p) => p.bpm > 0);
    pts.sort((a, b) => a.beat - b.beat);
    if (!pts.length) pts = [{ beat: 0, bpm: 120 }];
    if (pts[0].beat > 0) pts.unshift({ beat: 0, bpm: pts[0].bpm });
    this.pts = pts;
    this.secs = [0];
    for (let i = 1; i < pts.length; i++) this.secs.push(this.secs[i - 1] + (pts[i].beat - pts[i - 1].beat) * 60 / pts[i - 1].bpm);
  }
  _ib(beat) { let i = 0; while (i + 1 < this.pts.length && this.pts[i + 1].beat <= beat) i++; return i; }
  bpmAt(beat) { return this.pts[this._ib(beat)].bpm; }
  beatToSec(beat) { const i = this._ib(beat); const p = this.pts[i]; return this.secs[i] + (beat - p.beat) * 60 / p.bpm; }
  secToBeat(sec) {
    let i = 0;
    while (i + 1 < this.secs.length && this.secs[i + 1] <= sec) i++;
    const p = this.pts[i];
    return p.beat + (sec - this.secs[i]) * p.bpm / 60;
  }
}

/* ==================================================================== Api */
class ApiError extends Error {
  constructor(status, message, data) { super(message); this.status = status; this.data = data; }
}
const Api = {
  async request(method, url, body, { raw = false } = {}) {
    const init = { method, headers: {} };
    if (body !== undefined) {
      if (body instanceof ArrayBuffer || body instanceof Blob) {
        init.body = body;
        init.headers['Content-Type'] = 'application/octet-stream';
      } else {
        init.body = JSON.stringify(body);
        init.headers['Content-Type'] = 'application/json';
      }
    }
    let res;
    try {
      res = await fetch(url, init);
    } catch (e) {
      App.setConnected(false);
      throw new ApiError(0, 'Server unreachable');
    }
    App.setConnected(true);
    if (raw && res.ok) return res;
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = { error: text }; }
    if (!res.ok) throw new ApiError(res.status, (data && data.error) || res.statusText || `HTTP ${res.status}`, data);
    return data;
  },
  getProject: () => Api.request('GET', '/api/project'),
  putProject: (p) => Api.request('PUT', '/api/project', p),
  meta: () => Api.request('GET', '/api/meta'),
  history: () => Api.request('GET', '/api/history'),
  phonemes: () => Api.request('GET', '/api/phonemes'),
  tool: (name, args = {}) => Api.request('POST', '/api/tool', { name, args }),
  undo: () => Api.request('POST', '/api/undo', {}),
  redo: () => Api.request('POST', '/api/redo', {}),
  newProject: (opts = {}) => Api.request('POST', '/api/new', opts),
  save: (path) => Api.request('POST', '/api/save', path ? { path } : {}),
  splitLyrics: (text, language) => Api.request('POST', '/api/split_lyrics', { text, language }),
  ai: (message, reset = false) => Api.request('POST', '/api/ai', { message, reset }),
  importFile: (name, buf) => Api.request('POST', '/api/import?filename=' + encodeURIComponent(name), buf),
  async render(opts) {
    const res = await Api.request('POST', '/api/render', opts, { raw: true });
    let report = {};
    try { report = JSON.parse(res.headers.get('X-Render-Report') || '{}'); } catch { /* ignore */ }
    const buf = await res.arrayBuffer();
    return { buf, report };
  },
  async download(url) {
    const res = await Api.request('GET', url, undefined, { raw: true });
    const cd = res.headers.get('Content-Disposition') || '';
    const m = /filename="?([^";]+)"?/.exec(cd);
    return { blob: await res.blob(), filename: m ? m[1] : 'download' };
  },
};

/* ================================================================== Store */
function normalizeProject(p) {
  p = p || {};
  p.title = p.title ?? 'Untitled';
  p.tracks = Array.isArray(p.tracks) ? p.tracks : [];
  p.tempo = Array.isArray(p.tempo) && p.tempo.length ? p.tempo : [{ beat: 0, bpm: 120 }];
  p.time_signature = Array.isArray(p.time_signature) ? p.time_signature : [4, 4];
  p.master = p.master || {};
  p.metadata = p.metadata || {};
  p.key = p.key || 'C major';
  for (const t of p.tracks) {
    t.id = t.id || newId();
    t.notes = Array.isArray(t.notes) ? t.notes : [];
    t.curves = t.curves || {};
    for (const [k, c] of Object.entries(t.curves)) {
      if (Array.isArray(c)) t.curves[k] = { interpolation: 'cubic', points: c };
      else { c.points = c.points || []; c.interpolation = c.interpolation || 'cubic'; }
    }
    t.voice_params = t.voice_params || {};
    for (const n of t.notes) n.id = n.id || newId();
    sortNotes(t);
  }
  return p;
}

/** Client-side state: project JSON, selection, view, transport, history; pub/sub. */
class Store {
  constructor() {
    this.project = normalizeProject({});
    this.meta = { voices: [], styles: [], curves: {}, note_attributes: {}, voice_params: [], voice_defaults: {} };
    this.tempo = new TempoMap();
    this.trackId = null;
    this.selection = new Set();
    this.playhead = 0;
    this.loop = { on: false, start: 0, end: 0 };
    this.snap = 0.5;
    this.newNoteLen = 'auto';
    this.lastDuration = 1;
    this.version = 0;
    this.revision = null;
    this.history = { undo: null, redo: null };
    this.phonemes = {};
    this.phonemeErrors = {};
    this.clipboard = null;
    this.dirty = false;
    this.busy = false;
    this.path = null;
    this.view = { ppb: 80, scrollX: 0 };
    this._listeners = {};
    this._saveTimer = null;
    this._pending = false;
    this._inflight = 0;
    this._chain = Promise.resolve();
    this._phTimer = null;
  }
  on(evt, fn) { (this._listeners[evt] ||= []).push(fn); }
  emit(evt, detail) { for (const fn of this._listeners[evt] || []) fn(detail); }

  get track() {
    const ts = this.project.tracks;
    return ts.find((t) => t.id === this.trackId) || ts[0] || null;
  }
  trackById(id) { return this.project.tracks.find((t) => t.id === id) || null; }
  trackIndex(t) { return this.project.tracks.indexOf(t); }
  color(track) {
    if (!track) return TRACK_COLORS[0];
    const c = (this.project.metadata.track_colors || {})[track.id];
    return c || TRACK_COLORS[Math.max(0, this.trackIndex(track)) % TRACK_COLORS.length];
  }
  get beatsPerBar() { const [n, d] = this.project.time_signature; return n * 4 / d; }
  get beatLen() { return 4 / this.project.time_signature[1]; }
  get scale() { return parseKey(this.project.key); }
  projectEnd() {
    let e = 0;
    for (const t of this.project.tracks) for (const n of t.notes) e = Math.max(e, n.onset + n.duration);
    return e;
  }
  selectedNotes() {
    const t = this.track;
    return t ? t.notes.filter((n) => this.selection.has(n.id)) : [];
  }
  barBeat(beat) {
    const bpb = this.beatsPerBar, bl = this.beatLen;
    const bar = Math.floor(beat / bpb + EPS);
    const inBar = Math.max(0, beat - bar * bpb);
    const bt = Math.floor(inBar / bl + EPS);
    const tick = Math.round(((inBar - bt * bl) / bl) * 480);
    return { bar: bar + 1, beat: bt + 1, tick: Math.min(479, tick) };
  }
  fmtBeat(beat) { const b = this.barBeat(beat); return `${b.bar}.${b.beat}.${String(b.tick).padStart(3, '0')}`; }

  setProject(p, { keepView = true } = {}) {
    this.project = normalizeProject(p);
    this.tempo = new TempoMap(this.project.tempo);
    if (!this.trackById(this.trackId)) this.trackId = this.project.tracks[0]?.id ?? null;
    const t = this.track;
    const ids = new Set(t ? t.notes.map((n) => n.id) : []);
    this.selection = new Set([...this.selection].filter((id) => ids.has(id)));
    this.version++;
    this.emit('project', { keepView });
    this.refreshPhonemesSoon(10);
  }
  setTrack(id) {
    if (id === this.trackId) return;
    this.trackId = id;
    this.selection.clear();
    this.emit('selection');
  }
  select(ids, { add = false } = {}) {
    if (!add) this.selection.clear();
    for (const id of ids) this.selection.add(id);
    this.emit('selection');
  }

  /** Record an edit: bump version, notify views, schedule a debounced PUT. */
  commit(kind = 'notes') {
    if (kind === 'song') this.tempo = new TempoMap(this.project.tempo);
    this.version++;
    this.dirty = true;
    this.emit(kind);
    this.emit('dirty');
    this._pending = true;
    clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => this.flush(), 300);
  }
  /** Send any pending edit to the server; resolves when all PUTs are done. */
  flush() {
    if (!this._pending) return this._chain;
    clearTimeout(this._saveTimer);
    this._pending = false;
    const snapshot = JSON.parse(JSON.stringify(this.project));
    this._inflight++;
    this._chain = this._chain
      .then(() => Api.putProject(snapshot))
      .then((r) => {
        if (r && r.history) this.setHistory(r.history);
        if (r && r.revision != null) this.revision = r.revision;
        this.refreshPhonemesSoon();
      })
      .catch((e) => {
        toast('Could not sync the edit to the server: ' + e.message, 'err', 6000);
        status('Sync failed: ' + e.message, true);
      })
      .finally(() => { this._inflight--; });
    return this._chain;
  }
  get pending() { return this._pending || this._inflight > 0; }
  setHistory(h) {
    this.history = { undo: h.undo ?? null, redo: h.redo ?? null };
    if (h.revision != null) this.revision = h.revision;
    if (h.path !== undefined) this.path = h.path;
    this.emit('history');
  }
  async reload() {
    await this.flush();
    const p = await Api.getProject();
    this.setProject(p);
    await this.refreshHistory();
  }
  async refreshHistory() {
    try { this.setHistory(await Api.history()); } catch { /* offline */ }
  }
  async undo() { return this._undoRedo('undo'); }
  async redo() { return this._undoRedo('redo'); }
  async _undoRedo(which) {
    if (this.busy) return;
    await this.flush();
    try {
      const r = await (which === 'undo' ? Api.undo() : Api.redo());
      this.setProject(r.project);
      this.setHistory(r.history || {});
      this.dirty = true;
      this.emit('dirty');
      status(which === 'undo' ? 'Undo' : 'Redo');
    } catch (e) {
      status(e.status === 400 ? `Nothing to ${which}` : `${which} failed: ${e.message}`, e.status !== 400);
    }
  }
  refreshPhonemesSoon(ms = 120) {
    clearTimeout(this._phTimer);
    this._phTimer = setTimeout(async () => {
      try {
        const r = await Api.phonemes();
        this.phonemes = r.tracks || {};
        this.phonemeErrors = r.errors || {};
        this.emit('phonemes');
      } catch { /* not critical */ }
    }, ms);
  }
}

/* =============================================================== TimeGrid */
/** Grid line positions shared by the piano roll and the parameter lane. */
function timeGridLines(store, x0, x1, xOf, beatAt, ppb) {
  const bpb = store.beatsPerBar, bl = store.beatLen;
  let step = store.snap > 0 ? store.snap : 0.25;
  while (step * ppb < 7) step *= 2;
  const b0 = Math.max(0, Math.floor(beatAt(x0) / step) * step);
  const b1 = beatAt(x1);
  const out = [];
  const isMul = (b, unit) => Math.abs(b / unit - Math.round(b / unit)) < 1e-5;
  for (let i = Math.round(b0 / step); i * step <= b1 + EPS; i++) {
    const b = i * step;
    const kind = isMul(b, bpb) ? 'bar' : isMul(b, bl) ? 'beat' : 'sub';
    if (kind === 'beat' && bl * ppb < 5) continue;
    out.push({ x: Math.round(xOf(b)) + 0.5, beat: b, kind });
  }
  return out;
}
const GRID_COLORS = { bar: '#3a404c', beat: '#272b34', sub: '#1d2027' };

/* ============================================================== PianoRoll */
const KEYS_W = 58;
const RULER_H = 28;
const SB = 10; // scrollbar thickness

class PianoRoll {
  constructor(store, canvas, wrap) {
    this.s = store;
    this.cv = canvas;
    this.wrap = wrap;
    this.ctx = canvas.getContext('2d');
    this.noteH = 16;
    this.scrollY = 0;
    this.tool = 'draw';
    this.w = 0; this.h = 0; this.dpr = 1;
    this.drag = null;
    this.mouse = null;
    this.hover = null;
    this.showGhosts = true;
    this.showPhonemes = true;
    this.pressedKey = null;
    this.input = $('#lyric-input');
    this.editing = null;
    this._raf = 0;
    new ResizeObserver(() => this.resize()).observe(wrap);
    canvas.addEventListener('pointerdown', (e) => this.onDown(e));
    canvas.addEventListener('pointermove', (e) => this.onMove(e));
    canvas.addEventListener('pointerup', (e) => this.onUp(e));
    canvas.addEventListener('pointercancel', (e) => this.onUp(e));
    canvas.addEventListener('dblclick', (e) => this.onDbl(e));
    canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    canvas.addEventListener('contextmenu', (e) => this.onContext(e));
    canvas.addEventListener('pointerleave', () => { if (!this.drag) { this.mouse = null; this.hover = null; App.cursorInfo(''); this.invalidate(); } });
    for (const ev of ['project', 'notes', 'tracks', 'song', 'selection', 'transport', 'view', 'phonemes']) store.on(ev, () => this.invalidate());
    store.on('project', (d) => { if (!d || !d.keepView) this.centerOnNotes(); this.clampScroll(); });
    store.on('selection', () => { if (this.editing && !this.s.selection.has(this.editing.noteId)) this.closeLyric(false); });
    this.initLyricInput();
  }

  /* ---- geometry */
  get ppb() { return this.s.view.ppb; }
  get scrollX() { return this.s.view.scrollX; }
  set scrollX(v) { this.s.view.scrollX = v; }
  get gridW() { return Math.max(10, this.w - KEYS_W - SB); }
  get gridH() { return Math.max(10, this.h - RULER_H - SB); }
  xOf(beat) { return KEYS_W + beat * this.ppb - this.scrollX; }
  beatAt(x) { return (x - KEYS_W + this.scrollX) / this.ppb; }
  yOf(pitch) { return RULER_H + (127 - pitch) * this.noteH - this.scrollY; }
  pitchAt(y) { return clamp(127 - Math.floor((y - RULER_H + this.scrollY) / this.noteH), 0, 127); }
  contentBeats() {
    const bpb = this.s.beatsPerBar;
    const end = Math.max(this.s.projectEnd(), this.s.loop.end, this.s.playhead);
    return Math.max(Math.ceil(end / bpb + 8) * bpb, 32 * bpb);
  }
  clampScroll() {
    const maxX = Math.max(0, this.contentBeats() * this.ppb - this.gridW);
    this.scrollX = clamp(this.scrollX, 0, maxX);
    const maxY = Math.max(0, 128 * this.noteH - this.gridH);
    this.scrollY = clamp(this.scrollY, 0, maxY);
  }
  snapRound(b, e) {
    const s = this.s.snap;
    if (!s || (e && e.altKey)) return round6(b);
    return round6(Math.round(b / s) * s);
  }
  snapFloor(b, e) {
    const s = this.s.snap;
    if (!s || (e && e.altKey)) return round6(b);
    return round6(Math.floor(b / s + 1e-6) * s);
  }
  minDur() { return this.s.snap > 0 ? Math.min(this.s.snap, 0.25) : 1 / 32; }
  newLen() {
    const v = this.s.newNoteLen;
    if (v !== 'auto') return parseFrac(v);
    return this.s.lastDuration || Math.max(this.s.snap, 1);
  }

  resize() {
    const r = this.wrap.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.w = Math.max(1, Math.floor(r.width));
    this.h = Math.max(1, Math.floor(r.height));
    this.cv.width = Math.round(this.w * this.dpr);
    this.cv.height = Math.round(this.h * this.dpr);
    this.clampScroll();
    this.draw();
    this.s.emit('view');
  }
  invalidate() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this.draw(); });
  }
  local(e) {
    const r = this.cv.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  /* ---- view helpers */
  centerOnNotes() {
    const t = this.s.track;
    const pitches = t && t.notes.length ? t.notes.map((n) => n.pitch) : [60, 72];
    const mid = (Math.min(...pitches) + Math.max(...pitches)) / 2;
    this.scrollY = (127 - mid) * this.noteH - this.gridH / 2;
    if (t && t.notes.length) {
      const first = t.notes[0].onset;
      if (this.xOf(first) < KEYS_W || this.xOf(first) > KEYS_W + this.gridW * 0.8) this.scrollX = Math.max(0, first * this.ppb - 40);
    }
    this.clampScroll();
    this.s.emit('view');
  }
  zoomFit() {
    const t = this.s.track;
    if (!t || !t.notes.length) { this.centerOnNotes(); return; }
    const lo = Math.min(...t.notes.map((n) => n.onset));
    const hi = Math.max(...t.notes.map((n) => n.onset + n.duration));
    const pLo = Math.min(...t.notes.map((n) => n.pitch));
    const pHi = Math.max(...t.notes.map((n) => n.pitch));
    this.s.view.ppb = clamp(this.gridW / Math.max(1, hi - lo + 1), 8, 400);
    this.scrollX = Math.max(0, (lo - 0.5) * this.ppb);
    this.noteH = clamp(Math.floor(this.gridH / (pHi - pLo + 9)), 10, 20);
    this.scrollY = (127 - (pLo + pHi) / 2) * this.noteH - this.gridH / 2;
    this.clampScroll();
    this.s.emit('view');
  }
  zoomBy(f, anchorX) {
    const ax = anchorX ?? (KEYS_W + this.gridW / 2);
    const b = this.beatAt(ax);
    this.s.view.ppb = clamp(this.ppb * f, 6, 640);
    this.scrollX = b * this.ppb - (ax - KEYS_W);
    this.clampScroll();
    this.s.emit('view');
  }
  zoomYBy(f, anchorY) {
    const ay = anchorY ?? (RULER_H + this.gridH / 2);
    const p = 127 - (ay - RULER_H + this.scrollY) / this.noteH;
    this.noteH = clamp(this.noteH * f, 6, 44);
    this.scrollY = (127 - p) * this.noteH - (ay - RULER_H);
    this.clampScroll();
    this.s.emit('view');
  }
  ensureVisible(note) {
    const x0 = this.xOf(note.onset), x1 = this.xOf(note.onset + note.duration);
    if (x0 < KEYS_W + 10 || x1 > KEYS_W + this.gridW - 10) this.scrollX = Math.max(0, note.onset * this.ppb - this.gridW * 0.3);
    const y = this.yOf(note.pitch);
    if (y < RULER_H + 10 || y + this.noteH > RULER_H + this.gridH - 10) this.scrollY = (127 - note.pitch) * this.noteH - this.gridH / 2;
    this.clampScroll();
    this.s.emit('view');
  }
  followPlayhead(beat) {
    const x = this.xOf(beat);
    if (x > KEYS_W + this.gridW - 30 || x < KEYS_W) {
      this.scrollX = Math.max(0, beat * this.ppb - 40);
      this.clampScroll();
      this.s.emit('view');
    }
  }

  /* ---- hit testing */
  hitNote(x, y) {
    const t = this.s.track;
    if (!t) return null;
    for (let i = t.notes.length - 1; i >= 0; i--) {
      const n = t.notes[i];
      const nx = this.xOf(n.onset), nw = Math.max(3, n.duration * this.ppb), ny = this.yOf(n.pitch);
      if (x >= nx && x <= nx + nw && y >= ny && y <= ny + this.noteH) {
        const edge = x >= nx + nw - Math.min(8, Math.max(3, nw / 3));
        return { note: n, edge };
      }
    }
    return null;
  }
  loopEdgeAt(x) {
    const L = this.s.loop;
    if (!(L.end > L.start)) return null;
    if (Math.abs(x - this.xOf(L.start)) <= 5) return 'start';
    if (Math.abs(x - this.xOf(L.end)) <= 5) return 'end';
    return null;
  }

  /* ---- drawing */
  draw() {
    const c = this.ctx, W = this.w, H = this.h;
    if (!W || !H) return;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    c.save();
    c.beginPath();
    c.rect(KEYS_W, RULER_H, this.gridW, this.gridH);
    c.clip();
    this.drawRows(c);
    this.drawGrid(c);
    this.drawLoopShade(c);
    if (this.showGhosts) this.drawGhosts(c);
    this.drawNotes(c);
    this.drawBox(c);
    this.drawPlayhead(c);
    c.restore();
    this.drawRuler(c);
    this.drawKeys(c);
    this.drawCorner(c);
    this.drawScrollbars(c);
    if (!this.s.track) this.drawEmpty(c);
  }
  drawRows(c) {
    const scale = this.s.scale;
    const pTop = this.pitchAt(RULER_H), pBot = this.pitchAt(RULER_H + this.gridH);
    const hoverP = this.mouse && this.mouse.x > KEYS_W && this.mouse.y > RULER_H ? this.pitchAt(this.mouse.y) : null;
    for (let p = pBot; p <= pTop; p++) {
      const y = this.yOf(p);
      const pc = p % 12;
      let col = BLACK_KEYS.has(pc) ? '#14161b' : '#191c22';
      if (scale && !scale.has(pc)) col = BLACK_KEYS.has(pc) ? '#111317' : '#15171c';
      c.fillStyle = col;
      c.fillRect(KEYS_W, y, this.gridW, this.noteH);
      if (p === hoverP && !this.drag) { c.fillStyle = 'rgba(255,255,255,0.025)'; c.fillRect(KEYS_W, y, this.gridW, this.noteH); }
      c.fillStyle = pc === 0 ? '#2a2f39' : '#1a1d23';
      c.fillRect(KEYS_W, Math.round(y + this.noteH) - (pc === 0 ? 1 : 0.5), this.gridW, pc === 0 ? 1 : 0.5);
    }
  }
  drawGrid(c) {
    const lines = timeGridLines(this.s, KEYS_W, KEYS_W + this.gridW, (b) => this.xOf(b), (x) => this.beatAt(x), this.ppb);
    for (const ln of lines) {
      c.strokeStyle = GRID_COLORS[ln.kind];
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(ln.x, RULER_H);
      c.lineTo(ln.x, RULER_H + this.gridH);
      c.stroke();
    }
  }
  drawLoopShade(c) {
    const L = this.s.loop;
    if (!(L.end > L.start)) return;
    const x0 = this.xOf(L.start), x1 = this.xOf(L.end);
    c.fillStyle = L.on ? 'rgba(45,212,191,0.045)' : 'rgba(255,255,255,0.015)';
    c.fillRect(x0, RULER_H, x1 - x0, this.gridH);
    if (L.on) {
      c.strokeStyle = 'rgba(45,212,191,0.35)';
      c.beginPath();
      c.moveTo(Math.round(x0) + 0.5, RULER_H); c.lineTo(Math.round(x0) + 0.5, RULER_H + this.gridH);
      c.moveTo(Math.round(x1) + 0.5, RULER_H); c.lineTo(Math.round(x1) + 0.5, RULER_H + this.gridH);
      c.stroke();
    }
  }
  drawGhosts(c) {
    const active = this.s.track;
    for (const t of this.s.project.tracks) {
      if (t === active) continue;
      const col = this.s.color(t);
      c.fillStyle = rgba(col, t.mute ? 0.07 : 0.16);
      c.strokeStyle = rgba(col, t.mute ? 0.15 : 0.38);
      c.lineWidth = 1;
      for (const n of t.notes) {
        const x = this.xOf(n.onset), w = n.duration * this.ppb;
        if (x + w < KEYS_W || x > KEYS_W + this.gridW) continue;
        const y = this.yOf(n.pitch);
        if (y + this.noteH < RULER_H || y > RULER_H + this.gridH) continue;
        this.roundRect(c, x + 0.5, y + 1.5, Math.max(2, w - 1), this.noteH - 3, 3);
        c.fill();
        c.stroke();
      }
    }
  }
  drawNotes(c) {
    const t = this.s.track;
    if (!t) return;
    const col = this.s.color(t);
    const sel = this.s.selection;
    const ph = this.s.phonemes[t.id] || {};
    const errs = this.s.phonemeErrors[t.id] || {};
    const overlapping = new Set();
    for (let i = 1; i < t.notes.length; i++) {
      const a = t.notes[i - 1], b = t.notes[i];
      if (b.onset < a.onset + a.duration - EPS) { overlapping.add(a.id); overlapping.add(b.id); }
    }
    const nh = this.noteH;
    const fontPx = clamp(Math.round(nh * 0.68), 9, 13);
    for (const n of t.notes) {
      const x = this.xOf(n.onset), w = Math.max(3, n.duration * this.ppb);
      if (x + w < KEYS_W - 2 || x > KEYS_W + this.gridW + 2) continue;
      const y = this.yOf(n.pitch);
      if (y + nh < RULER_H - 14 || y > RULER_H + this.gridH) continue;
      const isSel = sel.has(n.id);
      const editing = this.editing && this.editing.noteId === n.id;
      // body
      this.roundRect(c, x + 0.5, y + 0.5, w - 1, nh - 1, Math.min(4, nh / 3));
      const g = c.createLinearGradient(0, y, 0, y + nh);
      if (isSel) { g.addColorStop(0, shade(col, 0.55)); g.addColorStop(1, shade(col, 0.3)); }
      else { g.addColorStop(0, shade(col, 0.08)); g.addColorStop(1, shade(col, -0.12)); }
      c.fillStyle = g;
      c.fill();
      c.lineWidth = isSel ? 1.5 : 1;
      c.strokeStyle = isSel ? '#ffffff' : shade(col, -0.45);
      c.stroke();
      if (overlapping.has(n.id)) {
        c.save(); c.setLineDash([3, 2]); c.strokeStyle = '#f0616d'; c.lineWidth = 1.5; c.stroke(); c.restore();
      }
      // resize grip
      if (w > 14 && nh >= 10) {
        c.fillStyle = 'rgba(0,0,0,0.22)';
        c.fillRect(x + w - 4, y + 3, 1.5, nh - 6);
      }
      // lyric
      if (nh >= 10 && w >= 16 && !editing) {
        c.save();
        c.beginPath(); c.rect(x + 1, y, w - 6, nh); c.clip();
        c.fillStyle = isSel ? '#06130f' : '#0b1512';
        c.font = `600 ${fontPx}px ${getComputedStyle(document.body).fontFamily}`;
        c.textBaseline = 'middle';
        c.fillText(n.lyric ?? '', x + 4, y + nh / 2 + 0.5);
        c.restore();
      }
      // phonemes above
      if (this.showPhonemes && nh >= 11 && this.ppb * n.duration > 18) {
        const p = errs[n.id] ? '⚠ ' + (n.phonemes || '?') : (ph[n.id] ?? '');
        if (p && p !== '-') {
          c.save();
          c.beginPath(); c.rect(x, y - 13, Math.max(w, 24), 13); c.clip();
          c.font = `10px ${'ui-monospace, Menlo, Consolas, monospace'}`;
          c.textBaseline = 'alphabetic';
          c.fillStyle = errs[n.id] ? '#f0616d' : (n.phonemes ? '#fcd34d' : 'rgba(190,200,215,0.62)');
          c.fillText(p, x + 2, y - 3);
          c.restore();
        }
      }
    }
  }
  drawBox(c) {
    const d = this.drag;
    if (!d || d.type !== 'box') return;
    const x = Math.min(d.x0, d.x1), y = Math.min(d.y0, d.y1), w = Math.abs(d.x1 - d.x0), h = Math.abs(d.y1 - d.y0);
    c.fillStyle = 'rgba(45,212,191,0.08)';
    c.fillRect(x, y, w, h);
    c.strokeStyle = 'rgba(45,212,191,0.8)';
    c.setLineDash([4, 3]);
    c.strokeRect(Math.round(x) + 0.5, Math.round(y) + 0.5, Math.round(w), Math.round(h));
    c.setLineDash([]);
  }
  drawPlayhead(c) {
    const x = Math.round(this.xOf(this.s.playhead)) + 0.5;
    if (x < KEYS_W || x > KEYS_W + this.gridW) return;
    c.strokeStyle = App.transport && App.transport.playing ? '#f5c542' : 'rgba(245,197,66,0.75)';
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(x, RULER_H);
    c.lineTo(x, RULER_H + this.gridH);
    c.stroke();
  }
  drawRuler(c) {
    const W = this.w;
    const g = c.createLinearGradient(0, 0, 0, RULER_H);
    g.addColorStop(0, '#20242c'); g.addColorStop(1, '#1a1d24');
    c.fillStyle = g;
    c.fillRect(KEYS_W, 0, W - KEYS_W, RULER_H);
    c.save();
    c.beginPath(); c.rect(KEYS_W, 0, this.gridW, RULER_H); c.clip();
    // loop band
    const L = this.s.loop;
    if (L.end > L.start) {
      const x0 = this.xOf(L.start), x1 = this.xOf(L.end);
      c.fillStyle = L.on ? 'rgba(45,212,191,0.32)' : 'rgba(140,150,165,0.18)';
      c.fillRect(x0, 2, x1 - x0, 10);
      c.fillStyle = L.on ? '#2dd4bf' : '#6b7383';
      c.fillRect(x0, 2, 2, 10);
      c.fillRect(x1 - 2, 2, 2, 10);
    }
    const bpb = this.s.beatsPerBar, bl = this.s.beatLen;
    const barPx = bpb * this.ppb;
    let every = 1;
    while (barPx * every < 34) every *= 2;
    const lines = timeGridLines(this.s, KEYS_W, KEYS_W + this.gridW, (b) => this.xOf(b), (x) => this.beatAt(x), this.ppb);
    c.font = `11px ${'ui-monospace, Menlo, Consolas, monospace'}`;
    c.textBaseline = 'alphabetic';
    for (const ln of lines) {
      if (ln.kind === 'bar') {
        const bar = Math.round(ln.beat / bpb);
        c.fillStyle = '#4a5160';
        c.fillRect(ln.x - 0.5, bar % every === 0 ? 12 : 18, 1, RULER_H);
        if (bar % every === 0) { c.fillStyle = '#aab3c2'; c.fillText(String(bar + 1), ln.x + 4, 24); }
      } else if (ln.kind === 'beat' && bl * this.ppb >= 12) {
        c.fillStyle = '#353b47';
        c.fillRect(ln.x - 0.5, 21, 1, RULER_H - 21);
      }
    }
    // playhead marker
    const px = this.xOf(this.s.playhead);
    c.fillStyle = '#f5c542';
    c.beginPath();
    c.moveTo(px - 6, RULER_H - 9); c.lineTo(px + 6, RULER_H - 9); c.lineTo(px, RULER_H - 1); c.closePath();
    c.fill();
    c.restore();
    c.fillStyle = '#0a0b0e';
    c.fillRect(KEYS_W, RULER_H - 1, W - KEYS_W, 1);
  }
  drawKeys(c) {
    // Realistic keyboard: white keys span 1.5 or 2 semitone rows, black keys sit on their own row.
    const nh = this.noteH;
    const pTop = this.pitchAt(RULER_H), pBot = this.pitchAt(RULER_H + this.gridH);
    const hoverP = this.mouse && this.mouse.x < KEYS_W && this.mouse.y > RULER_H && this.mouse.y < RULER_H + this.gridH ? this.pitchAt(this.mouse.y) : null;
    const WHITE = [[0, 0, 1.5], [2, 1.5, 3.5], [4, 3.5, 5], [5, 5, 6.5], [7, 6.5, 8.5], [9, 8.5, 10.5], [11, 10.5, 12]];
    const BW = Math.round(KEYS_W * 0.6);
    c.save();
    c.beginPath(); c.rect(0, RULER_H, KEYS_W, this.gridH); c.clip();
    c.fillStyle = '#c3c8d1';
    c.fillRect(0, RULER_H, KEYS_W, this.gridH);
    const o0 = Math.floor(pBot / 12) - 1, o1 = Math.floor(pTop / 12) + 1;
    c.font = `${clamp(Math.round(nh * 0.62), 8, 11)}px ${getComputedStyle(document.body).fontFamily}`;
    c.textBaseline = 'middle';
    c.textAlign = 'right';
    for (let o = o0; o <= o1; o++) {
      const base = o * 12;
      const yBase = this.yOf(base) + nh; // bottom edge of the octave's C row
      for (const [pc, lo, hi] of WHITE) {
        const p = base + pc;
        if (p < 0 || p > 127) continue;
        const top = yBase - hi * nh, bot = yBase - lo * nh;
        const g = c.createLinearGradient(0, 0, KEYS_W, 0);
        const lit = p === this.pressedKey ? '#5eead4' : (p === hoverP ? '#eefcf9' : null);
        g.addColorStop(0, lit || '#cfd3db'); g.addColorStop(1, lit || '#e6e9ee');
        c.fillStyle = g;
        c.fillRect(0, top, KEYS_W - 1, bot - top);
        c.fillStyle = '#8c939f';
        c.fillRect(0, Math.round(bot) - 1, KEYS_W, 1);
        if (pc === 0 && nh >= 8) {
          c.fillStyle = p === hoverP ? '#0f766e' : '#59606d';
          c.fillText(midiName(p), KEYS_W - 5, bot - Math.min(nh, (bot - top)) / 2);
        } else if (p === hoverP && nh >= 9) {
          c.fillStyle = '#0f766e';
          c.fillText(midiName(p), KEYS_W - 5, (top + bot) / 2);
        }
      }
      for (const pc of [1, 3, 6, 8, 10]) {
        const p = base + pc;
        if (p < 0 || p > 127) continue;
        const y = this.yOf(p);
        const g = c.createLinearGradient(0, 0, BW, 0);
        const lit = p === this.pressedKey ? '#14b8a6' : (p === hoverP ? '#3d4553' : null);
        g.addColorStop(0, lit || '#1b1e24'); g.addColorStop(0.85, lit || '#2b3039'); g.addColorStop(1, lit || '#15171b');
        c.fillStyle = g;
        c.fillRect(0, y + 0.5, BW, nh - 1);
        c.fillStyle = 'rgba(255,255,255,0.08)';
        c.fillRect(1, y + 1, BW - 3, 1);
      }
    }
    c.textAlign = 'left';
    c.restore();
    c.fillStyle = '#0a0b0e';
    c.fillRect(KEYS_W - 1, RULER_H, 1, this.gridH);
  }
  drawCorner(c) {
    c.fillStyle = '#1a1d24';
    c.fillRect(0, 0, KEYS_W, RULER_H);
    c.fillStyle = '#0a0b0e';
    c.fillRect(0, RULER_H - 1, KEYS_W, 1);
    c.fillRect(KEYS_W - 1, 0, 1, RULER_H);
    const t = this.s.track;
    if (t) {
      c.fillStyle = this.s.color(t);
      c.beginPath(); c.arc(10, RULER_H / 2, 3.5, 0, Math.PI * 2); c.fill();
      c.fillStyle = '#aab3c2';
      c.font = `600 10px ${getComputedStyle(document.body).fontFamily}`;
      c.textBaseline = 'middle';
      c.save(); c.beginPath(); c.rect(16, 0, KEYS_W - 20, RULER_H); c.clip();
      c.fillText(t.name, 17, RULER_H / 2 + 0.5);
      c.restore();
    }
  }
  scrollMetrics() {
    const contentW = Math.max(this.contentBeats() * this.ppb, this.gridW);
    const contentH = 128 * this.noteH;
    const hTrack = this.gridW, vTrack = this.gridH;
    const hThumb = Math.max(24, hTrack * this.gridW / contentW);
    const vThumb = Math.max(24, vTrack * this.gridH / contentH);
    const hPos = (contentW - this.gridW) > 0 ? (this.scrollX / (contentW - this.gridW)) * (hTrack - hThumb) : 0;
    const vPos = (contentH - this.gridH) > 0 ? (this.scrollY / (contentH - this.gridH)) * (vTrack - vThumb) : 0;
    return { contentW, contentH, hTrack, vTrack, hThumb, vThumb, hPos, vPos };
  }
  drawScrollbars(c) {
    const m = this.scrollMetrics();
    const yH = RULER_H + this.gridH, xV = KEYS_W + this.gridW;
    c.fillStyle = '#121419';
    c.fillRect(KEYS_W, yH, this.gridW + SB, SB);
    c.fillRect(xV, RULER_H, SB, this.gridH);
    c.fillRect(0, yH, KEYS_W, SB);
    const hot = this.drag && (this.drag.type === 'hscroll' || this.drag.type === 'vscroll') ? this.drag.type : null;
    c.fillStyle = hot === 'hscroll' ? '#56606f' : '#353c48';
    this.roundRect(c, KEYS_W + m.hPos + 1, yH + 2, m.hThumb - 2, SB - 4, 3); c.fill();
    c.fillStyle = hot === 'vscroll' ? '#56606f' : '#353c48';
    this.roundRect(c, xV + 2, RULER_H + m.vPos + 1, SB - 4, m.vThumb - 2, 3); c.fill();
    c.fillStyle = '#1a1d24';
    c.fillRect(xV, 0, SB, RULER_H);
  }
  drawEmpty(c) {
    c.fillStyle = 'rgba(15,17,21,0.7)';
    c.fillRect(KEYS_W, RULER_H, this.gridW, this.gridH);
    c.fillStyle = '#8790a1';
    c.font = `13px ${getComputedStyle(document.body).fontFamily}`;
    c.textAlign = 'center';
    c.fillText('No tracks yet - click "Add" in the track list to create one.', KEYS_W + this.gridW / 2, RULER_H + this.gridH / 2);
    c.textAlign = 'left';
  }
  roundRect(c, x, y, w, h, r) {
    r = Math.max(0, Math.min(r, w / 2, h / 2));
    c.beginPath();
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
  }

  /* ---- pointer interaction */
  onDown(e) {
    if (this.editing) this.commitLyric();
    if (isTyping()) document.activeElement.blur();
    App.closeMenus();
    const { x, y } = this.local(e);
    this.mouse = { x, y };
    if (e.button === 1 || (e.button === 0 && App.spaceHeld)) {
      e.preventDefault();
      this.startDrag(e, { type: 'pan', x0: x, y0: y, sx: this.scrollX, sy: this.scrollY });
      return;
    }
    if (e.button !== 0) return;
    const yH = RULER_H + this.gridH, xV = KEYS_W + this.gridW;
    if (y >= yH && x >= KEYS_W && x < xV) {
      const m = this.scrollMetrics();
      const onThumb = x >= KEYS_W + m.hPos && x <= KEYS_W + m.hPos + m.hThumb;
      if (!onThumb) { this.scrollX += (x < KEYS_W + m.hPos ? -1 : 1) * this.gridW * 0.8; this.clampScroll(); this.s.emit('view'); return; }
      this.startDrag(e, { type: 'hscroll', x0: x, sx: this.scrollX, m });
      return;
    }
    if (x >= xV && y >= RULER_H && y < yH) {
      const m = this.scrollMetrics();
      const onThumb = y >= RULER_H + m.vPos && y <= RULER_H + m.vPos + m.vThumb;
      if (!onThumb) { this.scrollY += (y < RULER_H + m.vPos ? -1 : 1) * this.gridH * 0.8; this.clampScroll(); this.s.emit('view'); return; }
      this.startDrag(e, { type: 'vscroll', y0: y, sy: this.scrollY, m });
      return;
    }
    if (y < RULER_H) {
      if (x < KEYS_W || x >= xV) return;
      const edge = this.loopEdgeAt(x);
      if (edge) { this.startDrag(e, { type: 'loopEdge', edge }); return; }
      this.startDrag(e, { type: 'ruler', x0: x, b0: this.beatAt(x), moved: false });
      return;
    }
    if (x < KEYS_W) {
      const p = this.pitchAt(y);
      this.pressedKey = p;
      App.preview(p);
      this.startDrag(e, { type: 'key', pitch: p });
      this.invalidate();
      return;
    }
    if (this.s.busy) return;
    const t = this.s.track;
    if (!t) return;
    const hit = this.hitNote(x, y);
    const sel = this.s.selection;
    if (hit) {
      const id = hit.note.id;
      let toggleOff = false;
      if (e.shiftKey) {
        if (sel.has(id)) toggleOff = true; else sel.add(id);
      } else if (!sel.has(id)) {
        sel.clear(); sel.add(id);
      }
      this.s.emit('selection');
      const notes = this.s.selectedNotes();
      const snap0 = notes.map((n) => ({ n, onset: n.onset, pitch: n.pitch, duration: n.duration }));
      if (hit.edge) {
        this.startDrag(e, { type: 'resize', grab: hit.note, grabEnd: hit.note.onset + hit.note.duration, b0: this.beatAt(x), orig: snap0, moved: false });
      } else {
        this.startDrag(e, {
          type: 'move', grab: hit.note, grabOnset: hit.note.onset, grabPitch: hit.note.pitch, b0: this.beatAt(x), p0: this.pitchAt(y),
          orig: snap0, moved: false, toggleOff, dup: e.ctrlKey || e.metaKey, duplicated: false, clickedId: id, shift: e.shiftKey,
        });
      }
      return;
    }
    if (this.tool === 'select' || e.shiftKey) {
      const base = e.shiftKey ? new Set(sel) : new Set();
      if (!e.shiftKey) { sel.clear(); this.s.emit('selection'); }
      this.startDrag(e, { type: 'box', x0: x, y0: y, x1: x, y1: y, base });
      return;
    }
    // draw tool: create a note
    const onset = Math.max(0, this.snapFloor(this.beatAt(x), e));
    const pitch = this.pitchAt(y);
    const note = { id: newId(), onset, duration: round6(this.newLen()), pitch, lyric: 'la' };
    t.notes.push(note);
    sortNotes(t);
    this.s.select([note.id]);
    this.s.emit('notes');
    this.startDrag(e, { type: 'create', note, anchor: onset, x0: x, moved: false });
  }
  startDrag(e, d) {
    this.drag = d;
    try { this.cv.setPointerCapture(e.pointerId); } catch { /* ignore */ }
  }
  onMove(e) {
    const { x, y } = this.local(e);
    this.mouse = { x, y };
    const d = this.drag;
    if (x > KEYS_W && y > RULER_H && x < KEYS_W + this.gridW && y < RULER_H + this.gridH) {
      App.cursorInfo(`${this.s.fmtBeat(Math.max(0, this.beatAt(x)))}  ${midiName(this.pitchAt(y))}`);
    }
    if (!d) {
      this.updateCursor(x, y);
      this.invalidate();
      return;
    }
    switch (d.type) {
      case 'create': {
        if (Math.abs(x - d.x0) > 3) d.moved = true;
        if (d.moved) {
          const end = this.snapRound(this.beatAt(x), e);
          d.note.duration = round6(Math.max(this.minDur(), end - d.anchor));
          this.s.emit('notes');
        }
        break;
      }
      case 'move': {
        const t = this.s.track;
        let delta = this.snapRound(d.grabOnset + (this.beatAt(x) - d.b0), e) - d.grabOnset;
        const minOn = Math.min(...d.orig.map((o) => o.onset));
        delta = Math.max(delta, -minOn);
        let dp = this.pitchAt(y) - d.p0;
        const pMin = Math.min(...d.orig.map((o) => o.pitch)), pMax = Math.max(...d.orig.map((o) => o.pitch));
        dp = clamp(dp, -pMin, 127 - pMax);
        if (!d.moved && (Math.abs(delta) > EPS || dp !== 0)) {
          d.moved = true;
          if (d.dup && !d.duplicated) {
            for (const o of d.orig) t.notes.push({ ...JSON.parse(JSON.stringify(o.n)), id: newId(), onset: o.onset, pitch: o.pitch });
            d.duplicated = true;
          }
        }
        if (d.moved) {
          const prevPitch = d.grab.pitch;
          for (const o of d.orig) { o.n.onset = round6(o.onset + delta); o.n.pitch = o.pitch + dp; }
          sortNotes(t);
          if (d.grab.pitch !== prevPitch) App.preview(d.grab.pitch, 0.18);
          this.s.emit('notes');
        }
        break;
      }
      case 'resize': {
        const newEnd = this.snapRound(d.grabEnd + (this.beatAt(x) - d.b0), e);
        const dd = newEnd - d.grabEnd;
        if (Math.abs(dd) > EPS) d.moved = true;
        for (const o of d.orig) o.n.duration = round6(Math.max(this.minDur(), o.duration + dd));
        this.s.emit('notes');
        break;
      }
      case 'box': {
        d.x1 = x; d.y1 = y;
        const b0 = this.beatAt(Math.min(d.x0, d.x1)), b1 = this.beatAt(Math.max(d.x0, d.x1));
        const pHi = this.pitchAt(Math.min(d.y0, d.y1)), pLo = this.pitchAt(Math.max(d.y0, d.y1));
        const ids = new Set(d.base);
        for (const n of this.s.track.notes) {
          if (n.onset < b1 && n.onset + n.duration > b0 && n.pitch >= pLo && n.pitch <= pHi) ids.add(n.id);
        }
        this.s.selection = ids;
        this.s.emit('selection');
        break;
      }
      case 'ruler': {
        if (Math.abs(x - d.x0) > 4) d.moved = true;
        if (d.moved) {
          const a = Math.max(0, this.snapRound(d.b0)), b = Math.max(0, this.snapRound(this.beatAt(x)));
          this.s.loop.start = Math.min(a, b);
          this.s.loop.end = Math.max(a, b);
          this.s.loop.on = this.s.loop.end > this.s.loop.start;
          this.s.emit('transport');
        }
        break;
      }
      case 'loopEdge': {
        const b = Math.max(0, this.snapRound(this.beatAt(x)));
        const L = this.s.loop;
        if (d.edge === 'start') L.start = Math.min(b, L.end - this.minDur());
        else L.end = Math.max(b, L.start + this.minDur());
        this.s.emit('transport');
        break;
      }
      case 'pan':
        this.scrollX = d.sx - (x - d.x0);
        this.scrollY = d.sy - (y - d.y0);
        this.clampScroll();
        this.s.emit('view');
        break;
      case 'hscroll': {
        const m = d.m;
        const ratio = (contentRange(m.contentW, this.gridW)) / Math.max(1, m.hTrack - m.hThumb);
        this.scrollX = d.sx + (x - d.x0) * ratio;
        this.clampScroll();
        this.s.emit('view');
        break;
      }
      case 'vscroll': {
        const m = d.m;
        const ratio = (contentRange(m.contentH, this.gridH)) / Math.max(1, m.vTrack - m.vThumb);
        this.scrollY = d.sy + (y - d.y0) * ratio;
        this.clampScroll();
        this.s.emit('view');
        break;
      }
      case 'key': {
        const p = this.pitchAt(y);
        if (p !== d.pitch) { d.pitch = p; this.pressedKey = p; App.preview(p); this.invalidate(); }
        break;
      }
      default:
        break;
    }
  }
  onUp(e) {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    try { this.cv.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    switch (d.type) {
      case 'create':
        if (this.s.newNoteLen === 'auto' && d.moved) this.s.lastDuration = d.note.duration;
        this.s.commit('notes');
        status(`Added note ${midiName(d.note.pitch)} at ${this.s.fmtBeat(d.note.onset)}`);
        break;
      case 'move':
        if (d.moved) this.s.commit('notes');
        else if (d.toggleOff) { this.s.selection.delete(d.clickedId); this.s.emit('selection'); }
        else if (!d.shift && this.s.selection.size > 1) this.s.select([d.clickedId]);
        break;
      case 'resize':
        if (d.moved) {
          if (this.s.newNoteLen === 'auto') this.s.lastDuration = d.grab.duration;
          this.s.commit('notes');
        }
        break;
      case 'ruler':
        if (!d.moved) App.transport.seek(Math.max(0, this.snapRound(this.beatAt(this.local(e).x))));
        else status(`Loop ${this.s.fmtBeat(this.s.loop.start)} - ${this.s.fmtBeat(this.s.loop.end)}`);
        App.transport.updateButtons();
        break;
      case 'loopEdge':
        App.transport.updateButtons();
        break;
      case 'key':
        this.pressedKey = null;
        break;
      default:
        break;
    }
    this.invalidate();
  }
  updateCursor(x, y) {
    let cur = 'default';
    if (y < RULER_H && x > KEYS_W) cur = this.loopEdgeAt(x) ? 'ew-resize' : 'pointer';
    else if (x < KEYS_W) cur = 'pointer';
    else if (x > KEYS_W + this.gridW || y > RULER_H + this.gridH) cur = 'default';
    else if (App.spaceHeld) cur = 'grab';
    else {
      const hit = this.hitNote(x, y);
      if (hit) cur = hit.edge ? 'ew-resize' : 'move';
      else cur = this.tool === 'draw' ? 'crosshair' : 'default';
    }
    this.cv.style.cursor = cur;
  }
  onDbl(e) {
    const { x, y } = this.local(e);
    if (y < RULER_H || x < KEYS_W) return;
    const hit = this.hitNote(x, y);
    if (hit) { this.openLyric(hit.note); return; }
    if (this.tool === 'select' && !this.s.busy && this.s.track) {
      const t = this.s.track;
      const note = { id: newId(), onset: Math.max(0, this.snapFloor(this.beatAt(x))), duration: round6(this.newLen()), pitch: this.pitchAt(y), lyric: 'la' };
      t.notes.push(note); sortNotes(t);
      this.s.select([note.id]);
      this.s.commit('notes');
    }
  }
  onWheel(e) {
    e.preventDefault();
    const { x, y } = this.local(e);
    const k = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 300 : 1;
    const dx = e.deltaX * k, dy = e.deltaY * k;
    if (e.ctrlKey || e.metaKey) this.zoomBy(Math.exp(-dy * 0.0016), Math.max(x, KEYS_W));
    else if (e.altKey) this.zoomYBy(Math.exp(-dy * 0.0016), y);
    else if (e.shiftKey) { this.scrollX += dy || dx; this.clampScroll(); this.s.emit('view'); }
    else { this.scrollX += dx; this.scrollY += dy; this.clampScroll(); this.s.emit('view'); }
  }
  onContext(e) {
    e.preventDefault();
    const { x, y } = this.local(e);
    if (x < KEYS_W || y < RULER_H || this.s.busy) return;
    const hit = this.hitNote(x, y);
    const beat = Math.max(0, this.snapRound(this.beatAt(x)));
    const items = [];
    if (hit) {
      if (!this.s.selection.has(hit.note.id)) this.s.select([hit.note.id]);
      const n = hit.note;
      items.push(['Edit lyric', 'Enter', () => this.openLyric(n)]);
      if (beat > n.onset + EPS && beat < n.onset + n.duration - EPS) items.push(['Split here', '', () => this.splitNote(n, beat)]);
      items.push(['Duplicate', 'Ctrl+D', () => this.duplicate()]);
      items.push(['Quantize to grid', 'Q', () => this.quantize()]);
      items.push('-');
      items.push(['Octave up', 'Shift+↑', () => this.transpose(12)]);
      items.push(['Octave down', 'Shift+↓', () => this.transpose(-12)]);
      items.push(['Clear attributes', '', () => this.clearAttrs()]);
      items.push('-');
      items.push(['Copy', 'Ctrl+C', () => this.copy()]);
      items.push(['Delete', 'Del', () => this.deleteSelection()]);
    } else {
      items.push(['Paste here', 'Ctrl+V', () => this.paste(beat)]);
      items.push(['Select all', 'Ctrl+A', () => this.selectAll()]);
      items.push(['Move playhead here', '', () => App.transport.seek(beat)]);
      items.push(['Set loop to this bar', '', () => {
        const bpb = this.s.beatsPerBar, b0 = Math.floor(beat / bpb) * bpb;
        Object.assign(this.s.loop, { on: true, start: b0, end: b0 + bpb });
        this.s.emit('transport'); App.transport.updateButtons();
      }]);
    }
    App.contextMenu(e.clientX, e.clientY, items);
  }

  /* ---- editing commands (keyboard / menus) */
  deleteSelection() {
    const t = this.s.track;
    if (!t || !this.s.selection.size) return;
    const n0 = t.notes.length;
    t.notes = t.notes.filter((n) => !this.s.selection.has(n.id));
    this.s.selection.clear();
    this.s.emit('selection');
    this.s.commit('notes');
    status(`Deleted ${n0 - t.notes.length} note(s)`);
  }
  transpose(semi) {
    const notes = this.s.selectedNotes();
    if (!notes.length) return;
    const lo = Math.min(...notes.map((n) => n.pitch)), hi = Math.max(...notes.map((n) => n.pitch));
    semi = clamp(semi, -lo, 127 - hi);
    if (!semi) return;
    for (const n of notes) n.pitch += semi;
    App.preview(notes[0].pitch, 0.18);
    this.s.commit('notes');
  }
  moveBy(beats) {
    const notes = this.s.selectedNotes();
    if (!notes.length) return;
    const lo = Math.min(...notes.map((n) => n.onset));
    beats = Math.max(beats, -lo);
    for (const n of notes) n.onset = round6(n.onset + beats);
    sortNotes(this.s.track);
    this.s.commit('notes');
  }
  resizeBy(beats) {
    const notes = this.s.selectedNotes();
    if (!notes.length) return;
    for (const n of notes) n.duration = round6(Math.max(this.minDur(), n.duration + beats));
    this.s.commit('notes');
  }
  quantize() {
    const notes = this.s.selectedNotes();
    const s = this.s.snap || 0.25;
    for (const n of notes) {
      const on = round6(Math.round(n.onset / s) * s);
      const end = round6(Math.max(on + s, Math.round((n.onset + n.duration) / s) * s));
      n.onset = on; n.duration = round6(end - on);
    }
    sortNotes(this.s.track);
    this.s.commit('notes');
  }
  clearAttrs() {
    for (const n of this.s.selectedNotes()) delete n.attrs;
    this.s.commit('notes');
  }
  splitNote(n, beat) {
    const t = this.s.track;
    const second = { ...JSON.parse(JSON.stringify(n)), id: newId(), onset: round6(beat), duration: round6(n.onset + n.duration - beat), lyric: '-' };
    delete second.phonemes;
    n.duration = round6(beat - n.onset);
    t.notes.push(second);
    sortNotes(t);
    this.s.select([second.id]);
    this.s.commit('notes');
  }
  selectAll() {
    const t = this.s.track;
    if (t) this.s.select(t.notes.map((n) => n.id));
  }
  copy() {
    const notes = this.s.selectedNotes();
    if (!notes.length) return false;
    const base = Math.min(...notes.map((n) => n.onset));
    this.s.clipboard = { base, span: Math.max(...notes.map((n) => n.onset + n.duration)) - base, notes: notes.map((n) => ({ ...JSON.parse(JSON.stringify(n)), onset: n.onset - base })) };
    status(`Copied ${notes.length} note(s)`);
    return true;
  }
  cut() { if (this.copy()) this.deleteSelection(); }
  paste(at) {
    const cb = this.s.clipboard, t = this.s.track;
    if (!cb || !t || this.s.busy) return;
    const start = at ?? this.snapRound(this.s.playhead);
    const ids = [];
    for (const n of cb.notes) {
      const c = { ...JSON.parse(JSON.stringify(n)), id: newId(), onset: round6(start + n.onset) };
      t.notes.push(c);
      ids.push(c.id);
    }
    sortNotes(t);
    this.s.select(ids);
    this.s.commit('notes');
    status(`Pasted ${ids.length} note(s) at ${this.s.fmtBeat(start)}`);
  }
  duplicate() {
    const notes = this.s.selectedNotes();
    if (!notes.length) return;
    this.copy();
    const end = Math.max(...notes.map((n) => n.onset + n.duration));
    const s = this.s.snap || 0;
    this.paste(s ? Math.ceil(end / s - 1e-6) * s : end);
  }
  selectStep(dir) {
    const t = this.s.track;
    if (!t || !t.notes.length) return;
    const sel = this.s.selectedNotes();
    let idx = sel.length ? t.notes.indexOf(dir > 0 ? sel[sel.length - 1] : sel[0]) + dir : (dir > 0 ? 0 : t.notes.length - 1);
    idx = clamp(idx, 0, t.notes.length - 1);
    this.s.select([t.notes[idx].id]);
    this.ensureVisible(t.notes[idx]);
  }

  /* ---- inline lyric editor */
  initLyricInput() {
    const inp = this.input;
    inp.addEventListener('keydown', (e) => {
      e.stopPropagation();
      if (e.key === 'Enter') { e.preventDefault(); this.commitLyric(); }
      else if (e.key === 'Escape') { e.preventDefault(); this.closeLyric(false); }
      else if (e.key === 'Tab') { e.preventDefault(); this.commitLyric(e.shiftKey ? -1 : 1); }
    });
    inp.addEventListener('blur', () => {
      const token = this.editing && this.editing.token;
      setTimeout(() => {
        if (this.editing && this.editing.token === token && document.activeElement !== inp) this.commitLyric();
      }, 0);
    });
  }
  openLyric(note) {
    const t = this.s.track;
    if (!t || this.s.busy) return;
    if (!this.s.selection.has(note.id) || this.s.selection.size !== 1) this.s.select([note.id]);
    this.ensureVisible(note);
    this.editing = { noteId: note.id, trackId: t.id, token: Math.random() };
    this.positionLyric(note);
    const inp = this.input;
    inp.hidden = false;
    inp.value = note.lyric ?? '';
    inp.focus();
    inp.select();
    this.invalidate();
  }
  positionLyric(note) {
    const x = this.xOf(note.onset), w = note.duration * this.ppb;
    const y = this.yOf(note.pitch) + this.noteH / 2 - 12;
    const width = clamp(w + 6, 76, 260);
    this.input.style.left = `${clamp(x - 2, KEYS_W, this.w - width - SB)}px`;
    this.input.style.top = `${clamp(y, RULER_H, this.h - 28)}px`;
    this.input.style.width = `${width}px`;
  }
  closeLyric(refocus = true) {
    this.editing = null;
    this.input.hidden = true;
    if (refocus && document.activeElement === this.input) this.input.blur();
    this.invalidate();
  }
  async commitLyric(advance = 0) {
    const ed = this.editing;
    if (!ed) return;
    this.editing = null;
    const track = this.s.trackById(ed.trackId);
    const note = track && track.notes.find((n) => n.id === ed.noteId);
    const text = this.input.value.trim();
    if (!advance) { this.input.hidden = true; this.input.blur(); }
    if (!note) { this.invalidate(); return; }
    let idx = track.notes.indexOf(note);
    let last = idx;
    if (text && text !== note.lyric) {
      if (/\s/.test(text)) {
        let tokens;
        try { tokens = (await Api.splitLyrics(text, track.language)).tokens; } catch { tokens = null; }
        if (!tokens || !tokens.length) tokens = text.split(/\s+/);
        idx = track.notes.indexOf(note);
        let i = idx;
        for (const tok of tokens) {
          if (i >= track.notes.length) break;
          track.notes[i].lyric = tok;
          i++;
        }
        last = i - 1;
        if (tokens.length > track.notes.length - idx) toast(`${tokens.length - (track.notes.length - idx)} lyric syllable(s) did not fit on the remaining notes`, 'warn');
      } else {
        note.lyric = text;
      }
      this.s.commit('notes');
    }
    if (advance) {
      const next = track.notes[advance > 0 ? last + 1 : idx - 1];
      if (next && this.s.track === track) { this.openLyric(next); return; }
      this.input.hidden = true;
      this.input.blur();
    }
    this.invalidate();
  }
}
function contentRange(content, visible) { return Math.max(0, content - visible); }

/* ============================================================== ParamLane */
class ParamLane {
  constructor(store, canvas, wrap, roll) {
    this.s = store;
    this.cv = canvas;
    this.wrap = wrap;
    this.roll = roll;
    this.ctx = canvas.getContext('2d');
    this.curve = 'pitchDelta';
    this.tool = 'draw';
    this.ranges = {};
    this.drag = null;
    this.mouse = null;
    this.w = 0; this.h = 0; this.dpr = 1;
    this._raf = 0;
    new ResizeObserver(() => this.resize()).observe(wrap);
    canvas.addEventListener('pointerdown', (e) => this.onDown(e));
    canvas.addEventListener('pointermove', (e) => this.onMove(e));
    canvas.addEventListener('pointerup', (e) => this.onUp(e));
    canvas.addEventListener('pointercancel', (e) => this.onUp(e));
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    canvas.addEventListener('dblclick', (e) => { if (this.local(e).x < KEYS_W) { delete this.ranges[this.curve]; this.invalidate(); } });
    canvas.addEventListener('pointerleave', () => { this.mouse = null; $('#lane-readout').textContent = ''; this.invalidate(); });
    for (const ev of ['project', 'notes', 'curves', 'tracks', 'song', 'selection', 'transport', 'view']) store.on(ev, () => this.invalidate());
    for (const ev of ['project', 'curves', 'selection', 'tracks']) store.on(ev, () => this.syncHeader());
    $('#lane-tools').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-ltool]');
      if (!b) return;
      this.tool = b.dataset.ltool;
      $$('#lane-tools button').forEach((x) => x.classList.toggle('active', x === b));
    });
    $('#lane-interp').addEventListener('change', (e) => {
      const c = this.curveObj(true);
      if (!c) return;
      c.interpolation = e.target.value;
      this.s.commit('curves');
    });
    $('#lane-clear').addEventListener('click', () => {
      const t = this.s.track;
      if (!t || !t.curves[this.curve] || !t.curves[this.curve].points.length) return;
      delete t.curves[this.curve];
      this.s.commit('curves');
      status(`Cleared ${CURVE_LABELS[this.curve] || this.curve} on ${t.name}`);
    });
  }
  buildTabs() {
    const host = $('#lane-tabs');
    host.innerHTML = '';
    for (const name of Object.keys(this.s.meta.curves || {})) {
      const spec = this.s.meta.curves[name];
      const b = el('button', { class: 'lane-tab' + (name === this.curve ? ' active' : ''), dataset: { curve: name }, title: `${CURVE_LABELS[name] || name}: ${spec.description} (${spec.min}..${spec.max} ${spec.unit})` },
        CURVE_TABS[name] || name);
      b.style.setProperty('--cc', CURVE_COLORS[name] || '#2dd4bf');
      b.addEventListener('click', () => { this.curve = name; this.syncHeader(); this.invalidate(); });
      host.append(b);
    }
    this.syncHeader();
  }
  syncHeader() {
    const t = this.s.track;
    $$('#lane-tabs .lane-tab').forEach((b) => {
      b.classList.toggle('active', b.dataset.curve === this.curve);
      const c = t && t.curves[b.dataset.curve];
      b.classList.toggle('has-data', !!(c && c.points.length));
    });
    const c = t && t.curves[this.curve];
    $('#lane-interp').value = c ? c.interpolation || 'cubic' : 'cubic';
  }
  get spec() { return (this.s.meta.curves || {})[this.curve] || { default: 0, min: -1, max: 1, unit: '' }; }
  range() {
    if (this.ranges[this.curve]) return this.ranges[this.curve];
    const sp = this.spec;
    if (this.curve === 'pitchDelta') return [-300, 300];
    if (this.curve === 'toneShift') return [-600, 600];
    if (this.curve === 'loudness') return [-24, 12];
    return [sp.min, sp.max];
  }
  get padT() { return 10; }
  get padB() { return 10; }
  yOf(v) { const [lo, hi] = this.range(); return this.padT + (1 - (v - lo) / (hi - lo)) * (this.h - this.padT - this.padB); }
  valAt(y) {
    const [lo, hi] = this.range();
    const v = lo + (1 - (y - this.padT) / (this.h - this.padT - this.padB)) * (hi - lo);
    return clamp(v, this.spec.min, this.spec.max);
  }
  xOf(b) { return this.roll.xOf(b); }
  beatAt(x) { return this.roll.beatAt(x); }
  get gridW() { return Math.max(10, this.w - KEYS_W - SB); }
  curveObj(create) {
    const t = this.s.track;
    if (!t) return null;
    if (!t.curves[this.curve] && create) t.curves[this.curve] = { interpolation: 'cubic', points: [] };
    return t.curves[this.curve] || null;
  }
  spacing() {
    let sp = 1 / 16;
    while (sp * this.roll.ppb < 2.5) sp *= 2;
    return sp;
  }
  evalAt(curve, beat) {
    const sp = this.spec;
    if (!curve || !curve.points.length) return sp.default;
    const pts = curve.points;
    if (beat <= pts[0][0]) return pts[0][1];
    if (beat >= pts[pts.length - 1][0]) return pts[pts.length - 1][1];
    let lo = 0, hi = pts.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (pts[mid][0] <= beat) lo = mid; else hi = mid; }
    const [x0, y0] = pts[lo], [x1, y1] = pts[hi];
    if (curve.interpolation === 'step') return y0;
    let u = (beat - x0) / Math.max(1e-9, x1 - x0);
    if (curve.interpolation !== 'linear') u = 0.5 - 0.5 * Math.cos(Math.PI * u);
    return y0 + (y1 - y0) * u;
  }
  resize() {
    const r = this.wrap.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.w = Math.max(1, Math.floor(r.width));
    this.h = Math.max(1, Math.floor(r.height));
    this.cv.width = Math.round(this.w * this.dpr);
    this.cv.height = Math.round(this.h * this.dpr);
    this.draw();
  }
  invalidate() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this.draw(); });
  }
  local(e) { const r = this.cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }

  draw() {
    const c = this.ctx, W = this.w, H = this.h;
    if (!W || !H) return;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    c.fillStyle = '#14161b';
    c.fillRect(0, 0, W, H);
    const t = this.s.track;
    const sp = this.spec;
    const col = CURVE_COLORS[this.curve] || '#2dd4bf';
    const [lo, hi] = this.range();
    c.save();
    c.beginPath(); c.rect(KEYS_W, 0, this.gridW, H); c.clip();
    // time grid
    for (const ln of timeGridLines(this.s, KEYS_W, KEYS_W + this.gridW, (b) => this.xOf(b), (x) => this.beatAt(x), this.roll.ppb)) {
      c.strokeStyle = GRID_COLORS[ln.kind];
      c.beginPath(); c.moveTo(ln.x, 0); c.lineTo(ln.x, H); c.stroke();
    }
    // value grid
    c.strokeStyle = '#20242b';
    for (let i = 1; i < 4; i++) {
      const y = Math.round(this.padT + (i / 4) * (H - this.padT - this.padB)) + 0.5;
      c.beginPath(); c.moveTo(KEYS_W, y); c.lineTo(KEYS_W + this.gridW, y); c.stroke();
    }
    // note context
    if (t) {
      const ncol = this.s.color(t);
      for (const n of t.notes) {
        const x = this.xOf(n.onset), w = n.duration * this.roll.ppb;
        if (x + w < KEYS_W || x > KEYS_W + this.gridW) continue;
        c.fillStyle = rgba(ncol, this.s.selection.has(n.id) ? 0.10 : 0.035);
        c.fillRect(x, 0, w, H);
        c.fillStyle = rgba(ncol, 0.45);
        c.fillRect(x + 1, H - 3, Math.max(1, w - 2), 2);
      }
    }
    // loop
    const L = this.s.loop;
    if (L.on && L.end > L.start) {
      c.fillStyle = 'rgba(45,212,191,0.04)';
      c.fillRect(this.xOf(L.start), 0, (L.end - L.start) * this.roll.ppb, H);
    }
    // default line
    const yDef = Math.round(this.yOf(sp.default)) + 0.5;
    c.strokeStyle = rgba(col, 0.45);
    c.setLineDash([5, 4]);
    c.beginPath(); c.moveTo(KEYS_W, yDef); c.lineTo(KEYS_W + this.gridW, yDef); c.stroke();
    c.setLineDash([]);
    // curve
    const curve = t && t.curves[this.curve];
    if (curve && curve.points.length) {
      c.beginPath();
      let first = true;
      for (let x = KEYS_W; x <= KEYS_W + this.gridW + 2; x += 2) {
        const y = this.yOf(this.evalAt(curve, this.beatAt(x)));
        if (first) { c.moveTo(x, y); first = false; } else c.lineTo(x, y);
      }
      c.strokeStyle = col;
      c.lineWidth = 1.7;
      c.stroke();
      c.lineTo(KEYS_W + this.gridW + 2, yDef);
      c.lineTo(KEYS_W, yDef);
      c.closePath();
      c.fillStyle = rgba(col, 0.12);
      c.fill();
      c.lineWidth = 1;
      const pts = curve.points;
      c.fillStyle = '#0f1115';
      c.strokeStyle = col;
      for (let i = 0; i < pts.length; i++) {
        const x = this.xOf(pts[i][0]);
        if (x < KEYS_W - 4 || x > KEYS_W + this.gridW + 4) continue;
        const gapL = i > 0 ? x - this.xOf(pts[i - 1][0]) : Infinity;
        const gapR = i < pts.length - 1 ? this.xOf(pts[i + 1][0]) - x : Infinity;
        if (Math.min(gapL, gapR) < 7) continue; // freehand strokes: no dots, just the line
        c.beginPath(); c.arc(x, this.yOf(pts[i][1]), 2.6, 0, Math.PI * 2); c.fill(); c.stroke();
      }
    } else if (t) {
      c.fillStyle = 'rgba(135,144,161,0.55)';
      c.font = `11.5px ${getComputedStyle(document.body).fontFamily}`;
      c.textAlign = 'center';
      c.fillText(`${CURVE_LABELS[this.curve] || this.curve}: automatic (default ${fmtNum(sp.default)} ${sp.unit}) - drag to draw, right-drag to erase`, KEYS_W + this.gridW / 2, H / 2 - 8);
      c.textAlign = 'left';
    }
    // drag previews
    const d = this.drag;
    if (d && d.mode === 'line') {
      c.strokeStyle = '#fff';
      c.setLineDash([4, 3]);
      c.beginPath(); c.moveTo(this.xOf(d.b0), this.yOf(d.v0)); c.lineTo(this.xOf(d.b1), this.yOf(d.v1)); c.stroke();
      c.setLineDash([]);
    }
    if (d && d.mode === 'erase') {
      c.fillStyle = 'rgba(240,97,109,0.10)';
      c.fillRect(this.xOf(d.lo), 0, (d.hi - d.lo) * this.roll.ppb, H);
    }
    // playhead
    const px = Math.round(this.xOf(this.s.playhead)) + 0.5;
    c.strokeStyle = 'rgba(245,197,66,0.75)';
    c.beginPath(); c.moveTo(px, 0); c.lineTo(px, H); c.stroke();
    // hover crosshair
    if (this.mouse && this.mouse.x > KEYS_W && !d) {
      c.strokeStyle = 'rgba(255,255,255,0.18)';
      c.beginPath(); c.moveTo(Math.round(this.mouse.x) + 0.5, 0); c.lineTo(Math.round(this.mouse.x) + 0.5, H); c.stroke();
    }
    c.restore();
    // gutter with value labels
    c.fillStyle = '#1a1d24';
    c.fillRect(0, 0, KEYS_W, H);
    c.fillStyle = '#0a0b0e';
    c.fillRect(KEYS_W - 1, 0, 1, H);
    c.fillStyle = '#121419';
    c.fillRect(KEYS_W + this.gridW, 0, SB, H);
    c.font = `10px ${'ui-monospace, Menlo, Consolas, monospace'}`;
    c.textAlign = 'right';
    const lbl = (v, y, color) => { c.fillStyle = color; c.fillText(fmtNum(v, Math.abs(hi - lo) < 5 ? 2 : 0), KEYS_W - 6, y); };
    c.textBaseline = 'top'; lbl(hi, 3, '#7c8595');
    c.textBaseline = 'bottom'; lbl(lo, H - 2, '#7c8595');
    c.textBaseline = 'middle';
    if (yDef > 18 && yDef < H - 18) lbl(sp.default, yDef, rgba(col, 0.95));
    c.fillStyle = '#5c6575';
    c.textAlign = 'left';
    c.textBaseline = 'middle';
    if (sp.unit) c.fillText(sp.unit, 5, H / 2);
    c.textAlign = 'left';
    c.textBaseline = 'alphabetic';
  }

  onDown(e) {
    const { x, y } = this.local(e);
    if (x < KEYS_W || x > KEYS_W + this.gridW) return;
    if (this.s.busy || !this.s.track) return;
    if (isTyping()) document.activeElement.blur();
    if (e.button === 1) return;
    const b = Math.max(0, this.beatAt(x)), v = this.valAt(y);
    const mode = (e.button === 2 || e.altKey || this.tool === 'erase') ? 'erase' : (e.shiftKey || this.tool === 'line') ? 'line' : 'draw';
    const curve = this.curveObj(mode !== 'erase');
    if (!curve && mode === 'erase') return;
    this.drag = { mode, b0: b, v0: v, b1: b, v1: v, lastB: b, lastV: v, lo: b, hi: b, changed: false };
    try { this.cv.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    if (mode === 'draw') this.drawSegment(b, v, b, v);
    if (mode === 'erase') this.eraseRange(b, b);
    this.s.emit('curves');
  }
  onMove(e) {
    const { x, y } = this.local(e);
    this.mouse = { x, y };
    const b = Math.max(0, this.beatAt(x)), v = this.valAt(y);
    const sp = this.spec;
    const curve = this.s.track && this.s.track.curves[this.curve];
    const shown = this.drag ? v : this.evalAt(curve, b);
    $('#lane-readout').textContent = x > KEYS_W ? `${CURVE_LABELS[this.curve] || this.curve}  ${this.s.fmtBeat(b)}  ${shown >= 0 && sp.min < 0 ? '+' : ''}${fmtNum(shown, 2)} ${sp.unit}` : '';
    const d = this.drag;
    if (!d) { this.cv.style.cursor = x < KEYS_W ? 'ns-resize' : (this.tool === 'erase' ? 'cell' : 'crosshair'); this.invalidate(); return; }
    if (d.mode === 'draw') { this.drawSegment(d.lastB, d.lastV, b, v); d.lastB = b; d.lastV = v; }
    else if (d.mode === 'line') { d.b1 = b; d.v1 = v; }
    else if (d.mode === 'erase') this.eraseRange(Math.min(d.b0, b), Math.max(d.b0, b));
    this.s.emit('curves');
  }
  onUp(e) {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    try { this.cv.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    const curve = this.curveObj(false);
    if (d.mode === 'line' && curve) {
      const [b0, v0, b1, v1] = d.b0 <= d.b1 ? [d.b0, d.v0, d.b1, d.v1] : [d.b1, d.v1, d.b0, d.v0];
      const sp = this.spacing();
      const pts = curve.points.filter((p) => p[0] < b0 - EPS || p[0] > b1 + EPS);
      const n = Math.max(1, Math.round((b1 - b0) / sp));
      for (let i = 0; i <= n; i++) {
        const u = i / n;
        pts.push([round6(b0 + (b1 - b0) * u), round6(v0 + (v1 - v0) * u)]);
      }
      curve.points = pts.sort((a, b) => a[0] - b[0]);
      d.lo = b0; d.hi = b1; d.changed = true;
    }
    if (curve && d.mode !== 'erase' && d.changed) this.anchor(curve, d.lo, d.hi);
    const t = this.s.track;
    if (t && t.curves[this.curve] && !t.curves[this.curve].points.length) delete t.curves[this.curve];
    if (d.changed) this.s.commit('curves');
    else this.invalidate();
  }
  /** Freehand: replace points between two mouse positions with points on a 1/16-beat grid. */
  drawSegment(bA, vA, bB, vB) {
    const curve = this.curveObj(true);
    const d = this.drag;
    const sp = this.spacing();
    if (bA > bB) [bA, vA, bB, vB] = [bB, vB, bA, vA];
    const g0 = Math.round(bA / sp) * sp, g1 = Math.round(bB / sp) * sp;
    const lo = g0 - sp / 2, hi = g1 + sp / 2;
    const pts = curve.points.filter((p) => p[0] < lo || p[0] >= hi);
    for (let g = g0; g <= g1 + EPS; g += sp) {
      const u = bB - bA > EPS ? clamp((g - bA) / (bB - bA), 0, 1) : 0;
      pts.push([round6(g), round6(vA + (vB - vA) * u)]);
    }
    curve.points = pts.sort((a, b) => a[0] - b[0]);
    d.lo = Math.min(d.lo, g0); d.hi = Math.max(d.hi, g1);
    d.changed = true;
  }
  eraseRange(b0, b1) {
    const curve = this.curveObj(false);
    const d = this.drag;
    if (!curve) return;
    const sp = this.spacing();
    const lo = Math.min(b0, d.lo) - sp / 2, hi = Math.max(b1, d.hi) + sp / 2;
    d.lo = Math.min(d.lo, b0); d.hi = Math.max(d.hi, b1);
    const n0 = curve.points.length;
    curve.points = curve.points.filter((p) => p[0] < lo || p[0] > hi);
    if (curve.points.length !== n0) d.changed = true;
  }
  /** Keep a drawn stroke local: return to the default value just outside it if nothing else is there. */
  anchor(curve, lo, hi) {
    const sp = this.spacing();
    const def = this.spec.default;
    const pts = curve.points;
    if (!pts.some((p) => p[0] < lo - EPS) && lo - sp >= 0) pts.push([round6(lo - sp), def]);
    if (!pts.some((p) => p[0] > hi + EPS)) pts.push([round6(hi + sp), def]);
    pts.sort((a, b) => a[0] - b[0]);
  }
  onWheel(e) {
    e.preventDefault();
    const k = e.deltaMode === 1 ? 16 : 1;
    const dy = e.deltaY * k, dx = e.deltaX * k;
    const { x, y } = this.local(e);
    if (e.ctrlKey || e.metaKey) { this.roll.zoomBy(Math.exp(-dy * 0.0016), Math.max(x, KEYS_W)); return; }
    if (e.altKey || x < KEYS_W) {
      const [lo, hi] = this.range();
      const sp = this.spec;
      const f = Math.exp(dy * 0.0016);
      const center = this.valAt(y);
      let nlo = center - (center - lo) * f, nhi = center + (hi - center) * f;
      nlo = Math.max(sp.min, nlo); nhi = Math.min(sp.max, nhi);
      if (nhi - nlo > (sp.max - sp.min) / 200) this.ranges[this.curve] = [nlo, nhi];
      this.invalidate();
      return;
    }
    this.roll.scrollX += dx || dy;
    this.roll.clampScroll();
    this.s.emit('view');
  }
}

/* ============================================================== TrackList */
/** Compact labelled slider row used by track cards and the master strip. */
function makeSlider(label, value, { min, max, step, def = 0, fmt, set, key, onCommit }) {
  const val = el('span', { class: 'tc-val' }, fmt(value));
  const r = el('input', { type: 'range', min, max, step, value, dataset: { key }, title: 'Double-click to reset' });
  setRangeFill(r);
  r.addEventListener('input', () => { set(parseFloat(r.value)); val.textContent = fmt(parseFloat(r.value)); setRangeFill(r); App.transport.invalidateCache(); });
  r.addEventListener('change', () => onCommit());
  r.addEventListener('dblclick', () => { r.value = def; set(def); val.textContent = fmt(def); setRangeFill(r); onCommit(); });
  return el('div', { class: 'tc-row' }, el('span', { class: 'tc-lbl' }, label), r, val);
}

class TrackList {
  constructor(store, host) {
    this.s = store;
    this.host = host;
    for (const ev of ['project', 'tracks']) store.on(ev, () => this.render());
    store.on('selection', () => this.markActive());
    $('#btn-add-track').addEventListener('click', () => this.addTrack());
  }
  render() {
    const keepFocus = this.host.contains(document.activeElement) ? document.activeElement.dataset.key : null;
    this.host.innerHTML = '';
    const meta = this.s.meta;
    if (!this.s.project.tracks.length) {
      this.host.append(el('div', { class: 'insp-empty' }, 'No tracks. ', el('b', {}, 'Add'), ' one to start writing.'));
    }
    this.s.project.tracks.forEach((t, i) => {
      const color = this.s.color(t);
      const card = el('div', { class: 'track-card' + (t === this.s.track ? ' active' : '') + (t.mute ? ' muted' : ''), dataset: { id: t.id } });
      card.style.setProperty('--tc', color);
      card.addEventListener('pointerdown', (e) => {
        if (e.target.closest('input,select,button')) { this.s.setTrack(t.id); return; }
        this.s.setTrack(t.id);
        if (App.roll) App.roll.centerOnNotes();
      });
      const swatch = el('button', { class: 'tc-swatch', title: 'Track color', onclick: (e) => { e.stopPropagation(); App.pickColor(color, (c) => { (this.s.project.metadata.track_colors ||= {})[t.id] = c; this.s.commit('tracks'); }); } });
      const name = el('input', { class: 'tc-name', value: t.name, spellcheck: false, dataset: { key: 'name-' + t.id }, title: 'Track name' });
      name.addEventListener('change', () => { t.name = name.value.trim() || `Track ${i + 1}`; this.s.commit('tracks'); });
      name.addEventListener('keydown', (e) => { if (e.key === 'Enter') name.blur(); e.stopPropagation(); });
      const mute = el('button', { class: 'ms-btn mute' + (t.mute ? ' on' : ''), title: 'Mute' }, 'M');
      mute.addEventListener('click', () => { t.mute = !t.mute; this.s.commit('tracks'); });
      const solo = el('button', { class: 'ms-btn solo' + (t.solo ? ' on' : ''), title: 'Solo' }, 'S');
      solo.addEventListener('click', () => { t.solo = !t.solo; this.s.commit('tracks'); });
      const del = el('button', { class: 'tc-del', title: 'Remove track' }, icon('x'));
      del.addEventListener('click', () => this.removeTrack(t));
      card.append(el('div', { class: 'tc-row' }, swatch, name, mute, solo, del));

      const voice = el('select', { title: 'Voice', dataset: { key: 'voice-' + t.id } },
        ...meta.voices.map((v) => el('option', { value: v.id, selected: v.id === t.voice }, v.name)));
      if (!meta.voices.some((v) => v.id === t.voice)) voice.append(el('option', { value: t.voice, selected: true }, t.voice));
      voice.addEventListener('change', () => { t.voice = voice.value; this.s.commit('tracks'); });
      const style = el('select', { title: 'Expression style', dataset: { key: 'style-' + t.id } },
        ...meta.styles.map((s) => el('option', { value: s.name, selected: s.name === t.style }, s.name)));
      if (!meta.styles.some((s) => s.name === t.style)) style.append(el('option', { value: t.style, selected: true }, t.style));
      style.addEventListener('change', () => { t.style = style.value; this.s.commit('tracks'); });
      card.append(el('div', { class: 'tc-row' }, voice, style));

      const commit = () => this.s.commit('tracks');
      card.append(makeSlider('Vol', t.gain_db ?? 0, { min: -36, max: 12, step: 0.5, def: 0, fmt: fmtDb, set: (v) => { t.gain_db = v; }, key: 'gain-' + t.id, onCommit: commit }));
      card.append(makeSlider('Pan', t.pan ?? 0, { min: -1, max: 1, step: 0.01, def: 0, fmt: fmtPan, set: (v) => { t.pan = v; }, key: 'pan-' + t.id, onCommit: commit }));
      this.host.append(card);
    });
    if (keepFocus) { const f = this.host.querySelector(`[data-key="${keepFocus}"]`); if (f) f.focus(); }
  }
  markActive() {
    const id = this.s.track && this.s.track.id;
    $$('.track-card', this.host).forEach((c) => c.classList.toggle('active', c.dataset.id === id));
  }
  addTrack() {
    if (this.s.busy) return;
    const p = this.s.project;
    const used = new Set(p.tracks.map((t) => t.voice));
    const voices = this.s.meta.voices;
    const v = voices.find((x) => !used.has(x.id)) || voices[p.tracks.length % Math.max(1, voices.length)] || { id: 'aurora', default_style: 'pop' };
    const t = {
      id: newId(), name: `Vocal ${p.tracks.length + 1}`, voice: v.id, language: 'en', style: v.default_style || 'pop',
      gain_db: 0, pan: 0, mute: false, solo: false, reverb_send: 0.25, seed: p.tracks.length * 7919 + 17, voice_params: {}, notes: [], curves: {},
    };
    p.tracks.push(t);
    this.s.trackId = t.id;
    this.s.selection.clear();
    this.s.commit('tracks');
    this.s.emit('selection');
    status(`Added track "${t.name}" (${v.name || v.id})`);
  }
  removeTrack(t) {
    if (this.s.busy) return;
    if (t.notes.length && !confirm(`Remove track "${t.name}" with ${t.notes.length} notes? (Undo restores it.)`)) return;
    const p = this.s.project;
    p.tracks = p.tracks.filter((x) => x !== t);
    if (p.metadata.track_colors) delete p.metadata.track_colors[t.id];
    if (this.s.trackId === t.id) { this.s.trackId = p.tracks[0]?.id ?? null; this.s.selection.clear(); }
    this.s.commit('tracks');
    this.s.emit('selection');
  }
}

/* ============================================================ MasterStrip */
class MasterStrip {
  constructor(store, host) {
    this.s = store;
    this.host = host;
    for (const ev of ['project', 'song']) store.on(ev, () => this.render());
  }
  render() {
    const m = this.s.project.master;
    this.host.innerHTML = '';
    const commit = () => this.s.commit('song');
    const gain = makeSlider('Gain', m.gain_db ?? 0, { min: -24, max: 12, step: 0.5, def: 0, fmt: fmtDb, set: (v) => { m.gain_db = v; }, key: 'm-gain', onCommit: commit });
    const room = makeSlider('Room', m.reverb_room ?? 0.55, { min: 0, max: 1, step: 0.01, def: 0.55, fmt: (v) => fmtNum(v, 2), set: (v) => { m.reverb_room = v; }, key: 'm-room', onCommit: commit });
    const wet = makeSlider('Reverb', m.reverb_wet ?? 1, { min: 0, max: 2, step: 0.01, def: 1, fmt: (v) => fmtNum(v, 2) + 'x', set: (v) => { m.reverb_wet = v; }, key: 'm-wet', onCommit: commit });
    const chk = (label, key) => {
      const i = el('input', { type: 'checkbox', checked: m[key] !== false });
      i.addEventListener('change', () => { m[key] = i.checked; commit(); });
      return el('label', { class: 'chk' }, i, label);
    };
    this.host.append(
      el('div', { class: 'master-title' }, el('span', {}, 'Master'), el('span', { class: 'tc-row', style: { gap: '10px' } }, chk('Normalize', 'normalize'), chk('Limiter', 'limiter'))),
      gain, room, wet,
    );
  }
}

/* ============================================================== Inspector */
class Inspector {
  constructor(store, host) {
    this.s = store;
    this.host = host;
    this.tab = 'note';
    this._t = 0;
    $('#insp-tabs').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-tab]');
      if (b) this.setTab(b.dataset.tab);
    });
    for (const ev of ['project', 'notes', 'tracks', 'song', 'selection', 'phonemes', 'history']) store.on(ev, () => this.schedule());
  }
  setTab(tab) {
    this.tab = tab;
    $$('#insp-tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
    this.render();
  }
  schedule() {
    if (this._t) return;
    this._t = requestAnimationFrame(() => { this._t = 0; this.render(); });
  }
  render() {
    // Do not rebuild under the user's fingers while they are typing in the inspector.
    const active = document.activeElement;
    const focusKey = this.host.contains(active) ? active.dataset.key : null;
    if (focusKey && active.dataset.editing === '1') return;
    const scroll = this.host.scrollTop;
    this.host.innerHTML = '';
    if (this.tab === 'note') this.renderNote();
    else if (this.tab === 'track') this.renderTrack();
    else this.renderProject();
    this.host.scrollTop = scroll;
    if (focusKey) { const f = this.host.querySelector(`[data-key="${focusKey}"]`); if (f) f.focus(); }
  }

  /* ---- field builders */
  field(label, control, { title, desc, err } = {}) {
    const f = el('div', { class: 'field', title: title || null }, el('label', {}, label), control);
    const out = [f];
    if (err) out.push(el('div', { class: 'err-text' }, err));
    if (desc) out.push(el('div', { class: 'desc-text' }, desc));
    return out;
  }
  section(t) { return el('div', { class: 'insp-section' }, t); }
  input(key, value, { placeholder = '', type = 'text', step, min, max, onCommit, parse }) {
    const i = el('input', { type, value: value ?? '', placeholder, step, min, max, dataset: { key }, spellcheck: false });
    const orig = i.value;
    i.addEventListener('input', () => { i.dataset.editing = '1'; i.classList.remove('invalid'); });
    i.addEventListener('keydown', (e) => {
      e.stopPropagation();
      if (e.key === 'Enter') i.blur();
      if (e.key === 'Escape') { i.value = orig; i.dataset.editing = ''; i.blur(); }
    });
    i.addEventListener('change', () => {
      i.dataset.editing = '';
      const raw = i.value.trim();
      const v = parse ? parse(raw) : raw;
      if (v === undefined) { i.classList.add('invalid'); return; }
      onCommit(v);
    });
    i.addEventListener('blur', () => { i.dataset.editing = ''; });
    return i;
  }
  numInput(key, value, opts) {
    return this.input(key, value == null ? '' : fmtNum(value, 6), {
      ...opts, type: 'text',
      parse: (raw) => {
        if (raw === '') return opts.allowBlank ? null : undefined;
        const v = Number(raw);
        if (!Number.isFinite(v)) return undefined;
        return opts.min != null || opts.max != null ? clamp(v, opts.min ?? -Infinity, opts.max ?? Infinity) : v;
      },
    });
  }
  select(key, options, value, onChange) {
    const s = el('select', { dataset: { key } }, ...options.map(([v, l]) => el('option', { value: v, selected: String(v) === String(value) }, l)));
    s.addEventListener('change', () => onChange(s.value));
    return s;
  }
  sliderField(key, value, def, { min, max, step }, onCommit) {
    const r = el('input', { type: 'range', min, max, step, value: value ?? def ?? min });
    setRangeFill(r);
    const n = this.numInput(key, value, { placeholder: def != null ? fmtNum(def, 3) : 'auto', allowBlank: true, min, max, onCommit });
    r.addEventListener('input', () => { n.value = fmtNum(parseFloat(r.value), 4); setRangeFill(r); });
    r.addEventListener('change', () => onCommit(parseFloat(r.value)));
    return el('div', { class: 'slider-row' }, r, n);
  }

  /* ---- Note tab */
  renderNote() {
    const H = this.host;
    const t = this.s.track;
    const notes = this.s.selectedNotes();
    if (!t) { H.append(el('div', { class: 'insp-empty' }, 'No track selected.')); return; }
    if (!notes.length) {
      H.append(el('div', { class: 'insp-empty', html: `No note selected.<br><b>Drag</b> on the grid to draw a note, <b>double-click</b> it to type a lyric, <b>Tab</b> to jump to the next note.<br><br>${t.notes.length} note(s) on <b>${escapeHtml(t.name)}</b>.` }));
      return;
    }
    const single = notes.length === 1 ? notes[0] : null;
    const ph = this.s.phonemes[t.id] || {};
    const errs = this.s.phonemeErrors[t.id] || {};
    const commit = () => this.s.commit('notes');
    const common = (get) => { const v0 = get(notes[0]); return notes.every((n) => JSON.stringify(get(n)) === JSON.stringify(v0)) ? v0 : undefined; };
    if (single) {
      H.append(el('div', { class: 'note-badge' },
        el('span', { class: 'nb-pitch' }, midiName(single.pitch)),
        el('span', { class: 'nb-lyric' }, single.lyric || '(rest)'),
        el('span', { class: 'nb-ph' }, single.phonemes ? `[${single.phonemes}]` : (ph[single.id] ? `/${ph[single.id]}/` : ''))));
    } else {
      H.append(el('div', { class: 'note-badge' }, el('span', { class: 'nb-pitch' }, `${notes.length}`), el('span', {}, 'notes selected'),
        el('span', { class: 'nb-ph' }, `${midiName(Math.min(...notes.map((n) => n.pitch)))}-${midiName(Math.max(...notes.map((n) => n.pitch)))}`)));
    }
    H.append(this.section('Note'));
    const lyr = common((n) => n.lyric);
    H.append(...this.field('Lyric', this.input('lyric', lyr ?? '', {
      placeholder: lyr === undefined ? 'mixed' : '', onCommit: (v) => { for (const n of notes) n.lyric = v; commit(); },
    }), { title: '"-" melisma, "+" next syllable of the previous word, "br" breath, "[s t aa r]" explicit phonemes' }));
    const phc = common((n) => n.phonemes || '');
    const auto = single ? (ph[single.id] || '') : '';
    const err = notes.map((n) => errs[n.id]).find(Boolean);
    H.append(...this.field('Phonemes', this.input('phonemes', phc ?? '', {
      placeholder: phc === undefined ? 'mixed' : (auto && auto !== '-' ? auto : 'automatic'),
      onCommit: (v) => { for (const n of notes) { if (v) n.phonemes = v; else delete n.phonemes; } commit(); },
    }), { title: 'Space-separated phoneme override (blank = automatic from the lyric)', err }));
    if (single) {
      H.append(...this.field('Pitch', this.input('pitch', midiName(single.pitch), {
        placeholder: 'C4 or 60', parse: (raw) => { const p = parsePitch(raw); return p == null || p < 0 || p > 127 ? undefined : p; },
        onCommit: (v) => { single.pitch = v; commit(); },
      })));
      H.append(...this.field('Onset', el('div', { class: 'pair' },
        this.numInput('onset', single.onset, { min: 0, onCommit: (v) => { single.onset = round6(v); sortNotes(t); commit(); } }),
        el('span', { class: 'unit' }, this.s.fmtBeat(single.onset))), { title: 'Start position in beats (quarter notes)' }));
    }
    const dur = common((n) => n.duration);
    H.append(...this.field('Duration', el('div', { class: 'pair' },
      this.numInput('duration', dur, { min: 1 / 64, placeholder: 'mixed', onCommit: (v) => { for (const n of notes) n.duration = round6(v); commit(); } }),
      el('span', { class: 'unit' }, 'beats'))));

    H.append(this.section('Expression'));
    const attrs = this.s.meta.note_attributes || {};
    for (const [name, spec] of Object.entries(attrs)) {
      const v = common((n) => (n.attrs || {})[name] ?? null);
      const setAttr = (val) => {
        for (const n of notes) {
          n.attrs = n.attrs || {};
          if (val === null || val === '' || val === undefined) delete n.attrs[name]; else n.attrs[name] = val;
          if (!Object.keys(n.attrs).length) delete n.attrs;
        }
        commit();
      };
      let control;
      if (spec.type === 'string') {
        control = this.select('attr-' + name, [['', 'auto (track)'], ...(this.s.meta.languages || []).map((l) => [l.id, l.name])], v ?? '', (val) => setAttr(val || null));
      } else {
        const ph2 = v === undefined ? 'mixed' : (spec.default != null ? `${fmtNum(spec.default, 3)} (default)` : 'auto');
        control = el('div', { class: 'pair' }, this.numInput('attr-' + name, v ?? null, { placeholder: ph2, allowBlank: true, onCommit: setAttr }),
          el('span', { class: 'unit' }, ATTR_UNITS[name] || ''));
      }
      H.append(...this.field(ATTR_LABELS[name] || name, control, { title: spec.description }));
    }
    H.append(el('div', { class: 'btn-row' },
      el('button', { class: 'mini-btn', onclick: () => App.roll.clearAttrs() }, 'Reset expression'),
      el('button', { class: 'mini-btn', onclick: () => App.roll.openLyric(notes[0]) }, 'Edit lyric'),
      el('button', { class: 'mini-btn danger', onclick: () => App.roll.deleteSelection() }, `Delete ${notes.length > 1 ? notes.length + ' notes' : 'note'}`)));
  }

  /* ---- Track tab */
  renderTrack() {
    const H = this.host;
    const t = this.s.track;
    const meta = this.s.meta;
    if (!t) { H.append(el('div', { class: 'insp-empty' }, 'No track selected.')); return; }
    const commit = () => this.s.commit('tracks');
    H.append(this.section('Track'));
    H.append(...this.field('Name', this.input('t-name', t.name, { onCommit: (v) => { t.name = v || t.name; commit(); } })));
    const voice = meta.voices.find((v) => v.id === t.voice);
    H.append(...this.field('Voice', this.select('t-voice', meta.voices.map((v) => [v.id, `${v.name} (${v.range})`]), t.voice, (v) => { t.voice = v; commit(); }),
      { desc: voice ? voice.description : '' }));
    const style = meta.styles.find((s) => s.name === t.style);
    H.append(...this.field('Style', this.select('t-style', meta.styles.map((s) => [s.name, s.name]), t.style, (v) => { t.style = v; commit(); }),
      { desc: style ? style.description : '' }));
    H.append(...this.field('Language', this.select('t-lang', (meta.languages || [{ id: 'en', name: 'English' }]).map((l) => [l.id, l.name]), t.language, (v) => { t.language = v; commit(); })));
    H.append(this.section('Mix'));
    H.append(...this.field('Gain (dB)', this.sliderField('t-gain', t.gain_db, 0, { min: -36, max: 12, step: 0.5 }, (v) => { t.gain_db = v ?? 0; commit(); })));
    H.append(...this.field('Pan', this.sliderField('t-pan', t.pan, 0, { min: -1, max: 1, step: 0.01 }, (v) => { t.pan = v ?? 0; commit(); })));
    H.append(...this.field('Reverb send', this.sliderField('t-rev', t.reverb_send, 0.25, { min: 0, max: 1, step: 0.01 }, (v) => { t.reverb_send = v ?? 0.25; commit(); })));
    H.append(...this.field('Seed', this.numInput('t-seed', t.seed, { onCommit: (v) => { t.seed = Math.round(v); commit(); } }), { title: 'Random seed for humanisation (drift, jitter)' }));
    H.append(this.section('Voice parameters'));
    const defs = (meta.voice_defaults || {})[t.voice] || {};
    for (const p of meta.voice_params || []) {
      const v = t.voice_params[p.name];
      H.append(...this.field(p.name.replace(/_/g, ' '), this.sliderField('vp-' + p.name, v ?? null, defs[p.name], p, (val) => {
        if (val === null || val === undefined) delete t.voice_params[p.name]; else t.voice_params[p.name] = val;
        commit();
      }), { title: p.description + (defs[p.name] != null ? ` (voice default ${defs[p.name]})` : '') }));
    }
    H.append(el('div', { class: 'btn-row' }, el('button', {
      class: 'mini-btn', onclick: () => {
        for (const k of Object.keys(t.voice_params)) if (!k.startsWith('style.')) delete t.voice_params[k];
        commit();
      },
    }, 'Reset voice parameters')));
    H.append(this.section('Style overrides'));
    const sv = (meta.style_values || {})[t.style] || {};
    for (const [name, label, unit] of STYLE_OVERRIDES) {
      if (!(name in sv) && meta.style_values) continue;
      const key = 'style.' + name;
      const v = t.voice_params[key];
      H.append(...this.field(label, el('div', { class: 'pair' },
        this.numInput('so-' + name, v ?? null, {
          placeholder: sv[name] != null ? `${fmtNum(sv[name], 3)} (${t.style})` : 'style default', allowBlank: true,
          onCommit: (val) => { if (val === null) delete t.voice_params[key]; else t.voice_params[key] = val; commit(); },
        }), el('span', { class: 'unit' }, unit)), { title: `Override the "${t.style}" style's ${name}` }));
    }
    H.append(el('div', { class: 'btn-row' }, el('button', {
      class: 'mini-btn', onclick: () => {
        for (const k of Object.keys(t.voice_params)) if (k.startsWith('style.')) delete t.voice_params[k];
        commit();
      },
    }, 'Reset style overrides')));
    const pitches = t.notes.map((n) => n.pitch);
    H.append(this.section('Info'));
    H.append(el('div', { class: 'desc-text', style: { margin: '0' } },
      `${t.notes.length} notes${pitches.length ? ` · range ${midiName(Math.min(...pitches))}-${midiName(Math.max(...pitches))}` : ''} · curves: ${Object.keys(t.curves).filter((k) => t.curves[k].points.length).join(', ') || 'none'}`));
  }

  /* ---- Project tab */
  renderProject() {
    const H = this.host;
    const p = this.s.project;
    const commit = () => this.s.commit('song');
    H.append(this.section('Song'));
    H.append(...this.field('Title', this.input('p-title', p.title, { onCommit: (v) => { p.title = v || 'Untitled'; commit(); } })));
    H.append(...this.field('Key', this.input('p-key', p.key, { placeholder: 'C major', onCommit: (v) => { p.key = v || 'C major'; commit(); } }), { title: 'e.g. "A minor", "F# major", "D dorian"' }));
    H.append(...this.field('Time signature', this.input('p-ts', p.time_signature.join('/'), {
      parse: (raw) => { const m = /^(\d+)\s*\/\s*(\d+)$/.exec(raw); return m && +m[1] > 0 && [1, 2, 4, 8, 16].includes(+m[2]) ? [+m[1], +m[2]] : undefined; },
      onCommit: (v) => { p.time_signature = v; commit(); },
    })));
    H.append(...this.field('Sample rate', this.select('p-sr', [[22050, '22.05 kHz'], [32000, '32 kHz'], [44100, '44.1 kHz'], [48000, '48 kHz']], p.sample_rate || 44100, (v) => { p.sample_rate = parseInt(v, 10); commit(); })));
    H.append(this.section('Tempo map'));
    p.tempo.forEach((pt, i) => {
      const row = el('div', { class: 'pair' },
        this.numInput(`tb-${i}`, pt.beat, { min: 0, onCommit: (v) => { if (i === 0) return; pt.beat = v; p.tempo.sort((a, b) => a.beat - b.beat); commit(); } }),
        this.numInput(`tv-${i}`, pt.bpm, { min: 20, max: 400, onCommit: (v) => { pt.bpm = v; commit(); } }),
        i === 0 ? el('span', { class: 'unit' }, 'bpm') : el('button', { class: 'mini-btn ghost', title: 'Remove tempo change', onclick: () => { p.tempo.splice(i, 1); commit(); } }, icon('x')));
      if (i === 0) $('input', row).disabled = true;
      H.append(...this.field(i === 0 ? 'Start' : `Change ${i}`, row, { title: 'beat / bpm' }));
    });
    H.append(el('div', { class: 'btn-row' }, el('button', {
      class: 'mini-btn', onclick: () => {
        const at = Math.max(this.s.beatsPerBar, Math.round(this.s.playhead / this.s.beatsPerBar) * this.s.beatsPerBar);
        if (p.tempo.some((x) => Math.abs(x.beat - at) < EPS)) return;
        p.tempo.push({ beat: at, bpm: this.s.tempo.bpmAt(at) });
        p.tempo.sort((a, b) => a.beat - b.beat);
        commit();
      },
    }, icon('plus'), 'Tempo change at playhead bar')));
    H.append(this.section('File'));
    H.append(el('div', { class: 'desc-text', style: { margin: '0 0 8px' } }, this.s.path ? `Saved to ${this.s.path}` : 'Not saved to disk yet.'));
    H.append(el('div', { class: 'btn-row' },
      el('button', { class: 'mini-btn', onclick: () => App.save() }, icon('save'), 'Save'),
      el('button', { class: 'mini-btn', onclick: () => App.save(true) }, 'Save as…')));
    const total = p.tracks.reduce((a, t) => a + t.notes.length, 0);
    H.append(this.section('Info'));
    H.append(el('div', { class: 'desc-text', style: { margin: '0' } },
      `${p.tracks.length} track(s) · ${total} notes · ${fmtTime(this.s.tempo.beatToSec(this.s.projectEnd()))} long · history ${this.s.history.undo ?? '?'} undo / ${this.s.history.redo ?? '?'} redo`));
  }
}

/* ============================================================== Transport */
class Transport {
  constructor(store) {
    this.s = store;
    this.ctx = null;
    this.out = null;
    this.src = null;
    this.playing = false;
    this.rendering = false;
    this.cancelled = false;
    this.cache = [];
    this.returnBeat = 0;
    this._raf = 0;
    this.btnPlay = $('#btn-play');
    $('#btn-play').addEventListener('click', () => this.toggle());
    $('#btn-stop').addEventListener('click', () => this.stop());
    $('#btn-start').addEventListener('click', () => this.seek(0));
    $('#btn-loop').addEventListener('click', () => this.toggleLoop());
    store.on('transport', () => this.updateLcd());
    store.on('song', () => this.updateLcd());
    store.on('project', () => { this.updateLcd(); });
  }
  ensureCtx() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) throw new Error('Web Audio is not available in this browser');
      this.ctx = new AC();
      this.out = this.ctx.createGain();
      this.out.connect(this.ctx.destination);
    }
    if (this.ctx.state === 'suspended') this.ctx.resume();
    return this.ctx;
  }
  invalidateCache() { this.cache = []; }
  toggle() { if (this.playing || this.rendering) this.stop(); else this.play(); }
  toggleLoop() {
    const L = this.s.loop;
    if (!(L.end > L.start)) {
      const bpb = this.s.beatsPerBar;
      L.start = Math.floor(this.s.playhead / bpb) * bpb;
      L.end = L.start + bpb * 2;
    }
    L.on = !L.on;
    this.s.emit('transport');
    this.updateButtons();
    status(L.on ? `Loop on: ${this.s.fmtBeat(L.start)} - ${this.s.fmtBeat(L.end)}` : 'Loop off');
  }
  updateButtons() {
    $('#btn-loop').classList.toggle('on', this.s.loop.on);
    this.btnPlay.classList.toggle('playing', this.playing);
    this.btnPlay.classList.toggle('rendering', this.rendering);
  }
  updateLcd() {
    const b = this.s.barBeat(this.s.playhead);
    $('#lcd-bars').textContent = `${String(b.bar).padStart(3, '0')} . ${b.beat} . ${String(b.tick).padStart(3, '0')}`;
    $('#lcd-time').textContent = fmtTime(this.s.tempo.beatToSec(this.s.playhead));
  }
  findCached(version, start, end) {
    return this.cache.find((c) => c.version === version && c.end === end && (end === null ? c.start <= start + EPS : Math.abs(c.start - start) < EPS));
  }
  async play() {
    try { this.ensureCtx(); } catch (e) { toast(e.message, 'err'); return; }
    if (!this.s.project.tracks.some((t) => t.notes.length)) { toast('Nothing to play yet: draw some notes first.', 'warn'); return; }
    await this.s.flush();
    const L = this.s.loop;
    const loop = L.on && L.end > L.start;
    const bpb = this.s.beatsPerBar;
    let from = this.s.playhead;
    if (!loop && from >= this.s.projectEnd() - EPS) from = 0;
    const start = loop ? L.start : Math.floor(from / bpb + EPS) * bpb;
    const end = loop ? L.end : null;
    const version = this.s.version;
    let entry = this.findCached(version, start, end);
    if (!entry) {
      this.rendering = true;
      this.cancelled = false;
      this.updateButtons();
      status('Rendering…');
      const t0 = performance.now();
      try {
        const body = { start_beat: start };
        if (end !== null) body.end_beat = end;
        const { buf, report } = await Api.render(body);
        const buffer = await this.ctx.decodeAudioData(buf);
        entry = { version, start, end, buffer, report };
        this.cache = [entry, ...this.cache.filter((c) => c.version === version)].slice(0, 4);
        App.showReport(report, (performance.now() - t0) / 1000);
      } catch (e) {
        this.rendering = false;
        this.updateButtons();
        const msg = e.status === 503 || e.status === 501 ? `Render engine unavailable: ${e.message}` : `Render failed: ${e.message}`;
        status(msg, true);
        toast(msg, 'err', 6000);
        return;
      }
      this.rendering = false;
      this.updateButtons();
      if (this.cancelled) { status('Render finished (playback cancelled)'); return; }
    }
    const offsetBeat = loop ? start : Math.max(start, from);
    this.startSource(entry, offsetBeat, loop);
  }
  startSource(entry, fromBeat, loop) {
    this.stopSource();
    const ctx = this.ctx;
    const tm = this.s.tempo;
    const src = ctx.createBufferSource();
    src.buffer = entry.buffer;
    src.connect(this.out);
    const renderStartSec = entry.report && entry.report.start_seconds != null ? entry.report.start_seconds : tm.beatToSec(entry.start);
    const offset = Math.max(0, tm.beatToSec(fromBeat) - renderStartSec);
    this.loopLen = 0;
    if (loop) {
      this.loopLen = Math.min(entry.buffer.duration, tm.beatToSec(entry.end) - renderStartSec);
      src.loop = true;
      src.loopStart = 0;
      src.loopEnd = this.loopLen;
    }
    this.t0 = ctx.currentTime + 0.04;
    this.renderStartSec = renderStartSec;
    this.offset = offset;
    src.start(this.t0, Math.min(offset, entry.buffer.duration));
    src.onended = () => { if (this.src === src) this.finish(); };
    this.src = src;
    this.returnBeat = fromBeat;
    this.playing = true;
    this.updateButtons();
    status(loop ? 'Playing loop' : `Playing from ${this.s.fmtBeat(fromBeat)}`);
    cancelAnimationFrame(this._raf);
    const tick = () => {
      if (!this.playing) return;
      const lat = (ctx.outputLatency || 0) + (ctx.baseLatency || 0);
      let el2 = Math.max(0, ctx.currentTime - this.t0 - lat) + this.offset;
      if (this.loopLen > 0) el2 %= this.loopLen;
      const beat = tm.secToBeat(this.renderStartSec + el2);
      this.s.playhead = beat;
      this.s.emit('transport');
      if (App.roll && !App.roll.drag) App.roll.followPlayhead(beat);
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }
  stopSource() {
    if (this.src) {
      this.src.onended = null;
      try { this.src.stop(); } catch { /* not started */ }
      this.src.disconnect();
      this.src = null;
    }
    cancelAnimationFrame(this._raf);
  }
  finish() {
    this.stopSource();
    this.playing = false;
    this.s.playhead = this.returnBeat;
    this.s.emit('transport');
    this.updateButtons();
    status('Stopped');
  }
  stop() {
    if (this.rendering) { this.cancelled = true; status('Playback cancelled (render continues in the background)'); }
    if (this.playing) { this.finish(); return; }
    if (!this.rendering) { this.s.playhead = 0; this.s.emit('transport'); }
    this.updateButtons();
  }
  seek(beat) {
    beat = Math.max(0, beat);
    const wasPlaying = this.playing;
    this.s.playhead = beat;
    this.s.emit('transport');
    if (wasPlaying) {
      const L = this.s.loop;
      if (L.on && L.end > L.start) { this.returnBeat = beat; return; }
      this.stopSource();
      this.playing = false;
      this.play();
    }
  }
}

/* =================================================================== Chat */
class Chat {
  constructor(store) {
    this.s = store;
    this.log = $('#chat-log');
    this.text = $('#chat-text');
    this.reset = false;
    this.available = false;
    this.sending = false;
    $('#chat-form').addEventListener('submit', (e) => { e.preventDefault(); this.send(); });
    this.text.addEventListener('keydown', (e) => {
      e.stopPropagation();
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.send(); }
    });
    this.text.addEventListener('input', () => { this.text.style.height = 'auto'; this.text.style.height = Math.min(120, this.text.scrollHeight + 2) + 'px'; });
    $('#chat-reset').addEventListener('click', () => { this.reset = true; this.log.innerHTML = ''; this.renderIntro(); status('New AI conversation'); });
  }
  setStatus(dir) {
    this.available = !!(dir && dir.available);
    const s = $('#ai-status');
    s.classList.toggle('on', this.available);
    s.classList.toggle('off', !this.available);
    $('.txt', s).textContent = this.available ? 'online' : 'offline';
    s.title = this.available ? 'Claude-powered director is ready' : (dir && dir.reason) || 'unavailable';
    this.reason = dir && dir.reason;
    if (!this.log.children.length) this.renderIntro();
  }
  renderIntro() {
    const box = el('div', { class: 'chat-empty' },
      'Describe what you want and the director edits the project with the same tools you use: melodies, lyrics, harmonies, expression, mixing.');
    for (const s of ['Write an 8-bar pop melody in C major with lyrics about city lights', 'Add a harmony a third above the lead from bar 5', 'Make the last phrase breathier and add vibrato to long notes']) {
      box.append(el('button', { class: 'sugg', onclick: () => { this.text.value = s; this.text.focus(); } }, s));
    }
    this.log.append(box);
    if (!this.available) this.add('system', `AI Director is offline. ${this.reason || 'Set ANTHROPIC_API_KEY to enable.'}\nRestart the studio with ANTHROPIC_API_KEY set (and the anthropic package installed) to chat with it.`);
  }
  add(role, text, actions) {
    const m = el('div', { class: 'msg ' + role });
    if (role === 'assistant') m.innerHTML = this.md(text || '(no reply)');
    else m.textContent = text;
    if (actions && actions.length) {
      const box = el('div', { class: 'actions' });
      for (const a of actions) {
        const { label, title, err } = this.describe(a);
        box.append(el('span', { class: 'action-chip' + (err ? ' err' : ''), title }, label));
      }
      m.append(box);
    }
    this.log.append(m);
    this.log.scrollTop = this.log.scrollHeight;
    return m;
  }
  describe(a) {
    if (typeof a === 'string') return { label: a, title: a, err: false };
    const name = a.tool || a.name || a.type || 'action';
    const args = a.input || a.args || a.arguments || {};
    const err = !!(a.error || a.is_error || a.ok === false);
    const brief = Object.entries(args).slice(0, 2).map(([k, v]) => `${k}=${typeof v === 'object' ? '…' : String(v).slice(0, 18)}`).join(' ');
    return { label: (err ? '✗ ' : '✓ ') + name + (brief ? ' ' + brief : ''), title: JSON.stringify(a, null, 1).slice(0, 1500), err };
  }
  md(s) {
    let h = escapeHtml(s);
    h = h.replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
    return h;
  }
  async send() {
    const msg = this.text.value.trim();
    if (!msg || this.sending) return;
    this.text.value = '';
    this.text.style.height = '';
    $('.chat-empty', this.log)?.remove();
    this.add('user', msg);
    const typing = el('div', { class: 'msg assistant typing' }, el('span'), el('span'), el('span'));
    this.log.append(typing);
    this.log.scrollTop = this.log.scrollHeight;
    this.sending = true;
    $('#chat-send').disabled = true;
    if (!this.available) {
      // Re-check (cheap) before calling the director; avoids a pointless 501 round trip.
      try { const m = await Api.meta(); this.available = !!(m.director && m.director.available); this.reason = m.director && m.director.reason; } catch { /* ignore */ }
      if (!this.available) {
        typing.remove();
        this.add('system', `AI Director is offline: ${this.reason || 'set ANTHROPIC_API_KEY to enable.'}`);
        this.sending = false;
        $('#chat-send').disabled = false;
        return;
      }
      this.setStatus({ available: true });
    }
    await this.s.flush();
    App.setBusy(true, 'AI Director is working…');
    try {
      const r = await Api.ai(msg, this.reset);
      this.reset = false;
      typing.remove();
      this.add('assistant', r.reply, r.actions);
      if (r.project) this.s.setProject(r.project);
      else await this.s.reload();
      if (r.history) this.s.setHistory(r.history);
      App.transport.invalidateCache();
      status(`AI Director: ${(r.actions || []).length} action(s)`);
    } catch (e) {
      typing.remove();
      if (e.status === 501) {
        this.setStatus({ available: false, reason: e.message });
        this.add('system', `AI Director is offline: ${e.message}`);
      } else {
        this.add('error', `Director error: ${e.message}`);
      }
      try { await this.s.reload(); } catch { /* ignore */ }
    } finally {
      this.sending = false;
      $('#chat-send').disabled = false;
      App.setBusy(false);
    }
  }
}

/* ==================================================================== App */
const App = {
  store: null, roll: null, lane: null, tracks: null, inspector: null, transport: null, chat: null,
  spaceHeld: false,
  connected: true,

  async init() {
    const s = (this.store = new Store());
    this.roll = new PianoRoll(s, $('#roll'), $('#roll-wrap'));
    this.lane = new ParamLane(s, $('#lane'), $('#lane-wrap'), this.roll);
    this.tracks = new TrackList(s, $('#track-list'));
    this.master = new MasterStrip(s, $('#master-strip'));
    this.inspector = new Inspector(s, $('#insp-body'));
    this.transport = new Transport(s);
    this.chat = new Chat(s);
    this.bindTopbar();
    this.bindEditorToolbar();
    this.bindKeys();
    this.bindMenus();
    this.bindDrop();
    this.bindSplitter();
    s.on('history', () => this.updateHistoryButtons());
    s.on('dirty', () => $('#dirty-dot').classList.toggle('on', s.dirty));
    s.on('selection', () => this.updateSelInfo());
    s.on('notes', () => { this.updateSelInfo(); this.transport.invalidateCache(); });
    s.on('project', () => { this.syncSongProps(); this.updateSelInfo(); });
    s.on('song', () => this.syncSongProps());
    window.addEventListener('beforeunload', () => {
      if (s.pending) navigator.sendBeacon('/api/project', new Blob([JSON.stringify(s.project)], { type: 'application/json' }));
    });
    try {
      const [meta, project] = await Promise.all([Api.meta(), Api.getProject()]);
      s.meta = meta;
      this.buildKeySelect();
      this.lane.buildTabs();
      this.chat.setStatus(meta.director);
      s.setProject(project, { keepView: false });
      await s.refreshHistory();
      this.roll.centerOnNotes();
      status(`Loaded "${s.project.title}"`);
    } catch (e) {
      status('Could not load project: ' + e.message, true);
      toast('Could not reach the studio server: ' + e.message, 'err', 8000);
    }
    this.transport.updateLcd();
    setInterval(() => this.poll(), 3000);
  },

  /* ---- server sync */
  async poll() {
    const s = this.store;
    if (s.pending || s.busy || this.roll.drag || this.lane.drag || this.roll.editing || this._polling) return;
    this._polling = true;
    try {
      const h = await Api.history();
      const changed = s.revision != null && h.revision != null && h.revision !== s.revision;
      s.setHistory(h);
      if (changed && !s.pending) {
        s.setProject(await Api.getProject());
        this.transport.invalidateCache();
        status('Project updated on the server (tool call) - reloaded');
      }
    } catch { /* connection status already updated */ }
    this._polling = false;
  },
  setConnected(ok) {
    if (ok === this.connected) return;
    this.connected = ok;
    const c = $('#sb-conn');
    c.classList.toggle('on', ok);
    c.classList.toggle('off', !ok);
    c.lastChild.textContent = ok ? 'server' : 'offline';
    if (!ok) status('Server connection lost', true);
  },
  setBusy(on, text) {
    this.store.busy = on;
    $('#busy-overlay').hidden = !on;
    if (text) $('#busy-text').textContent = text;
  },

  /* ---- top bar */
  bindTopbar() {
    const s = this.store;
    $('#btn-undo').addEventListener('click', () => s.undo());
    $('#btn-redo').addEventListener('click', () => s.redo());
    $('#btn-save').addEventListener('click', () => this.save());
    $('#btn-help').addEventListener('click', () => this.showModal('help-modal'));
    $$('.modal').forEach((m) => m.addEventListener('click', (e) => { if (e.target === m || e.target.closest('[data-close]')) m.hidden = true; }));
    const tempo = $('#in-tempo');
    tempo.addEventListener('change', () => {
      const v = parseFloat(tempo.value);
      if (!(v >= 20 && v <= 400)) { this.syncSongProps(); return; }
      s.project.tempo[0].bpm = v;
      s.commit('song');
      this.transport.invalidateCache();
    });
    tempo.addEventListener('keydown', (e) => { e.stopPropagation(); if (e.key === 'Enter') tempo.blur(); });
    $('#in-timesig').addEventListener('change', (e) => {
      s.project.time_signature = e.target.value.split('/').map(Number);
      s.commit('song');
    });
    $('#in-key').addEventListener('change', (e) => { s.project.key = e.target.value; s.commit('song'); });
    const title = $('#in-title');
    title.addEventListener('change', () => { s.project.title = title.value.trim() || 'Untitled'; s.commit('song'); });
    title.addEventListener('keydown', (e) => { e.stopPropagation(); if (e.key === 'Enter') title.blur(); });
  },
  buildKeySelect() {
    const sel = $('#in-key');
    sel.innerHTML = '';
    for (const mode of ['major', 'minor']) {
      const g = el('optgroup', { label: mode === 'major' ? 'Major' : 'Minor' });
      for (const t of KEY_TONICS) g.append(el('option', { value: `${t} ${mode}` }, `${t} ${mode}`));
      sel.append(g);
    }
  },
  syncSongProps() {
    const p = this.store.project;
    const tempo = $('#in-tempo');
    if (document.activeElement !== tempo) tempo.value = fmtNum(p.tempo[0].bpm, 2);
    tempo.title = p.tempo.length > 1 ? `${p.tempo.length - 1} tempo change(s) - edit in Project tab` : 'Tempo (BPM)';
    const ts = p.time_signature.join('/');
    const tsSel = $('#in-timesig');
    if (![...tsSel.options].some((o) => o.value === ts)) tsSel.append(el('option', {}, ts));
    tsSel.value = ts;
    const keySel = $('#in-key');
    if (![...keySel.querySelectorAll('option')].some((o) => o.value === p.key)) keySel.append(el('option', { value: p.key }, p.key));
    keySel.value = p.key;
    const title = $('#in-title');
    if (document.activeElement !== title) title.value = p.title;
    document.title = `${p.title} - Synth2 Vocal Studio`;
    $('#sb-path').textContent = this.store.path ? this.store.path : '';
  },
  updateHistoryButtons() {
    const h = this.store.history;
    $('#btn-undo').disabled = h.undo === 0;
    $('#btn-redo').disabled = h.redo === 0;
    $('#btn-undo').title = `Undo (Ctrl+Z)${h.undo != null ? ` - ${h.undo} step(s)` : ''}`;
    $('#btn-redo').title = `Redo (Ctrl+Y)${h.redo != null ? ` - ${h.redo} step(s)` : ''}`;
    $('#sb-path').textContent = this.store.path || '';
  },
  updateSelInfo() {
    const n = this.store.selection.size;
    const t = this.store.track;
    $('#sb-sel').textContent = t ? (n ? `${n} selected` : `${t.notes.length} notes`) : '';
  },
  cursorInfo(text) { $('#sb-cursor').textContent = text; },

  /* ---- editor toolbar */
  bindEditorToolbar() {
    const s = this.store;
    $('#tool-seg').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-tool]');
      if (b) this.setTool(b.dataset.tool);
    });
    const snap = $('#in-snap');
    for (const o of snap.options) { if (o.value === '0.6666666667') o.value = '2/3'; if (o.value === '0.3333333333') o.value = '1/3'; if (o.value === '0.1666666667') o.value = '1/6'; }
    snap.addEventListener('change', () => { s.snap = parseFrac(snap.value); s.emit('view'); status(`Snap: ${snap.options[snap.selectedIndex].text}`); });
    s.snap = parseFrac(snap.value);
    $('#in-notelen').addEventListener('change', (e) => { s.newNoteLen = e.target.value; });
    $('#chk-ghost').addEventListener('change', (e) => { this.roll.showGhosts = e.target.checked; this.roll.invalidate(); });
    $('#chk-phon').addEventListener('change', (e) => { this.roll.showPhonemes = e.target.checked; this.roll.invalidate(); });
    $('#btn-zoom-in').addEventListener('click', () => this.roll.zoomBy(1.4));
    $('#btn-zoom-out').addEventListener('click', () => this.roll.zoomBy(1 / 1.4));
    $('#btn-zoom-fit').addEventListener('click', () => this.roll.zoomFit());
  },
  setTool(tool) {
    this.roll.tool = tool;
    $$('#tool-seg button').forEach((b) => b.classList.toggle('active', b.dataset.tool === tool));
    status(tool === 'draw' ? 'Draw tool: drag to create notes (Shift+drag box-selects)' : 'Select tool: drag to box-select, double-click to add a note');
  },
  bindSplitter() {
    const sp = $('#splitter'), lane = $('#lane-panel');
    let drag = null;
    sp.addEventListener('pointerdown', (e) => { drag = { y: e.clientY, h: lane.getBoundingClientRect().height }; sp.classList.add('drag'); sp.setPointerCapture(e.pointerId); });
    sp.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const max = $('.editor').getBoundingClientRect().height - 220;
      lane.style.height = `${clamp(drag.h - (e.clientY - drag.y), 90, Math.max(100, max))}px`;
    });
    sp.addEventListener('pointerup', () => { drag = null; sp.classList.remove('drag'); });
  },

  /* ---- keyboard shortcuts */
  bindKeys() {
    window.addEventListener('keydown', (e) => this.onKey(e));
    document.addEventListener('change', (e) => {
      if (e.target.matches && e.target.matches('select, input[type=range], input[type=checkbox]') && !e.target.closest('.insp-body')) e.target.blur();
    });
    window.addEventListener('keyup', (e) => { if (e.code === 'Space') this.spaceHeld = false; });
    window.addEventListener('blur', () => { this.spaceHeld = false; });
  },
  onKey(e) {
    const s = this.store, roll = this.roll;
    const mod = e.ctrlKey || e.metaKey;
    const k = e.key;
    if (mod && k.toLowerCase() === 's') { e.preventDefault(); this.save(e.shiftKey); return; }
    if (mod && k.toLowerCase() === 'o') { e.preventDefault(); $('#file-input').click(); return; }
    if (isTyping(e.target)) return;
    if (!$('#help-modal').hidden && k === 'Escape') { $('#help-modal').hidden = true; return; }
    if (mod && k.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) s.redo(); else s.undo(); return; }
    if (mod && k.toLowerCase() === 'y') { e.preventDefault(); s.redo(); return; }
    if (s.busy && !['Escape', ' '].includes(k)) return;
    if (mod) {
      switch (k.toLowerCase()) {
        case 'a': e.preventDefault(); roll.selectAll(); return;
        case 'c': e.preventDefault(); roll.copy(); return;
        case 'x': e.preventDefault(); roll.cut(); return;
        case 'v': e.preventDefault(); roll.paste(); return;
        case 'd': e.preventDefault(); roll.duplicate(); return;
        default: return;
      }
    }
    const step = s.snap || 0.25;
    switch (k) {
      case ' ': e.preventDefault(); if (!e.repeat) this.transport.toggle(); return;
      case 'Delete': case 'Backspace': e.preventDefault(); roll.deleteSelection(); return;
      case 'ArrowUp': e.preventDefault(); roll.transpose(e.shiftKey ? 12 : 1); return;
      case 'ArrowDown': e.preventDefault(); roll.transpose(e.shiftKey ? -12 : -1); return;
      case 'ArrowLeft': e.preventDefault(); if (e.shiftKey) roll.resizeBy(-step); else roll.moveBy(-step); return;
      case 'ArrowRight': e.preventDefault(); if (e.shiftKey) roll.resizeBy(step); else roll.moveBy(step); return;
      case 'Enter': { const n = s.selectedNotes(); if (n.length) { e.preventDefault(); roll.openLyric(n[0]); } return; }
      case 'Tab': e.preventDefault(); roll.selectStep(e.shiftKey ? -1 : 1); return;
      case 'Escape': this.closeMenus(); if (this.transport.playing) this.transport.stop(); else s.select([]); return;
      case 'Home': e.preventDefault(); this.transport.seek(0); return;
      case 'End': e.preventDefault(); this.transport.seek(s.projectEnd()); return;
      default: break;
    }
    switch (k.toLowerCase()) {
      case 'l': this.transport.toggleLoop(); break;
      case 'd': this.setTool('draw'); break;
      case 'v': this.setTool('select'); break;
      case 'f': roll.zoomFit(); break;
      case 'q': roll.quantize(); break;
      case '?': this.showModal('help-modal'); break;
      case '=': case '+': roll.zoomBy(1.25); break;
      case '-': roll.zoomBy(0.8); break;
      default: break;
    }
  },

  /* ---- menus, files */
  bindMenus() {
    const menu = $('#file-menu');
    $('.menu-btn', menu).addEventListener('click', (e) => { e.stopPropagation(); menu.classList.toggle('open'); });
    document.addEventListener('pointerdown', (e) => {
      if (!e.target.closest('.menu')) menu.classList.remove('open');
      if (!e.target.closest('#ctx-menu')) $('#ctx-menu').hidden = true;
    });
    $('.menu-pop', menu).addEventListener('click', async (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      menu.classList.remove('open');
      if (b.dataset.export) return this.exportAs(b.dataset.export);
      switch (b.dataset.action) {
        case 'new': return this.newProject();
        case 'import': return $('#file-input').click();
        case 'save': return this.save();
        case 'save-as': return this.save(true);
        case 'download-wav': return this.downloadWav();
        case 'bounce': return this.bounce();
        default: return undefined;
      }
    });
    $('#file-input').addEventListener('change', async (e) => {
      const f = e.target.files[0];
      e.target.value = '';
      if (f) await this.importFile(f);
    });
    $('#sb-report').addEventListener('click', () => this.showReportModal());
  },
  closeMenus() {
    $('#file-menu').classList.remove('open');
    $('#ctx-menu').hidden = true;
  },
  contextMenu(x, y, items) {
    const m = $('#ctx-menu');
    m.innerHTML = '';
    for (const it of items) {
      if (it === '-') { m.append(el('div', { class: 'menu-sep' })); continue; }
      const [label, kbd, fn] = it;
      m.append(el('button', { onclick: () => { m.hidden = true; fn(); } }, el('span', {}, label), kbd ? el('kbd', {}, kbd) : null));
    }
    m.hidden = false;
    const r = m.getBoundingClientRect();
    m.style.left = `${Math.min(x, window.innerWidth - r.width - 6)}px`;
    m.style.top = `${Math.min(y, window.innerHeight - r.height - 6)}px`;
  },
  pickColor(current, onPick) {
    const inp = $('#color-input');
    inp.value = /^#[0-9a-f]{6}$/i.test(current) ? current : '#5eead4';
    const handler = () => { onPick(inp.value); inp.removeEventListener('change', handler); };
    inp.addEventListener('change', handler);
    inp.click();
  },
  async newProject() {
    if (!confirm('Start a new project? The current one stays in the undo history.')) return;
    await this.store.flush();
    try {
      const r = await Api.newProject({ title: 'Untitled', bpm: 120 });
      this.store.setProject(r.project, { keepView: false });
      this.store.setHistory(r.history || {});
      this.store.dirty = false;
      this.store.emit('dirty');
      this.transport.invalidateCache();
      status('New project');
    } catch (e) { toast('New project failed: ' + e.message, 'err'); }
  },
  async importFile(file) {
    await this.store.flush();
    status(`Importing ${file.name}…`);
    try {
      const buf = await file.arrayBuffer();
      const r = await Api.importFile(file.name, buf);
      this.store.setProject(r.project, { keepView: false });
      this.store.setHistory(r.history || {});
      this.transport.invalidateCache();
      this.roll.zoomFit();
      const n = r.project.tracks.reduce((a, t) => a + t.notes.length, 0);
      status(`Imported ${file.name}: ${r.project.tracks.length} track(s), ${n} notes`);
      toast(`Imported ${file.name}`);
    } catch (e) {
      status('Import failed: ' + e.message, true);
      toast('Import failed: ' + e.message, 'err', 7000);
    }
  },
  async exportAs(fmt) {
    await this.store.flush();
    try {
      const { blob, filename } = await Api.download('/api/export?format=' + encodeURIComponent(fmt));
      downloadBlob(blob, filename);
      status(`Exported ${filename}`);
    } catch (e) { toast(`Export (${fmt}) failed: ${e.message}`, 'err', 6000); }
  },
  async bounce() {
    await this.store.flush();
    status('Rendering the full song\u2026');
    this.transport.btnPlay.classList.add('rendering');
    try {
      const { buf, report } = await Api.render({});
      this.showReport(report);
      downloadBlob(new Blob([buf], { type: 'audio/wav' }), `${(this.store.project.title || 'render').replace(/[^\w.-]+/g, '_')}.wav`);
    } catch (e) { toast('Render failed: ' + e.message, 'err', 6000); status('Render failed: ' + e.message, true); }
    this.transport.btnPlay.classList.remove('rendering');
  },
  async downloadWav() {
    try {
      const { blob } = await Api.download('/api/render/last');
      downloadBlob(blob, `${(this.store.project.title || 'render').replace(/[^\w.-]+/g, '_')}.wav`);
    } catch (e) { toast(e.status === 404 ? 'Nothing rendered yet: press play first.' : 'Download failed: ' + e.message, 'warn'); }
  },
  async save(askPath = false) {
    const s = this.store;
    await s.flush();
    let path = null;
    if (askPath || !s.path) {
      const def = s.path || `${(s.project.title || 'untitled').replace(/[^\w.-]+/g, '_')}.s2p.json`;
      path = prompt('Save project to (path on the server machine; .s2p.json, .svp, .mid, .musicxml, .ust or .s2s):', def);
      if (!path) return;
    }
    try {
      const r = await Api.save(path);
      s.path = r.path;
      s.dirty = false;
      s.emit('dirty');
      this.syncSongProps();
      status(`Saved ${r.path}`);
      toast(`Saved ${r.path}`);
    } catch (e) { toast('Save failed: ' + e.message, 'err', 6000); status('Save failed: ' + e.message, true); }
  },
  bindDrop() {
    let hint = null;
    window.addEventListener('dragover', (e) => {
      if (!e.dataTransfer || ![...e.dataTransfer.types].includes('Files')) return;
      e.preventDefault();
      if (!hint) { hint = el('div', { class: 'drop-hint' }, 'Drop a .svp, .mid, .musicxml, .ust, .s2s or .json file to import'); document.body.append(hint); }
    });
    window.addEventListener('dragleave', (e) => { if (hint && (e.clientX <= 0 || e.clientY <= 0 || e.clientX >= innerWidth || e.clientY >= innerHeight)) { hint.remove(); hint = null; } });
    window.addEventListener('drop', (e) => {
      if (hint) { hint.remove(); hint = null; }
      const f = e.dataTransfer && e.dataTransfer.files[0];
      if (!f) return;
      e.preventDefault();
      this.importFile(f);
    });
  },

  /* ---- render report */
  showReport(report, wall) {
    this.lastReport = report;
    const warnings = (report.tracks || []).flatMap((t) => (t.warnings || []).map((w) => `${t.name || t.id}: ${w}`));
    const parts = [];
    if (report.duration_s != null) parts.push(`${fmtNum(report.duration_s, 2)} s audio`);
    const rt = report.render_time_s ?? wall;
    if (rt != null) parts.push(`rendered in ${fmtNum(rt, 2)} s`);
    if (report.realtime_factor != null) parts.push(`${fmtNum(report.realtime_factor, 1)}× RT`);
    if (report.peak_db != null) parts.push(`peak ${fmtNum(report.peak_db, 1)} dB`);
    const sb = $('#sb-report');
    sb.innerHTML = '';
    sb.append(el('span', {}, '● ' + parts.join(' · ')));
    if (warnings.length) sb.append(el('span', { class: 'warn' }, ` · ${warnings.length} warning(s)`));
    sb.title = warnings.length ? warnings.join('\n') : 'Click for the full render report';
    status('Render complete');
    const wkey = warnings.join('|');
    if (warnings.length && wkey !== this._lastWarnings) toast(warnings.slice(0, 3).join('\n'), 'warn', 6000);
    this._lastWarnings = wkey;
  },
  showReportModal() {
    if (!this.lastReport) return;
    let m = $('#report-modal');
    if (!m) {
      m = el('div', { class: 'modal', id: 'report-modal' }, el('div', { class: 'modal-card' },
        el('div', { class: 'modal-head' }, el('span', {}, 'Render report'), el('button', { class: 'mini-btn ghost', 'data-close': '' }, icon('x'))),
        el('pre', { class: 'report-pre' })));
      m.addEventListener('click', (e) => { if (e.target === m || e.target.closest('[data-close]')) m.hidden = true; });
      document.body.append(m);
    }
    $('.report-pre', m).textContent = JSON.stringify(this.lastReport, null, 2);
    m.hidden = false;
  },
  showModal(id) { $('#' + id).hidden = false; },

  /* ---- note preview tone */
  preview(pitch, dur = 0.32) {
    try {
      const ctx = this.transport.ensureCtx();
      const t = ctx.currentTime;
      const o = ctx.createOscillator(), g = ctx.createGain(), f = ctx.createBiquadFilter();
      o.type = 'sawtooth';
      o.frequency.value = 440 * Math.pow(2, (pitch - 69) / 12);
      f.type = 'lowpass'; f.frequency.value = 1800;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(0.08, t + 0.012);
      g.gain.exponentialRampToValueAtTime(0.0008, t + dur);
      o.connect(f); f.connect(g); g.connect(ctx.destination);
      o.start(t); o.stop(t + dur + 0.02);
    } catch { /* audio unavailable */ }
  },
};

window.addEventListener('DOMContentLoaded', () => App.init());
window.Synth2App = App; // handy for debugging and automated UI tests
