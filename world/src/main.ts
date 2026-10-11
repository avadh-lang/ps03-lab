// LabDex world: a walkable lab campus whose creatures ARE the lab's agents.
// Everything on screen is driven by the same /api/state + /ws events as the dashboard.
// @ts-nocheck  (esbuild bundles without type-checking; the vendored art/audio is PokeStonks code)
import * as Phaser from 'phaser';
import { artSteps } from './vendor/art/register';
import { CHAR_WALK_MAPPING } from './vendor/art/characters';
import { buildRouteMap, collisionGrid, tallGrassGrid } from './vendor/overworld/maps';
import { LAYER } from './vendor/overworld/mapTypes';
import { audio } from './vendor/audio';

const TILE = 16;
const TOKEN = new URLSearchParams(location.search).get('token');
const AUTH: Record<string, string> = TOKEN ? { authorization: 'Bearer ' + TOKEN } : {};
const $ = (s: string) => document.querySelector(s) as HTMLElement;
const esc = (s: any) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

async function api(path: string, opt: RequestInit = {}) {
  const r = await fetch(path, { ...opt, headers: { 'content-type': 'application/json', ...AUTH, ...(opt.headers || {}) } });
  if (!r.ok) {
    let m = r.statusText;
    try { const d = (await r.json()).detail; m = Array.isArray(d) ? d.map((x) => x.msg).join('; ') : d || m; } catch { /* keep statusText */ }
    throw new Error(m);
  }
  return r.json();
}

/* ------------------------------------------------------------------ creatures */
const NAME_KEY: Record<string, string> = {
  PaperReader: 'reader', Planner: 'planner', 'CodeRunner-Careful': 'careful', 'CodeRunner-Literal': 'literal',
  'CodeRunner-Hasty': 'hasty', Debugger: 'debugger', Analyst: 'analyst', HyperTuner: 'tuner',
};
const CAP_KEY: Record<string, string> = {
  paper_read: 'reader', experiment_plan: 'planner', whatif_design: 'planner', code_execute: 'literal',
  error_debug: 'debugger', repro_compare: 'analyst', hyperparam_search: 'tuner',
};
function creatureKey(a: any): string {
  const base = String(a.name || a.id).split('_v')[0];
  if (NAME_KEY[base]) return NAME_KEY[base];
  for (const c of a.capabilities || []) if (CAP_KEY[c]) return CAP_KEY[c];
  return 'literal';
}

/* ------------------------------------------------------------------ shared state */
const S: any = { agents: [], runs: [], approvals: [], lastId: 0, llm: null, mons: new Map(), scene: null, battle: null, shown: new Set(), claim: null };
let ws: WebSocket | null = null, retry = 0, syncT: any = null;

function feed(text: string, cls = '') {
  const f = $('#feed');
  const d = document.createElement('div');
  d.className = 'ln ' + cls;
  d.innerHTML = text;
  f.appendChild(d);
  while (f.children.length > 7) f.removeChild(f.firstChild!);
}

async function refresh() {
  try {
    const st = await api('/api/state');
    S.agents = st.agents; S.runs = st.runs; S.approvals = st.approvals; S.llm = st.llm_usage;
    if (!S.lastId) S.lastId = st.last_event_id;
    if (!S.claim) for (const e of st.events || []) if (e.type === 'paper') S.claim = e.claim;
    $('#conn').textContent = 'live'; $('#conn').className = 'pill on';
    renderHud();
    S.scene?.syncAgents();
  } catch { $('#conn').textContent = 'offline'; $('#conn').className = 'pill off'; }
}
const scheduleRefresh = () => { clearTimeout(syncT); syncT = setTimeout(refresh, 200); };

function narrate(e: any) {
  const n = (x: any) => `<b>${esc(x)}</b>`;
  switch (e.type) {
    case 'request': return feed(`> ${esc(e.text)}`, 'sys');
    case 'agent_spawned': return feed(`✦ a wild ${n(e.agent)} appeared!`, 'hit');
    case 'agent_retired': return feed(`✕ ${n(e.agent)} fainted: ${esc(e.reason)}`, 'err');
    case 'agent_refined': return feed(`★ ${n(e.old)} evolved into ${n(e.new)}`, 'hit');
    case 'agent_merged': return feed(`⊕ ${n(e.a)} + ${n(e.b)} fused into ${n(e.new)}`, 'hit');
    case 'run_started': return feed(`run #${e.run_id} started`, 'dim');
    case 'run_error': return feed(`run #${e.run_id} error: ${esc(e.signature)}`, 'err');
    case 'fix_recalled': return feed(`⟲ ${esc(e.message)}`, 'hit');
    case 'fix_saved': return feed(`fix saved to memory: ${esc(JSON.stringify(e.patch))}`, 'hit');
    case 'memory_hit': return feed(`★ ${esc(e.message)}`, 'hit');
    case 'run_done': return feed(`run #${e.run_id} [${esc(e.label)}] ${e.status} · ${e.result?.grok_epoch ? 'grokked @' + e.result.grok_epoch : 'no grok'}`, e.status === 'success' ? 'hit' : 'err');
    case 'run_killed': return feed(`run #${e.run_id} killed`, 'err');
    case 'repro_compare': return feed(`ReproComparator: ${esc(e.summary)}`, 'hit');
    case 'approval_requested': return feed(`⏸ ${n(e.name)} wants to join. Catch it?`, 'sys');
    case 'recovered': return feed('restart recovery closed interrupted work', 'err');
    case 'reset': return feed('lab reset', 'sys');
  }
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws?since=${S.lastId}${TOKEN ? '&token=' + TOKEN : ''}`);
  ws.onopen = () => { retry = 0; };
  ws.onclose = () => setTimeout(connect, Math.min(5000, 500 * 2 ** retry++));
  ws.onmessage = (m) => {
    const e = JSON.parse(m.data);
    if (e.type === 'heartbeat') return;
    if (e.id) S.lastId = Math.max(S.lastId, e.id);
    if (e.type === 'metric') { S.scene?.onMetric(e); S.battle?.onMetric(e); return; }
    if (e.type === 'paper') S.claim = e.claim;
    if (e.type === 'run_done' || e.type === 'run_killed' || e.type === 'run_failed') S.battle?.onRunEnd(e);
    narrate(e);
    S.scene?.onEvent(e);
    scheduleRefresh();
  };
}

/* ------------------------------------------------------------------ DOM HUD */
function renderHud() {
  const u = S.llm;
  $('#llm').textContent = u && u.configured ? `LLM ${Math.round(u.used / 1000)}k/${Math.round(u.cap / 1000)}k tok` : 'rule-based';
  const act = S.runs.filter((r: any) => r.status === 'running' || r.status === 'waiting');
  $('#runs').innerHTML = act.length
    ? act.map((r: any) => {
        const acc = S.scene?.accOf(r.id) ?? 0;
        return `<div class="run"><b>#${r.id} ${esc(r.label)}</b><span>${esc(r.agent_id || '')} · ${r.status}</span>
        <div class="bar"><i style="width:${Math.round(acc * 100)}%"></i></div>
        ${r.status === 'running' ? `<div><button class="mini" data-battle="${r.id}">Watch battle</button> <button class="mini red" data-kill="${r.id}">Kill</button></div>` : ''}</div>`;
      }).join('')
    : '<div class="empty">No runs. Ask the lab below.</div>';
  const pend = S.approvals.filter((a: any) => a.status === 'pending');
  $('#appr').innerHTML = pend.map((a: any) => `<div class="appr"><b>A wild ${esc(a.payload.name)} wants to join!</b>
    <span>needs ${esc(a.payload.capability)}</span>
    <div><button class="green" data-appr="${a.id}" data-y="1">Catch it</button><button class="red" data-appr="${a.id}" data-y="0">Let it go</button></div></div>`).join('');
}

function bindHud() {
  document.body.addEventListener('click', async (e) => {
    const b = (e.target as HTMLElement).closest('button');
    if (!b) return;
    try {
      if (b.dataset.battle) S.scene?.startBattle(+b.dataset.battle);
      if (b.id === 'backBtn') S.battle?.leave();
      if (b.dataset.kill) await api(`/api/runs/${b.dataset.kill}/kill`, { method: 'POST' });
      if (b.dataset.appr) await api(`/api/approvals/${b.dataset.appr}/decide`, { method: 'POST', body: JSON.stringify({ approve: b.dataset.y === '1', via: 'world' }) });
    } catch (err) { feed(esc(err.message), 'err'); }
    refresh();
  });
  $('#ask').addEventListener('submit', async (e) => {
    e.preventDefault();
    const t = ($('#askText') as HTMLTextAreaElement).value.trim();
    if (!t) return;
    try { await api('/api/requests', { method: 'POST', body: JSON.stringify({ text: t, preset: 'fast' }) }); audio.sfx('encounter'); }
    catch (err) { feed(esc(err.message), 'err'); }
  });
  $('#mute').addEventListener('click', () => {
    audio.setMuted(!audio.isMuted());
    $('#mute').textContent = audio.isMuted() ? '🔇 Sound off' : '🔊 Sound on';
  });
  const start = () => { audio.unlock().then(() => audio.bgm('overworld')); window.removeEventListener('pointerdown', start); window.removeEventListener('keydown', start); };
  window.addEventListener('pointerdown', start); window.addEventListener('keydown', start);
  $('#notebook').setAttribute('href', '/' + (TOKEN ? '?token=' + TOKEN : ''));
}

/* ------------------------------------------------------------------ scenes */
class Boot extends Phaser.Scene {
  constructor() { super('Boot'); }
  create() {
    const steps = artSteps(this);
    const built = buildRouteMap('tech', 'labdex', { routeName: 'LabDex Campus' });
    this.registry.set('map', built);
    this.cache.tilemap.add(built.key, { format: Phaser.Tilemaps.Formats.TILED_JSON, data: built.json });
    let i = 0;
    const next = () => {
      if (i >= steps.length) { $('#loading').classList.add('gone'); this.scene.start('World'); return; }
      steps[i++].run();
      ($('#loadbar') as HTMLElement).style.width = Math.round((i / steps.length) * 100) + '%';
      $('#loadlabel').textContent = steps[i - 1].label;
      this.time.delayedCall(0, next);
    };
    next();
  }
}

interface Mon {
  agent: any; key: string; img: Phaser.GameObjects.Image; label: Phaser.GameObjects.Text; hp: Phaser.GameObjects.Graphics;
  star: Phaser.GameObjects.Image | null; tx: number; ty: number; home: { x: number; y: number }; moving: boolean;
  runId: number | null; slot: number | null; acc: number; fainted: boolean; version: number;
}

class World extends Phaser.Scene {
  constructor() { super('World'); }
  built: any; w = 0; h = 0; blocked: Uint8Array; tall: Uint8Array; dist: Int16Array;
  player: Phaser.GameObjects.Sprite; px = 0; py = 0; pMoving = false; facing = 'down';
  homes: Array<{ x: number; y: number }> = []; slots: Array<{ x: number; y: number; used: number | null }> = [];
  keys: any; cursors: any; metrics = new Map<number, number>(); animFlip = 0; layers: any = {};

  create() {
    S.scene = this;
    const built = (this.built = this.registry.get('map'));
    const meta = built.meta; this.w = meta.width; this.h = meta.height;
    this.blocked = collisionGrid(built); this.tall = tallGrassGrid(built);

    const tilemap = this.make.tilemap({ key: built.key });
    const tileset = tilemap.addTilesetImage('tiles', 'tiles_tech', TILE, TILE, 0, 0)!;
    for (const l of built.json.layers) {
      if (l.type !== 'tilelayer') continue;
      const layer = tilemap.createLayer(l.name, tileset, 0, 0);
      if (layer) this.layers[l.name] = layer;
    }
    this.layers[LAYER.collision]?.setVisible(false);
    this.layers[LAYER.anim1]?.setVisible(false);
    this.time.addEvent({ delay: 460, loop: true, callback: () => {
      this.animFlip = this.animFlip ? 0 : 1;
      this.layers[LAYER.anim0]?.setVisible(this.animFlip === 0);
      this.layers[LAYER.anim1]?.setVisible(this.animFlip === 1);
    } });

    // BFS distances from the spawn: every spot we use is reachable on foot
    this.dist = new Int16Array(this.w * this.h).fill(-1);
    const q = [[meta.spawn.x, meta.spawn.y]]; this.dist[meta.spawn.y * this.w + meta.spawn.x] = 0;
    for (let k = 0; k < q.length; k++) {
      const [x, y] = q[k];
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nx = x + dx, ny = y + dy;
        if (!this.free(nx, ny) || this.dist[ny * this.w + nx] >= 0) continue;
        this.dist[ny * this.w + nx] = this.dist[y * this.w + x] + 1; q.push([nx, ny]);
      }
    }
    const reach = [];
    for (let y = 0; y < this.h; y++) for (let x = 0; x < this.w; x++) if (this.dist[y * this.w + x] > 1) reach.push({ x, y, d: this.dist[y * this.w + x] });
    const pick = (cands: any[], n: number, gap: number) => {
      const out: any[] = [];
      for (const c of cands) { if (out.every((o) => Math.max(Math.abs(o.x - c.x), Math.abs(o.y - c.y)) >= gap)) out.push(c); if (out.length >= n) break; }
      return out;
    };
    this.homes = pick(reach.filter((c) => !this.tall[c.y * this.w + c.x] && c.d >= 3 && c.d <= 16).sort((a, b) => a.d - b.d), 12, 3);
    this.slots = pick(reach.filter((c) => this.tall[c.y * this.w + c.x]).sort((a, b) => a.d - b.d), 6, 3).map((c) => ({ x: c.x, y: c.y, used: null }));

    this.px = meta.spawn.x; this.py = meta.spawn.y;
    this.player = this.add.sprite(this.px * TILE, this.py * TILE - 8, 'char_player_a', 1).setOrigin(0, 0);
    this.player.setDepth(this.py * TILE + 100);
    const cam = this.cameras.main;
    cam.setBounds(0, 0, this.w * TILE, this.h * TILE);
    cam.startFollow(this.player, true, 0.2, 0.2, -8, -4);
    const zoom = () => cam.setZoom(Math.max(2, Math.min(4, Math.round(Math.min(this.scale.width / 480, this.scale.height / 320)))));
    zoom(); this.scale.on('resize', zoom);

    this.input.keyboard!.disableGlobalCapture();
    this.cursors = this.input.keyboard!.createCursorKeys();
    this.keys = this.input.keyboard!.addKeys('W,A,S,D');
    this.time.addEvent({ delay: 1800, loop: true, callback: () => this.wander() });
    this.syncAgents();
    refresh();
  }

  free(x: number, y: number) { return x >= 0 && y >= 0 && x < this.w && y < this.h && !this.blocked[y * this.w + x]; }
  tileAt(m: Mon, x: number, y: number) { return { x: x * TILE + 8, y: y * TILE + 15 }; }
  accOf(runId: number) { return this.metrics.get(runId) ?? 0; }

  /* ---- player ---- */
  update() {
    if (S.battle || this.pMoving || document.activeElement?.tagName === 'TEXTAREA' || document.activeElement?.tagName === 'INPUT') return;
    let dx = 0, dy = 0, face = this.facing;
    if (this.cursors.left.isDown || this.keys.A.isDown) { dx = -1; face = 'left'; }
    else if (this.cursors.right.isDown || this.keys.D.isDown) { dx = 1; face = 'right'; }
    else if (this.cursors.up.isDown || this.keys.W.isDown) { dy = -1; face = 'up'; }
    else if (this.cursors.down.isDown || this.keys.S.isDown) { dy = 1; face = 'down'; }
    if (!dx && !dy) return;
    this.facing = face;
    const m = CHAR_WALK_MAPPING[face];
    const nx = this.px + dx, ny = this.py + dy;
    if (!this.free(nx, ny)) { this.player.setFrame(m.standing); audio.sfx('bump'); return; }
    this.pMoving = true; this.px = nx; this.py = ny;
    this.player.setFrame(this.animFlip ? m.leftFoot : m.rightFoot);
    this.tweens.add({
      targets: this.player, x: nx * TILE, y: ny * TILE - 8, duration: 140, onComplete: () => {
        this.pMoving = false; this.player.setFrame(m.standing); this.player.setDepth(ny * TILE + 100);
        if (this.tall[ny * this.w + nx]) audio.sfx('step_grass');
      },
    });
  }

  /* ---- creatures ---- */
  syncAgents() {
    if (!this.homes.length) return;
    const seen = new Set<string>();
    S.agents.forEach((a: any, i: number) => {
      seen.add(a.id);
      let m: Mon | undefined = S.mons.get(a.id);
      if (!m) { m = this.spawnMon(a, this.homes[i % this.homes.length]); S.mons.set(a.id, m); }
      m.agent = a;
      if (a.status === 'retired' && !m.fainted) this.faint(m, false);
      if (a.version > m.version) { m.version = a.version; this.evolveFx(m); }
      this.relabel(m);
    });
    for (const [id, m] of [...S.mons]) if (!seen.has(id)) { this.destroyMon(m); S.mons.delete(id); }
    // runs: running runs pull their creature onto the training field; finished runs send it home
    const running = new Map<string, any>();
    for (const r of S.runs) if (r.status === 'running' && r.agent_id) running.set(r.agent_id, r);
    for (const [id, m] of S.mons as Map<string, Mon>) {
      const r = running.get(id);
      if (r && m.runId !== r.id && !m.fainted) this.toField(m, r.id);
      if (!r && m.runId !== null) this.toHome(m);
    }
    // a baseline run is the reproduction: pull the player into the battle automatically (once per run)
    for (const r of S.runs) if (r.status === 'running' && r.label === 'baseline' && !S.shown.has(r.id) && !S.battle) this.startBattle(r.id);
    renderHud();
  }

  startBattle(runId: number) {
    if (S.battle) return;
    S.shown.add(runId);
    this.scene.launch('Battle', { runId });
  }

  spawnMon(a: any, home: { x: number; y: number }): Mon {
    const key = creatureKey(a);
    const tex = `mon_${key}_front`;
    const x = home.x * TILE + 8, y = home.y * TILE + 15;
    const img = this.add.image(x, y, tex).setOrigin(0.5, 1).setDisplaySize(26, 26).setDepth(y);
    const label = this.add.text(x, y - 30, '', { fontFamily: 'monospace', fontSize: '5px', color: '#ffffff', stroke: '#14120d', strokeThickness: 1.5 }).setOrigin(0.5, 1).setResolution(6).setDepth(y + 500);
    const hp = this.add.graphics().setDepth(y + 500);
    const m: Mon = { agent: a, key, img, label, hp, star: null, tx: home.x, ty: home.y, home, moving: false, runId: null, slot: null, acc: 0, fainted: false, version: a.version };
    img.setScale(0); this.tweens.add({ targets: img, displayWidth: 26, displayHeight: 26, duration: 360, ease: 'Back.easeOut' });
    if (a.version > 1 || a.origin === 'merged') this.addStar(m);
    this.spark(x, y - 10);
    audio.sfx('encounter');
    return m;
  }
  addStar(m: Mon) { if (!m.star) m.star = this.add.image(m.img.x + 9, m.img.y - 24, 'spark').setTint(0xffc21a).setScale(1.4).setDepth(m.img.depth + 600); }
  destroyMon(m: Mon) { m.img.destroy(); m.label.destroy(); m.hp.destroy(); m.star?.destroy(); }

  relabel(m: Mon) {
    const a = m.agent, lv = (a.n_tasks || 0) + 1;
    const running = m.runId !== null;
    const short = a.id.replace('CodeRunner-', 'CR-').replace('_v', ' v');
    m.label.setText(`${short.length > 15 ? short.slice(0, 14) + '…' : short} Lv${lv}`);
    m.hp.clear();
    const val = running ? m.acc : Math.max(0, Math.min(1, a.avg_reward ?? 0));
    const x = m.img.x - 11, y = m.img.y - 28;
    m.hp.fillStyle(0x14120d, 1).fillRect(x - 1, y - 1, 24, 5);
    m.hp.fillStyle(0xffffff, 1).fillRect(x, y, 22, 3);
    m.hp.fillStyle(running ? 0x8b6cf0 : val < 0.3 ? 0xff5436 : val < 0.6 ? 0xffc21a : 0x169b48, 1).fillRect(x, y, Math.max(1, Math.round(22 * val)), 3);
    m.label.setPosition(m.img.x, m.img.y - 30);
    if (m.star) m.star.setPosition(m.img.x + 10, m.img.y - 22);
    const d = m.img.y; m.img.setDepth(d); m.label.setDepth(d + 500); m.hp.setDepth(d + 500);
  }

  bfsPath(from: { x: number; y: number }, to: { x: number; y: number }) {
    const key = (x: number, y: number) => y * this.w + x, prev = new Map<number, number>([[key(from.x, from.y), -1]]), q = [[from.x, from.y]];
    for (let k = 0; k < q.length; k++) {
      const [x, y] = q[k];
      if (x === to.x && y === to.y) break;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nx = x + dx, ny = y + dy;
        if (!this.free(nx, ny) || prev.has(key(nx, ny))) continue;
        prev.set(key(nx, ny), key(x, y)); q.push([nx, ny]);
      }
    }
    const out: Array<{ x: number; y: number }> = [];
    let cur = key(to.x, to.y);
    if (!prev.has(cur)) return out;
    while (cur !== -1) { out.push({ x: cur % this.w, y: Math.floor(cur / this.w) }); cur = prev.get(cur)!; }
    return out.reverse().slice(1);
  }

  walk(m: Mon, to: { x: number; y: number }, then?: () => void) {
    const path = this.bfsPath({ x: m.tx, y: m.ty }, to);
    if (!path.length) { then?.(); return; }
    m.moving = true;
    this.tweens.killTweensOf(m.img);               // a spawn pop-in must not fight the walk bounce over the same size
    m.img.setDisplaySize(26, 26);
    const step = (i: number) => {
      if (i >= path.length || m.fainted) { m.moving = false; m.img.setDisplaySize(26, 26); then?.(); return; }
      const t = path[i], px = t.x * TILE + 8, py = t.y * TILE + 15;
      this.tweens.add({ targets: m.img, x: px, y: py, duration: 150, onUpdate: () => this.relabel(m), onComplete: () => { m.tx = t.x; m.ty = t.y; step(i + 1); } });
      this.tweens.add({ targets: m.img, displayHeight: { from: 26, to: 22 }, duration: 75, yoyo: true });
    };
    step(0);
  }

  toField(m: Mon, runId: number) {
    const slot = this.slots.find((s) => s.used === null);
    if (!slot) return;
    slot.used = runId; m.runId = runId; m.slot = this.slots.indexOf(slot); m.acc = this.metrics.get(runId) ?? 0;
    this.walk(m, slot, () => { this.relabel(m); });
  }
  toHome(m: Mon) {
    if (m.slot !== null) this.slots[m.slot].used = null;
    m.runId = null; m.slot = null; m.acc = 0;
    this.walk(m, m.home, () => this.relabel(m));
  }
  wander() {
    for (const m of S.mons.values() as Iterable<Mon>) {
      if (m.fainted || m.moving || m.runId !== null || Math.random() < 0.45) continue;
      const dx = Math.floor(Math.random() * 3) - 1, dy = Math.floor(Math.random() * 3) - 1, nx = m.tx + dx, ny = m.ty + dy;
      if ((!dx && !dy) || Math.abs(nx - m.home.x) > 1 || Math.abs(ny - m.home.y) > 1 || !this.free(nx, ny) || this.tall[ny * this.w + nx]) continue;
      this.walk(m, { x: nx, y: ny });
    }
  }

  /* ---- effects ---- */
  spark(x: number, y: number) {
    const em = this.add.particles(x, y, 'spark', { speed: { min: 30, max: 90 }, lifespan: 600, scale: { start: 1.2, end: 0 }, quantity: 14, emitting: false });
    em.setDepth(9999); em.explode(14); this.time.delayedCall(900, () => em.destroy());
  }
  faint(m: Mon, loud = true) {
    m.fainted = true; m.img.setTintFill(0x9aa3a6);
    this.tweens.add({ targets: m.img, angle: 90, alpha: 0.55, duration: 500 });
    if (loud) audio.sfx('faint');
    if (m.slot !== null) { this.slots[m.slot].used = null; m.slot = null; m.runId = null; }
  }
  evolveFx(m: Mon) {
    m.img.setTintFill(0xffffff);
    this.tweens.add({ targets: m.img, displayWidth: 30, displayHeight: 30, duration: 400, yoyo: true, onComplete: () => m.img.clearTint() });
    this.addStar(m); this.spark(m.img.x, m.img.y - 10); audio.sfx('levelup');
  }

  /* ---- live events ---- */
  onMetric(e: any) {
    this.metrics.set(e.run_id, e.test_acc);
    for (const m of S.mons.values() as Iterable<Mon>) if (m.runId === e.run_id) { m.acc = e.test_acc; this.relabel(m); }
    if (!this._hudT) this._hudT = setTimeout(() => { this._hudT = null; renderHud(); }, 400);
  }
  _hudT: any = null;
  onEvent(e: any) {
    switch (e.type) {
      case 'run_done': {
        const m = [...S.mons.values()].find((x: Mon) => x.runId === e.run_id) as Mon | undefined;
        if (m) this.spark(m.img.x, m.img.y - 10);
        audio.sfx(e.status === 'success' ? (e.result?.grok_epoch ? 'levelup' : 'win') : 'lose');
        break;
      }
      case 'agent_retired': { const m = S.mons.get(e.agent); if (m) this.faint(m); break; }
      case 'run_error': audio.sfx('error'); break;
      case 'fix_recalled': case 'memory_hit': audio.sfx('heal'); break;
      case 'approval_requested': audio.sfx('encounter'); break;
      case 'reset': for (const m of S.mons.values()) this.destroyMon(m); S.mons.clear(); this.metrics.clear(); this.slots.forEach((s) => (s.used = null)); break;
    }
  }
}


/* ------------------------------------------------------------------ battle: a run versus the paper's claim */
const GROK_LINE = 0.95;
const MONO = '"JetBrains Mono", ui-monospace, Menlo, monospace';
// Two layouts of the same scene: wide screens use landscape, phones (narrow/portrait) stack the fighters.
const LAND = { BW: 960, BH: 640, bg: [480, 320, 960, 640], foePlat: [700, 330], foeShadow: [700, 318, 170], foe: [700, 306, 230], allyPlat: [250, 505], allyShadow: [250, 486, 220], ally: [250, 490, 270],
  foePlate: [40, 24, 470, 126], foeName: [56, 32, 20], foeBar: [56, 70, 430], foeSub: [56, 98, 15], allyPlate: [520, 340, 390, 130], allyName: [540, 352, 20], allyBar: [536, 388, 350], allyAcc: [540, 418, 16],
  dialog: [24, 512, 912, 112], msg: [48, 536, 860, 22], clouds: [[130, 90, 150], [560, 60, 120], [820, 130, 170]] };
const PORT = { BW: 520, BH: 800, bg: [260, 400, 1200, 800], foePlat: [390, 322], foeShadow: [390, 310, 150], foe: [390, 300, 200], allyPlat: [150, 580], allyShadow: [150, 560, 190], ally: [150, 564, 230],
  foePlate: [14, 14, 492, 112], foeName: [28, 22, 18], foeBar: [28, 60, 464], foeSub: [28, 88, 14], allyPlate: [14, 604, 492, 92], allyName: [28, 610, 18], allyBar: [28, 640, 464], allyAcc: [28, 666, 14],
  dialog: [14, 706, 492, 84], msg: [26, 716, 468, 17], clouds: [[100, 150, 110], [380, 110, 100]] };

class Battle extends Phaser.Scene {
  constructor() { super('Battle'); }
  runId = 0; run: any; agent: any; total = 3000; epoch = 0; testAcc = 0; trainAcc = 0; L: any = LAND;
  foeHp = 1; allyHp = 1; memorised = false; ended = false; lastHitAcc = 0;
  g: Phaser.GameObjects.Graphics; foe: Phaser.GameObjects.Image; ally: Phaser.GameObjects.Image;
  msg: Phaser.GameObjects.Text; foeTxt: Phaser.GameObjects.Text; foeSub: Phaser.GameObjects.Text; allyTxt: Phaser.GameObjects.Text; accTxt: Phaser.GameObjects.Text;
  queue: string[] = []; typing = false;

  init(d: any) {
    this.runId = d.runId;
    this.ended = false; this.memorised = false; this.epoch = 0; this.testAcc = 0; this.trainAcc = 0; this.lastHitAcc = 0; this.queue = []; this.typing = false;
  }

  create() {
    S.battle = this;
    const TOP = 56, cam = this.cameras.main;
    this.L = this.scale.width / (this.scale.height - TOP) < 1.15 ? PORT : LAND;
    const L = this.L;
    cam.setBackgroundColor('#0f1f5c');
    const fit = () => { const w = this.scale.width, h = this.scale.height - TOP; cam.setViewport(0, TOP, w, h); cam.setZoom(Math.min(w / L.BW, h / L.BH)); cam.centerOn(L.BW / 2, L.BH / 2); };
    fit(); this.scale.on('resize', fit);
    this.run = S.runs.find((r: any) => r.id === this.runId) || {};
    this.agent = S.agents.find((a: any) => a.id === this.run.agent_id) || { id: this.run.agent_id || 'Agent', n_tasks: 0, name: 'x', capabilities: ['code_execute'] };
    this.total = (this.run.effective?.epochs || this.run.requested?.epochs || 3000);

    this.add.image(L.bg[0], L.bg[1], 'bg_battle_tech').setDisplaySize(L.bg[2], L.bg[3]).setDepth(0);
    for (const c of L.clouds) {
      const cl = this.add.ellipse(c[0], c[1], c[2], 34, 0xffffff, 0.5).setDepth(1);
      this.tweens.add({ targets: cl, x: c[0] + 60, duration: 9000 + c[0] * 6, yoyo: true, repeat: -1, ease: 'Sine.easeInOut' });
    }
    this.add.image(L.foePlat[0], L.foePlat[1], 'platform_foe_tech').setDepth(2);
    this.add.image(L.allyPlat[0], L.allyPlat[1], 'platform_ally_tech').setDepth(2);
    this.add.ellipse(L.foeShadow[0], L.foeShadow[1], L.foeShadow[2], 26, 0x000000, 0.22).setDepth(3);
    this.add.ellipse(L.allyShadow[0], L.allyShadow[1], L.allyShadow[2], 30, 0x000000, 0.22).setDepth(3);
    this.foe = this.add.image(L.foe[0], L.foe[1], 'mon_reader_front').setDisplaySize(L.foe[2], L.foe[2]).setDepth(10).setOrigin(0.5, 1);
    this.ally = this.add.image(L.ally[0], L.ally[1], `mon_${creatureKey(this.agent)}_back`).setDisplaySize(L.ally[2], L.ally[2]).setDepth(10).setOrigin(0.5, 1);
    this.tweens.add({ targets: this.foe, y: L.foe[1] - 6, duration: 1100, yoyo: true, repeat: -1, ease: 'Sine.easeInOut' });
    this.tweens.add({ targets: this.ally, y: L.ally[1] + 6, duration: 1300, yoyo: true, repeat: -1, ease: 'Sine.easeInOut' });

    this.g = this.add.graphics().setDepth(20);
    const t = (x: number, y: number, size: number, color = '#14120d', bold = true) =>
      this.add.text(x, y, '', { fontFamily: MONO, fontSize: size + 'px', color, fontStyle: bold ? 'bold' : 'normal' }).setDepth(30).setResolution(2);
    this.foeTxt = t(L.foeName[0], L.foeName[1], L.foeName[2]);
    this.foeSub = this.add.text(L.foeSub[0], L.foeSub[1], '', { fontFamily: MONO, fontSize: L.foeSub[2] + 'px', color: '#14120d', lineSpacing: 4 }).setDepth(30).setResolution(2);
    this.allyTxt = t(L.allyName[0], L.allyName[1], L.allyName[2]);
    this.accTxt = t(L.allyAcc[0], L.allyAcc[1], L.allyAcc[2], '#14120d', false);
    this.msg = this.add.text(L.msg[0], L.msg[1], '', { fontFamily: MONO, fontSize: L.msg[3] + 'px', color: '#14120d', fontStyle: 'bold', wordWrap: { width: L.msg[2] }, lineSpacing: 6 }).setDepth(40).setResolution(2);

    const claim = S.claim;
    this.say(`A wild paper appeared! ${claim ? 'It claims grokking at ~' + claim.grok_epoch + (claim.verified ? '' : ' (unverified)') : ''}`);
    this.say(`Go, ${this.agent.id}! Run #${this.runId} ${this.run.label || ''}.`);
    audio.bgm('battle'); audio.sfx('encounter');
    ($('#backBtn') as HTMLElement).hidden = false; document.body.classList.add('inbattle');
    this.events.once(Phaser.Scenes.Events.SHUTDOWN, () => { S.battle = null; ($('#backBtn') as HTMLElement).hidden = true; document.body.classList.remove('inbattle'); this.scale.off('resize', fit); audio.bgm('overworld'); });
    this.draw();

    // catch up if we joined mid-run or the run already finished
    api(`/api/runs/${this.runId}/metrics`).then((ms: any[]) => {
      if (!ms.length) return;
      const last = ms[ms.length - 1];
      this.epoch = last.epoch; this.testAcc = last.test_acc; this.trainAcc = last.train_acc;
      this.memorised = ms.some((m) => m.train_acc >= 0.99);
      this.lastHitAcc = this.testAcc;
      if (this.ended) { this.foeHp = this.testAcc >= GROK_LINE ? 0 : Math.max(0, 1 - this.testAcc / GROK_LINE); this.allyHp = Math.max(0, 1 - this.epoch / this.total); this.draw(); }
      else this.apply(false);
    }).catch(() => {});
    if (this.run.status && this.run.status !== 'running' && this.run.status !== 'waiting') this.finish(this.run.status, this.run.result?.grok_epoch);
  }

  say(text: string) { this.queue.push(text); if (!this.typing) this.next(); }
  next() {
    const t = this.queue.shift();
    if (!t) { this.typing = false; return; }
    this.typing = true; let i = 0;
    const ev = this.time.addEvent({ delay: 18, loop: true, callback: () => {
      this.msg.setText(t.slice(0, ++i));
      if (i >= t.length) { ev.remove(false); this.time.delayedCall(Math.max(900, 25 * t.length), () => this.next()); }
    } });
  }

  onMetric(e: any) {
    if (e.run_id !== this.runId || this.ended) return;
    this.epoch = e.epoch; this.testAcc = e.test_acc; this.trainAcc = e.train_acc;
    if (!this.memorised && e.train_acc >= 0.99) {
      this.memorised = true;
      this.say(`${this.agent.id} memorised the training set! Test accuracy is only ${Math.round(e.test_acc * 100)}%.`);
    }
    this.apply(true);
    if (e.test_acc >= GROK_LINE && !this.ended) this.finish('success', e.epoch);
  }
  onRunEnd(e: any) {
    if (e.run_id !== this.runId || this.ended) return;
    this.finish(e.type === 'run_killed' ? 'killed' : e.status || 'failed', e.result?.grok_epoch, e);
  }

  apply(fx: boolean) {
    this.foeHp = Math.max(0, 1 - this.testAcc / GROK_LINE);
    this.allyHp = Math.max(0, 1 - this.epoch / this.total);
    if (fx && this.testAcc - this.lastHitAcc >= 0.04) {
      const big = this.testAcc - this.lastHitAcc >= 0.12;
      this.lastHitAcc = this.testAcc;
      audio.sfx(big ? 'crit' : 'hit');
      this.tweens.add({ targets: this.foe, x: this.L.foe[0] + 12, duration: 50, yoyo: true, repeat: 3 });
      this.foe.setTintFill(0xffffff); this.time.delayedCall(90, () => !this.ended && this.foe.clearTint());
      if (big) this.say(`A critical jump! Test accuracy is ${Math.round(this.testAcc * 100)}%.`);
    }
    this.draw();
  }

  finish(status: string, grokEpoch?: number, e?: any) {
    if (this.ended) return; this.ended = true;
    this.queue = [];
    const L = this.L;
    if (status === 'success' && (grokEpoch || this.testAcc >= GROK_LINE)) {
      this.foeHp = 0; this.draw();
      this.say(`Grokked at epoch ${grokEpoch || this.epoch}!${S.claim ? ` The paper claims ~${S.claim.grok_epoch}${S.claim.verified ? '' : ' (unverified)'}.` : ''}`);
      this.tweens.add({ targets: this.foe, y: L.foe[1] + 90, alpha: 0, duration: 900, delay: 400 });
      audio.bgm('victory'); audio.sfx('win');
      for (let i = 0; i < 4; i++) this.time.delayedCall(i * 250, () => this.sparkAt(L.foe[0] - 90 + Math.random() * 180, L.foe[1] - 160 + Math.random() * 120));
    } else {
      const why = status === 'killed' ? 'The run was stopped.' : status === 'failed' ? `The run failed${e?.error ? ': ' + e.error : ''}.` : `Out of epochs: no grok within ${this.total}.`;
      this.say(`${this.agent.id} fainted. ${why}`);
      this.tweens.add({ targets: this.ally, y: L.BH + 40, alpha: 0, angle: 12, duration: 900, delay: 300 });
      audio.bgm('defeat'); audio.sfx('faint');
    }
    this.time.delayedCall(9000, () => this.leave());
  }
  sparkAt(x: number, y: number) {
    const em = this.add.particles(x, y, 'spark', { speed: { min: 60, max: 180 }, lifespan: 800, scale: { start: 3, end: 0 }, quantity: 16, emitting: false });
    em.setDepth(60); em.explode(16); this.time.delayedCall(1000, () => em.destroy());
  }
  leave() { if (this.scene.isActive('Battle')) this.scene.stop('Battle'); }

  draw() {
    const L = this.L, g = this.g; g.clear();
    const plate = (r: number[]) => { g.fillStyle(0x14120d, 1).fillRect(r[0] - 6, r[1] - 6, r[2] + 12, r[3] + 12); g.fillStyle(0xfff8e6, 1).fillRect(r[0], r[1], r[2], r[3]); };
    const bar = (b: number[], pct: number, color: number) => {
      g.fillStyle(0x14120d, 1).fillRect(b[0] - 3, b[1] - 3, b[2] + 6, 22);
      g.fillStyle(0xffffff, 1).fillRect(b[0], b[1], b[2], 16);
      g.fillStyle(color, 1).fillRect(b[0], b[1], Math.round(b[2] * Math.max(0, Math.min(1, pct))), 16);
    };
    const hue = (v: number) => (v > 0.5 ? 0x169b48 : v > 0.2 ? 0xffc21a : 0xff5436);
    plate(L.foePlate); plate(L.allyPlate);
    this.foeTxt.setText('THE PAPER');
    bar(L.foeBar, this.foeHp, hue(this.foeHp));
    const claim = S.claim ? `claim ~${S.claim.grok_epoch}${S.claim.verified ? '' : ' [unverified]'}` : 'claim';
    this.foeSub.setText(`test acc ${Math.round(this.testAcc * 100)}% → goal 95%\n${claim}`);
    this.allyTxt.setText(`${this.agent.id.length > 20 ? this.agent.id.slice(0, 19) + '…' : this.agent.id}  Lv${(this.agent.n_tasks || 0) + 1}`);
    bar(L.allyBar, this.allyHp, hue(this.allyHp));
    this.accTxt.setText(`epoch ${this.epoch}/${this.total} · train ${Math.round(this.trainAcc * 100)}%`);
    const d = L.dialog;
    g.fillStyle(0x14120d, 1).fillRect(d[0], d[1], d[2], d[3]);
    g.fillStyle(0xfff8e6, 1).fillRect(d[0] + 8, d[1] + 8, d[2] - 16, d[3] - 16);
  }
}

/* ------------------------------------------------------------------ boot */
bindHud();
feed('Welcome to the LabDex campus. Ask the lab a question to send your creatures to work.', 'sys');
new Phaser.Game({
  type: Phaser.AUTO, parent: 'game', backgroundColor: '#0f1f5c', pixelArt: true, roundPixels: true,
  scale: { mode: Phaser.Scale.RESIZE, width: window.innerWidth, height: window.innerHeight },
  scene: [Boot, World, Battle],
});
connect();
setInterval(refresh, 3000);
