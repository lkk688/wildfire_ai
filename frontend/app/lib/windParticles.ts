// Map layers drawn from the hourly HRRR tiles (lib/weatherStore.ts) at the hour picked on the time slider:
//   WindParticles: particles drift with the 10 m wind and leave short trails, coloured by speed (like weather apps)
//   ScalarField:   one variable (temperature, humidity, gusts, rain, cloud, fire weather) as a smooth colour field
import L from 'leaflet';
import { WIND_CLASSES } from './legend';
import { WeatherStore } from './weatherStore';
import { WxVar, colorFor } from './weatherVars';

// A particle keeps its last TRAIL positions in Leaflet layer pixels (fixed to the ground while panning; reset on zoom).
type P = { xs: number[]; ys: number[]; age: number; life: number };
const TRAIL = 18;
const K = 0.2; // screen pixels per frame per mph: the same visual speed at every zoom
const ALPHAS = [0.12, 0.3, 0.55, 0.9]; // older -> newer part of each trail

function makeCanvas(map: L.Map, pane: string, z: string) {
  const p = map.getPane(pane) || map.createPane(pane);
  p.style.zIndex = z;
  p.style.pointerEvents = 'none';
  const c = L.DomUtil.create('canvas', '', p);
  Object.assign(c.style, { pointerEvents: 'none', maxWidth: 'none', maxHeight: 'none', position: 'absolute' });
  return c;
}

function sizeCanvas(map: L.Map, c: HTMLCanvasElement, ctx: CanvasRenderingContext2D) {
  const s = map.getSize(), dpr = window.devicePixelRatio || 1;
  c.width = s.x * dpr;
  c.height = s.y * dpr;
  c.style.width = `${s.x}px`;
  c.style.height = `${s.y}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  L.DomUtil.setPosition(c, map.containerPointToLayerPoint([0, 0]));
  return s;
}

export class WindParticles extends L.Layer {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private parts: P[] = [];
  private raf = 0;
  private w = 0;
  private h = 0;
  private hidden = false;

  constructor(private store: WeatherStore, private hour: number) {
    super();
  }

  setHour(h: number) {
    this.hour = h;
  }

  onAdd(map: L.Map) {
    this._map = map;
    this.canvas = makeCanvas(map, 'windPane', '450'); // above rasters and polygons, below markers and tooltips
    this.ctx = this.canvas.getContext('2d');
    map.on('move', this.onMove, this);
    map.on('resize', this.onResize, this);
    map.on('zoomstart', this.onZoomStart, this);
    map.on('zoomend', this.onZoomEnd, this);
    this.onResize();
    this.loop();
    return this;
  }

  onRemove(map: L.Map) {
    cancelAnimationFrame(this.raf);
    map.off('move', this.onMove, this);
    map.off('resize', this.onResize, this);
    map.off('zoomstart', this.onZoomStart, this);
    map.off('zoomend', this.onZoomEnd, this);
    this.canvas?.remove();
    this.canvas = this.ctx = null;
    return this;
  }

  private onResize() {
    if (!this._map || !this.canvas || !this.ctx) return;
    const s = sizeCanvas(this._map, this.canvas, this.ctx);
    this.w = s.x;
    this.h = s.y;
    this.parts = [];
  }

  private onMove() {
    if (this._map && this.canvas) L.DomUtil.setPosition(this.canvas, this._map.containerPointToLayerPoint([0, 0]));
  }

  private onZoomStart() {
    this.hidden = true;
    if (this.canvas) this.canvas.style.visibility = 'hidden';
  }

  private onZoomEnd() {
    this.hidden = false;
    this.parts = [];
    this.onMove();
    this.ctx?.clearRect(0, 0, this.w, this.h);
    if (this.canvas) this.canvas.style.visibility = 'visible';
  }

  private spawn(p?: P): P {
    const map = this._map as L.Map;
    const q = p || ({ xs: [], ys: [], age: 0, life: 0 } as P);
    let lp = map.containerPointToLayerPoint([0, 0]);
    for (let t = 0; t < 8; t++) {
      lp = map.containerPointToLayerPoint([Math.random() * this.w, Math.random() * this.h]);
      const ll = map.layerPointToLatLng(lp);
      if (this.store.sampleUV(ll.lat, ll.lng, this.hour)) break;
    }
    q.xs = [lp.x];
    q.ys = [lp.y];
    q.age = 0;
    q.life = 50 + Math.floor(Math.random() * 70);
    return q;
  }

  private loop = () => {
    this.raf = requestAnimationFrame(this.loop);
    const map = this._map, ctx = this.ctx;
    if (!map || !ctx || this.hidden || !this.w) return;
    const want = Math.min(2200, Math.round((this.w * this.h) / 600));
    while (this.parts.length < want) {
      const p = this.spawn();
      p.age = Math.floor(Math.random() * p.life); // stagger so they don't all respawn together
      this.parts.push(p);
    }
    const o = map.containerPointToLayerPoint([0, 0]); // canvas top-left in layer pixels
    const paths = WIND_CLASSES.map(() => ALPHAS.map(() => new Path2D()));
    for (const p of this.parts) {
      const n = p.xs.length, x = p.xs[n - 1], y = p.ys[n - 1];
      const ll = map.layerPointToLatLng([x, y]);
      const s = this.store.sampleUV(ll.lat, ll.lng, this.hour);
      const off = x < o.x - 40 || y < o.y - 40 || x > o.x + this.w + 40 || y > o.y + this.h + 40;
      if (!s || off || ++p.age > p.life) {
        this.spawn(p);
        continue;
      }
      p.xs.push(x + s.u * K);
      p.ys.push(y - s.v * K);
      if (p.xs.length > TRAIL) {
        p.xs.shift();
        p.ys.shift();
      }
      const mph = Math.hypot(s.u, s.v);
      const c = WIND_CLASSES.findIndex((k) => mph < k.max);
      const m = p.xs.length;
      const life = Math.min(1, p.age / 10, (p.life - p.age) / 10); // fade in at birth, out at the end
      for (let k = 1; k < m; k++) {
        const b = Math.min(ALPHAS.length - 1, Math.floor((k / m) * ALPHAS.length * life));
        const path = paths[c][b];
        path.moveTo(p.xs[k - 1] - o.x, p.ys[k - 1] - o.y);
        path.lineTo(p.xs[k] - o.x, p.ys[k] - o.y);
      }
    }
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.lineWidth = 1.4;
    ctx.lineCap = 'round';
    paths.forEach((byAlpha, i) => {
      ctx.strokeStyle = WIND_CLASSES[i].color;
      byAlpha.forEach((path, b) => {
        ctx.globalAlpha = ALPHAS[b];
        ctx.stroke(path);
      });
    });
    ctx.globalAlpha = 1;
  };
}

const CELL = 6; // screen pixels per sample; the low-res image is smoothed when scaled up

export class ScalarField extends L.Layer {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private offscreen = document.createElement('canvas');
  private raf = 0;
  private unsub: (() => void) | null = null;

  constructor(private store: WeatherStore, private v: WxVar, private hour: number) {
    super();
  }

  set(v: WxVar, hour: number) {
    if (v === this.v && hour === this.hour) return;
    this.v = v;
    this.hour = hour;
    this.schedule();
  }

  onAdd(map: L.Map) {
    this._map = map;
    this.canvas = makeCanvas(map, 'wxFieldPane', '350'); // above the basemap, below vector overlays
    this.canvas.style.opacity = '0.7';
    this.ctx = this.canvas.getContext('2d');
    map.on('moveend resize', this.schedule, this);
    map.on('zoomstart', this.hide, this);
    this.unsub = this.store.subscribe(this.schedule.bind(this));
    this.schedule();
    return this;
  }

  onRemove(map: L.Map) {
    cancelAnimationFrame(this.raf);
    map.off('moveend resize', this.schedule, this);
    map.off('zoomstart', this.hide, this);
    this.unsub?.();
    this.canvas?.remove();
    this.canvas = this.ctx = null;
    return this;
  }

  private hide() {
    if (this.canvas) this.canvas.style.visibility = 'hidden';
  }

  private schedule() {
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(() => this.draw());
  }

  private draw() {
    const map = this._map, c = this.canvas, ctx = this.ctx;
    if (!map || !c || !ctx) return;
    const s = sizeCanvas(map, c, ctx);
    const cw = Math.ceil(s.x / CELL) + 1, ch = Math.ceil(s.y / CELL) + 1;
    this.offscreen.width = cw;
    this.offscreen.height = ch;
    const octx = this.offscreen.getContext('2d');
    if (!octx) return;
    const img = octx.createImageData(cw, ch);
    // Web Mercator: longitude depends only on x, latitude only on y
    const lngs = Array.from({ length: cw }, (_, x) => map.containerPointToLatLng([x * CELL, 0]).lng);
    const lats = Array.from({ length: ch }, (_, y) => map.containerPointToLatLng([0, y * CELL]).lat);
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const smp = this.store.sample(lats[y], lngs[x], this.hour);
        if (!smp) continue;
        const [r, g, b, a] = colorFor(this.v, smp[this.v]);
        const o = (y * cw + x) * 4;
        img.data[o] = r;
        img.data[o + 1] = g;
        img.data[o + 2] = b;
        img.data[o + 3] = a;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.clearRect(0, 0, s.x, s.y);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(this.offscreen, 0, 0, cw * CELL, ch * CELL);
    c.style.visibility = 'visible';
  }
}
