// Weather on fixed map tiles (/api/weather/grid-tile), shared by the colour field, the wind animation, the time slider
// and the hover readout: NOAA HRRR hourly for the past 24 h and next 48 h, then the National Blend of Models every
// 3-6 h to about 10 days. Tiles go through the browser cache (lib/cache.ts), so a tile seen once paints instantly when
// the map comes back to it.
import { swrJSON } from './cache';
import { WxSample, fosberg } from './weatherVars';

export const TILE = 6;
export const STEPS = [0.03, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0];
const TTL = 20 * 60 * 1000; // new model runs arrive about hourly
const MAX_TILES = 30;
const CONCURRENT = 6; // tiles in flight at once (served from the grid on our own server)
const MAX_GAP = 6; // hours between model steps that may be interpolated

type Field = 'temp_f10' | 'rh' | 'u_mph10' | 'v_mph10' | 'gust_mph10' | 'precip_in1000' | 'cloud';
export type Tile = {
  step: number; ti: number; tj: number; lat0: number; lon0: number; n: number;
  times: number[]; src: string; now_hour: number;
  vars: Record<Field, (number | null)[]>; scale: Record<Field, number>;
  runs_utc: Partial<Record<'hrrr' | 'nbm', string>>; outside_points: number; source: string;
};

export type WxStatus = {
  tiles: number; loading: number; failed: number; error: string | null; step: number; fromCache: number;
  runs: Partial<Record<'hrrr' | 'nbm', string>>;
};

export type WxRange = { first: number; last: number; now: number; hrrrLast: number | null };

const key = (step: number, ti: number, tj: number) => `${step}|${ti}|${tj}`;
const RAD = Math.PI / 180;

export class WeatherStore {
  private tiles = new Map<string, Tile>();
  private loading = new Set<string>();
  private failed = new Map<string, string>();
  private cachedKeys = new Set<string>();
  private loadedAt = new Map<string, number>();
  private queue: Array<[number, number, number]> = []; // waiting tiles, nearest the screen centre first
  private active = 0;
  private lastView: [number, number, number, number] | null = null;
  private retryTimer: ReturnType<typeof setTimeout> | undefined;
  private subs = new Set<() => void>();
  private hoverSubs = new Set<() => void>();
  step = 0.05;
  version = 0;
  hover: { lat: number; lng: number } | null = null;
  hoverVersion = 0;

  subscribe = (fn: () => void) => {
    this.subs.add(fn);
    return () => {
      this.subs.delete(fn);
    };
  };
  subscribeHover = (fn: () => void) => {
    this.hoverSubs.add(fn);
    return () => {
      this.hoverSubs.delete(fn);
    };
  };
  getVersion = () => this.version;
  getHoverVersion = () => this.hoverVersion;

  private emit() {
    this.version++;
    this.subs.forEach((f) => f());
  }

  setHover(p: { lat: number; lng: number } | null) {
    this.hover = p;
    this.hoverVersion++;
    this.hoverSubs.forEach((f) => f());
  }

  /** Load the tiles that cover this box (min_lon, min_lat, max_lon, max_lat) at a spacing that suits its size. */
  setView(w: number, s: number, e: number, n: number) {
    this.lastView = [w, s, e, n];
    w = Math.max(w, -180);
    e = Math.min(e, 180);
    s = Math.max(s, -85);
    n = Math.min(n, 85);
    const span = Math.max(e - w, (n - s) * 1.3);
    let step = STEPS.find((st) => span / st <= 24) ?? STEPS[STEPS.length - 1];
    const count = (st: number) => {
      const sz = TILE * st;
      return (Math.floor(n / sz) - Math.floor(s / sz) + 1) * (Math.floor(e / sz) - Math.floor(w / sz) + 1);
    };
    while (count(step) > MAX_TILES && step !== STEPS[STEPS.length - 1]) step = STEPS[STEPS.indexOf(step) + 1];
    const changed = step !== this.step;
    this.step = step;
    const sz = TILE * step;
    // centre tiles first so the middle of the screen paints first
    const want: Array<[number, number, number]> = [];
    const cl = (s + n) / 2, cg = (w + e) / 2;
    for (let ti = Math.floor(s / sz); ti <= Math.floor(n / sz); ti++)
      for (let tj = Math.floor(w / sz); tj <= Math.floor(e / sz); tj++)
        want.push([ti, tj, Math.hypot((ti + 0.5) * sz - cl, (tj + 0.5) * sz - cg)]);
    want.sort((a, b) => a[2] - b[2]);
    // replace the waiting list with what this view needs (tiles already in flight carry on)
    this.queue = want.map(([ti, tj]) => [step, ti, tj] as [number, number, number])
      .filter(([st, ti, tj]) => { const k = key(st, ti, tj); return !this.loading.has(k) && Date.now() - (this.loadedAt.get(k) ?? 0) >= TTL; });
    this.pump();
    if (changed) this.emit();
  }

  private pump() {
    while (this.active < CONCURRENT && this.queue.length) {
      const [st, ti, tj] = this.queue.shift() as [number, number, number];
      this.load(st, ti, tj);
    }
  }

  private load(step: number, ti: number, tj: number) {
    const k = key(step, ti, tj);
    if (this.loading.has(k) || Date.now() - (this.loadedAt.get(k) ?? 0) < TTL) return;
    this.loading.add(k);
    this.active++;
    this.emit();
    swrJSON<{ tile: Tile }>(
      `/api/weather/grid-tile?step=${step}&ti=${ti}&tj=${tj}`,
      TTL,
      (d, meta) => {
        this.tiles.set(k, d.tile);
        this.loadedAt.set(k, meta.fetchedAt);
        if (!meta.cached) this.failed.delete(k);
        if (meta.cached) this.cachedKeys.add(k);
        else this.cachedKeys.delete(k);
        this.emit();
      },
      (err, hadCopy) => {
        this.failed.set(k, `${hadCopy ? 'showing cached copy; ' : ''}${String(err.message || err)}`);
        this.loadedAt.set(k, Date.now() - TTL + 60 * 1000); // eligible again in about a minute
        clearTimeout(this.retryTimer);
        this.retryTimer = setTimeout(() => this.lastView && this.setView(...this.lastView), 62 * 1000);
      },
      () => {
        this.loading.delete(k);
        this.active--;
        this.emit();
        this.pump();
      },
    );
  }

  status(): WxStatus {
    const cur = [...this.tiles.values()].filter((t) => t.step === this.step);
    const runs: WxStatus['runs'] = {};
    for (const t of cur) for (const [k, v] of Object.entries(t.runs_utc || {})) {
      const kk = k as 'hrrr' | 'nbm';
      if (!runs[kk] || (v as string) > (runs[kk] as string)) runs[kk] = v as string;
    }
    return {
      tiles: cur.length, loading: this.loading.size, failed: this.failed.size, step: this.step, runs,
      error: [...this.failed.values()][0] ?? null,
      fromCache: [...this.cachedKeys].filter((k) => k.startsWith(`${this.step}|`)).length,
    };
  }

  /** Hours (epoch hours, UTC) with data in the loaded tiles, the current hour and the last HRRR hour. */
  range(): WxRange | null {
    let first = Infinity, last = -Infinity, now = -Infinity, hrrrLast = -Infinity;
    for (const t of this.tiles.values()) {
      if (!t.times?.length) continue;
      first = Math.min(first, t.times[0]);
      last = Math.max(last, t.times[t.times.length - 1]);
      now = Math.max(now, t.now_hour);
      const h = t.src.lastIndexOf('h');
      if (h >= 0) hrrrLast = Math.max(hrrrLast, t.times[h]);
    }
    return Number.isFinite(first) && Number.isFinite(last)
      ? { first, last, now, hrrrLast: Number.isFinite(hrrrLast) ? hrrrLast : null } : null;
  }

  /** Which model an hour comes from, and whether it falls between two of the model's time steps. */
  sourceAt(hour: number): { model: 'hrrr' | 'nbm'; between: boolean } | null {
    for (const t of this.tiles.values()) {
      const k = this.timeIndex(t, hour);
      if (k) return { model: t.src[k.k0] === 'n' || t.src[k.k1] === 'n' ? 'nbm' : 'hrrr', between: k.w > 0 && k.w < 1 };
    }
    return null;
  }

  // Bracketing time steps of a tile for an hour (k0, k1 and weight w of k1), memoised per tile.
  private tIdx = new WeakMap<Tile, { hour: number; r: { k0: number; k1: number; w: number } | null }>();
  private timeIndex(t: Tile, hour: number) {
    const m = this.tIdx.get(t);
    if (m && m.hour === hour) return m.r;
    const T = t.times;
    let r: { k0: number; k1: number; w: number } | null = null;
    if (T.length && hour >= T[0] && hour <= T[T.length - 1]) {
      let lo = 0, hi = T.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (T[mid] <= hour) lo = mid;
        else hi = mid;
      }
      if (T[lo] === hour) r = { k0: lo, k1: lo, w: 0 };
      else if (T[hi] === hour) r = { k0: hi, k1: hi, w: 0 };
      else if (T[hi] - T[lo] <= MAX_GAP) r = { k0: lo, k1: hi, w: (hour - T[lo]) / (T[hi] - T[lo]) };
    }
    this.tIdx.set(t, { hour, r });
    return r;
  }

  // Tile, bilinear corners and time weights for a point, trying the current spacing first, then others loaded.
  private locate(lat: number, lng: number, hour: number) {
    const order = [this.step, ...STEPS.filter((s) => s !== this.step).sort((a, b) => Math.abs(Math.log(a / this.step)) - Math.abs(Math.log(b / this.step)))];
    for (const step of order) {
      const sz = TILE * step;
      const t = this.tiles.get(key(step, Math.floor(lat / sz), Math.floor(lng / sz)));
      if (!t) continue;
      const k = this.timeIndex(t, hour);
      if (!k) continue;
      const a = (lat - t.lat0) / step, b = (lng - t.lon0) / step;
      const i = Math.min(TILE - 1, Math.max(0, Math.floor(a))), j = Math.min(TILE - 1, Math.max(0, Math.floor(b)));
      const nn = t.n * t.n, c = i * t.n + j;
      return { t, c: [c, c + 1, c + t.n, c + t.n + 1], b0: k.k0 * nn, b1: k.k1 * nn, w: k.w, fy: a - i, fx: b - j };
    }
    return null;
  }

  private val(L: NonNullable<ReturnType<WeatherStore['locate']>>, f: Field): number | null {
    const arr = L.t.vars[f], { c, fx, fy } = L;
    const bw = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
    let out = 0;
    for (const [base, tw] of L.w > 0 ? [[L.b0, 1 - L.w], [L.b1, L.w]] : [[L.b0, 1]]) {
      for (let q = 0; q < 4; q++) {
        const x = arr[base + c[q]];
        if (x == null) return null;
        out += tw * bw[q] * x;
      }
    }
    return out / (L.t.scale?.[f] ?? 1);
  }

  /** Wind (mph) towards east / north at a point and hour, or null. Fast path for the particle animation. */
  sampleUV(lat: number, lng: number, hour: number): { u: number; v: number } | null {
    const L = this.locate(lat, lng, hour);
    if (!L) return null;
    const u = this.val(L, 'u_mph10'), v = this.val(L, 'v_mph10');
    return u == null || v == null ? null : { u, v };
  }

  /** Every variable at a point and hour (display units), or null where there is no data. */
  sample(lat: number, lng: number, hour: number): WxSample | null {
    const L = this.locate(lat, lng, hour);
    if (!L) return null;
    const temp = this.val(L, 'temp_f10'), rh = this.val(L, 'rh'), u = this.val(L, 'u_mph10'), v = this.val(L, 'v_mph10');
    if (temp == null || rh == null || u == null || v == null) return null;
    const gust = this.val(L, 'gust_mph10'), precip = this.val(L, 'precip_in1000'), cloud = this.val(L, 'cloud');
    const wspd = Math.hypot(u, v);
    const wdir = ((Math.atan2(-u, -v) / RAD) + 360) % 360;
    return { temp, rh, wspd, wdir, gust: gust ?? 0, precip: precip ?? 0, cloud: cloud ?? 0, ffwi: fosberg(temp, rh, wspd) };
  }
}
