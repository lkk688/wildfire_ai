// Hourly HRRR weather on fixed map tiles (/api/weather/forecast-tile), shared by the colour field, the wind animation,
// the time slider and the hover readout. Tiles go through the browser cache (lib/cache.ts), so a tile seen once
// paints instantly when the map comes back to it.
import { swrJSON } from './cache';
import { WxSample, fosberg } from './weatherVars';

export const TILE = 6;
export const STEPS = [0.03, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0];
const TTL = 40 * 60 * 1000;
const MAX_TILES = 20;
const CONCURRENT = 3; // tiles in flight at once (Open-Meteo's free tier is rate-limited)

export type Tile = {
  step: number; ti: number; tj: number; lat0: number; lon0: number; n: number; t0: number; hours: number; now_hour: number;
  first_hour: number | null; last_hour: number | null;
  vars: Record<'temp_f' | 'rh' | 'wspd_mph10' | 'wdir' | 'gust_mph10' | 'precip_in1000' | 'cloud', (number | null)[]>;
  fetched_utc: string | null; stale_points: number; missing_points: number; upstream_error: string | null; source: string;
};

export type WxStatus = {
  tiles: number; loading: number; failed: number; stale: boolean; fetchedAt: string | null; error: string | null;
  step: number; fromCache: number;
};

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
    let step = STEPS.find((st) => span / st <= 12) ?? STEPS[STEPS.length - 1];
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
      `/api/weather/forecast-tile?step=${step}&ti=${ti}&tj=${tj}`,
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
    const fetched = cur.map((t) => t.fetched_utc).filter(Boolean).sort() as string[];
    return {
      tiles: cur.length, loading: this.loading.size, failed: this.failed.size, step: this.step,
      stale: cur.some((t) => t.stale_points > 0), fetchedAt: fetched[0] ?? null,
      error: [...this.failed.values()][0] ?? cur.find((t) => t.upstream_error)?.upstream_error ?? null,
      fromCache: [...this.cachedKeys].filter((k) => k.startsWith(`${this.step}|`)).length,
    };
  }

  /** Hours (epoch hours, UTC) with data in the loaded tiles, and the current hour. */
  range(): { first: number; last: number; now: number } | null {
    let first = Infinity, last = -Infinity, now = -Infinity;
    for (const t of this.tiles.values()) {
      if (t.first_hour != null) first = Math.min(first, t.first_hour);
      if (t.last_hour != null) last = Math.max(last, t.last_hour);
      now = Math.max(now, t.now_hour);
    }
    return Number.isFinite(first) && Number.isFinite(last) ? { first, last, now } : null;
  }

  // Tile + bilinear weights for a point, trying the current spacing first, then others already loaded.
  private locate(lat: number, lng: number, hour: number) {
    const order = [this.step, ...STEPS.filter((s) => s !== this.step).sort((a, b) => Math.abs(Math.log(a / this.step)) - Math.abs(Math.log(b / this.step)))];
    for (const step of order) {
      const sz = TILE * step;
      const t = this.tiles.get(key(step, Math.floor(lat / sz), Math.floor(lng / sz)));
      if (!t) continue;
      const h = hour - t.t0;
      if (h < 0 || h >= t.hours) continue;
      const a = (lat - t.lat0) / step, b = (lng - t.lon0) / step;
      const i = Math.min(TILE - 1, Math.max(0, Math.floor(a))), j = Math.min(TILE - 1, Math.max(0, Math.floor(b)));
      const nn = t.n * t.n, base = h * nn + i * t.n + j;
      return { t, idx: [base, base + 1, base + t.n, base + t.n + 1], fy: a - i, fx: b - j };
    }
    return null;
  }

  private lerp(arr: (number | null)[], idx: number[], fx: number, fy: number): number | null {
    const a = arr[idx[0]], b = arr[idx[1]], c = arr[idx[2]], d = arr[idx[3]];
    if (a == null || b == null || c == null || d == null) return null;
    return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy;
  }

  /** Wind (mph) towards east / north at a point and hour, or null. Fast path for the particle animation. */
  sampleUV(lat: number, lng: number, hour: number): { u: number; v: number } | null {
    const L = this.locate(lat, lng, hour);
    if (!L) return null;
    const sp = L.t.vars.wspd_mph10, dr = L.t.vars.wdir;
    let u = 0, v = 0;
    const w = [(1 - L.fx) * (1 - L.fy), L.fx * (1 - L.fy), (1 - L.fx) * L.fy, L.fx * L.fy];
    for (let q = 0; q < 4; q++) {
      const s = sp[L.idx[q]], d = dr[L.idx[q]];
      if (s == null || d == null) return null;
      const to = (d + 180) * RAD;
      u += w[q] * (s / 10) * Math.sin(to);
      v += w[q] * (s / 10) * Math.cos(to);
    }
    return { u, v };
  }

  /** Every variable at a point and hour (display units), or null where there is no data. */
  sample(lat: number, lng: number, hour: number): WxSample | null {
    const L = this.locate(lat, lng, hour);
    if (!L) return null;
    const V = L.t.vars;
    const g = (a: (number | null)[]) => this.lerp(a, L.idx, L.fx, L.fy);
    const temp = g(V.temp_f), rh = g(V.rh), gust = g(V.gust_mph10), precip = g(V.precip_in1000), cloud = g(V.cloud);
    const uv = this.sampleUV(lat, lng, hour);
    if (temp == null || rh == null || uv == null) return null;
    const wspd = Math.hypot(uv.u, uv.v);
    const wdir = ((Math.atan2(-uv.u, -uv.v) / RAD) + 360) % 360;
    return {
      temp, rh, wspd, wdir, gust: (gust ?? 0) / 10, precip: (precip ?? 0) / 1000, cloud: cloud ?? 0,
      ffwi: fosberg(temp, rh, wspd),
    };
  }
}
