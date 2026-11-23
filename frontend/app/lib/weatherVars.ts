// Weather variables shown as colour fields on the map, with their colour scales. No Leaflet here (used by the legend).

export type WxVar = 'temp' | 'rh' | 'gust' | 'precip' | 'cloud' | 'ffwi';
export type WxSample = {
  temp: number; rh: number; wspd: number; wdir: number; gust: number; precip: number; cloud: number; ffwi: number;
};
type Stop = [number, string, number?]; // value, colour, alpha (default 1)

export const WX_VARS: Record<WxVar, { label: string; short: string; unit: string; stops: Stop[]; ticks: number[]; note: string; fmt: (v: number) => string }> = {
  temp: {
    label: 'Temperature', short: 'Temp', unit: '°F', note: '2 m above ground',
    stops: [[10, '#313695'], [25, '#4575b4'], [40, '#74add1'], [50, '#abd9e9'], [60, '#e0f3f8'], [68, '#fee090'], [78, '#fdae61'], [88, '#f46d43'], [98, '#d73027'], [110, '#a50026']],
    ticks: [20, 40, 60, 80, 100], fmt: (v) => `${Math.round(v)}°F`,
  },
  rh: {
    label: 'Relative humidity', short: 'Humidity', unit: '%', note: 'Below ~15% fuels dry out fast',
    stops: [[5, '#8c2d04'], [15, '#cc4c02'], [25, '#fe9929'], [35, '#fee391'], [50, '#c7eae5'], [70, '#5ab4ac'], [95, '#01665e']],
    ticks: [10, 25, 50, 75, 95], fmt: (v) => `${Math.round(v)}%`,
  },
  gust: {
    label: 'Wind gusts', short: 'Gusts', unit: 'mph', note: '10 m above ground',
    stops: [[0, '#fff7f3', 0], [8, '#fcc5c0', 0.4], [15, '#fa9fb5', 0.8], [25, '#f768a1'], [35, '#dd3497'], [50, '#ae017e'], [70, '#7a0177']],
    ticks: [10, 25, 40, 55, 70], fmt: (v) => `${Math.round(v)} mph`,
  },
  precip: {
    label: 'Precipitation', short: 'Rain', unit: 'in/h', note: 'Hourly total; clear = dry',
    stops: [[0.004, '#bfdbfe', 0], [0.01, '#93c5fd', 0.7], [0.03, '#60a5fa', 0.9], [0.08, '#2563eb'], [0.2, '#1e3a8a'], [0.4, '#7c3aed'], [0.8, '#c026d3']],
    ticks: [0.01, 0.08, 0.4], fmt: (v) => `${v.toFixed(2)} in/h`,
  },
  cloud: {
    label: 'Cloud cover', short: 'Cloud', unit: '%', note: 'Total cloud; clear = sky clear',
    stops: [[5, '#cbd5e1', 0], [40, '#cbd5e1', 0.35], [100, '#f8fafc', 0.85]],
    ticks: [25, 50, 75, 100], fmt: (v) => `${Math.round(v)}%`,
  },
  ffwi: {
    label: 'Fire weather (Fosberg index)', short: 'FFWI', unit: '', note: 'From temperature, humidity and wind; ≥50 is extreme',
    stops: [[0, '#1a9850', 0.15], [10, '#66bd63', 0.55], [20, '#d9ef8b'], [30, '#fee08b'], [40, '#fdae61'], [50, '#f46d43'], [65, '#d73027'], [80, '#a50026']],
    ticks: [10, 30, 50, 70], fmt: (v) => `${Math.round(v)}`,
  },
};
export const WX_ORDER: WxVar[] = ['temp', 'rh', 'gust', 'precip', 'cloud', 'ffwi'];

/** Fosberg Fire Weather Index (Fosberg 1978) from air temperature (°F), relative humidity (%) and wind speed (mph). */
export function fosberg(tF: number, rh: number, mph: number): number {
  const h = Math.max(0, Math.min(100, rh));
  const m = h < 10 ? 0.03229 + 0.281073 * h - 0.000578 * h * tF
    : h <= 50 ? 2.22749 + 0.160107 * h - 0.01478 * tF
      : 21.0606 + 0.005565 * h * h - 0.00035 * h * tF - 0.483199 * h;
  const r = Math.max(0, m) / 30;
  const eta = 1 - 2 * r + 1.5 * r * r - 0.5 * r * r * r;
  return Math.min(100, Math.max(0, (eta * Math.sqrt(1 + mph * mph)) / 0.3002));
}

export function ffwiClass(v: number) {
  return v >= 50 ? 'extreme' : v >= 30 ? 'high' : v >= 20 ? 'elevated' : 'low';
}

const hex = (c: string) => [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
const cache = new Map<WxVar, Array<[number, number[], number]>>();

/** RGBA (0-255) for a value on a variable's scale; alpha 0 below the first stop's alpha. */
export function colorFor(v: WxVar, x: number): [number, number, number, number] {
  let st = cache.get(v);
  if (!st) {
    st = WX_VARS[v].stops.map(([val, c, a]) => [val, hex(c), a ?? 1] as [number, number[], number]);
    cache.set(v, st);
  }
  if (x <= st[0][0]) return [st[0][1][0], st[0][1][1], st[0][1][2], st[0][2] * 255];
  for (let i = 1; i < st.length; i++) {
    if (x <= st[i][0]) {
      const [v0, c0, a0] = st[i - 1], [v1, c1, a1] = st[i];
      const f = (x - v0) / (v1 - v0);
      return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f, (a0 + (a1 - a0) * f) * 255];
    }
  }
  const l = st[st.length - 1];
  return [l[1][0], l[1][1], l[1][2], l[2] * 255];
}

/** Position (0-1) of a value on the legend bar; the colour stops are spaced evenly, so uneven scales stay readable. */
export function legendPos(v: WxVar, x: number) {
  const st = WX_VARS[v].stops, n = st.length - 1;
  if (x <= st[0][0]) return 0;
  for (let i = 1; i <= n; i++) if (x <= st[i][0]) return (i - 1 + (x - st[i - 1][0]) / (st[i][0] - st[i - 1][0])) / n;
  return 1;
}

/** CSS linear-gradient for the legend (same colours as the map, stops evenly spaced). */
export function gradientCss(v: WxVar) {
  const s = WX_VARS[v].stops;
  return `linear-gradient(to right, ${s
    .map(([, c, a], i) => {
      const [r, g, b] = hex(c);
      return `rgba(${r},${g},${b},${Math.max(a ?? 1, 0.15)}) ${((i / (s.length - 1)) * 100).toFixed(1)}%`;
    })
    .join(', ')})`;
}

export const COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
export const cardinal = (deg: number) => COMPASS[Math.round((((deg % 360) + 360) % 360) / 22.5) % 16];
