'use client';

import React, { useEffect, useState, useSyncExternalStore } from 'react';
import { Pause, Play, RotateCcw, Loader2, CloudOff } from 'lucide-react';
import { WeatherStore } from '../lib/weatherStore';
import { WX_VARS, WxVar, cardinal, ffwiClass } from '../lib/weatherVars';

const H = 3600 * 1000;
const fmtDay = (h: number) => new Date(h * H).toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' });
const fmtHour = (h: number) => new Date(h * H).toLocaleTimeString([], { hour: 'numeric' });
const rel = (d: number) => (d === 0 ? 'now' : Math.abs(d) < 48 ? (d > 0 ? `in ${d} h` : `${-d} h ago`) : `in ${Math.floor(d / 24)} d ${d % 24} h`);
const runLabel = (iso?: string) => (iso ? `${iso.slice(11, 13)}z ${new Date(iso).toLocaleDateString([], { month: 'short', day: 'numeric' })}` : '—');

// Time slider for the weather layers (like a weather app): the past 24 h and about 10 days ahead. HRRR, hourly, up
// to 48 h; then the National Blend of Models (NBM), whose steps are 3 h and later 6 h apart (values in between are
// interpolated, and the readout says so).
export default function TimeSlider({
  store, hour, setHour, field, center,
}: {
  store: WeatherStore;
  hour: number | null;
  setHour: (h: number | null | ((p: number | null) => number | null)) => void;
  field: WxVar | 'none';
  center: { lat: number; lng: number; name: string };
}) {
  useSyncExternalStore(store.subscribe, store.getVersion, () => 0);
  useSyncExternalStore(store.subscribeHover, store.getHoverVersion, () => 0);
  const [playing, setPlaying] = useState(false);
  const range = store.range();
  const st = store.status();
  const h = hour ?? range?.now ?? null;

  useEffect(() => {
    if (!playing || !range) return;
    const t = setInterval(() => setHour((p) => {
      const cur = p ?? range.now;
      const step = range.hrrrLast != null && cur >= range.hrrrLast ? 3 : 1; // hourly while HRRR, then NBM's steps
      return cur + step > range.last ? range.first : cur + step;
    }), 550);
    return () => clearInterval(t);
  }, [playing, range, setHour]);

  if (!range || h == null) {
    return (
      <div id="time-slider" className="absolute bottom-4 left-16 right-[15.5rem] z-30 rounded-2xl bg-slate-950/90 border border-slate-700/80 backdrop-blur px-4 py-3 text-xs text-slate-400 flex items-center gap-2">
        {st.failed && !st.loading ? <CloudOff className="w-4 h-4 text-rose-400" /> : <Loader2 className="w-4 h-4 animate-spin" />}
        {st.failed && !st.loading ? `Weather data unavailable: ${st.error}` : 'Loading NOAA weather for this view…'}
      </div>
    );
  }

  const p = store.hover || center;
  const s = store.sample(p.lat, p.lng, h);
  const d = h - range.now;
  const src = store.sourceAt(h);
  const long = range.last - range.first > 96;
  const ticks: number[] = [];
  for (let x = range.first; x <= range.last; x++) if (new Date(x * H).getHours() % (long ? 12 : 6) === 0) ticks.push(x);
  const pos = (x: number) => `${((x - range.first) / (range.last - range.first)) * 100}%`;
  const hEnd = range.hrrrLast ?? range.last;
  const val = (v: WxVar) => (s ? WX_VARS[v].fmt(s[v]) : '—');

  return (
    <div id="time-slider" className="absolute bottom-4 left-16 right-[15.5rem] z-30 rounded-2xl bg-slate-950/90 border border-slate-700/80 shadow-2xl backdrop-blur px-4 pt-2.5 pb-2">
      <div className="flex items-center gap-3">
        <button
          onClick={() => setPlaying(!playing)}
          aria-label={playing ? 'Pause' : 'Play'}
          className="w-8 h-8 shrink-0 rounded-full bg-sky-500 hover:bg-sky-400 text-white flex items-center justify-center cursor-pointer"
        >
          {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
        </button>
        <div className="min-w-[150px]">
          <div className="text-sm font-bold text-white leading-tight">{fmtDay(h)} · {fmtHour(h)}</div>
          <div className="text-[10px] leading-tight">
            <span className={d < 0 ? 'text-slate-400' : d === 0 ? 'text-emerald-300' : 'text-sky-300'}>{rel(d)}</span>
            <span className="text-slate-500"> · {d < 0 ? 'past (HRRR)' : src?.model === 'nbm' ? `NBM forecast${src.between ? ', between model steps' : ''}` : 'HRRR forecast'}</span>
          </div>
        </div>
        {d !== 0 && (
          <button onClick={() => { setPlaying(false); setHour(null); }} className="shrink-0 px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-[11px] font-semibold text-slate-200 flex items-center gap-1 cursor-pointer">
            <RotateCcw className="w-3 h-3" /> Now
          </button>
        )}
        <div id="weather-readout" className="flex-1 min-w-0 text-[11px] text-slate-300 truncate text-right">
          <span className="text-slate-500">{store.hover ? 'At cursor' : `At ${center.name}`}:</span>{' '}
          {s ? (
            <>
              <b className={field === 'temp' ? 'text-white' : ''}>{val('temp')}</b> · RH <b className={field === 'rh' ? 'text-white' : ''}>{val('rh')}</b> · wind {cardinal(s.wdir)} {Math.round(s.wspd)} mph, gusts{' '}
              <b className={field === 'gust' ? 'text-white' : ''}>{Math.round(s.gust)}</b> · rain <b className={field === 'precip' ? 'text-white' : ''}>{s.precip.toFixed(2)}</b> in · cloud {Math.round(s.cloud)}% · fire weather{' '}
              <b className={field === 'ffwi' ? 'text-white' : ''}>{Math.round(s.ffwi)} ({ffwiClass(s.ffwi)})</b>
            </>
          ) : (
            <span className="text-slate-500">no data here</span>
          )}
        </div>
      </div>

      <div className="relative mt-1.5 h-9">
        {/* past part of the track is shaded */}
        {/* track: past | HRRR hourly forecast | NBM forecast */}
        <div className="absolute top-[9px] left-0 right-0 h-1.5 rounded-full bg-violet-400/25" />
        <div className="absolute top-[9px] h-1.5 bg-sky-500/35" style={{ left: pos(range.now), width: `calc(${pos(hEnd)} - ${pos(range.now)})` }} />
        <div className="absolute top-[9px] left-0 h-1.5 rounded-l-full bg-slate-600/70" style={{ width: pos(range.now) }} />
        <div className="absolute top-[3px] w-0.5 h-[18px] bg-emerald-400 rounded" style={{ left: pos(range.now) }} title="Now" />
        {ticks.map((x) => {
          const midnight = new Date(x * H).getHours() === 0;
          if (long && !midnight) {
            return <div key={x} className="absolute top-[18px] w-px h-1 bg-slate-600 pointer-events-none" style={{ left: pos(x) }} />;
          }
          return (
            <div key={x} className="absolute top-[18px] -translate-x-1/2 flex flex-col items-center pointer-events-none" style={{ left: pos(x) }}>
              <div className={`w-px ${midnight ? 'h-2 bg-slate-300' : 'h-1.5 bg-slate-500'}`} />
              <div className={`text-[9px] leading-none mt-0.5 whitespace-nowrap ${midnight ? 'text-slate-200 font-semibold' : 'text-slate-500'}`}>
                {midnight ? new Date(x * H).toLocaleDateString([], { weekday: 'short', day: 'numeric' }) : fmtHour(x)}
              </div>
            </div>
          );
        })}
        <input
          type="range"
          aria-label="Forecast hour"
          min={range.first}
          max={range.last}
          step={1}
          value={Math.min(range.last, Math.max(range.first, h))}
          onChange={(e) => {
            setPlaying(false);
            const v = Number(e.target.value);
            setHour(v === range.now ? null : v);
          }}
          className="wx-range absolute inset-x-0 top-0 w-full h-6 cursor-pointer"
        />
      </div>
      <div className="flex justify-between text-[9px] text-slate-500 -mt-0.5">
        <span>
          <span className="inline-block w-2 h-1.5 rounded-sm bg-sky-500/70 align-middle" /> HRRR 3 km, hourly (run {runLabel(st.runs.hrrr)}) ·{' '}
          <span className="inline-block w-2 h-1.5 rounded-sm bg-violet-400/70 align-middle" /> NBM 2.5 km, 3-6 h steps (run {runLabel(st.runs.nbm)}) · NOAA open data · local time · {st.loading ? <span className="text-sky-300">loading {st.loading} tile{st.loading > 1 ? 's' : ''}…</span> : `${st.tiles} tiles${st.fromCache ? ` (${st.fromCache} from cache)` : ''}`}
          {st.failed > 0 && <span className="text-amber-300"> · {st.failed} tile{st.failed > 1 ? 's' : ''} not loaded, retrying</span>}
        </span>
        <span className="whitespace-nowrap">Hover the map for values · drag, or ←/→</span>
      </div>
    </div>
  );
}
