'use client';

import React, { useEffect, useState } from 'react';
import {
  Flame, Layers, Eye, ShieldAlert, Sparkles, MapPin, Shield, Wind, BookOpen, Home, Mountain, Droplet, Radio, Scale,
  AlertTriangle, CheckCircle2, Loader2, Info, Thermometer,
} from 'lucide-react';
import type { LayerToggles } from './MapView';
import { bellwetherCategories } from '../lib/legend';
import { LayerState, REGIONS, getJSON } from '../lib/api';

interface SidebarProps {
  activeLayer: '1yr' | '5yr' | 'none';
  setActiveLayer: (layer: '1yr' | '5yr' | 'none') => void;
  selectedRegion: string;
  setSelectedRegion: (region: string) => void;
  layers: LayerToggles;
  setLayers: (fn: (prev: LayerToggles) => LayerToggles) => void;
  layerStatus: Record<string, LayerState>;
  opacity: number;
  setOpacity: (val: number) => void;
  layerStats: any;
  onOpenDataCatalog: () => void;
  onOpenModelEvaluation: () => void;
  onSwitchToVision?: () => void;
}

const LAYERS: Array<{ id: keyof LayerToggles; title: string; source: string; icon: React.ReactNode; tone: string }> = [
  { id: 'perimeters', title: 'Fire perimeters & incidents', source: 'NIFC WFIGS · CAL FIRE', icon: <Flame className="w-4 h-4" />, tone: 'bg-rose-500/10 text-rose-400 border-rose-500/20' },
  { id: 'hotspots', title: 'Satellite hotspots (24 h)', source: 'NASA FIRMS VIIRS 375 m', icon: <Radio className="w-4 h-4" />, tone: 'bg-amber-500/10 text-amber-400 border-amber-500/20' },
  { id: 'fuel', title: 'Fuel moisture (RAWS)', source: 'CDEC, 10-h fuel sticks, hourly', icon: <Droplet className="w-4 h-4" />, tone: 'bg-teal-500/10 text-teal-400 border-teal-500/20' },
  { id: 'wind', title: 'Wind (HRRR)', source: 'NOAA HRRR 3 km via Open-Meteo', icon: <Wind className="w-4 h-4" />, tone: 'bg-blue-500/10 text-blue-400 border-blue-500/20' },
  { id: 'fhsz', title: 'Fire Hazard Severity Zones', source: 'CAL FIRE SRA 2023 · LRA 2025', icon: <Shield className="w-4 h-4" />, tone: 'bg-red-500/10 text-red-400 border-red-500/20' },
  { id: 'wrc', title: 'Annual burn probability', source: 'USDA Wildfire Risk to Communities, 30 m', icon: <Flame className="w-4 h-4" />, tone: 'bg-blue-500/10 text-blue-300 border-blue-500/20' },
  { id: 'slope', title: 'Terrain slope', source: 'USGS 3DEP DEM', icon: <Mountain className="w-4 h-4" />, tone: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20' },
  { id: 'ndvi', title: 'Vegetation (NDVI)', source: 'Sentinel-2, latest clear scene', icon: <Sparkles className="w-4 h-4" />, tone: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20' },
  { id: 'buildings', title: 'Building footprints', source: 'Microsoft US Building Footprints', icon: <Home className="w-4 h-4" />, tone: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' },
];

// Classes drawn by backend/live_sources.py (wrc_bp_overlay); below 0.05 %/yr is left clear.
const WRC_LEGEND: Array<[string, string]> = [
  ['0.05–0.2 %/yr', '#BFDBFE'], ['0.2–0.5', '#60A5FA'], ['0.5–1', '#2563EB'], ['≥1 %/yr', '#1E3A8A'],
];

function StatusLine({ s }: { s?: LayerState }) {
  if (!s || s.state === 'off') return null;
  const cls =
    s.state === 'ok' ? 'text-emerald-400' : s.state === 'error' ? 'text-rose-400' : s.state === 'info' ? 'text-amber-300' : 'text-slate-400';
  const icon =
    s.state === 'ok' ? <CheckCircle2 className="w-3 h-3 shrink-0" /> : s.state === 'error' ? <AlertTriangle className="w-3 h-3 shrink-0" /> :
    s.state === 'info' ? <Info className="w-3 h-3 shrink-0" /> : <Loader2 className="w-3 h-3 shrink-0 animate-spin" />;
  return (
    <div className={`mt-1.5 flex items-start gap-1 text-[10px] leading-snug ${cls}`}>
      {icon}
      <span>{s.state === 'loading' ? 'Loading from source…' : s.detail}</span>
    </div>
  );
}

function Conditions({ region }: { region: string }) {
  const [w, setW] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setW(null);
    setErr(null);
    getJSON(`/api/weather/live?region=${region}`)
      .then((d) => setW(d.weather))
      .catch((e) => setErr(String(e.message || e)));
  }, [region]);
  return (
    <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs space-y-1.5">
      <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider text-slate-400">
        <span className="flex items-center gap-1.5"><Thermometer className="w-3.5 h-3.5 text-blue-400" /> Current conditions</span>
        <span className="normal-case font-normal text-slate-500">{w ? `HRRR ${w.valid_time_utc}` : ''}</span>
      </div>
      {err && <div className="text-rose-400 text-[11px]">Weather source unavailable: {err}</div>}
      {!w && !err && <div className="text-slate-500 text-[11px]">Loading…</div>}
      {w && (
        <>
          <div className="grid grid-cols-3 gap-2 text-slate-200">
            <div><div className="text-[10px] text-slate-500">Temp</div><b>{w.temperature_f}°F</b></div>
            <div><div className="text-[10px] text-slate-500">RH</div><b>{w.relative_humidity_pct}%</b></div>
            <div><div className="text-[10px] text-slate-500">Wind</div><b>{w.wind_direction_cardinal} {w.wind_speed_mph}</b> <span className="text-slate-500">g{w.wind_gust_mph} mph</span></div>
          </div>
          {w.alerts?.length ? (
            w.alerts.map((a: any, i: number) => (
              <div key={i} className={`p-1.5 rounded-lg text-[11px] font-semibold ${a.event?.includes('Red Flag') ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40' : 'bg-amber-500/10 text-amber-300 border border-amber-500/30'}`}>
                {a.event}{a.ends ? ` until ${new Date(a.ends).toLocaleString()}` : ''}
              </div>
            ))
          ) : (
            <div className="text-[10px] text-slate-500">No active NWS alerts at the region centre.</div>
          )}
        </>
      )}
    </div>
  );
}

export default function Sidebar({
  activeLayer, setActiveLayer, selectedRegion, setSelectedRegion, layers, setLayers, layerStatus, opacity, setOpacity,
  layerStats, onOpenDataCatalog, onOpenModelEvaluation, onSwitchToVision,
}: SidebarProps) {
  // Bellwether categories for the model shown on the map (same cut-offs as the overlay)
  const bwWhich: '1yr' | '5yr' = activeLayer === '5yr' ? '5yr' : '1yr';
  const riskCategories = bellwetherCategories(bwWhich);

  return (
    <aside className="w-96 glass-panel h-screen flex flex-col z-20 shadow-2xl overflow-y-auto border-r border-slate-800">
      <div className="p-5 border-b border-slate-800/80 bg-slate-900/40">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2.5 rounded-xl bg-orange-500/20 text-orange-400 border border-orange-500/30">
            <Flame className="w-6 h-6" />
          </div>
          <div>
            <h1 className="font-bold text-lg text-slate-100 tracking-wide">Wildfire AI Platform</h1>
            <p className="text-xs text-orange-400 font-medium">Live public data · research prototype</p>
          </div>
        </div>
        <p className="text-[12px] text-slate-300 leading-snug">
          Wildfire hazard and today&apos;s fire conditions for three Bay Area communities at the edge of wildland. Pick a region,
          switch layers on, and click the map.
        </p>

        <div className="mt-3">
          <label className="text-[11px] font-semibold uppercase text-slate-400 tracking-wider mb-1 flex items-center gap-1.5">
            <MapPin className="w-3.5 h-3.5 text-orange-400" /> Region
          </label>
          <select
            value={selectedRegion}
            onChange={(e) => setSelectedRegion(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 text-slate-100 text-xs font-semibold rounded-xl p-2.5 outline-none focus:border-orange-500 cursor-pointer"
          >
            {Object.entries(REGIONS).map(([k, r]) => (
              <option key={k} value={k}>{r.name}</option>
            ))}
          </select>
        </div>

        <button onClick={onOpenModelEvaluation} className="w-full mt-2.5 py-2.5 px-3 rounded-xl bg-gradient-to-r from-orange-500/25 via-slate-800 to-purple-500/25 hover:from-orange-500/35 hover:to-purple-500/35 border border-orange-500/40 text-xs font-bold text-orange-200 flex items-center justify-center gap-2">
          <Scale className="w-4 h-4 text-orange-400" /> Bellwether vs FHSZ analysis
        </button>
        <button onClick={onOpenDataCatalog} className="w-full mt-2 py-2 px-3 rounded-xl bg-slate-800/70 hover:bg-slate-800 border border-slate-700 text-xs font-semibold text-slate-300 flex items-center justify-center gap-2">
          <BookOpen className="w-4 h-4 text-amber-400" /> Data sources & live status
        </button>
        {onSwitchToVision && (
          <button onClick={onSwitchToVision} className="w-full mt-2 py-2.5 px-3 rounded-xl bg-gradient-to-r from-purple-950/60 via-slate-900 to-indigo-950/60 border border-purple-500/40 text-xs font-bold text-purple-200 flex items-center justify-center gap-2">
            <Eye className="w-4 h-4 text-purple-400" /> VLM evaluation (held-out fires)
          </button>
        )}
      </div>

      <div className="p-5 space-y-4 flex-1">
        <Conditions region={selectedRegion} />

        <div>
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2.5">
            <Layers className="w-4 h-4 text-orange-400" /> Bellwether hazard model (100 m)
          </div>
          <p className="text-[11px] text-slate-400 -mt-1 mb-2 leading-snug">
            Chance that each 100 m cell burns within the next 1 or 5 years. Partner model, available in the pilot area only.
          </p>
          <div className="grid grid-cols-3 gap-2">
            {([['1yr', '1-year'], ['5yr', '5-year'], ['none', 'Off']] as const).map(([k, label]) => (
              <button
                key={k}
                onClick={() => setActiveLayer(k)}
                className={`p-2.5 rounded-xl border text-xs font-bold ${activeLayer === k ? 'bg-orange-500/20 border-orange-500/60 text-orange-300' : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:text-slate-200'}`}
              >
                {label}
              </button>
            ))}
          </div>
          <StatusLine s={layerStatus.bellwether} />
        </div>

        <div className="space-y-2">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Live layers</div>
          {LAYERS.map((l) => (
            <label key={l.id} className="block p-2.5 rounded-xl bg-slate-900/50 border border-slate-800 cursor-pointer hover:border-slate-700">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <div className={`p-1.5 rounded-lg border ${l.tone}`}>{l.icon}</div>
                  <div>
                    <div className="text-xs font-semibold text-slate-200">{l.title}</div>
                    <div className="text-[10px] text-slate-400">{l.source}</div>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={layers[l.id]}
                  onChange={(e) => setLayers((prev) => ({ ...prev, [l.id]: e.target.checked }))}
                  className="w-4 h-4 accent-orange-500 cursor-pointer"
                />
              </div>
              <StatusLine s={layerStatus[l.id]} />
              {l.id === 'wrc' && layers.wrc && (
                <div className="mt-1.5 flex flex-wrap gap-x-2.5 gap-y-1 text-[10px] text-slate-400">
                  {WRC_LEGEND.map(([label, color]) => (
                    <span key={label} className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: color }} />{label}</span>
                  ))}
                </div>
              )}
            </label>
          ))}
          <div className="text-[10px] text-slate-500 px-1">Click anywhere on the map for a point summary from all sources.</div>
        </div>

        <div className="space-y-1.5 bg-slate-900/40 p-2.5 rounded-xl border border-slate-800/80">
          <div className="flex items-center justify-between text-xs text-slate-300">
            <span className="flex items-center gap-1.5 font-medium"><Eye className="w-3.5 h-3.5 text-slate-400" /> Bellwether opacity</span>
            <span className="font-mono text-orange-400 font-bold">{Math.round(opacity * 100)}%</span>
          </div>
          <input type="range" min="0.1" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(parseFloat(e.target.value))} className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-orange-500" />
        </div>

        <div className="space-y-1.5">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <ShieldAlert className="w-4 h-4 text-orange-400" /> Bellwether {bwWhich === '5yr' ? '5-year' : '1-year'} categories
          </div>
          <div className="space-y-0.5 bg-slate-900/60 p-2 rounded-xl border border-slate-800/80 text-xs">
            {riskCategories.map((cat) => {
              const stat = layerStats?.[cat.name];
              return (
                <div key={cat.name} className="flex items-center justify-between py-0.5">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: cat.color }} />
                    <span className="font-medium text-slate-200">{cat.name}</span>
                  </div>
                  <div className="flex items-center gap-3 font-mono text-slate-400">
                    <span>{cat.range}</span>
                    {stat && <span className="text-orange-400 font-bold w-12 text-right">{stat.percentage}%</span>}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="text-[10px] text-slate-500 px-1">Annual probability of a 100 m cell being inside a fire perimeter; percentages are the share of cells with values in view.</div>
        </div>
      </div>
    </aside>
  );
}
