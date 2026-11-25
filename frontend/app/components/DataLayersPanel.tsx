'use client';

import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Database, Eye, Loader2 } from 'lucide-react';
import { GEO_GROUPS, GeoLayer, useGeoCatalog } from '../lib/geodata';

interface Props {
  geoLayers: string[];
  setGeoLayers: (fn: (prev: string[]) => string[]) => void;
  geoOpacity: number;
  setGeoOpacity: (v: number) => void;
  mapZoom: number;
}

// Layers kept in the local California cache. Tiles come from /api/geodata/tiles (fetched from the public source the
// first time a place is viewed, from disk afterwards).
export default function DataLayersPanel({ geoLayers, setGeoLayers, geoOpacity, setGeoOpacity, mapZoom }: Props) {
  const { layers, error } = useGeoCatalog();
  const [open, setOpen] = useState<Record<string, boolean>>({ [GEO_GROUPS[0]]: true });
  const toggle = (id: string, on: boolean) => setGeoLayers((prev) => (on ? [...prev.filter((x) => x !== id), id] : prev.filter((x) => x !== id)));
  const groups = GEO_GROUPS.map((g) => [g, layers.filter((l) => l.group === g)] as [string, GeoLayer[]]).filter(([, ls]) => ls.length);

  return (
    <div id="data-layers-panel" className="space-y-2">
      <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
        <Database className="w-4 h-4 text-sky-400" /> California data (local cache)
      </div>
      <p className="text-[11px] text-slate-400 -mt-1 leading-snug">
        Statewide public layers stored on our server. Each tile is downloaded from its source once, then read from disk.
      </p>
      {error && <div className="text-[11px] text-rose-400">Catalog unavailable: {error}</div>}
      {!error && layers.length === 0 && (
        <div className="flex items-center gap-1.5 text-[11px] text-slate-400"><Loader2 className="w-3 h-3 animate-spin" /> Loading catalog…</div>
      )}
      {groups.map(([g, ls]) => {
        const nOn = ls.filter((l) => geoLayers.includes(l.id)).length;
        return (
          <div key={g} className="rounded-xl bg-slate-900/50 border border-slate-800">
            <button
              onClick={() => setOpen((o) => ({ ...o, [g]: !o[g] }))}
              className="w-full flex items-center justify-between px-2.5 py-2 text-xs font-semibold text-slate-200 cursor-pointer"
            >
              <span className="flex items-center gap-1.5 text-left">
                {open[g] ? <ChevronDown className="w-3.5 h-3.5 shrink-0" /> : <ChevronRight className="w-3.5 h-3.5 shrink-0" />}
                {g}
              </span>
              {nOn > 0 && <span className="text-[10px] font-bold text-sky-300">{nOn} on</span>}
            </button>
            {open[g] && (
              <div className="px-2.5 pb-2 space-y-1.5">
                {ls.map((l) => {
                  const on = geoLayers.includes(l.id);
                  const tooFar = on && l.min_zoom !== undefined && mapZoom < l.min_zoom;
                  const minZoom = l.min_zoom ?? 0;
                  const lazy = l.type === 'raster' && !l.prefetch;
                  return (
                    <label key={l.id} className={`block ${l.available === false ? 'opacity-50' : 'cursor-pointer'}`} title={l.notes || ''}>
                      <div className="flex items-start gap-2">
                        <input
                          type="checkbox"
                          disabled={l.available === false}
                          checked={on}
                          onChange={(e) => toggle(l.id, e.target.checked)}
                          className="mt-0.5 w-3.5 h-3.5 accent-sky-500 cursor-pointer shrink-0"
                        />
                        <div className="min-w-0">
                          <div className="text-[11px] font-medium text-slate-200 leading-tight">{l.title}</div>
                          <div className="text-[10px] text-slate-500 leading-tight">
                            {l.source.split(',')[0]} · {l.date}
                            {lazy ? ' · loads where you look' : ''}
                            {l.available === false ? ' · not built yet' : ''}
                          </div>
                          {tooFar && <div className="text-[10px] text-amber-300 leading-tight">Zoom in to see (zoom {minZoom}+, now {mapZoom})</div>}
                        </div>
                      </div>
                    </label>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
      {geoLayers.length > 0 && (
        <div className="space-y-1 bg-slate-900/40 p-2.5 rounded-xl border border-slate-800/80">
          <div className="flex items-center justify-between text-xs text-slate-300">
            <span className="flex items-center gap-1.5 font-medium"><Eye className="w-3.5 h-3.5 text-slate-400" /> Data layer opacity</span>
            <span className="font-mono text-sky-300 font-bold">{Math.round(geoOpacity * 100)}%</span>
          </div>
          <input type="range" min="0.1" max="1" step="0.05" value={geoOpacity} onChange={(e) => setGeoOpacity(parseFloat(e.target.value))} className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500" />
          <button onClick={() => setGeoLayers(() => [])} className="text-[10px] text-slate-400 underline cursor-pointer">Turn all data layers off</button>
        </div>
      )}
    </div>
  );
}
