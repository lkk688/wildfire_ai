'use client';

import React, { useState } from 'react';
import { ChevronDown, ChevronUp, ListTree } from 'lucide-react';
import type { LayerToggles } from './MapView';
import { LAYER_LEGENDS, LEGEND_ORDER, Swatch, bellwetherLegend } from '../lib/legend';
import type { LayerState } from '../lib/api';
import { WX_VARS, WxVar, gradientCss, legendPos } from '../lib/weatherVars';

function Mark({ s }: { s: Swatch }) {
  const shape = s.shape || 'box';
  if (shape === 'dot') return <span className="w-2.5 h-2.5 rounded-full shrink-0 border border-slate-900" style={{ background: s.color }} />;
  if (shape === 'ring') return <span className="w-2.5 h-2.5 rounded-full shrink-0 border-[1.5px] border-white" style={{ background: s.color }} />;
  if (shape === 'line') return <span className="w-4 h-[3px] rounded shrink-0" style={{ background: s.color }} />;
  if (shape === 'dash') return <span className="w-4 h-0 shrink-0 border-t-2 border-dashed" style={{ borderColor: s.color }} />;
  if (shape === 'arrow')
    return (
      <svg width="10" height="14" viewBox="0 0 22 34" className="shrink-0">
        <path d="M11 0 L20 12 L13 12 L13 34 L9 34 L9 12 L2 12 Z" fill={s.color} stroke="#0f172a" strokeWidth="1.5" />
      </svg>
    );
  return <span className="w-3 h-3 rounded-sm shrink-0" style={{ background: s.color }} />;
}

// Legend for the layers that are currently switched on (colours match what the map draws).
export default function MapLegend({ activeLayer, layers, layerStatus, wxField }: { activeLayer: '1yr' | '5yr' | 'none'; layers: LayerToggles; layerStatus: Record<string, LayerState>; wxField: WxVar | 'none' }) {
  const [open, setOpen] = useState(true);
  const items = [
    ...(activeLayer !== 'none' ? [bellwetherLegend(activeLayer)] : []),
    ...LEGEND_ORDER.filter((k) => (layers as Record<string, boolean>)[k])
      .map((k) => (layerStatus[k]?.swatches ? { ...LAYER_LEGENDS[k], swatches: layerStatus[k].swatches as Swatch[] } : LAYER_LEGENDS[k]))
      .filter((it) => it.swatches.length > 0),
  ];
  return (
    <div id="map-legend" className="absolute left-3 top-[72px] z-[500] w-64 max-h-[calc(100%-200px)] overflow-y-auto rounded-xl bg-slate-950/90 border border-slate-700/80 shadow-2xl backdrop-blur">
      <button onClick={() => setOpen(!open)} className="w-full flex items-center justify-between px-3 py-2 text-[11px] font-bold uppercase tracking-wider text-slate-200 cursor-pointer">
        <span className="flex items-center gap-1.5"><ListTree className="w-3.5 h-3.5 text-orange-400" /> What the colours mean</span>
        {open ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </button>
      {open && (
        <div className="px-3 pb-3 space-y-2.5">
          {wxField !== 'none' && (
            <div>
              <div className="text-[11px] font-semibold text-slate-100 leading-tight">
                {WX_VARS[wxField].label}{WX_VARS[wxField].unit ? ` (${WX_VARS[wxField].unit})` : ''} at the slider hour
              </div>
              <div className="text-[10px] text-slate-400 leading-tight">NOAA HRRR · {WX_VARS[wxField].note}</div>
              <div className="mt-1 h-2.5 rounded" style={{ background: gradientCss(wxField) }} />
              <div className="relative h-3 text-[9px] text-slate-400">
                {WX_VARS[wxField].ticks.map((t) => (
                  <span key={t} className="absolute -translate-x-1/2" style={{ left: `${legendPos(wxField, t) * 100}%` }}>{t}</span>
                ))}
              </div>
            </div>
          )}
          {items.length === 0 && wxField === 'none' && <div className="text-[11px] text-slate-400">No layers are on. Turn some on in the left panel.</div>}
          {items.map((it) => (
            <div key={it.id}>
              <div className="text-[11px] font-semibold text-slate-100 leading-tight">{it.title}</div>
              {it.note && <div className="text-[10px] text-slate-400 leading-tight">{it.note}</div>}
              <div className={`mt-1 grid ${it.swatches.some((s) => s.label.length > 17) ? 'grid-cols-1' : 'grid-cols-2'} gap-x-2 gap-y-0.5`}>
                {it.swatches.map((s) => (
                  <div key={s.label} className="flex items-center gap-1.5 text-[10px] text-slate-300 min-w-0">
                    <Mark s={s} />
                    <span className="truncate">{s.label}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
