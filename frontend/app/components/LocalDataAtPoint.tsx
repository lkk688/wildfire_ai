'use client';

import React, { useEffect, useState } from 'react';
import { Database, Loader2 } from 'lucide-react';
import { getJSON } from '../lib/api';
import { compass } from '../lib/geodata';

// Layers summarised in the point card (a fast subset; the agent can query all of them)
const RASTERS = [
  'fuel_model', 'canopy_cover', 'canopy_height_lf', 'slope', 'worldcover',
  'burn_probability', 'flame_length', 'exposure_type', 'cond_risk_structures',
  'fw_erc_p90', 'fw_wind_p90', 'fw_dir_mode', 'fw_rmin_p10',
];
const VECTORS = ['fhsz', 'fire_history', 'ignitions', 'treatments_calfire', 'treatments_rx', 'treatments_calvtp', 'treatments_federal'];

const fmtVal = (id: string, v: any, units?: string) => {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'object' && 'label' in v) return v.label;
  if (id === 'burn_probability') return `${(v * 100).toFixed(2)}% per year`;
  if (id === 'fw_dir_mode') return `from ${compass(v)} (${Math.round(v)}°)`;
  if (id === 'exposure_type') return v >= 0.99 ? 'direct flame' : v > 0 ? `embers only (${v.toFixed(2)})` : 'not exposed';
  return `${typeof v === 'number' ? (Number.isInteger(v) ? v : Math.abs(v) < 10 ? v.toFixed(2) : v.toFixed(0)) : v} ${units || ''}`.trim();
};

function Row({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div className="flex items-start justify-between gap-3 py-0.5">
      <span className="text-slate-400 shrink-0">{label}</span>
      <span className="text-right text-slate-200">
        {value}
        {sub && <span className="block text-[9px] text-slate-500">{sub}</span>}
      </span>
    </div>
  );
}

export default function LocalDataAtPoint({ lat, lng }: { lat: number; lng: number }) {
  const [d, setD] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let alive = true; // the parent remounts this component (key) for a new point, so state starts empty
    getJSON(`/api/geodata/point?lat=${lat}&lng=${lng}&layers=${[...RASTERS, ...VECTORS].join(',')}`)
      .then((r) => alive && setD(r))
      .catch((e) => alive && setErr(String(e.message || e)));
    return () => {
      alive = false;
    };
  }, [lat, lng]);

  const head = (
    <div className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-400 mb-1">
      <Database className="w-3.5 h-3.5 text-sky-400" /> California data at this point (local cache)
    </div>
  );
  if (err) return <div id="local-data-point" className="p-2.5 rounded-xl bg-slate-900/50 border border-slate-800 text-xs">{head}<div className="text-rose-400">{err}</div></div>;
  if (!d) return <div id="local-data-point" className="p-2.5 rounded-xl bg-slate-900/50 border border-slate-800 text-xs">{head}<div className="flex items-center gap-1.5 text-slate-400"><Loader2 className="w-3 h-3 animate-spin" /> Reading cache…</div></div>;

  const r = d.rasters || {};
  const v = d.vectors || {};
  const fh = v.fhsz?.containing?.[0];
  const fires = v.fire_history?.containing || [];
  const ign = v.ignitions;
  const ignKey = ign && Object.keys(ign).find((k) => k.startsWith('count_within'));
  const treat = ['treatments_calfire', 'treatments_rx', 'treatments_calvtp', 'treatments_federal'].flatMap((k) => {
    const t = v[k];
    if (!t) return [];
    const near = Object.keys(t).find((x) => x.startsWith('within_'));
    return [...(t.containing || []).map((f: any) => ({ f, where: 'here', k })), ...((near && t[near]) || []).map((f: any) => ({ f, where: `${f.distance_m} m away`, k }))];
  });
  const rose = d.fire_weather_wind_rose?.from_direction_share as Record<string, number> | undefined;
  const val = (id: string) => (r[id] ? fmtVal(id, r[id].value, r[id].units) : '—');

  return (
    <div id="local-data-point" className="p-2.5 rounded-xl bg-slate-900/50 border border-slate-800 text-xs space-y-2">
      {head}
      <div>
        <div className="text-[10px] font-semibold text-slate-500 uppercase">Fuels and terrain (LANDFIRE 2025)</div>
        <Row label="Fuel model" value={val('fuel_model')} />
        <Row label="Canopy cover / height" value={`${val('canopy_cover')} / ${val('canopy_height_lf')}`} />
        <Row label="Slope" value={val('slope')} />
        <Row label="Land cover (WorldCover)" value={val('worldcover')} />
      </div>
      <div>
        <div className="text-[10px] font-semibold text-slate-500 uppercase">Modelled hazard (USFS WRC 2024)</div>
        <Row label="Burn probability" value={val('burn_probability')} />
        <Row label="Flame length if burning" value={val('flame_length')} />
        <Row label="Exposure" value={val('exposure_type')} />
      </div>
      <div>
        <div className="text-[10px] font-semibold text-slate-500 uppercase">Fire weather 2000-2025 (gridMET)</div>
        <Row label="Wind on fire-weather days (p90)" value={val('fw_wind_p90')} />
        <Row label="Usual direction then" value={val('fw_dir_mode')} />
        <Row label="Low humidity then (p10)" value={val('fw_rmin_p10')} />
        {rose && (
          <div className="flex items-end gap-1 h-8 mt-1" title="Share of fire-weather days with wind from each direction">
            {Object.entries(rose).map(([k, s]) => (
              <div key={k} className="flex flex-col items-center flex-1">
                <div className="w-full bg-sky-500/70 rounded-sm" style={{ height: `${Math.max(2, s * 60)}px` }} />
                <span className="text-[8px] text-slate-500">{k}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div>
        <div className="text-[10px] font-semibold text-slate-500 uppercase">Official zones and history</div>
        <Row label="FHSZ" value={fh ? `${fh.FHSZ_Description} (${fh.responsibility_area})` : 'not in a Moderate+ zone'} />
        <Row label="Past fires here" value={fires.length ? fires.slice(0, 3).map((f: any) => `${f.FIRE_NAME || '?'} ${f.YEAR_ || ''}`).join(', ') : 'none on record'} sub={fires.length > 3 ? `+${fires.length - 3} more` : undefined} />
        <Row
          label="Ignitions within 5 km (1992-)"
          value={ignKey ? ign[ignKey] : '—'}
          sub={ign?.nearest?.[0] ? `nearest ${ign.nearest[0].distance_m} m: ${ign.nearest[0].IncidentName || 'unnamed'} (${String(ign.nearest[0].discovered ?? '').slice(0, 4)}, ${ign.nearest[0].cause || 'cause ?'})` : undefined}
        />
        <Row label="Fuel treatments here / within 200 m" value={treat.length ? `${treat.length}` : 'none recorded'} sub={treat.slice(0, 2).map((t) => `${t.f.TREATMENT_NAME || t.f.TreatmentName || t.f.Treatment_Type || 'treatment'} (${t.where})`).join('; ') || undefined} />
      </div>
      {d.errors && <div className="text-[10px] text-amber-300">Not available: {Object.keys(d.errors).join(', ')}</div>}
    </div>
  );
}
