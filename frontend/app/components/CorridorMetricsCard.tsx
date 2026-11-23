'use client';

import React, { useState } from 'react';
import { Home, Flame, X, Wind, Shield, TreePine, History, Droplet, Mountain, Loader2, AlertTriangle, Layers } from 'lucide-react';

interface CorridorMetricsCardProps {
  data: any;
  onClose: () => void;
}

const pct = (v: number | null | undefined, d = 3) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(d)}%`);

function Row({ icon, title, children, source }: { icon: React.ReactNode; title: string; children: React.ReactNode; source?: string }) {
  return (
    <div className="p-2.5 rounded-xl bg-slate-900/50 border border-slate-800">
      <div className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-400 mb-1">{icon} {title}</div>
      <div className="text-xs text-slate-200">{children}</div>
      {source && <div className="text-[10px] text-slate-500 mt-1">{source}</div>}
    </div>
  );
}

export default function CorridorMetricsCard({ data, onClose }: CorridorMetricsCardProps) {
  // Replacement cost is an explicit user assumption (no parcel or assessor data is available).
  const [costPerSqft, setCostPerSqft] = useState<string>('');
  if (!data) return null;
  const { center } = data;
  const err = (k: string) => data.errors?.[k];
  const w = data.live_weather;
  const bw = data.bellwether;
  const b = data.buildings_in_radius;
  const st = data.nearest_fuel_station;
  const slope = data.terrain_slope_share_pct;
  const sqft = b ? b.footprint_area_m2 * 10.7639 : 0;
  const cost = parseFloat(costPerSqft);

  return (
    <div className="absolute top-5 right-5 w-[430px] max-h-[88vh] overflow-y-auto glass-panel rounded-2xl p-5 shadow-2xl z-30 border border-slate-700/80 space-y-2.5">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-orange-500/20 text-orange-400 border border-orange-500/30"><Flame className="w-5 h-5" /></div>
          <div>
            <h3 className="font-bold text-sm text-slate-100">Location summary</h3>
            <p className="text-xs text-slate-400 font-mono">{center.lat.toFixed(5)}, {center.lng.toFixed(5)}</p>
          </div>
        </div>
        <button onClick={onClose} className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800"><X className="w-4 h-4" /></button>
      </div>

      {data.loading && (
        <div className="flex items-center gap-2 text-xs text-slate-400 py-6 justify-center"><Loader2 className="w-4 h-4 animate-spin" /> Querying live sources…</div>
      )}
      {data.error && <div className="text-xs text-rose-400 flex items-center gap-1.5"><AlertTriangle className="w-4 h-4" /> {data.error}</div>}

      {!data.loading && !data.error && (
        <>
          <Row icon={<Wind className="w-3.5 h-3.5 text-blue-400" />} title="Weather now" source={w ? `${w.source}, valid ${w.valid_time_utc}` : err('live_weather')}>
            {w ? (
              <>
                {w.temperature_f}°F · RH {w.relative_humidity_pct}% · wind {w.wind_direction_cardinal} {w.wind_speed_mph} mph (gust {w.wind_gust_mph})
                {w.alerts?.length ? (
                  <div className="mt-1 text-rose-300 font-semibold">{w.alerts.map((a: any) => a.event).join(' · ')}</div>
                ) : (
                  <div className="mt-0.5 text-[11px] text-slate-500">No active NWS alerts here.</div>
                )}
              </>
            ) : 'Unavailable'}
          </Row>

          <div className="grid grid-cols-2 gap-2">
            <Row icon={<Layers className="w-3.5 h-3.5 text-orange-400" />} title="Bellwether" source="Partner GeoTIFF, 2026-04 release">
              {bw ? (
                <>
                  1-yr <b>{pct(bw.prob_1yr)}</b> · 5-yr <b>{pct(bw.prob_5yr, 2)}</b>
                  <div className="text-[11px] text-orange-300">{bw.category_1yr}</div>
                </>
              ) : (
                <span className="text-slate-500">No value here (outside the shared raster or masked)</span>
              )}
            </Row>
            <Row icon={<Shield className="w-3.5 h-3.5 text-red-400" />} title="CAL FIRE FHSZ" source={data.fhsz?.responsibility_area ? `CAL FIRE, ${data.fhsz.responsibility_area}` : 'CAL FIRE SRA 2023 / LRA 2025'}>
              {data.fhsz ? <b>{data.fhsz.zone}</b> : <span className="text-slate-500">{err('fhsz') || 'Unavailable'}</span>}
            </Row>
          </div>

          <Row icon={<Flame className="w-3.5 h-3.5 text-blue-400" />} title="USDA annual burn probability" source={data.usda_burn_probability ? 'USDA Forest Service, Wildfire Risk to Communities (2024 release; landscape as of 2020)' : err('usda_burn_probability') || 'USDA Wildfire Risk to Communities'}>
            {data.usda_burn_probability ? (
              <>
                <b>{data.usda_burn_probability.pct_per_year.toFixed(2)}%</b> per year
                {data.usda_burn_probability.one_in_n_years ? <span className="text-slate-400"> (about 1 in {data.usda_burn_probability.one_in_n_years.toLocaleString()} years)</span> : null}
              </>
            ) : (
              <span className="text-slate-500">No value here (water or outside coverage)</span>
            )}
          </Row>

          <Row icon={<TreePine className="w-3.5 h-3.5 text-emerald-400" />} title={`Land cover within ${Math.round(data.land_cover?.radius_m || data.radius_meters)} m`} source={data.land_cover?.source || err('land_cover')}>
            {data.land_cover ? (
              <>
                At point: <b>{data.land_cover.at_point}</b>
                <div className="text-[11px] text-slate-400">
                  {Object.entries(data.land_cover.within_radius_pct).map(([k, v]) => `${k} ${v}%`).join(' · ')}
                </div>
              </>
            ) : 'Unavailable'}
          </Row>

          <div className="grid grid-cols-2 gap-2">
            <Row icon={<Home className="w-3.5 h-3.5 text-emerald-400" />} title={`Buildings within ${data.radius_feet} ft`} source="Microsoft US Building Footprints">
              {b ? (
                <>
                  <b>{b.count}</b> footprints · {Math.round(sqft).toLocaleString()} sq ft
                  <div className="text-[11px] text-slate-400">nearest {b.nearest_m?.toFixed(0)} m</div>
                </>
              ) : 'Unavailable'}
            </Row>
            <Row icon={<Mountain className="w-3.5 h-3.5 text-indigo-400" />} title="Slope within ~200 m" source="USGS 3DEP DEM">
              {slope ? (
                <>
                  ≥30%: <b>{((slope['30_50'] || 0) + (slope.ge50 || 0)).toFixed(0)}%</b> of area
                  <div className="text-[11px] text-slate-400">&lt;15%: {slope.lt15}% · 15–30%: {slope['15_30']}%</div>
                </>
              ) : 'Unavailable'}
            </Row>
          </div>

          <Row icon={<Droplet className="w-3.5 h-3.5 text-teal-400" />} title="Nearest RAWS fuel moisture" source={st ? `${st.source}; ${st.fuel_moisture_10h_pct_time_pst} PST` : err('nearest_fuel_station')}>
            {st ? (
              <>
                <b>{st.fuel_moisture_10h_pct}%</b> 10-h ({st.dryness_class}) at {st.station_name}, {st.distance_km} km away
              </>
            ) : 'No reporting station nearby'}
          </Row>

          <Row icon={<History className="w-3.5 h-3.5 text-amber-400" />} title="Past fires covering this point (1950+)" source="CAL FIRE FRAP perimeters">
            {data.historic_fires?.length
              ? data.historic_fires.map((f: any) => `${f.name} ${f.year} (${Math.round(f.acres).toLocaleString()} ac)`).join(' · ')
              : data.historic_fires ? 'None recorded' : 'Unavailable'}
          </Row>

          {b && (
            <div className="p-2.5 rounded-xl bg-slate-950/50 border border-slate-800 text-xs space-y-1">
              <div className="text-[11px] font-semibold text-slate-400">Replacement-cost estimate (your assumption)</div>
              <div className="flex items-center gap-2">
                <span className="text-slate-400">$/sq ft</span>
                <input
                  value={costPerSqft}
                  onChange={(e) => setCostPerSqft(e.target.value)}
                  placeholder="enter"
                  className="w-20 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-100"
                />
                <span className="text-slate-200">{Number.isFinite(cost) && cost > 0 ? `≈ $${Math.round(sqft * cost).toLocaleString()} for footprint area` : ''}</span>
              </div>
              <div className="text-[10px] text-slate-500">Footprint area × your cost; no parcel, assessor or insurance data are used.</div>
            </div>
          )}
          {data.corridor_note && <div className="text-[10px] text-slate-500">Orange circle: {data.corridor_note}</div>}
        </>
      )}
    </div>
  );
}
