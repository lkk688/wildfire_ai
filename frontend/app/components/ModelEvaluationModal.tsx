'use client';

import React, { useEffect, useState } from 'react';
import { X, Layers, Scale, AlertTriangle, CheckCircle2, FileText, MapPin, FlaskConical, Loader2 } from 'lucide-react';
import { getJSON } from '../lib/api';

interface ModelEvaluationModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedRegion: string;
}

type Tab = 'analysis' | 'models' | 'evidence' | 'guidance';
const CATS = ['Very Low', 'Low', 'Moderate', 'Significant', 'High', 'Very High', 'Extreme'];
const CAT_COLORS = ['#228B22', '#90EE90', '#FFFF66', '#FFCC00', '#FF8000', '#EE2222', '#9400D3'];

export default function ModelEvaluationModal({ isOpen, onClose, selectedRegion }: ModelEvaluationModalProps) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('analysis');

  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    setError(null);
    getJSON(`/api/model-evaluation?region=${selectedRegion}`)
      .then((d) => setData(d.data))
      .catch((e) => setError(String(e.message || e)))
      .finally(() => setLoading(false));
  }, [isOpen, selectedRegion]);

  if (!isOpen) return null;
  const a = data?.analysis;

  const tabs: Array<[Tab, string, React.ReactNode]> = [
    ['analysis', 'Bellwether vs FHSZ (computed)', <Scale key="a" className="w-3.5 h-3.5" />],
    ['models', 'What each model measures', <Layers key="m" className="w-3.5 h-3.5" />],
    ['evidence', 'Our held-out VLM results', <FlaskConical key="e" className="w-3.5 h-3.5" />],
    ['guidance', 'Guidance', <FileText key="g" className="w-3.5 h-3.5" />],
  ];

  return (
    <div className="fixed inset-0 z-[2000] flex items-center justify-center p-3 sm:p-6 bg-slate-950/85 backdrop-blur-md">
      <div className="glass-panel w-full max-w-6xl max-h-[92vh] rounded-3xl p-6 shadow-2xl border border-slate-700/80 flex flex-col overflow-hidden bg-slate-900/95 text-slate-100">
        <div className="flex items-start justify-between pb-4 border-b border-slate-800 shrink-0">
          <div>
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-slate-800 text-slate-300 border border-slate-700">
              {data?.region_name || selectedRegion}
            </span>
            <h2 className="text-xl font-extrabold text-slate-100 mt-1">Model comparison from real layers</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Bellwether 1-year probabilities compared with CAL FIRE Fire Hazard Severity Zones on the same 100 m grid.
            </p>
          </div>
          <button onClick={onClose} className="p-2.5 rounded-xl text-slate-400 hover:text-slate-100 hover:bg-slate-800"><X className="w-5 h-5" /></button>
        </div>

        <div className="flex items-center gap-2 py-3 border-b border-slate-800/80 shrink-0 overflow-x-auto text-xs font-semibold">
          {tabs.map(([id, label, icon]) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`px-3.5 py-2 rounded-xl flex items-center gap-2 whitespace-nowrap ${tab === id ? 'bg-orange-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'}`}
            >
              {icon} {label}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto py-4 pr-1 space-y-4 text-xs">
          {loading && (
            <div className="py-20 flex flex-col items-center gap-3 text-slate-400">
              <Loader2 className="w-8 h-8 animate-spin text-orange-500" />
              <span>Computing from the Bellwether raster and the FHSZ, perimeter, land-cover and footprint services…</span>
            </div>
          )}
          {error && <div className="py-10 text-center text-rose-400">Could not compute the comparison: {error}</div>}

          {!loading && data && tab === 'analysis' && (
            !a?.available ? (
              <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-200">{a?.reason}</div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <Stat label="Cells analysed (100 m)" value={a.cells.toLocaleString()} sub={`Bellwether values for ${a.bellwether_valid_share_pct}% of the region`} />
                  <Stat label="Rank correlation" value={a.rank_correlation_spearman ?? '—'} sub="Spearman, Bellwether vs FHSZ class" />
                  <Stat label="Bellwether High+, FHSZ none/Moderate" value={`${a.disagreement_area_pct.bellwether_high_fhsz_low}%`} sub="of analysed area" />
                  <Stat label="FHSZ Very High, Bellwether Low" value={`${a.disagreement_area_pct.fhsz_very_high_bellwether_low}%`} sub="of analysed area" />
                </div>

                <div className="overflow-x-auto rounded-2xl border border-slate-800">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="bg-slate-950/80 text-slate-400 text-[10px] uppercase tracking-wider">
                        <th className="p-2.5">FHSZ class</th><th className="p-2.5">Area</th><th className="p-2.5">Bellwether mean / median (1 yr)</th>
                        <th className="p-2.5">High or above</th><th className="p-2.5">Burned since 2000</th><th className="p-2.5 w-1/3">Bellwether category mix</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {a.crosstab.map((r: any) => (
                        <tr key={r.fhsz}>
                          <td className="p-2.5 font-bold text-slate-200">{r.fhsz}</td>
                          <td className="p-2.5">{r.area_pct}%</td>
                          <td className="p-2.5 font-mono">{r.bellwether_1yr_mean_pct}% / {r.bellwether_1yr_median_pct}%</td>
                          <td className="p-2.5">{r.share_high_or_above_pct}%</td>
                          <td className="p-2.5">{r.burned_since_2000_pct}%</td>
                          <td className="p-2.5">
                            <div className="flex h-3 rounded overflow-hidden">
                              {CATS.map((c, i) => (
                                <div key={c} title={`${c}: ${r.bellwether_category_pct[c]}%`} style={{ width: `${r.bellwether_category_pct[c]}%`, background: CAT_COLORS[i] }} />
                              ))}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div>
                  <h3 className="text-sm font-bold text-slate-100 mb-2">Largest disagreement areas</h3>
                  {a.disagreement_areas.length === 0 && <div className="text-slate-400">No disagreement areas in this region.</div>}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {a.disagreement_areas.map((c: any, i: number) => (
                      <div key={i} className="p-3.5 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-1.5">
                        <div className="text-[10px] font-bold uppercase tracking-wider text-rose-300">{c.type}</div>
                        <div className="flex items-center gap-1.5 text-slate-200"><MapPin className="w-3.5 h-3.5 text-orange-400" /> {c.lat}, {c.lon} · {c.area_ha} ha</div>
                        <div className="text-slate-300">Bellwether mean {c.bellwether_1yr_mean_pct}%/yr · FHSZ {c.fhsz}</div>
                        <div className="text-slate-400">Buildings within 300 m: {c.buildings_within_300m ?? '—'}</div>
                        <div className="text-slate-400">
                          Land cover: {c.land_cover ? Object.entries(c.land_cover).slice(0, 3).map(([k, v]) => `${k} ${v}%`).join(' · ') : '—'}
                        </div>
                        <div className="text-slate-400">
                          Past fires: {c.past_fires?.length ? c.past_fires.map((f: any) => `${f.name} ${f.year}`).join(', ') : c.past_fires ? 'none recorded' : '—'}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/50 border border-slate-800 text-slate-400">
                  {a.note} Sources: Bellwether {a.bellwether_window}; CAL FIRE FHSZ {a.fhsz_sources.join(', ')}; CAL FIRE FRAP perimeters;
                  ESA WorldCover 2021; Microsoft building footprints.
                </div>
              </div>
            )
          )}

          {!loading && data && tab === 'models' && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {data.models.map((m: any) => (
                  <div key={m.id} className="p-4 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-2">
                    <span className="text-[10px] font-bold px-2.5 py-1 rounded-full uppercase" style={{ backgroundColor: `${m.color}20`, color: m.color, border: `1px solid ${m.color}40` }}>{m.badge}</span>
                    <h3 className="text-base font-bold text-slate-100">{m.name}</h3>
                    <div className="text-[11px] text-orange-400">{m.type} · {m.developer}</div>
                    <p className="text-slate-300 bg-slate-950/50 p-2.5 rounded-xl border border-slate-800/60">{m.core_methodology}</p>
                    <div className="text-[11px] text-slate-400">Output: {m.output_metric} · Resolution: {m.spatial_resolution} · Updates: {m.update_cadence}</div>
                    <ul className="space-y-0.5">
                      {m.strengths.map((s: string) => <li key={s} className="text-emerald-300 flex gap-1.5"><CheckCircle2 className="w-3 h-3 mt-0.5 shrink-0" />{s}</li>)}
                      {m.limitations.map((s: string) => <li key={s} className="text-amber-300 flex gap-1.5"><AlertTriangle className="w-3 h-3 mt-0.5 shrink-0" />{s}</li>)}
                    </ul>
                    <div className="text-[11px] text-slate-500">In this app: {m.in_this_app}</div>
                  </div>
                ))}
              </div>
              <div className="overflow-x-auto rounded-2xl border border-slate-800">
                <table className="w-full text-left border-collapse">
                  <thead><tr className="bg-slate-950/80 text-slate-400 text-[10px] uppercase"><th className="p-3">Dimension</th><th className="p-3">Bellwether</th><th className="p-3">CAL FIRE FHSZ</th><th className="p-3">Insurance models</th></tr></thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {data.comparison_matrix.map((r: any) => (
                      <tr key={r.dimension}><td className="p-3 font-bold text-slate-200">{r.dimension}</td><td className="p-3 text-slate-300">{r.bellwether}</td><td className="p-3 text-slate-300">{r.calfire}</td><td className="p-3 text-slate-300">{r.insurance}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {!loading && data && tab === 'evidence' && (
            data.research_evidence?.available ? (
              <div className="space-y-3">
                <p className="text-slate-300">
                  Held-out results from this project's evaluation of vision-language models against CAL FIRE damage inspections
                  (fires split by event and time; numbers generated from the experiment outputs).
                </p>
                {data.research_evidence.items.map((it: any) => (
                  <div key={it.task} className="p-3.5 rounded-xl bg-slate-900/70 border border-slate-800">
                    <div className="font-bold text-slate-100">{it.task}</div>
                    <div className="text-orange-300 font-mono mt-1">{it.result}</div>
                    <div className="text-slate-400 mt-1">{it.meaning}</div>
                  </div>
                ))}
                <div className="text-[11px] text-slate-500">Source: {data.research_evidence.source}</div>
              </div>
            ) : <div className="text-slate-400">No research results available on this server.</div>
          )}

          {!loading && data && tab === 'guidance' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[['For fire marshals', data.guidance.for_fire_marshals], ['For homeowners', data.guidance.for_homeowners]].map(([t, items]: any) => (
                <div key={t} className="p-4 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-2">
                  <div className="text-sm font-bold text-orange-300">{t}</div>
                  {items.map((x: string) => <p key={x} className="text-slate-300">{x}</p>)}
                </div>
              ))}
              <a href="https://www.insurance.ca.gov/01-consumers/200-wrr/Safer-from-Wildfires.cfm" target="_blank" rel="noreferrer" className="text-orange-400 hover:text-orange-300 font-semibold">
                CDI Safer from Wildfires ↗
              </a>
            </div>
          )}
        </div>

        <div className="pt-3 border-t border-slate-800/80 flex justify-end shrink-0">
          <button onClick={onClose} className="px-4 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs">Close</button>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
      <div className="text-[11px] text-slate-400">{label}</div>
      <div className="text-xl font-black text-slate-100 mt-0.5">{value}</div>
      {sub && <div className="text-[10px] text-slate-500">{sub}</div>}
    </div>
  );
}
