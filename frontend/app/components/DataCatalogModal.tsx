'use client';

import { useGeoCatalog } from '../lib/geodata';
import React, { useEffect, useState } from 'react';
import { BookOpen, X, CheckCircle2, AlertTriangle, Loader2, RefreshCw } from 'lucide-react';
import { getJSON } from '../lib/api';

interface DataCatalogModalProps {
  isOpen: boolean;
  onClose: () => void;
  region?: string;
}

export default function DataCatalogModal({ isOpen, onClose, region = 'san_bruno' }: DataCatalogModalProps) {
  const [sources, setSources] = useState<any[]>([]);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const check = () => {
    setChecking(true);
    setError(null);
    getJSON(`/api/data-sources/status?region=${region}`)
      .then((d) => setSources(d.sources))
      .catch((e) => setError(String(e.message || e)))
      .finally(() => setChecking(false));
  };

  useEffect(() => {
    if (!isOpen) return;
    getJSON('/api/data-catalog').then((d) => setSources(d.catalog)).catch(() => undefined);
    check();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, region]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[2000] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md">
      <div className="glass-panel w-full max-w-4xl max-h-[85vh] rounded-3xl p-6 shadow-2xl border border-slate-700/80 flex flex-col overflow-hidden">
        <div className="flex items-center justify-between pb-4 border-b border-slate-800 shrink-0">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-orange-500/20 text-orange-400 border border-orange-500/30"><BookOpen className="w-6 h-6" /></div>
            <div>
              <h2 className="text-lg font-bold text-slate-100">Data sources & live status</h2>
              <p className="text-xs text-slate-400">Every layer comes from the source below. The status is checked now by fetching from each source.</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={check} disabled={checking} className="px-3 py-2 rounded-xl bg-slate-800 border border-slate-700 text-xs text-slate-200 flex items-center gap-1.5 disabled:opacity-50">
              <RefreshCw className={`w-3.5 h-3.5 ${checking ? 'animate-spin' : ''}`} /> Re-check
            </button>
            <button onClick={onClose} className="p-2 rounded-xl text-slate-400 hover:text-slate-100 hover:bg-slate-800"><X className="w-5 h-5" /></button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto py-4 space-y-3 pr-2">
          {error && <div className="text-xs text-rose-400">Status check failed: {error}</div>}
          {sources.map((s) => {
            const st = s.status;
            return (
              <div key={s.id} className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800/90 space-y-2">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h3 className="font-bold text-sm text-slate-100">{s.name}</h3>
                    <div className="text-[11px] text-slate-400 mt-0.5">
                      <span className="text-orange-400 font-semibold">{s.provider}</span> · {s.resolution} · {s.cadence}
                    </div>
                  </div>
                  {!st ? (
                    <span className="px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 text-[11px] flex items-center gap-1"><Loader2 className="w-3 h-3 animate-spin" /> checking</span>
                  ) : st.ok ? (
                    <span className="px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[11px] font-semibold flex items-center gap-1 shrink-0">
                      <CheckCircle2 className="w-3 h-3" /> reachable{st.latency_s !== undefined ? ` · ${st.latency_s}s` : ''}
                    </span>
                  ) : (
                    <span className="px-2.5 py-1 rounded-full bg-rose-500/10 border border-rose-500/30 text-rose-300 text-[11px] font-semibold flex items-center gap-1 shrink-0">
                      <AlertTriangle className="w-3 h-3" /> unavailable
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">{s.description}</p>
                <div className="text-[11px] text-slate-500">Access: {s.access}</div>
                {st && (st.ok ? (
                  <div className="text-[11px] text-slate-400">Sample from this check: <span className="font-mono text-slate-300">{String(st.sample)}</span></div>
                ) : st.error ? (
                  <div className="text-[11px] text-rose-400">{st.error}</div>
                ) : null)}
              </div>
            );
          })}
          <GeoCacheTable />
        </div>

        <div className="pt-3 border-t border-slate-800 shrink-0 flex items-center justify-between text-xs text-slate-400">
          <span>Bellwether data are shared with the pilot partners and are not redistributed.</span>
          <button onClick={onClose} className="px-5 py-2 rounded-xl bg-orange-500 text-white font-bold hover:bg-orange-600">Close</button>
        </div>
      </div>
    </div>
  );
}

// Layers kept in the local California cache (backend/geodata): provenance for every layer the map and agent read.
function GeoCacheTable() {
  const { layers, error } = useGeoCatalog();
  if (error) return <div className="text-xs text-rose-400">California data cache catalog unavailable: {error}</div>;
  if (!layers.length) return null;
  return (
    <div id="geo-cache-sources" className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800/90 space-y-2">
      <h3 className="font-bold text-sm text-slate-100">California data cache ({layers.length} layers)</h3>
      <p className="text-xs text-slate-300 leading-relaxed">
        Public statewide layers stored on this server, in a tile pyramid for rasters and GeoParquet for vectors. Layers marked
        &quot;statewide&quot; are downloaded in full; the others are fetched from the source the first time a place is viewed and kept.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-[11px] text-left">
          <thead className="text-slate-400">
            <tr><th className="py-1 pr-2">Layer</th><th className="pr-2">Source</th><th className="pr-2">Date</th><th className="pr-2">Res.</th><th>Cache</th></tr>
          </thead>
          <tbody className="text-slate-300">
            {layers.map((l) => (
              <tr key={l.id} className="border-t border-slate-800/80 align-top">
                <td className="py-1 pr-2">{l.url ? <a href={l.url} target="_blank" rel="noreferrer" className="underline decoration-slate-600">{l.title}</a> : l.title}</td>
                <td className="pr-2 text-slate-400">{l.source}</td>
                <td className="pr-2 text-slate-400">{l.date}</td>
                <td className="pr-2 text-slate-400">{l.type === 'raster' ? `${l.res} m` : l.geom}</td>
                <td className="text-slate-400">{l.type === 'vector' ? (l.available ? 'statewide' : 'not built') : l.prefetch ? 'statewide' : 'on view'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
