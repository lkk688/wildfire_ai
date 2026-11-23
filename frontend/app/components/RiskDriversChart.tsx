'use client';

import React, { useEffect, useState } from 'react';
import { BarChart2, ChevronDown, ChevronUp } from 'lucide-react';
import { getJSON } from '../lib/api';

export default function RiskDriversChart({ region, raised = false }: { region: string; raised?: boolean }) {
  const [data, setData] = useState<any>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    setData(null);
    getJSON(`/api/risk-factors?region=${region}`)
      .then((d) => setData(d.factors))
      .catch(() => setData({ available: false, reason: 'Risk-factor data unavailable' }));
  }, [region]);

  if (!data) return null;

  const List = ({ items, color }: { items: any[]; color: string }) => (
    <div className="space-y-1.5">
      {items.slice(0, 4).map((it) => (
        <div key={it.feature}>
          <div className="flex justify-between text-[11px] text-slate-300">
            <span className="truncate pr-2">{it.feature}</span>
            <span className="font-mono text-slate-400" title="share of cells where it is in the top 5">{it.cells_in_top5_pct}% of cells</span>
          </div>
          <div className="w-full bg-slate-900/80 rounded-full h-1.5 overflow-hidden">
            <div className={`${color} h-1.5 rounded-full`} style={{ width: `${Math.min(it.weight_share_pct * 2.5, 100)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <div className={`absolute ${raised ? 'bottom-[104px]' : 'bottom-5'} left-16 w-96 glass-panel rounded-2xl p-4 shadow-2xl z-20 border border-slate-800`}>
      <button onClick={() => setOpen(!open)} className="w-full flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider">
        <span className="flex items-center gap-2"><BarChart2 className="w-4 h-4 text-orange-400" /> Bellwether risk factors in this region</span>
        {open ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
      </button>
      {open && (
        !data.available ? (
          <div className="mt-2 text-[11px] text-slate-500">{data.reason}</div>
        ) : (
          <div className="mt-3 space-y-3">
            <div>
              <div className="text-[10px] font-bold text-rose-300 uppercase mb-1">Raising the probability</div>
              <List items={data.contributing} color="bg-gradient-to-r from-orange-500 to-rose-500" />
            </div>
            <div>
              <div className="text-[10px] font-bold text-emerald-300 uppercase mb-1">Lowering the probability</div>
              <List items={data.mitigating} color="bg-gradient-to-r from-emerald-500 to-teal-400" />
            </div>
            <div className="text-[10px] text-slate-500">Bar: share of total attribution weight; right: share of cells where the factor is in the top 5. {data.model} model, partner GeoTIFF.</div>
          </div>
        )
      )}
    </div>
  );
}
