'use client';

import React, { useEffect, useState } from 'react';
import { Eye, Map, X, MousePointerClick, Layers, MapPin, Bot, ListChecks, ImageIcon, ChevronRight, Play } from 'lucide-react';
import { getJSON } from '../lib/api';

type Mode = 'map' | 'vision';

function Step({ n, icon, children }: { n: number; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <li className="flex gap-2.5 items-start">
      <span className="w-5 h-5 shrink-0 rounded-full bg-slate-800 border border-slate-600 text-[11px] font-bold text-slate-200 flex items-center justify-center">{n}</span>
      <span className="text-[13px] text-slate-300 leading-snug flex gap-1.5">
        <span className="mt-0.5 text-slate-400">{icon}</span>
        <span>{children}</span>
      </span>
    </li>
  );
}

// Best score per task, computed from the stored per-case outputs (/api/heldout/summary); nothing hard-coded.
function useFindings(open: boolean) {
  const [f, setF] = useState<{ t3: number; t3Label: string; t2: number; t2Label: string } | null>(null);
  useEffect(() => {
    if (!open || f) return;
    getJSON('/api/heldout/summary')
      .then((s) => {
        type M = { label: string; input: string; auroc?: number | null; pair_accuracy?: number | null };
        const best = (ms: M[], k: 'auroc' | 'pair_accuracy') =>
          ms.reduce<M | null>((a, m) => ((m[k] ?? 0) > (a?.[k] ?? 0) ? m : a), null);
        const a = best(s.t3.models as M[], 'auroc');
        const b = best((s.t2.models as M[]).filter((m) => m.input !== 'metadata only (no image)'), 'pair_accuracy');
        if (a?.auroc != null && b?.pair_accuracy != null)
          setF({ t3: a.auroc, t3Label: `${a.label}, ${a.input}`, t2: b.pair_accuracy, t2Label: b.label });
      })
      .catch(() => setF(null));
  }, [open, f]);
  return f;
}

export default function GuideModal({ open, onClose, onGo }: { open: boolean; onClose: () => void; onGo: (m: Mode) => void }) {
  const f = useFindings(open);
  useEffect(() => {
    if (!open) return;
    const k = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [open, onClose]);
  if (!open) return null;

  return (
    <div id="guide-modal" className="fixed inset-0 z-[3000] bg-black/70 backdrop-blur-sm flex items-center justify-center p-4" onClick={onClose}>
      <div className="w-full max-w-5xl max-h-[92vh] overflow-y-auto rounded-2xl bg-slate-900 border border-slate-700 shadow-2xl p-6" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-xl font-extrabold text-white">Wildfire AI: two ways to look at wildfire risk</h2>
            <p className="text-sm text-slate-400 mt-1 max-w-3xl">
              Everything on this site is real: live public data feeds, CAL FIRE damage inspections, and the stored answers of
              AI models. Nothing is simulated. Pick a view to start; you can reopen this guide from the <b>Guide</b> button at the top.
            </p>
          </div>
          <button onClick={onClose} aria-label="Close guide" className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 cursor-pointer">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-5">
          <section className="rounded-2xl border border-orange-500/40 bg-gradient-to-b from-orange-500/10 to-slate-900 p-5 flex flex-col">
            <div className="flex items-center gap-2">
              <div className="p-2 rounded-xl bg-orange-500/20 text-orange-400 border border-orange-500/30"><Map className="w-5 h-5" /></div>
              <div>
                <h3 className="font-bold text-white">Risk map (live data)</h3>
                <p className="text-xs text-orange-200/80">Where could fire start or spread, and what are conditions right now?</p>
              </div>
            </div>
            <p className="text-[13px] text-slate-300 mt-3">
              Wildfire hazard and today&apos;s fire conditions for three Bay Area communities at the edge of wildland.
            </p>
            <ol className="mt-3 space-y-2">
              <Step n={1} icon={<MapPin className="w-3.5 h-3.5" />}>Pick a <b>region</b> in the left panel.</Step>
              <Step n={2} icon={<Layers className="w-3.5 h-3.5" />}>
                Turn <b>layers</b> on and off: hazard maps, fires, satellite heat detections, fuel dryness, wind. The <b>legend</b> on the map explains every colour.
              </Step>
              <Step n={3} icon={<MousePointerClick className="w-3.5 h-3.5" />}><b>Click any spot</b> on the map to see everything known about it.</Step>
              <Step n={4} icon={<Bot className="w-3.5 h-3.5" />}>Ask the <b>AI copilot</b> (bottom right); it answers from the same live data.</Step>
            </ol>
            <div className="flex-1" />
            <button id="guide-go-map" onClick={() => onGo('map')} className="mt-4 w-full py-2.5 rounded-xl bg-gradient-to-r from-orange-500 to-rose-600 text-white text-sm font-bold flex items-center justify-center gap-1.5 cursor-pointer">
              Open the risk map <ChevronRight className="w-4 h-4" />
            </button>
          </section>

          <section className="rounded-2xl border border-purple-500/40 bg-gradient-to-b from-purple-500/10 to-slate-900 p-5 flex flex-col">
            <div className="flex items-center gap-2">
              <div className="p-2 rounded-xl bg-purple-500/20 text-purple-300 border border-purple-500/30"><Eye className="w-5 h-5" /></div>
              <div>
                <h3 className="font-bold text-white">VLM evaluation (held-out fires)</h3>
                <p className="text-xs text-purple-200/80">Can AI tell from aerial photos which homes burn?</p>
              </div>
            </div>
            <p className="text-[13px] text-slate-300 mt-3">
              Vision-language models answer questions about real homes in 2024–25 California fires they were never trained on,
              checked against CAL FIRE&apos;s inspection of each home.
            </p>
            <ol className="mt-3 space-y-2">
              <Step n={1} icon={<ListChecks className="w-3.5 h-3.5" />}>
                Choose a question: <b>after the fire</b> (was this home destroyed?) or <b>before the fire</b> (which of two neighbours burned?).
              </Step>
              <Step n={2} icon={<ImageIcon className="w-3.5 h-3.5" />}>
                Look at the photos, then at each model&apos;s answer: <span className="text-emerald-300 font-semibold">green = right</span>, <span className="text-rose-300 font-semibold">red = wrong</span>.
              </Step>
              <Step n={3} icon={<ChevronRight className="w-3.5 h-3.5" />}>Browse cases with the arrows, or show only the ones a model got wrong.</Step>
              <Step n={4} icon={<Play className="w-3.5 h-3.5" />}>Press <b>Run</b> to ask a model the same question live.</Step>
            </ol>
            {f && (
              <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                <div className="rounded-xl bg-slate-950/70 border border-emerald-500/30 p-2.5">
                  <div className="text-slate-400">After the fire</div>
                  <div className="text-lg font-extrabold text-emerald-300">{f.t3.toFixed(2)} <span className="text-xs font-semibold text-slate-400">AUROC</span></div>
                  <div className="text-slate-400 leading-tight">Damage is easy to see. (1 = perfect, 0.5 = coin flip)</div>
                </div>
                <div className="rounded-xl bg-slate-950/70 border border-amber-500/30 p-2.5">
                  <div className="text-slate-400">Before the fire</div>
                  <div className="text-lg font-extrabold text-amber-300">{Math.round(f.t2 * 100)}% <span className="text-xs font-semibold text-slate-400">right</span></div>
                  <div className="text-slate-400 leading-tight">Best model picking the home that burned; a coin flip gets 50%.</div>
                </div>
              </div>
            )}
            <div className="flex-1" />
            <button id="guide-go-vision" onClick={() => onGo('vision')} className="mt-4 w-full py-2.5 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 text-white text-sm font-bold flex items-center justify-center gap-1.5 cursor-pointer">
              Open the VLM evaluation <ChevronRight className="w-4 h-4" />
            </button>
          </section>
        </div>
      </div>
    </div>
  );
}
