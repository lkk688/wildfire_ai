'use client';

// VLM evaluation view: held-out test fires (2024-25).
// Every value comes from a stored output of this project's evaluation (output/demo/heldout_cases.json, written by
// scripts/demo/export_heldout_cases.py from output/experiments/*), CAL FIRE DINS labels and Esri Wayback crops with
// verified capture dates. Metrics are computed by the backend from the same per-case outputs that are listed here.
// The "Run now" panel re-sends a case to a model with the exact research prompt and images.

import { API_BASE, getJSON } from '../lib/api';

import React, { useEffect, useMemo, useState } from 'react';
import {
  Eye,
  Layers,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  XCircle,
  Play,
  Crosshair,
  MapPin,
  FileText,
  EyeOff,
  Info,
} from 'lucide-react';

// ---------------------------------------------------------------- types

type Task = 't3' | 't2';

interface ModelInfo {
  key: string;
  label: string;
  input: string;
  kind: string;
  source: string;
  n: number;
  n_unparsed: number;
  auroc?: number | null;
  accuracy_at_0_5?: number | null;
  pair_accuracy?: number | null;
}

interface Summary {
  source: string;
  note?: string;
  t3: { models: ModelInfo[]; n: number; split_file: string; events: [string, number][] };
  t2: { models: ModelInfo[]; n: number; split_file: string; events: [string, number][] };
  live_models: { id: string; label: string; configured: boolean }[];
}

interface Building {
  global_id: string;
  structure_type: string | null;
  year_built: number | string | null;
  dins_damage: string;
  label: 'Destroyed' | 'Survived';
  lat3: number | null;
  lon3: number | null;
  pre: { capture_date: string | null; provider: string | null; days_before_fire: number | null; release: string | null };
  post?: { capture_date: string | null; provider: string | null; days_after_fire: number | null; release: string | null };
}

interface FireInfo {
  event: string;
  fire: string;
  county: string;
  fire_start: string;
}

interface T3Pred {
  p: number;
  parsed: boolean;
  evidence?: string[];
  ts?: string;
  model_id?: string;
}

type T3Case = FireInfo & Building & { preds: Record<string, T3Pred> };

interface T2Pred {
  p_destroyed_one: number;
  parsed: boolean;
  differences?: string[];
  ts?: string;
}

type T2Case = FireInfo & {
  pair_id: string;
  dist_m: number;
  destroyed_is: 'A' | 'B';
  destroyed: Building;
  survived: Building;
  preds: Record<string, T2Pred>;
};

interface LiveResult {
  success: boolean;
  error?: string | null;
  model_id?: string;
  latency_sec?: number;
  prompt_version?: string;
  prompt?: string;
  parsed?: boolean;
  p?: number;
  evidence?: string[];
  p_destroyed_one?: number;
  differences?: string[];
}

interface VisionBenchmarkViewProps {
  onSwitchToMap: () => void;
}

// ---------------------------------------------------------------- helpers

const REF_MODEL: Record<Task, string> = { t3: 'sonnet_post', t2: 'sonnet_image' };

function pct(v: number | null | undefined, digits = 0): string {
  return v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`;
}

function f3(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : v.toFixed(3).replace(/^0/, '');
}

function SourceLine({ children }: { children: React.ReactNode }) {
  return <div className="text-[10px] text-slate-500 leading-snug">Source: {children}</div>;
}

function Fact({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="bg-slate-950/60 px-2.5 py-1.5 rounded-lg border border-slate-800/80 min-w-0">
      <span className="text-slate-400 block text-[10px]">{label}</span>
      <span className="font-semibold text-xs text-slate-200 break-words">{value}</span>
    </div>
  );
}

// Probability bar: one hue, 0.5 tick marks chance / decision threshold.
function ProbBar({ p, tone }: { p: number; tone: 'hit' | 'miss' | 'neutral' }) {
  const fill = tone === 'hit' ? 'bg-emerald-400' : tone === 'miss' ? 'bg-rose-400' : 'bg-slate-400';
  return (
    <div className="relative h-2 w-full rounded-full bg-slate-800">
      <div className={`absolute left-0 top-0 h-2 rounded-full ${fill}`} style={{ width: `${Math.max(2, p * 100)}%` }} />
      <div className="absolute top-[-2px] h-3 w-px bg-slate-300/70" style={{ left: '50%' }} />
    </div>
  );
}

function CaptureBadge({ kind, date, extra }: { kind: 'pre' | 'post'; date: string | null; extra?: string }) {
  return (
    <span
      className={`px-2 py-0.5 rounded-md text-[11px] font-bold backdrop-blur-md border ${
        kind === 'pre' ? 'bg-emerald-950/90 text-emerald-300 border-emerald-500/40' : 'bg-rose-950/90 text-rose-300 border-rose-500/40'
      }`}
    >
      {kind === 'pre' ? 'Pre-fire' : 'Post-fire'} · captured {date || '—'}
      {extra ? ` · ${extra}` : ''}
    </span>
  );
}

function Crop({ gid, epoch, badge, reticle, label }: { gid: string; epoch: 'pre' | 'post'; badge: React.ReactNode; reticle: boolean; label?: React.ReactNode }) {
  return (
    <div className="relative w-full aspect-square bg-black overflow-hidden rounded-xl border border-slate-800">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={`${API_BASE}/api/heldout/image/${gid}?epoch=${epoch}`} alt={`${epoch}-fire aerial crop`} className="w-full h-full object-cover" />
      {reticle && (
        <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
          <div className="w-12 h-12 rounded-full border-2 border-amber-400/90" />
        </div>
      )}
      <div className="absolute top-2 left-2 flex flex-wrap gap-1.5">{badge}</div>
      {label && <div className="absolute bottom-2 left-2 right-2">{label}</div>}
    </div>
  );
}

// ---------------------------------------------------------------- component

export default function VisionBenchmarkView({ onSwitchToMap }: VisionBenchmarkViewProps) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [t3, setT3] = useState<T3Case[]>([]);
  const [t2, setT2] = useState<T2Case[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [task, setTask] = useState<Task>('t3');
  const [eventFilter, setEventFilter] = useState<string>('all');
  const [labelFilter, setLabelFilter] = useState<'all' | 'Destroyed' | 'Survived'>('all');
  const [onlyMisses, setOnlyMisses] = useState<boolean>(false);
  const [index, setIndex] = useState<number>(0);
  const [reveal, setReveal] = useState<boolean>(false);
  const [reticle, setReticle] = useState<boolean>(true);
  const [showSources, setShowSources] = useState<boolean>(false);
  const [liveModel, setLiveModel] = useState<string>('claude-sonnet-5-5');
  const [liveCondition, setLiveCondition] = useState<string>('postfire_image');
  const [liveLoading, setLiveLoading] = useState<boolean>(false);
  const [live, setLive] = useState<LiveResult | null>(null);
  const [showPrompt, setShowPrompt] = useState<boolean>(false);
  const [pendingCase, setPendingCase] = useState<string | null>(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const t = params.get('task');
    if (t === 't2' || t === 't3') setTask(t);
    setPendingCase(params.get('case'));
    Promise.all([getJSON('/api/heldout/summary'), getJSON('/api/heldout/cases?task=t3'), getJSON('/api/heldout/cases?task=t2')])
      .then(([s, a, b]) => {
        setSummary(s);
        setT3(a.cases || []);
        setT2(b.cases || []);
      })
      .catch((e) => setLoadError(`Could not load held-out cases from ${API_BASE}: ${String(e)}`));
  }, []);

  const models = summary ? summary[task].models : [];
  const ref = REF_MODEL[task];

  const filtered = useMemo(() => {
    if (task === 't3') {
      return t3.filter((c) => {
        if (eventFilter !== 'all' && c.event !== eventFilter) return false;
        if (labelFilter !== 'all' && c.label !== labelFilter) return false;
        if (onlyMisses) {
          const p = c.preds[ref];
          if (!p || (p.p >= 0.5) === (c.label === 'Destroyed')) return false;
        }
        return true;
      });
    }
    return t2.filter((c) => {
      if (eventFilter !== 'all' && c.event !== eventFilter) return false;
      if (onlyMisses) {
        const p = c.preds[ref];
        if (!p || p.p_destroyed_one > 0.5) return false;
      }
      return true;
    });
  }, [task, t3, t2, eventFilter, labelFilter, onlyMisses, ref]);

  // deep link: ?mode=vision&task=t2&case=p0012
  useEffect(() => {
    if (!pendingCase || filtered.length === 0) return;
    const i = (filtered as (T3Case | T2Case)[]).findIndex((c) => ('pair_id' in c ? c.pair_id : c.global_id) === pendingCase);
    if (i >= 0) setIndex(i);
    setPendingCase(null);
  }, [pendingCase, filtered]);

  const resetCase = () => {
    setLive(null);
    setShowPrompt(false);
    setReveal(false);
  };
  const switchTask = (t: Task) => {
    setTask(t);
    setEventFilter('all');
    setLabelFilter('all');
    setOnlyMisses(false);
    setIndex(0);
    setLiveCondition(t === 't3' ? 'postfire_image' : 'image');
    resetCase();
  };
  const go = (d: number) => {
    if (filtered.length === 0) return;
    setIndex((i) => (i + d + filtered.length) % filtered.length);
    resetCase();
  };

  const cur = filtered[Math.min(index, Math.max(filtered.length - 1, 0))] as T3Case | T2Case | undefined;
  const liveConfigured = summary?.live_models.filter((m) => m.configured) || [];

  const runLive = async () => {
    if (!cur) return;
    setLiveLoading(true);
    setLive(null);
    try {
      const body = { task, case_id: task === 't3' ? (cur as T3Case).global_id : (cur as T2Case).pair_id, model: liveModel, condition: liveCondition };
      const res = await fetch(`${API_BASE}/api/heldout/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const data = await res.json();
      setLive(res.ok ? data : { success: false, error: data?.detail || `HTTP ${res.status}` });
    } catch (e) {
      setLive({ success: false, error: String(e) });
    } finally {
      setLiveLoading(false);
    }
  };

  const metricOf = (m: ModelInfo) => (task === 't3' ? m.auroc : m.pair_accuracy) ?? 0;
  const sortedModels = [...models].sort((a, b) => metricOf(b) - metricOf(a));
  const events = summary ? summary[task].events : [];

  // ------------------------------------------------------------ render
  return (
    <div className="flex-1 h-full flex flex-col bg-slate-950 text-slate-100 overflow-y-auto">
      <div className="h-16 shrink-0 bg-slate-950" />

      <header className="border-b border-slate-800 bg-slate-900/80 px-6 py-3 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center">
            <Eye className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-base font-bold text-white tracking-wide">VLM evaluation · held-out fires (2024–25)</h1>
              <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">Labels: CAL FIRE DINS</span>
              <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-sky-500/15 text-sky-300 border border-sky-500/30">Imagery: Esri Wayback, capture dates verified</span>
            </div>
            <p className="text-xs text-slate-400">
              Every building below is from a fire the models were never tuned on. Predictions are the stored outputs of the paper&apos;s runs; you can re-run any case now.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={onSwitchToMap} className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 cursor-pointer">
            <Layers className="w-3.5 h-3.5 text-orange-400" />
            <span>Risk map</span>
          </button>
          <button onClick={() => setShowSources(true)} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/80 cursor-pointer">
            <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
            <span>How this is built</span>
          </button>
        </div>
      </header>

      {/* Task tabs */}
      <nav className="px-6 pt-3 flex gap-2 flex-wrap">
        {(
          [
            { k: 't3', t: 'Post-fire: was this home destroyed?', s: `T3 · ${summary?.t3.n ?? '…'} buildings` },
            { k: 't2', t: 'Pre-fire: which of two neighbours burned?', s: `T2 · ${summary?.t2.n ?? '…'} same-fire pairs` },
          ] as const
        ).map((x) => (
          <button
            key={x.k}
            id={`task-${x.k}`}
            onClick={() => switchTask(x.k)}
            className={`px-4 py-2 rounded-xl text-left border cursor-pointer ${
              task === x.k ? 'bg-purple-600/25 border-purple-500/60 text-white' : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <div className="text-sm font-bold">{x.t}</div>
            <div className="text-[11px] opacity-80">{x.s}</div>
          </button>
        ))}
      </nav>

      {/* Scoreboard computed from the per-case outputs */}
      <section className="px-6 py-3">
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
          <div className="flex items-baseline justify-between flex-wrap gap-2 mb-2">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
              {task === 't3' ? 'AUROC on the 2024–25 test fires (0.5 = chance)' : 'Pair accuracy on 403 same-fire pairs (0.5 = chance)'}
            </span>
            <span className="text-[11px] text-slate-400">
              {task === 't3'
                ? 'Damage is easy to see after the fire: trained 4B/9B models match the best zero-shot model.'
                : 'Before the fire, no model gets much above chance once exposure is controlled; metadata alone gets .578.'}
            </span>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-x-8 gap-y-1">
            {sortedModels.map((m) => {
              const v = metricOf(m);
              return (
                <div key={m.key} className="grid grid-cols-[minmax(0,15rem)_1fr_3rem] items-center gap-3 py-0.5" title={`${m.label} · ${m.input} · n=${m.n}${m.n_unparsed ? `, ${m.n_unparsed} unparsed counted as 0.5` : ''} · ${m.source}`}>
                  <div className="min-w-0">
                    <div className="text-xs text-slate-200 truncate">{m.label}</div>
                    <div className="text-[10px] text-slate-500 truncate">
                      {m.input} · {m.kind} · n={m.n}
                    </div>
                  </div>
                  <div className="relative h-2.5 rounded-full bg-slate-800">
                    <div className="absolute left-0 top-0 h-2.5 rounded-full bg-indigo-400" style={{ width: `${Math.max(0, (v - 0.4) / 0.6) * 100}%` }} />
                    <div className="absolute top-[-3px] h-4 w-px bg-slate-300/70" style={{ left: `${((0.5 - 0.4) / 0.6) * 100}%` }} />
                  </div>
                  <div className="text-xs font-mono text-slate-100 text-right">{f3(v)}</div>
                </div>
              );
            })}
          </div>
          <div className="mt-2 flex justify-between flex-wrap gap-2">
            <SourceLine>
              per-case outputs in {summary?.source || 'output/demo/heldout_cases.json'}; split {summary?.[task].split_file}. Axis starts at 0.4; tick = 0.5. Unparsed answers count as 0.5.
            </SourceLine>
          </div>
        </div>
      </section>

      {/* Filters and navigation */}
      <section className="px-6 pb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          <select
            id="event-filter"
            value={eventFilter}
            onChange={(e) => {
              setEventFilter(e.target.value);
              setIndex(0);
              resetCase();
            }}
            className="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1.5 max-w-[18rem]"
          >
            <option value="all">All fires ({summary?.[task].n ?? 0})</option>
            {events.map(([e, n]) => (
              <option key={e} value={e}>
                {e.replace(' | ', ' Fire · ')} ({n})
              </option>
            ))}
          </select>
          {task === 't3' &&
            (['all', 'Destroyed', 'Survived'] as const).map((l) => (
              <button
                key={l}
                onClick={() => {
                  setLabelFilter(l);
                  setIndex(0);
                  resetCase();
                }}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium cursor-pointer ${labelFilter === l ? 'bg-orange-500 text-white' : 'bg-slate-800/60 text-slate-300 border border-slate-700/60'}`}
              >
                {l === 'all' ? 'All labels' : l}
              </button>
            ))}
          <label className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer ml-1">
            <input
              id="only-misses"
              type="checkbox"
              checked={onlyMisses}
              onChange={(e) => {
                setOnlyMisses(e.target.checked);
                setIndex(0);
                resetCase();
              }}
              className="accent-orange-500"
            />
            Only cases Claude Sonnet got wrong
          </label>
        </div>
        <div className="flex items-center gap-2">
          <button id="case-prev" onClick={() => go(-1)} className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700" title="Previous case">
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="text-xs text-slate-400 font-mono">
            {filtered.length ? Math.min(index + 1, filtered.length) : 0} of {filtered.length}
          </span>
          <button id="case-next" onClick={() => go(1)} className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700" title="Next case">
            <ChevronRight className="w-4 h-4" />
          </button>
          <button
            onClick={() => setReticle(!reticle)}
            className={`px-2 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1 border cursor-pointer ${reticle ? 'bg-amber-500/20 text-amber-300 border-amber-500/50' : 'bg-slate-800 text-slate-400 border-slate-700'}`}
            title="Mark the DINS point (image centre)"
          >
            <Crosshair className="w-3.5 h-3.5" /> DINS point
          </button>
        </div>
      </section>

      {loadError ? (
        <div className="flex-1 flex items-center justify-center text-rose-300 text-sm py-20 px-6 text-center">{loadError}</div>
      ) : !summary ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 text-slate-400 py-20">
          <div className="w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm">Loading held-out cases…</span>
        </div>
      ) : !cur ? (
        <div className="flex-1 flex items-center justify-center text-slate-500 text-sm py-20">No cases match the current filter.</div>
      ) : (
        <main className="px-6 pb-28 grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
          {/* Left: imagery */}
          <div className="xl:col-span-7 space-y-3">
            {task === 't3' ? (
              <T3Images c={cur as T3Case} reticle={reticle} />
            ) : (
              <T2Images c={cur as T2Case} reticle={reticle} reveal={reveal} />
            )}
            <CaseFacts task={task} c={cur} reveal={reveal} setReveal={setReveal} />
          </div>

          {/* Right: stored outputs + live re-run */}
          <div className="xl:col-span-5 space-y-3">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-2.5">
              <div className="flex items-center justify-between gap-2">
                <h3 className="text-sm font-bold text-white">Stored model outputs for this case</h3>
                <span className="text-[10px] text-slate-500">{task === 't3' ? 'p(destroyed)' : task === 't2' && reveal ? 'p(the destroyed one)' : 'choice A / B'}</span>
              </div>
              {task === 't3' ? <T3Preds c={cur as T3Case} models={models} /> : <T2Preds c={cur as T2Case} models={models} reveal={reveal} />}
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-2.5">
              <div className="flex items-center gap-2">
                <Play className="w-4 h-4 text-emerald-400" />
                <div>
                  <span className="text-xs font-bold text-white block">Run this case now</span>
                  <span className="text-[11px] text-slate-400">Same research prompt and the same verified images as the stored run.</span>
                </div>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <select id="live-model" value={liveModel} onChange={(e) => setLiveModel(e.target.value)} className="flex-1 min-w-[9rem] bg-slate-800 border border-slate-700 text-xs rounded-lg px-2.5 py-1.5 text-slate-200">
                  {summary.live_models.map((m) => (
                    <option key={m.id} value={m.id} disabled={!m.configured}>
                      {m.label}
                      {m.configured ? '' : ' (not configured on backend)'}
                    </option>
                  ))}
                </select>
                <select value={liveCondition} onChange={(e) => setLiveCondition(e.target.value)} className="flex-1 min-w-[9rem] bg-slate-800 border border-slate-700 text-xs rounded-lg px-2.5 py-1.5 text-slate-200">
                  {task === 't3' ? (
                    <>
                      <option value="postfire_image">Post-fire image</option>
                      <option value="prepost_image">Pre + post images</option>
                    </>
                  ) : (
                    <>
                      <option value="image">Two pre-fire images</option>
                      <option value="meta">Metadata only</option>
                    </>
                  )}
                </select>
                <button
                  id="live-run"
                  onClick={runLive}
                  disabled={liveLoading || liveConfigured.length === 0}
                  className="px-4 py-1.5 rounded-lg text-xs font-bold bg-emerald-600 hover:bg-emerald-700 text-white disabled:opacity-50 flex items-center gap-1.5 cursor-pointer"
                >
                  {liveLoading ? <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                  {liveLoading ? 'Running…' : 'Run'}
                </button>
              </div>
              {liveConfigured.length === 0 && <div className="text-[11px] text-amber-300">No model API key is configured on the backend.</div>}
              {live && <LiveCard task={task} live={live} c={cur} reveal={reveal} showPrompt={showPrompt} setShowPrompt={setShowPrompt} />}
            </div>
          </div>
        </main>
      )}

      {showSources && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setShowSources(false)}>
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-3xl w-full p-6 space-y-3 max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2 text-white font-bold text-base">
                <BookOpen className="w-5 h-5 text-cyan-400" />
                <span>How this view is built</span>
              </div>
              <button onClick={() => setShowSources(false)} className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white cursor-pointer">
                ✕
              </button>
            </div>
            {[
              ['Labels', 'CAL FIRE Damage Inspection (DINS) records. Destroyed = Destroyed (>50%) or Major (26-50%); otherwise survived.'],
              ['Splits', 'By fire event and forward in time: models are trained or tuned on 2015-23 fires and tested on 2024-25 fires, with a 100 m buffer between splits.'],
              ['Imagery', 'Esri World Imagery Wayback crops (zoom 18, 512 px, ~150 m across). The capture date at each building is read from the Wayback metadata. Pre-fire = newest capture before the fire start; post-fire = earliest capture within 365 days after. Unverifiable crops are dropped, never replaced.'],
              ['T3 (post-fire)', 'The model sees the post-fire crop (or pre + post) and gives p(destroyed). Zero-shot runs: scripts/eval/run_vlm_conditions.py (Claude on a fixed 400-building subset). Qwen3.5 LoRA: scripts/train/sft_postfire_qwen.py. DINOv2 probe: frozen encoder + logistic regression.'],
              ['T2 (pre-fire pairs)', 'Two homes 15-100 m apart in the same fire, exactly one destroyed. Same fire, weather and timing, so the pair isolates what is visible on the building and yard. scripts/eval/run_vlm_pairs.py; the A/B order is seeded and shown here exactly as the model saw it.'],
              ['Privacy', 'Addresses, APNs and assessed values are not sent to the browser. Coordinates are rounded to ~100 m.'],
            ].map(([t, d]) => (
              <div key={t} className="bg-slate-950/80 p-3 rounded-xl border border-slate-800 text-xs text-slate-300">
                <span className="font-bold text-slate-100 block mb-1">{t}</span>
                <p>{d}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- sub-views

function T3Images({ c, reticle }: { c: T3Case; reticle: boolean }) {
  return (
    <div className="grid grid-cols-2 gap-3">
      <Crop gid={c.global_id} epoch="pre" reticle={reticle} badge={<CaptureBadge kind="pre" date={c.pre.capture_date} />} />
      <Crop
        gid={c.global_id}
        epoch="post"
        reticle={reticle}
        badge={<CaptureBadge kind="post" date={c.post?.capture_date ?? null} extra={c.post?.days_after_fire != null ? `${c.post.days_after_fire} d after start` : undefined} />}
      />
    </div>
  );
}

function T2Images({ c, reticle, reveal }: { c: T2Case; reticle: boolean; reveal: boolean }) {
  const a = c.destroyed_is === 'A' ? c.destroyed : c.survived;
  const b = c.destroyed_is === 'A' ? c.survived : c.destroyed;
  const tag = (x: Building, name: string) => (
    <div className="flex items-center justify-between">
      <span className="px-2 py-0.5 rounded-md text-xs font-bold bg-slate-950/90 text-white border border-slate-600">Building {name}</span>
      {reveal && (
        <span className={`px-2 py-0.5 rounded-md text-xs font-bold border ${x.label === 'Destroyed' ? 'bg-rose-950/90 text-rose-200 border-rose-500/50' : 'bg-emerald-950/90 text-emerald-200 border-emerald-500/50'}`}>
          DINS: {x.dins_damage}
        </span>
      )}
    </div>
  );
  return (
    <div className="grid grid-cols-2 gap-3">
      <Crop gid={a.global_id} epoch="pre" reticle={reticle} badge={<CaptureBadge kind="pre" date={a.pre.capture_date} />} label={tag(a, 'A')} />
      <Crop gid={b.global_id} epoch="pre" reticle={reticle} badge={<CaptureBadge kind="pre" date={b.pre.capture_date} />} label={tag(b, 'B')} />
    </div>
  );
}

function CaseFacts({ task, c, reveal, setReveal }: { task: Task; c: T3Case | T2Case; reveal: boolean; setReveal: (v: boolean) => void }) {
  const lat = task === 't3' ? (c as T3Case).lat3 : (c as T2Case).destroyed.lat3;
  const lon = task === 't3' ? (c as T3Case).lon3 : (c as T2Case).destroyed.lon3;
  const mapHref = lat !== null && lon !== null ? `/?lat=${lat}&lng=${lon}&z=14&layers=perimeters,buildings,hotspots&bw=none` : null;
  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 space-y-2.5">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div>
          <h3 className="text-sm font-bold text-white">
            {c.fire} Fire · {c.county} County
          </h3>
          <span className="text-[11px] text-slate-400">started {c.fire_start}</span>
        </div>
        {task === 't3' ? (
          <span className={`px-3 py-1 rounded-full text-xs font-bold border ${(c as T3Case).label === 'Destroyed' ? 'bg-rose-500/20 text-rose-300 border-rose-500/40' : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'}`}>
            DINS: {(c as T3Case).dins_damage}
          </span>
        ) : (
          <button
            id="reveal-btn"
            onClick={() => setReveal(!reveal)}
            className={`px-3 py-1 rounded-full text-xs font-bold border flex items-center gap-1.5 cursor-pointer ${reveal ? 'bg-slate-800 text-slate-300 border-slate-600' : 'bg-orange-500/20 text-orange-200 border-orange-500/50'}`}
          >
            {reveal ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
            {reveal ? 'Hide outcome' : 'Reveal which one burned'}
          </button>
        )}
      </div>
      {task === 't3' ? (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <Fact label="Structure" value={(c as T3Case).structure_type || 'not recorded'} />
          <Fact label="Year built" value={(c as T3Case).year_built || 'not recorded'} />
          <Fact label="Pre-fire capture" value={`${(c as T3Case).pre.capture_date} (${(c as T3Case).pre.provider || '—'})`} />
          <Fact label="Post-fire capture" value={`${(c as T3Case).post?.capture_date} (${(c as T3Case).post?.provider || '—'})`} />
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          <Fact label="Distance between them" value={`${(c as T2Case).dist_m} m`} />
          <Fact label="Building A" value={`${((c as T2Case).destroyed_is === 'A' ? (c as T2Case).destroyed : (c as T2Case).survived).structure_type || '—'}`} />
          <Fact label="Building B" value={`${((c as T2Case).destroyed_is === 'A' ? (c as T2Case).survived : (c as T2Case).destroyed).structure_type || '—'}`} />
        </div>
      )}
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <SourceLine>CAL FIRE DINS; Esri World Imagery Wayback (capture dates from Wayback metadata).</SourceLine>
        {mapHref && (
          <a href={mapHref} className="text-[11px] text-sky-300 hover:text-sky-200 flex items-center gap-1">
            <MapPin className="w-3 h-3" /> Open this area on the risk map
          </a>
        )}
      </div>
    </div>
  );
}

function T3Preds({ c, models }: { c: T3Case; models: ModelInfo[] }) {
  const truth = c.label === 'Destroyed';
  const withEvidence = models.filter((m) => c.preds[m.key]?.evidence?.length);
  return (
    <div className="space-y-2">
      <div className="space-y-1.5">
        {models.map((m) => {
          const p = c.preds[m.key];
          if (!p) {
            return (
              <div key={m.key} className="grid grid-cols-[minmax(0,12rem)_1fr_3.5rem_1.25rem] items-center gap-2 opacity-50">
                <div className="text-[11px] text-slate-400 truncate">{m.label}</div>
                <div className="text-[10px] text-slate-500">not in this run&apos;s subset</div>
                <div />
                <div />
              </div>
            );
          }
          const hit = p.p >= 0.5 === truth;
          return (
            <div key={m.key} className="grid grid-cols-[minmax(0,12rem)_1fr_3.5rem_1.25rem] items-center gap-2" title={m.source}>
              <div className="min-w-0">
                <div className="text-[11px] text-slate-200 truncate">{m.label}</div>
                <div className="text-[10px] text-slate-500 truncate">{m.input}</div>
              </div>
              <ProbBar p={p.p} tone={hit ? 'hit' : 'miss'} />
              <div className="text-xs font-mono text-right text-slate-100">{p.parsed ? pct(p.p) : 'n/p'}</div>
              {hit ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : <XCircle className="w-4 h-4 text-rose-400" />}
            </div>
          );
        })}
      </div>
      <div className="text-[10px] text-slate-500">Bar = p(destroyed); tick = 0.5 threshold. Green = agrees with DINS, red = disagrees. n/p = answer not parsed (counted as 0.5).</div>
      {withEvidence.slice(0, 2).map((m) => (
        <div key={m.key} className="bg-slate-950/70 border border-slate-800 rounded-xl p-2.5">
          <div className="text-[11px] font-semibold text-slate-300 mb-1 flex items-center gap-1">
            <FileText className="w-3 h-3" /> {m.label} ({m.input}): key evidence
          </div>
          <ul className="list-disc pl-4 space-y-0.5 text-[11px] text-slate-300">
            {c.preds[m.key].evidence!.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

function T2Preds({ c, models, reveal }: { c: T2Case; models: ModelInfo[]; reveal: boolean }) {
  const sonnet = c.preds['sonnet_image'];
  return (
    <div className="space-y-2">
      <div className="space-y-1.5">
        {models.map((m) => {
          const p = c.preds[m.key];
          if (!p) return null;
          // p(A destroyed) as the model answered it
          const pA = c.destroyed_is === 'A' ? p.p_destroyed_one : 1 - p.p_destroyed_one;
          const choice = pA > 0.5 ? 'A' : pA < 0.5 ? 'B' : 'tie';
          const hit = p.p_destroyed_one > 0.5;
          return (
            <div key={m.key} className="grid grid-cols-[minmax(0,12rem)_1fr_4.5rem_1.25rem] items-center gap-2" title={m.source}>
              <div className="min-w-0">
                <div className="text-[11px] text-slate-200 truncate">{m.label}</div>
                <div className="text-[10px] text-slate-500 truncate">{m.input}</div>
              </div>
              {reveal ? <ProbBar p={p.p_destroyed_one} tone={hit ? 'hit' : p.p_destroyed_one === 0.5 ? 'neutral' : 'miss'} /> : <ProbBar p={pA} tone="neutral" />}
              <div className="text-xs font-mono text-right text-slate-100">{p.parsed ? (reveal ? pct(p.p_destroyed_one) : `A ${pct(pA)}`) : 'n/p'}</div>
              {reveal ? (
                choice === 'tie' ? <span className="text-[10px] text-slate-400">tie</span> : hit ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : <XCircle className="w-4 h-4 text-rose-400" />
              ) : (
                <span className="text-[11px] font-bold text-slate-300">{choice}</span>
              )}
            </div>
          );
        })}
      </div>
      <div className="text-[10px] text-slate-500">
        {reveal ? 'Bar = probability the model gave to the home that was actually destroyed; tick = 0.5.' : 'Bar = p(A was destroyed) as answered; letter = the model’s pick. Reveal the outcome to score it.'}
      </div>
      {sonnet?.differences?.length ? (
        <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-2.5">
          <div className="text-[11px] font-semibold text-slate-300 mb-1 flex items-center gap-1">
            <FileText className="w-3 h-3" /> Claude Sonnet 5.5 (two pre-fire images): differences it cited
          </div>
          <ul className="list-disc pl-4 space-y-0.5 text-[11px] text-slate-300">
            {sonnet.differences.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function LiveCard({ task, live, c, reveal, showPrompt, setShowPrompt }: { task: Task; live: LiveResult; c: T3Case | T2Case; reveal: boolean; showPrompt: boolean; setShowPrompt: (v: boolean) => void }) {
  if (!live.success) return <div className="text-xs text-rose-400">Error: {live.error || 'request failed'}</div>;
  let headline: React.ReactNode;
  if (task === 't3') {
    const truth = (c as T3Case).label === 'Destroyed';
    const hit = (live.p ?? 0.5) >= 0.5 === truth;
    headline = (
      <div className="grid grid-cols-[1fr_3.5rem_1.25rem] items-center gap-2">
        <ProbBar p={live.p ?? 0.5} tone={hit ? 'hit' : 'miss'} />
        <div className="text-xs font-mono text-right">{live.parsed ? pct(live.p) : 'n/p'}</div>
        {hit ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : <XCircle className="w-4 h-4 text-rose-400" />}
      </div>
    );
  } else {
    const pTrue = live.p_destroyed_one ?? 0.5;
    const pA = (c as T2Case).destroyed_is === 'A' ? pTrue : 1 - pTrue;
    headline = (
      <div className="text-xs text-slate-200">
        Picked <span className="font-bold">{pA > 0.5 ? 'A' : pA < 0.5 ? 'B' : 'neither (tie)'}</span> · p(A destroyed) {pct(pA)}
        {reveal && <span className={pTrue > 0.5 ? 'text-emerald-400' : 'text-rose-400'}>{pTrue > 0.5 ? ' · correct' : ' · wrong'}</span>}
      </div>
    );
  }
  const notes = (task === 't3' ? live.evidence : live.differences) || [];
  return (
    <div className="border border-emerald-500/40 rounded-xl p-3 space-y-2 bg-slate-950/60">
      <div className="flex items-center justify-between text-[11px]">
        <span className="font-bold text-emerald-300">Live answer · {live.model_id}</span>
        <span className="text-slate-400 font-mono">{live.latency_sec}s · prompt {live.prompt_version}</span>
      </div>
      {headline}
      {notes.length > 0 && (
        <ul className="list-disc pl-4 space-y-0.5 text-[11px] text-slate-300">
          {notes.slice(0, 3).map((e, i) => (
            <li key={i}>{String(e)}</li>
          ))}
        </ul>
      )}
      <button onClick={() => setShowPrompt(!showPrompt)} className="text-[11px] text-purple-300 hover:text-purple-200 flex items-center gap-1 cursor-pointer">
        <Info className="w-3 h-3" /> {showPrompt ? 'Hide prompt' : 'Show the exact prompt sent'}
      </button>
      {showPrompt && <pre className="p-2 rounded-lg bg-slate-950 text-[10px] text-slate-300 max-h-60 overflow-y-auto whitespace-pre-wrap border border-slate-800">{live.prompt}</pre>}
    </div>
  );
}
