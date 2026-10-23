'use client';

// VLM benchmark (dev set) view.
// Rule for this file: every value shown comes from a real source (CAL FIRE DINS records, Esri Wayback imagery with
// per-location capture dates, ERA5 reanalysis, stored model outputs, paper/numbers.tex) and its source is shown next
// to it. Fields that earlier versions generated from the true outcome label are null in the API and their sections
// are not rendered; they reappear only if real data is ever supplied.

import { API_BASE, getJSON } from '../lib/api';

import React, { useState, useEffect } from 'react';
import {
  Eye,
  Flame,
  ShieldCheck,
  AlertTriangle,
  Cpu,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  Play,
  Info,
  Layers,
  MapPin,
  Wind,
  CheckCircle2,
  XCircle,
  ExternalLink,
  BookOpen,
  Maximize2,
  Columns,
  Crosshair,
  ZoomIn,
  Landmark,
  Camera,
  FileText,
  PanelRightClose,
  PanelRightOpen,
  Copy,
  FlaskConical,
} from 'lucide-react';

// ---------------------------------------------------------------- types

type EpochKey = 'prefire' | 'postfire' | 'current';

interface EpochEntry {
  epoch: string;
  label: string;
  selection_rule?: string;
  url: string | null;
  available: boolean;
  exists?: boolean;
  capture_date: string | null;
  unavailable_reason: string | null;
  source: string;
  release_date?: string | null;
  release_identifier?: string | null;
  capture_source?: string | null;
  capture_provider?: string | null;
  capture_resolution_m?: number | null;
  days_before_fire?: number | null;
}

interface FireWeather {
  available: boolean;
  reason?: string;
  source: string;
  incident_start?: string;
  window_start?: string;
  window_end?: string;
  max_wind_mph?: number | null;
  max_wind_time_local?: string | null;
  max_wind_dir_deg?: number | null;
  max_wind_dir_compass?: string | null;
  max_gust_mph?: number | null;
  max_gust_time_local?: string | null;
  max_gust_dir_deg?: number | null;
  max_gust_dir_compass?: string | null;
  min_rh_pct?: number | null;
  min_rh_time_local?: string | null;
  max_temp_f?: number | null;
  max_temp_time_local?: string | null;
  grid_latitude?: number;
  grid_longitude?: number;
  grid_elevation_m?: number;
  request_url?: string;
}

interface Chapter7AInfo {
  built_after_2008?: boolean | null;
  code_status: string;
  standard_summary: string;
  regulatory_basis: string;
}

interface AssessorRecord {
  apn: string;
  year_built: number | string | null;
  assessed_improved_value?: number | null;
  assessed_value_display?: string;
  chapter_7a?: Chapter7AInfo;
  source?: string;
}

interface StreetViewInfo {
  has_image: boolean;
  provenance_badge?: string;
  provenance_label?: string;
  explanation?: string;
  google_maps_url: string;
  google_coords_url?: string;
}

type Supplied = Record<string, unknown> | null | undefined;

interface CaseItem {
  id: string | number;
  incident: string;
  county: string;
  city?: string;
  street_address?: string;
  latitude: number;
  longitude: number;
  year_built: string | number;
  structure_type: string;
  roof_construction: string;
  vent_screen: string;
  exterior_siding: string;
  window_pane?: string;
  deck_on_grade: string;
  deck_elevated: string;
  fence_attached: string;
  eaves: string;
  survival_label: 'Destroyed' | 'Survived';
  raw_damage: string;
  ignition_point: string;
  ignition_source: string;
  defensive_actions: string;
  incident_start?: string | null;
  multi_epoch_images?: Record<EpochKey, EpochEntry> & { streetview?: EpochEntry };
  fire_weather?: FireWeather;
  assessor_record?: AssessorRecord;
  streetview?: StreetViewInfo;
  // Removed (label-conditioned) fields. Null today; rendered only if real data is supplied.
  bellwether_prediction?: Supplied;
  academic_simulations?: Supplied;
  insurance_actuarial?: Supplied;
  structured_assessment?: Supplied;
  expert_diagnosis?: Supplied;
  incident_forensics?: Supplied;
}

interface ModelOutput {
  predicted_survival: 'Destroyed' | 'Survived' | 'Unknown';
  predicted_vulnerability: string;
  predicted_mechanism: string;
  is_survival_correct: boolean;
  locus_match: boolean;
  latency_sec: number;
  thinking_trace?: string;
  analysis_text?: string;
}

interface ModelScoreboardEntry {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  false_alarm_rate: number;
  locus_attribution_accuracy: number;
  confusion_matrix: { TP: number; FP: number; TN: number; FN: number };
  total_evaluated?: number;
  evaluated_at?: string | null;
}

interface RunNotes {
  input?: string;
  prompt_weather_text?: string;
  dataset?: string;
  trace?: string;
}

interface ResearchItem {
  task: string;
  result: string;
  meaning: string;
}

interface ResearchEvidence {
  available: boolean;
  source?: string;
  items?: ResearchItem[];
}

interface VisionBenchmarkViewProps {
  onSwitchToMap: () => void;
}

// Models with stored runs (baseline_evaluation_*.json) — also the models the live call supports.
const MODELS: { key: string; label: string }[] = [
  { key: 'MiniMax-M3', label: 'MiniMax-M3' },
  { key: 'claude-haiku-5-5', label: 'Claude Haiku 5.5' },
  { key: 'claude-sonnet-5-5', label: 'Claude Sonnet 5.5' },
  { key: 'Qwen3.8-27B', label: 'Qwen3.8-27B' },
];

const REJECT_REASONS: Record<string, string> = {
  no_postfire_capture_within_lag: 'No Wayback capture of this location within 365 days after the fire',
  capture_not_before_fire: 'The newest release before the fire serves a capture dated after the fire start',
  capture_too_stale: 'Newest capture before the fire is more than 5 years old',
  built_after_capture: 'Newest capture before the fire predates the building (YEARBUILT)',
  no_release_before_fire: 'No Wayback release before the fire',
  no_capture_metadata: 'Wayback metadata has no capture date at this location',
  missing_tile_in_release: 'The release has no tile at this location',
};

const REMOVED_SECTIONS: { field: keyof CaseItem; title: string }[] = [
  { field: 'bellwether_prediction', title: 'Bellwether hazard values' },
  { field: 'academic_simulations', title: 'Fire-spread simulations' },
  { field: 'insurance_actuarial', title: 'Insurance / underwriting' },
  { field: 'structured_assessment', title: 'Structured assessment' },
  { field: 'incident_forensics', title: 'Incident forensics' },
  { field: 'expert_diagnosis', title: 'Expert diagnosis' },
];

// ---------------------------------------------------------------- helpers

function reasonText(r: string | null | undefined): string {
  if (!r) return 'Unavailable';
  return REJECT_REASONS[r] || r.replace(/_/g, ' ');
}

function fmtLocal(t: string | null | undefined): string {
  if (!t) return '—';
  return t.replace('T', ' ');
}

function fmtNum(v: number | null | undefined, digits = 0): string {
  return v === null || v === undefined ? '—' : v.toFixed(digits);
}

function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const r = 6371;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * r * Math.asin(Math.sqrt(a));
}

function hasDefensiveAction(c: CaseItem): boolean {
  const a = (c.defensive_actions || '').trim().toLowerCase();
  return a !== '' && a !== 'none' && a !== 'unknown';
}

function flatten(obj: unknown, prefix = ''): [string, string][] {
  if (obj === null || obj === undefined) return [];
  if (typeof obj !== 'object') return [[prefix || 'value', String(obj)]];
  if (Array.isArray(obj)) return [[prefix, obj.map((x) => (typeof x === 'object' ? JSON.stringify(x) : String(x))).join('; ')]];
  return Object.entries(obj as Record<string, unknown>).flatMap(([k, v]) => flatten(v, prefix ? `${prefix}.${k}` : k));
}

function SourceLine({ children }: { children: React.ReactNode }) {
  return <div className="text-[10px] text-slate-500 leading-snug">Source: {children}</div>;
}

function Fact({ label, value, accent }: { label: string; value: React.ReactNode; accent?: string }) {
  return (
    <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800/80 min-w-0">
      <span className="text-slate-400 block text-[10px]">{label}</span>
      <span className={`font-semibold text-xs break-words ${accent || 'text-slate-200'}`}>{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------- component

export default function VisionBenchmarkView({ onSwitchToMap }: VisionBenchmarkViewProps) {
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [modelResults, setModelResults] = useState<Record<string, Record<string, ModelOutput>>>({});
  const [scoreboard, setScoreboard] = useState<ModelScoreboardEntry[]>([]);
  const [runNotes, setRunNotes] = useState<RunNotes>({});
  const [research, setResearch] = useState<ResearchEvidence | null>(null);
  const [researchError, setResearchError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [filterType, setFilterType] = useState<'all' | 'survived' | 'destroyed' | 'defended'>('all');
  const [loading, setLoading] = useState<boolean>(true);
  const [expandedThinking, setExpandedThinking] = useState<Record<string, boolean>>({});
  const [showDataSourceModal, setShowDataSourceModal] = useState<boolean>(false);
  const [selectedTraceModal, setSelectedTraceModal] = useState<{ modelName: string; trace: string; caseId: string | number } | null>(null);
  const [copyFeedback, setCopyFeedback] = useState<string | null>(null);
  const [liveModel, setLiveModel] = useState<string>('claude-sonnet-5-5');
  const [liveLoading, setLiveLoading] = useState<boolean>(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [liveResult, setLiveResult] = useState<any>(null);
  const [selectedEpoch, setSelectedEpoch] = useState<EpochKey>('prefire');
  const [viewPerspective, setViewPerspective] = useState<'aerial' | 'street'>('aerial');
  const [splitComparison, setSplitComparison] = useState<boolean>(false);
  const [zoomLevel, setZoomLevel] = useState<'1x' | '2x'>('1x');
  const [showTargetReticle, setShowTargetReticle] = useState<boolean>(true);
  const [isReasoningCollapsed, setIsReasoningCollapsed] = useState<boolean>(false);

  const handleCopyTrace = (text: string, id: string) => {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      navigator.clipboard.writeText(text);
      setCopyFeedback(id);
      setTimeout(() => setCopyFeedback(null), 2000);
    }
  };

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      const [casesRes, resultsRes, metricsRes, researchRes] = await Promise.allSettled([
        getJSON('/api/vlm/benchmark/cases'),
        getJSON('/api/vlm/benchmark/results'),
        getJSON('/api/vlm/benchmark/metrics'),
        getJSON('/api/vlm/research-results'),
      ]);
      if (casesRes.status === 'fulfilled') {
        const items: CaseItem[] = casesRes.value?.items || [];
        setCases(items);
        if (typeof window !== 'undefined') {
          const targetId = new URLSearchParams(window.location.search).get('case');
          if (targetId) {
            const idx = items.findIndex((c) => String(c.id) === String(targetId));
            if (idx !== -1) setCurrentIndex(idx);
          }
        }
      } else {
        setLoadError(`Could not load cases from ${API_BASE}: ${String(casesRes.reason)}`);
      }
      if (resultsRes.status === 'fulfilled') {
        setModelResults(resultsRes.value?.cases || {});
        setRunNotes(resultsRes.value?.run_notes || {});
      }
      if (metricsRes.status === 'fulfilled') setScoreboard(metricsRes.value?.scoreboard || []);
      if (researchRes.status === 'fulfilled') setResearch(researchRes.value);
      else setResearchError(String(researchRes.reason));
      setLoading(false);
    }
    loadData();
  }, []);

  const filteredCases = cases.filter((item) => {
    if (filterType === 'survived') return item.survival_label === 'Survived';
    if (filterType === 'destroyed') return item.survival_label === 'Destroyed';
    if (filterType === 'defended') return hasDefensiveAction(item);
    return true;
  });

  const currentCase = filteredCases[currentIndex] || filteredCases[0];
  const currentCaseResults = currentCase ? modelResults[String(currentCase.id)] || {} : {};
  const epochs = currentCase?.multi_epoch_images;
  const nDestroyed = cases.filter((c) => c.survival_label === 'Destroyed').length;
  const nSurvived = cases.length - nDestroyed;
  const survivedFires = Array.from(new Set(cases.filter((c) => c.survival_label === 'Survived').map((c) => c.incident.trim())));
  const destroyedFires = Array.from(new Set(cases.filter((c) => c.survival_label === 'Destroyed').map((c) => c.incident.trim())));

  const resetCaseState = () => {
    setLiveResult(null);
    setExpandedThinking({});
  };
  const handlePrev = () => {
    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : filteredCases.length - 1));
    resetCaseState();
  };
  const handleNext = () => {
    setCurrentIndex((prev) => (prev < filteredCases.length - 1 ? prev + 1 : 0));
    resetCaseState();
  };

  const handleRunLivePrediction = async () => {
    if (!currentCase) return;
    setLiveLoading(true);
    setLiveResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/vlm/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: liveModel, case_id: String(currentCase.id) }),
      });
      const data = await res.json();
      setLiveResult(res.ok ? data : { success: false, error: data?.detail || `HTTP ${res.status}` });
    } catch (err) {
      setLiveResult({ success: false, error: String(err) });
    } finally {
      setLiveLoading(false);
    }
  };

  // ------------------------------------------------------------ image tile
  const renderAerialTile = (ep: EpochEntry | undefined, compact: boolean) => {
    if (!currentCase) return null;
    if (!ep || !ep.available || !ep.url) {
      return (
        <div className={`w-full ${compact ? 'aspect-square' : 'h-[560px]'} bg-slate-950 flex flex-col items-center justify-center gap-2 p-6 text-center`}>
          <AlertTriangle className="w-6 h-6 text-amber-400" />
          <span className="text-sm font-semibold text-amber-200">{ep?.label || 'Image'} imagery unavailable</span>
          <span className="text-xs text-slate-400 max-w-md">{reasonText(ep?.unavailable_reason)}</span>
          <span className="text-[10px] text-slate-500">No other image is substituted.</span>
        </div>
      );
    }
    return (
      <div className={`relative ${compact ? 'aspect-square' : 'h-[560px]'} w-full bg-black overflow-hidden`}>
        <img
          src={`${API_BASE}${ep.url}`}
          alt={`${ep.label} aerial image, case ${currentCase.id}, captured ${ep.capture_date}`}
          className={`w-full h-full object-contain transition-transform duration-500 origin-center ${zoomLevel === '2x' ? 'scale-[2.0]' : ''}`}
        />
        {showTargetReticle && (
          <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
            <div className="w-10 h-10 rounded-full border-2 border-amber-400/90" />
            <div className="absolute w-1 h-1 rounded-full bg-amber-400" />
          </div>
        )}
        <div className="absolute top-2 left-2 flex flex-wrap gap-1.5">
          <span
            className={`px-2 py-0.5 rounded-md text-[11px] font-bold backdrop-blur-md border ${
              ep.epoch === 'prefire'
                ? 'bg-emerald-950/90 text-emerald-300 border-emerald-500/40'
                : ep.epoch === 'postfire'
                ? 'bg-rose-950/90 text-rose-300 border-rose-500/40'
                : 'bg-cyan-950/90 text-cyan-300 border-cyan-500/40'
            }`}
          >
            {ep.label} · captured {ep.capture_date}
          </span>
        </div>
        <div className="absolute bottom-2 left-2 right-2 bg-slate-950/85 backdrop-blur-md px-2 py-1 rounded text-[10px] text-slate-300 truncate">
          {ep.capture_provider || '—'} {ep.capture_source ? `(${ep.capture_source})` : ''}
          {ep.capture_resolution_m ? ` · ${ep.capture_resolution_m} m source resolution` : ''} · Wayback {ep.release_identifier}
        </div>
      </div>
    );
  };

  const epochCaption = (ep: EpochEntry | undefined) => {
    if (!ep || !currentCase) return null;
    let timing = '';
    if (ep.available && ep.days_before_fire !== null && ep.days_before_fire !== undefined) {
      timing = ep.days_before_fire >= 0 ? `${ep.days_before_fire} days before the fire start` : `${-ep.days_before_fire} days after the fire start`;
    }
    return (
      <div className="bg-slate-950/90 px-3.5 py-2 border-t border-slate-800 text-[11px] space-y-0.5">
        <div className="text-slate-300">
          <span className="font-semibold">{ep.label}:</span> {ep.selection_rule}.{' '}
          {ep.available ? (
            <>
              Capture date <span className="font-mono text-slate-100">{ep.capture_date}</span>
              {timing ? ` (${timing})` : ''}; release {ep.release_identifier} published {ep.release_date}.
            </>
          ) : (
            <span className="text-amber-300">{reasonText(ep.unavailable_reason)}.</span>
          )}
        </div>
        <SourceLine>{ep.source}. Fire start {currentCase.incident_start || '—'} from CAL FIRE DINS.</SourceLine>
      </div>
    );
  };

  // ------------------------------------------------------------ render
  return (
    <div className="flex-1 h-full flex flex-col bg-slate-950 text-slate-100 overflow-y-auto">
      {/* room for the page-level mode switcher that floats over the top of this view */}
      <div className="h-16 shrink-0 bg-slate-950" />
      <div className="bg-amber-500/15 border-b border-amber-500/40 px-6 py-2 text-[11px] text-amber-200">
        Legacy demo on a 50-case development set. The stored model runs are text-only and the dev set is confounded (all{' '}
        {nSurvived || 25} survived cases come from one fire), so its scores are not results. Held-out results are shown in the
        panel below. Sections whose values were generated from the true outcome label have been removed.
      </div>

      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-rose-600 flex items-center justify-center">
            <Eye className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-base font-bold text-white tracking-wide">VLM benchmark · development set</h1>
              <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-orange-500/20 text-orange-300 border border-orange-500/30">
                Not a held-out evaluation
              </span>
              <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                Labels: CAL FIRE DINS
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Browse the 50 dev-set structures: DINS inspection record, verified aerial imagery, ERA5 fire weather and stored model outputs.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={onSwitchToMap}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 cursor-pointer"
          >
            <Layers className="w-3.5 h-3.5 text-orange-400" />
            <span>Risk map</span>
          </button>
          <button
            onClick={() => setShowDataSourceModal(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/80 cursor-pointer"
          >
            <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
            <span>Data sources</span>
          </button>
        </div>
      </header>

      {/* Held-out results (this project's real evaluation numbers) */}
      <section className="bg-slate-900/40 border-b border-slate-800/60 px-6 py-3 space-y-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <FlaskConical className="w-4 h-4 text-emerald-400" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300">Held-out results (this project)</span>
          </div>
          {research?.source && <span className="text-[10px] text-slate-500">Source: {research.source}</span>}
        </div>
        {research?.available && research.items ? (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-2">
            {research.items.map((it) => (
              <div key={it.task} className="bg-slate-900/80 border border-slate-800 rounded-xl p-2.5 space-y-1">
                <div className="text-[11px] font-semibold text-slate-200">{it.task}</div>
                <div className="text-[11px] font-mono text-emerald-300">{it.result}</div>
                <div className="text-[11px] text-slate-400">{it.meaning}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-[11px] text-slate-500">
            {research && !research.available
              ? 'Held-out numbers unavailable: paper/numbers.tex not found on the backend.'
              : researchError
              ? `Held-out numbers unavailable: ${researchError}`
              : 'Loading…'}
          </div>
        )}
      </section>

      {/* Dev-set scoreboard of the stored text-only runs */}
      <section className="bg-slate-900/20 border-b border-slate-800/60 px-6 py-3">
        <div className="flex items-start justify-between mb-2 gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-purple-400" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
              Stored text-only runs on this dev set ({nDestroyed} destroyed, {nSurvived} survived)
            </span>
          </div>
          <span className="text-[11px] text-amber-300/90 max-w-3xl">
            Confounded: destroyed cases come from {destroyedFires.length} fires, survived cases from{' '}
            {survivedFires.length === 1 ? `one fire (${survivedFires[0]})` : `${survivedFires.length} fires`}. Models saw no image.
          </span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {scoreboard.map((item) => (
            <div key={item.model_name} className="bg-slate-900/80 border border-slate-800 rounded-xl p-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold text-white">{item.model_name}</span>
                <span className="text-[10px] text-slate-500">
                  n={item.total_evaluated ?? '—'}
                  {item.evaluated_at ? ` · run ${item.evaluated_at.slice(0, 10)}` : ''}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center pt-1 border-t border-slate-800/60">
                <div>
                  <div className="text-[10px] text-slate-400">Accuracy</div>
                  <div className="text-sm font-bold text-slate-100">{item.accuracy}%</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-400">FPR on survived</div>
                  <div className="text-sm font-bold text-slate-100">{item.false_alarm_rate}%</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-400">Ignition-point match</div>
                  <div className="text-sm font-bold text-slate-100">{item.locus_attribution_accuracy}%</div>
                </div>
              </div>
              <div className="mt-2 text-[10px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800/40">
                <span>TP={item.confusion_matrix.TP} FP={item.confusion_matrix.FP}</span>
                <span>TN={item.confusion_matrix.TN} FN={item.confusion_matrix.FN}</span>
              </div>
            </div>
          ))}
          {scoreboard.length === 0 && !loading && <div className="text-[11px] text-slate-500">No stored runs found.</div>}
        </div>
        <div className="mt-2">
          <SourceLine>
            baseline_evaluation_*.json (src/vlm_benchmark/evaluate_baselines.py). {runNotes.prompt_weather_text ? `Note: ${runNotes.prompt_weather_text}.` : ''}
          </SourceLine>
        </div>
      </section>

      {/* Case navigation */}
      <section className="bg-slate-900/60 border-b border-slate-800/80 px-6 py-2.5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-xs text-slate-400 mr-1 font-medium">Filter:</span>
          {(
            [
              { key: 'all', label: `All (${cases.length})` },
              { key: 'destroyed', label: `Destroyed (${nDestroyed})` },
              { key: 'survived', label: `Survived (${nSurvived})` },
              { key: 'defended', label: `Defensive action recorded (${cases.filter(hasDefensiveAction).length})` },
            ] as const
          ).map((pill) => (
            <button
              key={pill.key}
              onClick={() => {
                setFilterType(pill.key);
                setCurrentIndex(0);
                resetCaseState();
              }}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                filterType === pill.key
                  ? 'bg-orange-500 text-white'
                  : 'bg-slate-800/60 text-slate-300 hover:bg-slate-800 border border-slate-700/60'
              }`}
            >
              {pill.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={handlePrev} className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700" title="Previous case">
            <ChevronLeft className="w-4 h-4" />
          </button>
          <select
            value={currentCase ? String(currentCase.id) : ''}
            onChange={(e) => {
              const idx = filteredCases.findIndex((c) => String(c.id) === e.target.value);
              if (idx !== -1) {
                setCurrentIndex(idx);
                resetCaseState();
              }
            }}
            className="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-orange-500 max-w-[22rem]"
          >
            {filteredCases.map((c) => (
              <option key={c.id} value={String(c.id)}>
                #{c.id} · {c.incident.trim()} Fire · DINS: {c.raw_damage}
              </option>
            ))}
          </select>
          <button onClick={handleNext} className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700" title="Next case">
            <ChevronRight className="w-4 h-4" />
          </button>
          <span className="text-xs text-slate-400 font-mono ml-1">
            {filteredCases.length > 0 ? Math.min(currentIndex + 1, filteredCases.length) : 0} of {filteredCases.length}
          </span>
          <div className="h-4 w-[1px] bg-slate-800 mx-1" />
          <button
            onClick={() => setIsReasoningCollapsed(!isReasoningCollapsed)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold border cursor-pointer bg-slate-800 text-slate-300 border-slate-700 hover:text-white"
          >
            {isReasoningCollapsed ? (
              <>
                <PanelRightOpen className="w-3.5 h-3.5 text-purple-400" />
                <span>Show model outputs</span>
              </>
            ) : (
              <>
                <PanelRightClose className="w-3.5 h-3.5 text-orange-400" />
                <span>Hide model outputs</span>
              </>
            )}
          </button>
        </div>
      </section>

      {loading ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 text-slate-400 py-20">
          <div className="w-8 h-8 border-4 border-orange-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm">Loading dev-set cases…</span>
        </div>
      ) : loadError ? (
        <div className="flex-1 flex items-center justify-center text-rose-300 text-sm py-20">{loadError}</div>
      ) : currentCase ? (
        <main className="flex-1 p-6 space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left column */}
            <div className={`${isReasoningCollapsed ? 'lg:col-span-12' : 'lg:col-span-8'} space-y-4`}>
              {/* Imagery */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
                <div className="bg-slate-950 p-2.5 border-b border-slate-800 flex items-center justify-between gap-2 flex-wrap">
                  <div className="flex items-center gap-1.5 bg-slate-900 p-1 rounded-xl border border-slate-800">
                    <button
                      onClick={() => setViewPerspective('aerial')}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 cursor-pointer ${
                        viewPerspective === 'aerial' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      <Layers className="w-3.5 h-3.5" />
                      <span>Aerial (Esri Wayback)</span>
                    </button>
                    <button
                      onClick={() => {
                        setViewPerspective('street');
                        setSplitComparison(false);
                      }}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 cursor-pointer ${
                        viewPerspective === 'street' ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      <Camera className="w-3.5 h-3.5" />
                      <span>Street-level photo</span>
                      {!currentCase.streetview?.has_image && <span className="text-[10px] text-slate-500 font-normal">(none)</span>}
                    </button>
                  </div>
                  <span className="text-[10px] font-mono text-slate-400">
                    {currentCase.latitude.toFixed(5)}, {currentCase.longitude.toFixed(5)} · zoom 18 · 512 px
                  </span>
                </div>

                {viewPerspective === 'aerial' && (
                  <div>
                    <div className="bg-slate-950/80 p-2 border-b border-slate-800 flex items-center justify-between gap-2 text-xs flex-wrap">
                      <div className="flex items-center gap-1 flex-wrap">
                        {(['prefire', 'postfire', 'current'] as EpochKey[]).map((key) => {
                          const ep = epochs?.[key];
                          const active = !splitComparison && selectedEpoch === key;
                          return (
                            <button
                              key={key}
                              onClick={() => {
                                setSelectedEpoch(key);
                                setSplitComparison(false);
                              }}
                              title={ep?.available ? `${ep.selection_rule}` : reasonText(ep?.unavailable_reason)}
                              className={`px-2.5 py-1.5 rounded-lg font-semibold flex items-center gap-1.5 whitespace-nowrap cursor-pointer border ${
                                active ? 'bg-slate-700/60 text-white border-slate-500' : 'text-slate-400 hover:text-slate-200 border-transparent'
                              }`}
                            >
                              <span
                                className={`w-2 h-2 rounded-full ${
                                  !ep?.available ? 'bg-slate-600' : key === 'prefire' ? 'bg-emerald-400' : key === 'postfire' ? 'bg-rose-400' : 'bg-cyan-400'
                                }`}
                              />
                              <span>{ep?.label || key}</span>
                              <span className={`font-mono text-[10px] ${ep?.available ? 'text-slate-300' : 'text-slate-500'}`}>
                                {ep?.available ? ep.capture_date : 'unavailable'}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <button
                          onClick={() => setShowTargetReticle(!showTargetReticle)}
                          className={`px-2 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1 cursor-pointer border ${
                            showTargetReticle ? 'bg-amber-500/20 text-amber-300 border-amber-500/50' : 'bg-slate-800 text-slate-400 border-slate-700'
                          }`}
                          title="Mark the DINS point (image centre)"
                        >
                          <Crosshair className="w-3.5 h-3.5" />
                          <span className="hidden sm:inline">DINS point</span>
                        </button>
                        <button
                          onClick={() => setZoomLevel(zoomLevel === '1x' ? '2x' : '1x')}
                          className={`px-2 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1 cursor-pointer border ${
                            zoomLevel === '2x' ? 'bg-orange-600 text-white border-orange-500' : 'bg-slate-800 text-slate-400 border-slate-700'
                          }`}
                          title="Digital 2x enlargement of the same crop (no extra detail)"
                        >
                          <ZoomIn className="w-3.5 h-3.5" />
                          <span>2x</span>
                        </button>
                        <button
                          onClick={() => setSplitComparison(!splitComparison)}
                          className={`px-2.5 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 cursor-pointer border ${
                            splitComparison ? 'bg-orange-600 text-white border-orange-500' : 'bg-slate-800 text-slate-300 border-slate-700'
                          }`}
                          title="Pre-fire and post-fire side by side"
                        >
                          <Columns className="w-3.5 h-3.5" />
                          <span className="hidden sm:inline">Pre / post</span>
                        </button>
                      </div>
                    </div>

                    {splitComparison ? (
                      <>
                        <div className="grid grid-cols-2 divide-x divide-slate-800 bg-slate-950">
                          <div>{renderAerialTile(epochs?.prefire, true)}</div>
                          <div>{renderAerialTile(epochs?.postfire, true)}</div>
                        </div>
                        {epochCaption(epochs?.prefire)}
                        {epochCaption(epochs?.postfire)}
                      </>
                    ) : (
                      <>
                        {renderAerialTile(epochs?.[selectedEpoch], false)}
                        {epochCaption(epochs?.[selectedEpoch])}
                      </>
                    )}
                  </div>
                )}

                {viewPerspective === 'street' && (
                  <div>
                    {currentCase.streetview?.has_image ? (
                      <div className="relative h-[560px] w-full bg-black overflow-hidden">
                        <img
                          src={`${API_BASE}/api/vlm/benchmark/image/${currentCase.id}?epoch=streetview`}
                          alt={`Street-level photo near case ${currentCase.id} (capture date unknown)`}
                          className="w-full h-full object-contain"
                        />
                        <div className="absolute top-3 left-3">
                          <span className="px-2.5 py-1 rounded-md text-xs font-bold bg-amber-950/90 text-amber-200 border border-amber-500/40 flex items-center gap-1.5">
                            <AlertTriangle className="w-3.5 h-3.5" />
                            <span>Capture date unknown: may show the building before or after the fire</span>
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="aspect-video w-full bg-slate-950 flex flex-col items-center justify-center gap-2 text-center p-6">
                        <Camera className="w-6 h-6 text-slate-500" />
                        <span className="text-sm text-slate-300">No street-level photo is stored for this case.</span>
                      </div>
                    )}
                    <div className="bg-slate-950/90 px-3.5 py-2 border-t border-slate-800 text-[11px] space-y-1">
                      <div className="text-slate-300">{currentCase.streetview?.explanation}</div>
                      <div className="flex gap-3 flex-wrap">
                        {currentCase.streetview?.google_maps_url && (
                          <a href={currentCase.streetview.google_maps_url} target="_blank" rel="noopener noreferrer" className="text-blue-300 hover:text-blue-200 flex items-center gap-1">
                            Open Street View at this point <ExternalLink className="w-3 h-3" />
                          </a>
                        )}
                        {currentCase.streetview?.google_coords_url && (
                          <a href={currentCase.streetview.google_coords_url} target="_blank" rel="noopener noreferrer" className="text-blue-300 hover:text-blue-200 flex items-center gap-1">
                            Open coordinates in Google Maps <ExternalLink className="w-3 h-3" />
                          </a>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className={isReasoningCollapsed ? 'grid grid-cols-1 lg:grid-cols-2 gap-4' : 'space-y-4'}>
                {/* DINS inspection record */}
                <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3 gap-2 flex-wrap">
                    <div className="flex items-center gap-2">
                      <ShieldCheck className="w-5 h-5 text-emerald-400" />
                      <div>
                        <h3 className="text-sm font-bold text-white">CAL FIRE DINS inspection record</h3>
                        <span className="text-[11px] text-slate-400">
                          {currentCase.incident.trim()} Fire · {currentCase.county} County · started {currentCase.incident_start || '—'}
                        </span>
                      </div>
                    </div>
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-bold ${
                        currentCase.survival_label === 'Survived'
                          ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                          : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                      }`}
                    >
                      Label: {currentCase.survival_label}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                    <Fact label="DINS damage" value={currentCase.raw_damage} />
                    <Fact label="Where fire started on structure" value={currentCase.ignition_point || '—'} accent="text-amber-300" />
                    <Fact label="What it started from" value={currentCase.ignition_source || '—'} />
                    <Fact label="Defensive actions" value={currentCase.defensive_actions || '—'} />
                    <Fact label="Structure type" value={currentCase.structure_type} />
                    <Fact label="Address" value={currentCase.street_address || '—'} />
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    <Fact label="Roof" value={currentCase.roof_construction} />
                    <Fact label="Vent screen" value={currentCase.vent_screen} />
                    <Fact label="Exterior siding" value={currentCase.exterior_siding} />
                    <Fact label="Eaves" value={currentCase.eaves} />
                    <Fact label="Window pane" value={currentCase.window_pane || '—'} />
                    <Fact label="Deck on grade" value={currentCase.deck_on_grade} />
                    <Fact label="Deck elevated" value={currentCase.deck_elevated} />
                    <Fact label="Fence attached" value={currentCase.fence_attached} />
                  </div>
                  <SourceLine>
                    CAL FIRE DINS (OBJECTID {currentCase.id}). The dev-set label is a binarisation of DAMAGE (&quot;Destroyed&quot; = Destroyed or
                    Major; otherwise &quot;Survived&quot;). DINS attributes are recorded after the fire.
                  </SourceLine>
                </div>

                {/* Fire weather (ERA5) */}
                <FireWeatherCard c={currentCase} />

                {/* Parcel record from DINS */}
                <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-3">
                  <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
                    <Landmark className="w-5 h-5 text-indigo-400" />
                    <h3 className="text-sm font-bold text-white">Parcel attributes</h3>
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    <Fact label="APN" value={currentCase.assessor_record?.apn ?? '—'} />
                    <Fact label="Year built" value={(currentCase.assessor_record?.year_built ?? currentCase.year_built) || '—'} />
                    <Fact label="Assessed improved value" value={currentCase.assessor_record?.assessed_value_display ?? '—'} />
                  </div>
                  {currentCase.assessor_record?.chapter_7a && (
                    <div className="text-[11px] text-slate-400 leading-relaxed">
                      <span className="font-semibold text-slate-300">{currentCase.assessor_record.chapter_7a.code_status}.</span>{' '}
                      {currentCase.assessor_record.chapter_7a.standard_summary}
                    </div>
                  )}
                  <SourceLine>{currentCase.assessor_record?.source || 'CAL FIRE DINS record'}.</SourceLine>
                </div>

                {/* Removed, label-conditioned sections: rendered only if real data is supplied */}
                {REMOVED_SECTIONS.map(({ field, title }) => {
                  const data = currentCase[field] as Supplied;
                  if (!data) return null;
                  return (
                    <div key={String(field)} className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-2">
                      <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
                        <FileText className="w-4 h-4 text-slate-400" />
                        <h3 className="text-sm font-bold text-white">{title}</h3>
                      </div>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-1 text-[11px]">
                        {flatten(data).map(([k, v]) => (
                          <div key={k} className="flex justify-between gap-2 border-b border-slate-800/40 py-0.5">
                            <span className="text-slate-400">{k}</span>
                            <span className="text-slate-200 text-right break-words">{v}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Right column: stored model outputs + live call */}
            <div className={`${isReasoningCollapsed ? 'hidden' : 'lg:col-span-4'} space-y-4`}>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <Flame className="w-5 h-5 text-orange-400" />
                  <h3 className="text-sm font-bold text-white">Stored model outputs for this case</h3>
                </div>
                <p className="text-[11px] text-slate-400">
                  Text-only prompt ({runNotes.input || 'DINS attributes, no image'}). Responses are stored truncated to 500 characters.
                </p>
              </div>

              <div className="grid grid-cols-1 gap-3">
                {MODELS.map(({ key, label }) => {
                  const res = currentCaseResults[key];
                  const isExpanded = expandedThinking[key] || false;
                  const traceText = res ? res.thinking_trace || res.analysis_text || '' : '';
                  return (
                    <div key={key} className="bg-slate-900 border border-slate-800 rounded-2xl p-3.5 space-y-2.5">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-white">{label}</span>
                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
                          {res ? `${res.latency_sec}s` : 'no record'}
                        </span>
                      </div>
                      {res ? (
                        <>
                          <div className="flex items-center justify-between bg-slate-950/70 p-2 rounded-xl border border-slate-800 text-xs gap-2 flex-wrap">
                            <span className="text-slate-400">Prediction</span>
                            <div className="flex items-center gap-1.5">
                              <span
                                className={`font-bold px-2 py-0.5 rounded ${
                                  res.predicted_survival === 'Destroyed' ? 'bg-rose-500/20 text-rose-300' : 'bg-emerald-500/20 text-emerald-300'
                                }`}
                              >
                                {res.predicted_survival}
                              </span>
                              {res.is_survival_correct ? (
                                <span className="flex items-center text-[11px] text-emerald-400 gap-0.5">
                                  <CheckCircle2 className="w-3.5 h-3.5" /> matches DINS label
                                </span>
                              ) : (
                                <span className="flex items-center text-[11px] text-rose-400 gap-0.5">
                                  <XCircle className="w-3.5 h-3.5" /> differs from DINS label
                                </span>
                              )}
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-2">
                            <Fact
                              label="Predicted weak point"
                              value={
                                <>
                                  {res.predicted_vulnerability || '—'}{' '}
                                  <span className="text-[10px] font-normal text-slate-500">
                                    {res.locus_match ? '(matches DINS)' : currentCase.ignition_point && !['none', 'unknown'].includes(currentCase.ignition_point.toLowerCase()) ? '(DINS differs)' : '(DINS: none/unknown)'}
                                  </span>
                                </>
                              }
                            />
                            <Fact label="Predicted mechanism" value={res.predicted_mechanism || '—'} />
                          </div>
                          {traceText && (
                            <div className="space-y-1.5">
                              <div className="flex items-center justify-between gap-1">
                                <button
                                  onClick={() => setExpandedThinking((p) => ({ ...p, [key]: !p[key] }))}
                                  className="flex items-center gap-1 text-[11px] text-purple-300 hover:text-purple-200 cursor-pointer"
                                >
                                  <span>{isExpanded ? 'Hide response text' : 'Show response text'}</span>
                                  {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                                </button>
                                <div className="flex items-center gap-1">
                                  <button
                                    onClick={() => handleCopyTrace(traceText, key)}
                                    className="px-1.5 py-0.5 rounded text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center gap-1 border border-slate-700 cursor-pointer"
                                  >
                                    <Copy className="w-3 h-3" />
                                    <span>{copyFeedback === key ? 'Copied' : 'Copy'}</span>
                                  </button>
                                  <button
                                    onClick={() => setSelectedTraceModal({ modelName: label, trace: traceText, caseId: currentCase.id })}
                                    className="px-1.5 py-0.5 rounded text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center gap-1 border border-slate-700 cursor-pointer"
                                  >
                                    <Maximize2 className="w-3 h-3" />
                                    <span>Open</span>
                                  </button>
                                </div>
                              </div>
                              {isExpanded ? (
                                <div className="p-2.5 rounded-xl bg-slate-950 text-[11px] font-mono text-slate-300 max-h-72 overflow-y-auto border border-slate-800 whitespace-pre-wrap">
                                  {traceText}
                                </div>
                              ) : (
                                <p className="text-[11px] text-slate-500 line-clamp-2">{traceText}</p>
                              )}
                            </div>
                          )}
                        </>
                      ) : (
                        <div className="p-3 bg-slate-950/40 rounded-xl text-center text-xs text-slate-500">No stored output for this case</div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Live call */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-2.5">
                <div className="flex items-center gap-2">
                  <Play className="w-4 h-4 text-emerald-400" />
                  <div>
                    <span className="text-xs font-bold text-white block">Live model call</span>
                    <span className="text-[11px] text-slate-400">Same text-only prompt as the stored runs, sent now to the selected model.</span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <select
                    value={liveModel}
                    onChange={(e) => setLiveModel(e.target.value)}
                    className="flex-1 bg-slate-800 border border-slate-700 text-xs rounded-lg px-3 py-1.5 text-slate-200 focus:outline-none focus:border-orange-500"
                  >
                    {MODELS.map((m) => (
                      <option key={m.key} value={m.key}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={handleRunLivePrediction}
                    disabled={liveLoading}
                    className="px-4 py-1.5 rounded-lg text-xs font-bold bg-orange-600 hover:bg-orange-700 text-white disabled:opacity-50 flex items-center gap-1.5 cursor-pointer"
                  >
                    {liveLoading ? (
                      <>
                        <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        <span>Running…</span>
                      </>
                    ) : (
                      <span>Run</span>
                    )}
                  </button>
                </div>
                <div className="text-[10px] text-slate-500">Requires the model&apos;s API key or endpoint to be configured on the backend.</div>
              </div>

              {liveResult && (
                <div className="bg-slate-900/90 border border-orange-500/40 rounded-2xl p-4 space-y-2.5">
                  <div className="flex items-center justify-between text-xs border-b border-slate-800 pb-2">
                    <span className="font-bold text-orange-300">Live result {liveResult.model_name ? `(${liveResult.model_name})` : ''}</span>
                    {liveResult.latency_sec !== undefined && <span className="text-slate-400 font-mono">{liveResult.latency_sec}s</span>}
                  </div>
                  {liveResult.success ? (
                    <div className="space-y-2 text-xs">
                      <div className="grid grid-cols-3 gap-2">
                        <Fact label="Prediction" value={liveResult.predicted_survival} />
                        <Fact label="Weak point" value={liveResult.predicted_vulnerability} />
                        <Fact label="Mechanism" value={liveResult.predicted_mechanism} />
                      </div>
                      {liveResult.is_correct !== null && liveResult.is_correct !== undefined && (
                        <div className={liveResult.is_correct ? 'text-emerald-400 text-[11px]' : 'text-rose-400 text-[11px]'}>
                          {liveResult.is_correct ? 'Matches' : 'Differs from'} the DINS label ({currentCase.survival_label}).
                        </div>
                      )}
                      {(liveResult.thinking_trace || liveResult.analysis_text) && (
                        <div className="p-2.5 rounded-xl bg-slate-950 font-mono text-[11px] text-slate-300 max-h-72 overflow-y-auto border border-slate-800 whitespace-pre-wrap">
                          {liveResult.thinking_trace || liveResult.analysis_text}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-xs text-rose-400">Error: {liveResult.error || 'request failed'}</div>
                  )}
                </div>
              )}
            </div>
          </div>
        </main>
      ) : (
        <div className="flex-1 flex items-center justify-center text-slate-500 text-sm py-20">No cases match the current filter.</div>
      )}

      {/* Data sources modal */}
      {showDataSourceModal && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setShowDataSourceModal(false)}>
          <div
            className="bg-slate-900 border border-slate-800 rounded-2xl max-w-3xl w-full p-6 space-y-3 max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2 text-white font-bold text-base">
                <BookOpen className="w-5 h-5 text-cyan-400" />
                <span>Data sources for this view</span>
              </div>
              <button onClick={() => setShowDataSourceModal(false)} className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white cursor-pointer">
                ✕
              </button>
            </div>
            <div className="space-y-2.5 text-xs text-slate-300 leading-relaxed">
              {[
                {
                  t: 'Labels and building attributes: CAL FIRE DINS',
                  d: 'Damage Inspection (DINS) records: damage class, where the fire started on the structure, defensive actions, roof/vent/siding attributes, APN, year built, assessed value, incident start date. Inspected after the fire.',
                },
                {
                  t: 'Aerial imagery: Esri World Imagery Wayback',
                  d: 'For each case the capture date at the structure point is read from the Wayback release metadata (SRC_DATE). Pre-fire = latest capture before the fire start; post-fire = earliest capture within 365 days after; latest = newest release. A crop is never replaced by current imagery; if no verified capture exists the epoch is shown as unavailable. Built by scripts/data/fetch_dev_case_imagery.py.',
                },
                {
                  t: 'Fire weather: ERA5 reanalysis via Open-Meteo',
                  d: 'Hourly 10 m wind, gusts, relative humidity and temperature at the case location for the incident start date and the next 2 days (local time). ERA5 is a ~25-31 km grid and can understate local gusts. No historical Red Flag Warning data is used. Built by scripts/data/fetch_dev_case_weather.py.',
                },
                {
                  t: 'Street-level photos',
                  d: 'Stored for a few cases only. Capture dates are not recorded, so a photo may show the building before or after the fire.',
                },
                {
                  t: 'Stored model runs',
                  d: `${runNotes.input || 'Text-only prompt'}. ${runNotes.prompt_weather_text ? runNotes.prompt_weather_text.charAt(0).toUpperCase() + runNotes.prompt_weather_text.slice(1) + '.' : ''} ${runNotes.dataset || ''}`,
                },
                {
                  t: 'Held-out results',
                  d: `${research?.source || 'paper/numbers.tex'}. Evaluated on held-out fires (by-fire splits), not on this dev set.`,
                },
              ].map((s) => (
                <div key={s.t} className="bg-slate-950/80 p-3 rounded-xl border border-slate-800">
                  <span className="font-bold text-slate-100 block mb-1">{s.t}</span>
                  <p>{s.d}</p>
                </div>
              ))}
              <div className="bg-slate-950/90 p-3 rounded-xl border border-amber-500/40 text-[11px] text-amber-200 flex gap-2">
                <Info className="w-4 h-4 shrink-0 mt-0.5" />
                <span>
                  Removed: Bellwether values, fire-spread simulations, insurance/underwriting figures, structured assessments, expert
                  diagnoses and incident forensics shown by earlier versions. They were generated from the true outcome label.
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Response text modal */}
      {selectedTraceModal && (
        <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-4" onClick={() => setSelectedTraceModal(null)}>
          <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-4xl w-full max-h-[90vh] flex flex-col overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
              <div>
                <h3 className="text-sm font-bold text-white">{selectedTraceModal.modelName} · stored response</h3>
                <p className="text-[11px] text-slate-400">
                  Case #{selectedTraceModal.caseId} · {selectedTraceModal.trace.length} characters (truncated at storage)
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleCopyTrace(selectedTraceModal.trace, 'modal')}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center gap-1.5 cursor-pointer"
                >
                  <Copy className="w-3.5 h-3.5" />
                  <span>{copyFeedback === 'modal' ? 'Copied' : 'Copy'}</span>
                </button>
                <button onClick={() => setSelectedTraceModal(null)} className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-700 hover:bg-slate-600 text-white cursor-pointer">
                  Close
                </button>
              </div>
            </div>
            <div className="p-6 overflow-y-auto flex-1 text-slate-200 font-mono text-xs leading-relaxed whitespace-pre-wrap">{selectedTraceModal.trace}</div>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- fire weather card

function FireWeatherCard({ c }: { c: CaseItem }) {
  const w = c.fire_weather;
  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3 gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <Wind className="w-5 h-5 text-sky-400" />
          <div>
            <h3 className="text-sm font-bold text-white">Fire weather at this location</h3>
            <span className="text-[11px] text-slate-400">
              {w?.available ? `${w.window_start} to ${w.window_end} (fire start + 2 days), local time` : 'Unavailable'}
            </span>
          </div>
        </div>
      </div>
      {w?.available ? (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <Fact
              label="Max 10 m wind"
              value={
                <>
                  {fmtNum(w.max_wind_mph, 1)} mph from {w.max_wind_dir_compass} ({w.max_wind_dir_deg}°)
                  <span className="block text-[10px] font-normal text-slate-500">{fmtLocal(w.max_wind_time_local)}</span>
                </>
              }
            />
            <Fact
              label="Max gust"
              value={
                <>
                  {fmtNum(w.max_gust_mph, 1)} mph from {w.max_gust_dir_compass} ({w.max_gust_dir_deg}°)
                  <span className="block text-[10px] font-normal text-slate-500">{fmtLocal(w.max_gust_time_local)}</span>
                </>
              }
            />
            <Fact
              label="Min relative humidity"
              value={
                <>
                  {fmtNum(w.min_rh_pct)}%<span className="block text-[10px] font-normal text-slate-500">{fmtLocal(w.min_rh_time_local)}</span>
                </>
              }
            />
            <Fact
              label="Max temperature"
              value={
                <>
                  {fmtNum(w.max_temp_f, 1)} °F<span className="block text-[10px] font-normal text-slate-500">{fmtLocal(w.max_temp_time_local)}</span>
                </>
              }
            />
          </div>
          <SourceLine>
            {w.source}. Grid point {w.grid_latitude?.toFixed(3)}, {w.grid_longitude?.toFixed(3)}
            {w.grid_latitude !== undefined && w.grid_longitude !== undefined
              ? ` (${haversineKm(c.latitude, c.longitude, w.grid_latitude, w.grid_longitude).toFixed(1)} km from the structure)`
              : ''}
            . Not a record of conditions at the structure; no Red Flag Warning data.{' '}
            {w.request_url && (
              <a href={w.request_url} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 inline-flex items-center gap-0.5">
                API request <ExternalLink className="w-2.5 h-2.5" />
              </a>
            )}
          </SourceLine>
        </>
      ) : (
        <div className="text-xs text-slate-400">
          {w?.reason || 'No fire-weather record for this case.'} <SourceLine>{w?.source || 'ERA5 reanalysis via Open-Meteo'}</SourceLine>
        </div>
      )}
      <div className="flex items-start gap-1.5 text-[10px] text-slate-500">
        <MapPin className="w-3 h-3 mt-0.5 shrink-0" />
        <span>Fuel type and fuel moisture are not shown: no per-case source is available in this dataset.</span>
      </div>
    </div>
  );
}
