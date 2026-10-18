'use client';

import React, { useState, useEffect } from 'react';
import {
  Eye,
  Flame,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Cpu,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  Play,
  RotateCcw,
  Sparkles,
  Info,
  Layers,
  MapPin,
  Wind,
  CheckCircle2,
  XCircle,
  HelpCircle,
  ExternalLink,
  BookOpen,
  Maximize2,
  Minimize2,
} from 'lucide-react';

interface CaseItem {
  id: string | number;
  incident: string;
  county: string;
  latitude: number;
  longitude: number;
  year_built: string | number;
  structure_type: string;
  roof_construction: string;
  vent_screen: string;
  exterior_siding: string;
  deck_on_grade: string;
  deck_elevated: string;
  fence_attached: string;
  eaves: string;
  survival_label: 'Destroyed' | 'Survived';
  raw_damage: string;
  ignition_point: string;
  ignition_source: string;
  defensive_actions: string;
  image_url: string;
  has_image: boolean;
  expert_diagnosis: {
    title: string;
    diagnosis: string;
    recommendation: string;
  };
}

interface ModelOutput {
  predicted_survival: 'Destroyed' | 'Survived' | 'Unknown';
  predicted_vulnerability: string;
  predicted_mechanism: string;
  is_survival_correct: boolean;
  locus_match: boolean;
  latency_sec: number;
  thinking_trace_snippet?: string;
  full_reasoning_trace?: string;
  analysis_text?: string;
}

interface ModelScoreboardEntry {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  false_alarm_rate: number;
  locus_attribution_accuracy: number;
  confusion_matrix: {
    TP: number;
    FP: number;
    TN: number;
    FN: number;
  };
}

interface VisionBenchmarkViewProps {
  onSwitchToMap: () => void;
}

const MODEL_CONFIGS: Record<string, { label: string; badge: string; color: string; border: string }> = {
  'MiniMax-M3': {
    label: 'MiniMax-M3',
    badge: 'CoT Deductive (2026)',
    color: 'from-amber-500/20 to-orange-500/10 text-orange-300',
    border: 'border-orange-500/30',
  },
  'claude-haiku-5-5': {
    label: 'Claude Haiku 5.5',
    badge: 'Anthropic Fast Native Reasoning',
    color: 'from-cyan-500/20 to-blue-500/10 text-cyan-300',
    border: 'border-cyan-500/30',
  },
  'claude-sonnet-5-5': {
    label: 'Claude Sonnet 5.5',
    badge: 'Anthropic Frontier Reasoning',
    color: 'from-purple-500/20 to-indigo-500/10 text-purple-300',
    border: 'border-purple-500/30',
  },
  'Qwen3.8-27B': {
    label: 'Qwen3.8-27B',
    badge: 'Remote vLLM Open Weights',
    color: 'from-emerald-500/20 to-teal-500/10 text-emerald-300',
    border: 'border-emerald-500/30',
  },
};

export default function VisionBenchmarkView({ onSwitchToMap }: VisionBenchmarkViewProps) {
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [modelResults, setModelResults] = useState<Record<string, Record<string, ModelOutput>>>({});
  const [scoreboard, setScoreboard] = useState<ModelScoreboardEntry[]>([]);
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [filterType, setFilterType] = useState<'all' | 'survived' | 'destroyed' | 'saves'>('all');
  const [loading, setLoading] = useState<boolean>(true);
  const [expandedThinking, setExpandedThinking] = useState<Record<string, boolean>>({});
  const [showDataSourceModal, setShowDataSourceModal] = useState<boolean>(false);
  const [liveModel, setLiveModel] = useState<string>('claude-sonnet-5-5');
  const [liveLoading, setLiveLoading] = useState<boolean>(false);
  const [liveResult, setLiveResult] = useState<any>(null);

  // Fetch benchmark data from backend
  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const [casesRes, resultsRes, metricsRes] = await Promise.all([
          fetch('http://localhost:8000/api/vlm/benchmark/cases').catch(() => null),
          fetch('http://localhost:8000/api/vlm/benchmark/results').catch(() => null),
          fetch('http://localhost:8000/api/vlm/benchmark/metrics').catch(() => null),
        ]);

        if (casesRes && casesRes.ok) {
          const casesData = await casesRes.json();
          setCases(casesData.items || []);
        }

        if (resultsRes && resultsRes.ok) {
          const resData = await resultsRes.json();
          setModelResults(resData.cases || {});
        }

        if (metricsRes && metricsRes.ok) {
          const metricsData = await metricsRes.json();
          setScoreboard(metricsData.scoreboard || []);
        }
      } catch (err) {
        console.error('Error fetching VLM benchmark data:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  // Filter cases based on user selection
  const filteredCases = cases.filter((item) => {
    if (filterType === 'survived') return item.survival_label === 'Survived';
    if (filterType === 'destroyed') return item.survival_label === 'Destroyed';
    if (filterType === 'saves') {
      return (
        item.survival_label === 'Survived' &&
        (item.defensive_actions.toLowerCase().includes('hand crew') ||
          item.defensive_actions.toLowerCase().includes('engine') ||
          item.vent_screen.includes('<= 1/8'))
      );
    }
    return true;
  });

  const currentCase = filteredCases[currentIndex] || cases[0];
  const currentCaseResults = currentCase ? modelResults[String(currentCase.id)] || {} : {};

  const toggleThinking = (modelName: string) => {
    setExpandedThinking((prev) => ({ ...prev, [modelName]: !prev[modelName] }));
  };

  const handlePrev = () => {
    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : filteredCases.length - 1));
    setLiveResult(null);
  };

  const handleNext = () => {
    setCurrentIndex((prev) => (prev < filteredCases.length - 1 ? prev + 1 : 0));
    setLiveResult(null);
  };

  const handleRunLivePrediction = async () => {
    if (!currentCase) return;
    setLiveLoading(true);
    setLiveResult(null);
    try {
      const res = await fetch('http://localhost:8000/api/vlm/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: liveModel,
          case_id: String(currentCase.id),
        }),
      });
      const data = await res.json();
      setLiveResult(data);
    } catch (err) {
      setLiveResult({ success: false, error: String(err) });
    } finally {
      setLiveLoading(false);
    }
  };

  return (
    <div className="flex-1 h-full flex flex-col bg-slate-950 text-slate-100 overflow-y-auto">
      {/* Top Header & Application Bar */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md px-6 py-3.5 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-30">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-rose-600 flex items-center justify-center shadow-lg shadow-orange-500/20">
            <Eye className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-white tracking-wide">
                WildfireReason-VLM
              </h1>
              <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-orange-500/20 text-orange-400 border border-orange-500/30">
                CVPR 2026 Frontier Track
              </span>
              <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                CAL FIRE DINS Ground-Truth
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Zero-Shot Forensic Ignition Attribution & Defect Discovery Engine across 4 Frontier Reasoning Models
            </p>
          </div>
        </div>

        {/* Global Action Bar */}
        <div className="flex items-center gap-3">
          {/* Switch to GIS Map View */}
          <button
            onClick={onSwitchToMap}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all shadow-sm"
          >
            <Layers className="w-3.5 h-3.5 text-orange-400" />
            <span>Switch to 3D GIS Map</span>
          </button>

          {/* Data Source & Methodology Button */}
          <button
            onClick={() => setShowDataSourceModal(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/80 transition-all"
          >
            <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
            <span>Data Sources & Benchmark Protocol</span>
          </button>
        </div>
      </header>

      {/* Cross-Model Benchmark Scoreboard (Top Metric Ribbon) */}
      <section className="bg-slate-900/40 border-b border-slate-800/60 px-6 py-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-purple-400" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
              Cross-Model Empirical Scoreboard (50 Ground-Truth Cases: 25 Destroyed vs 25 Survived)
            </span>
          </div>
          <span className="text-[11px] text-rose-400 font-medium">
            ⚠️ Key Finding: All frontier models perform at ~50% (coin-toss) with 88%–100% False Alarm Rates
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {scoreboard.map((item) => {
            const isQwen = item.model_name.includes('Qwen');
            const isSonnet = item.model_name.includes('sonnet');
            const isHaiku = item.model_name.includes('haiku');
            return (
              <div
                key={item.model_name}
                className="bg-slate-900/80 border border-slate-800 rounded-xl p-3 hover:border-slate-700 transition-all shadow-sm relative overflow-hidden"
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-bold text-white tracking-wide">
                    {item.model_name}
                  </span>
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                      isSonnet
                        ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                        : isHaiku
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                        : isQwen
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                        : 'bg-orange-500/20 text-orange-300 border border-orange-500/30'
                    }`}
                  >
                    {isSonnet ? 'SOTA Frontier' : isHaiku ? 'Fast CoT' : isQwen ? 'Open vLLM' : 'Commercial CoT'}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center pt-1 border-t border-slate-800/60">
                  <div>
                    <div className="text-[10px] text-slate-400">Accuracy</div>
                    <div className="text-sm font-bold text-slate-100">{item.accuracy}%</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-rose-400 font-medium">False Alarm</div>
                    <div className="text-sm font-bold text-rose-400">{item.false_alarm_rate}%</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-400">Locus Acc</div>
                    <div className="text-sm font-bold text-amber-300">
                      {item.locus_attribution_accuracy}%
                    </div>
                  </div>
                </div>

                <div className="mt-2 text-[10px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800/40">
                  <span>CM: TP={item.confusion_matrix.TP} | FP={item.confusion_matrix.FP}</span>
                  <span>TN={item.confusion_matrix.TN} | FN={item.confusion_matrix.FN}</span>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Case Navigation & Filtering Sub-Header */}
      <section className="bg-slate-900/60 border-b border-slate-800/80 px-6 py-2.5 flex flex-wrap items-center justify-between gap-3">
        {/* Filter Pills */}
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-slate-400 mr-1 font-medium">Filters:</span>
          {(
            [
              { key: 'all', label: `All (${cases.length})` },
              { key: 'survived', label: 'Survived (Defect Zero)' },
              { key: 'destroyed', label: 'Destroyed' },
              { key: 'saves', label: 'Tactical Firefighter Saves' },
            ] as const
          ).map((t) => (
            <button
              key={t.key}
              onClick={() => {
                setFilterType(t.key);
                setCurrentIndex(0);
                setLiveResult(null);
              }}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                filterType === t.key
                  ? 'bg-orange-500 text-white shadow-md shadow-orange-500/20'
                  : 'bg-slate-800/60 text-slate-300 hover:bg-slate-800 border border-slate-700/60'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Case Selector Dropdown & Stepper */}
        <div className="flex items-center gap-2">
          <button
            onClick={handlePrev}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all"
            title="Previous Case"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>

          <select
            value={currentCase ? currentCase.id : ''}
            onChange={(e) => {
              const idx = filteredCases.findIndex((c) => String(c.id) === e.target.value);
              if (idx !== -1) {
                setCurrentIndex(idx);
                setLiveResult(null);
              }
            }}
            className="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-orange-500"
          >
            {filteredCases.map((c, i) => (
              <option key={c.id} value={c.id}>
                Case #{c.id} - {c.survival_label} ({c.incident} Fire) {c.defensive_actions !== 'None' ? '★ Saved by Hand Crew' : ''}
              </option>
            ))}
          </select>

          <button
            onClick={handleNext}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all"
            title="Next Case"
          >
            <ChevronRight className="w-4 h-4" />
          </button>

          <span className="text-xs text-slate-400 font-mono ml-1">
            {filteredCases.length > 0 ? currentIndex + 1 : 0} of {filteredCases.length}
          </span>
        </div>
      </section>

      {/* Main Case Inspector Workstation */}
      {loading ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 text-slate-400">
          <div className="w-8 h-8 border-4 border-orange-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm">Loading Ground-Truth DINS Cases & Model Reasoning Matrix...</span>
        </div>
      ) : currentCase ? (
        <main className="flex-1 p-6 space-y-6">
          {/* Top Split: Left Parcel Imagery & GT Details vs Right Multi-Model Predictions */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column: Aerial Imagery & Forensic GT Card (5 Cols) */}
            <div className="lg:col-span-5 space-y-4">
              {/* High-Resolution Aerial Imagery Card */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
                <div className="relative aspect-video w-full bg-slate-950 overflow-hidden group">
                  <img
                    src={`http://localhost:8000/api/vlm/benchmark/image/${currentCase.id}`}
                    alt={`High-Res Aerial Imagery for Case ${currentCase.id}`}
                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                    onError={(e) => {
                      // Fallback image placeholder
                      (e.target as HTMLElement).style.display = 'none';
                    }}
                  />
                  {/* Image Overlays */}
                  <div className="absolute top-3 left-3 flex flex-wrap gap-1.5">
                    <span className="px-2 py-0.5 rounded-md text-[11px] font-bold bg-slate-900/90 text-slate-200 backdrop-blur-md border border-white/10 shadow-md">
                      ESRI World Imagery (Zoom 18)
                    </span>
                    <span className="px-2 py-0.5 rounded-md text-[11px] font-medium bg-black/70 text-slate-300 backdrop-blur-md">
                      0.3m / Pixel Orthorectified
                    </span>
                  </div>

                  <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between text-[11px] bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10">
                    <div className="flex items-center gap-1.5 text-slate-300">
                      <MapPin className="w-3.5 h-3.5 text-rose-400" />
                      <span>{currentCase.latitude.toFixed(4)}°N, {currentCase.longitude.toFixed(4)}°W</span>
                    </div>
                    <span className="text-slate-400 font-medium">
                      {currentCase.incident} Fire, {currentCase.county} Co.
                    </span>
                  </div>
                </div>

                {/* Weather & Meteorological Forcing Banner */}
                <div className="bg-rose-950/20 border-t border-rose-900/30 p-3 flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2 text-rose-300">
                    <Wind className="w-4 h-4 text-rose-400 animate-pulse" />
                    <span className="font-semibold">Red Flag Warning Forcing:</span>
                    <span>Gusts 25-35 mph | DFM &lt; 6%</span>
                  </div>
                  <span className="text-rose-400/80 font-mono text-[11px]">Chaparral/Timber Fuel</span>
                </div>
              </div>

              {/* CAL FIRE DINS Forensic Ground-Truth Panel */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-5 h-5 text-emerald-400" />
                    <h3 className="text-sm font-bold text-white tracking-wide">
                      CAL FIRE DINS Forensic Inspection
                    </h3>
                  </div>
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-extrabold uppercase tracking-wider ${
                      currentCase.survival_label === 'Survived'
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm shadow-emerald-500/20'
                        : 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-sm shadow-rose-500/20'
                    }`}
                  >
                    Actual GT: {currentCase.survival_label}
                  </span>
                </div>

                {/* Key Ground Truth Facts Grid */}
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                    <span className="text-slate-400 block text-[11px]">Inspection Damage Level</span>
                    <span className="font-semibold text-slate-200">{currentCase.raw_damage}</span>
                  </div>

                  <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                    <span className="text-slate-400 block text-[11px]">Actual Forensic Ignition Point</span>
                    <span className="font-semibold text-amber-300">{currentCase.ignition_point || 'None'}</span>
                  </div>

                  <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80 col-span-2">
                    <span className="text-slate-400 block text-[11px]">Emergency Responder Tactical Actions</span>
                    <span className={`font-semibold ${currentCase.defensive_actions !== 'None' ? 'text-cyan-300' : 'text-slate-400'}`}>
                      {currentCase.defensive_actions}
                      {currentCase.defensive_actions !== 'None' && ' (Active Fire Suppression Save)'}
                    </span>
                  </div>
                </div>

                {/* Building Hardening Specs */}
                <div className="pt-2 border-t border-slate-800/60 space-y-2">
                  <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">
                    Physical Structural Hardening Envelope
                  </span>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Year Built:</span>
                      <span className="text-slate-200 font-medium">{currentCase.year_built}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Structure Type:</span>
                      <span className="text-slate-200 font-medium">{currentCase.structure_type.replace('Single Family Residence', 'SFR')}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Roof Material:</span>
                      <span className="text-slate-200 font-medium">{currentCase.roof_construction}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Vent Mesh:</span>
                      <span className={`font-semibold ${currentCase.vent_screen.includes('<= 1/8') ? 'text-emerald-400' : 'text-slate-200'}`}>
                        {currentCase.vent_screen}
                      </span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Exterior Siding:</span>
                      <span className="text-slate-200 font-medium">{currentCase.exterior_siding}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Ground Deck:</span>
                      <span className="text-slate-200 font-medium">{currentCase.deck_on_grade}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Eaves Design:</span>
                      <span className="text-slate-200 font-medium">{currentCase.eaves}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-400">Attached Fence:</span>
                      <span className="text-slate-200 font-medium">{currentCase.fence_attached}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column: Multi-Model Deductive Reasoning Matrix (7 Cols) */}
            <div className="lg:col-span-7 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Flame className="w-5 h-5 text-orange-400" />
                  <h3 className="text-sm font-bold text-white tracking-wide">
                    Multi-Model Deductive Reasoning & Prediction Matrix
                  </h3>
                </div>
                <span className="text-xs text-slate-400">
                  Pre-computed Zero-Shot Chain-of-Thought (CoT)
                </span>
              </div>

              {/* 4 Models Display Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.keys(MODEL_CONFIGS).map((modelKey) => {
                  const cfg = MODEL_CONFIGS[modelKey];
                  const res = currentCaseResults[modelKey];
                  const isCorrect = res ? res.is_survival_correct : false;
                  const isExpanded = expandedThinking[modelKey] || false;

                  return (
                    <div
                      key={modelKey}
                      className={`bg-slate-900 border ${cfg.border} rounded-2xl p-4 shadow-lg flex flex-col justify-between space-y-3 transition-all hover:border-slate-600`}
                    >
                      {/* Model Card Header */}
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-bold text-white tracking-wide">
                            {cfg.label}
                          </span>
                          <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${cfg.color}`}>
                            {cfg.badge}
                          </span>
                        </div>

                        {/* Prediction vs Ground Truth Outcome */}
                        {res ? (
                          <div className="space-y-2">
                            <div className="flex items-center justify-between bg-slate-950/70 p-2.5 rounded-xl border border-slate-800">
                              <span className="text-xs text-slate-400">Predicted Survival:</span>
                              <div className="flex items-center gap-1.5">
                                <span
                                  className={`text-xs font-bold px-2 py-0.5 rounded ${
                                    res.predicted_survival === 'Destroyed'
                                      ? 'bg-rose-500/20 text-rose-300'
                                      : 'bg-emerald-500/20 text-emerald-300'
                                  }`}
                                >
                                  {res.predicted_survival}
                                </span>
                                {isCorrect ? (
                                  <span className="flex items-center text-[11px] text-emerald-400 font-semibold gap-0.5">
                                    <CheckCircle2 className="w-3.5 h-3.5" /> Correct
                                  </span>
                                ) : (
                                  <span className="flex items-center text-[11px] text-rose-400 font-semibold gap-0.5">
                                    <XCircle className="w-3.5 h-3.5" />
                                    {currentCase.survival_label === 'Survived' ? 'False Alarm' : 'Miss'}
                                  </span>
                                )}
                              </div>
                            </div>

                            {/* Attributions: Vulnerability & Failure Mechanism */}
                            <div className="grid grid-cols-2 gap-2 text-[11px]">
                              <div className="bg-slate-950/50 p-2 rounded-lg border border-slate-800/60">
                                <span className="text-slate-400 block text-[10px]">Primary Vulnerability</span>
                                <span className="font-semibold text-amber-300">
                                  {res.predicted_vulnerability || 'Unknown'}
                                </span>
                              </div>
                              <div className="bg-slate-950/50 p-2 rounded-lg border border-slate-800/60">
                                <span className="text-slate-400 block text-[10px]">Failure Mechanism</span>
                                <span className="font-semibold text-orange-300">
                                  {res.predicted_mechanism || 'Unknown'}
                                </span>
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="p-4 bg-slate-950/40 rounded-xl text-center text-xs text-slate-500">
                            No evaluation record for this case
                          </div>
                        )}
                      </div>

                      {/* Collapsible Reasoning Trace (Thinking tokens) */}
                      {res && (
                        <div className="pt-2 border-t border-slate-800/80">
                          <button
                            onClick={() => toggleThinking(modelKey)}
                            className="w-full flex items-center justify-between text-left text-xs font-medium text-slate-300 hover:text-white py-1 transition-colors"
                          >
                            <span className="flex items-center gap-1.5 text-purple-300">
                              <Sparkles className="w-3.5 h-3.5" />
                              <span>{isExpanded ? 'Hide' : 'View'} CoT Thinking Trace</span>
                            </span>
                            {isExpanded ? (
                              <ChevronUp className="w-3.5 h-3.5 text-slate-400" />
                            ) : (
                              <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
                            )}
                          </button>

                          {isExpanded && (
                            <div className="mt-2 p-2.5 rounded-xl bg-slate-950 text-[11px] font-mono text-slate-300 max-h-48 overflow-y-auto border border-purple-500/20 leading-relaxed">
                              <div className="text-[10px] text-purple-400 font-bold mb-1 uppercase tracking-wider">
                                Deductive Reasoning Trace (&lt;think&gt; tokens):
                              </div>
                              {res.thinking_trace_snippet || res.full_reasoning_trace || 'No raw trace captured'}
                            </div>
                          )}

                          {/* Analysis Excerpt */}
                          {res.analysis_text && !isExpanded && (
                            <p className="mt-2 text-[11px] text-slate-400 line-clamp-3 leading-relaxed italic bg-slate-950/30 p-2 rounded-lg">
                              "{res.analysis_text}"
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Live Inference Testing Bar */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col md:flex-row items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <Play className="w-4 h-4 text-emerald-400" />
                  <div>
                    <span className="text-xs font-bold text-white block">Interactive Live Model Query</span>
                    <span className="text-[11px] text-slate-400">Trigger on-demand inference against current case</span>
                  </div>
                </div>

                <div className="flex items-center gap-2.5 w-full md:w-auto">
                  <select
                    value={liveModel}
                    onChange={(e) => setLiveModel(e.target.value)}
                    className="bg-slate-800 border border-slate-700 text-xs rounded-lg px-3 py-1.5 text-slate-200 focus:outline-none focus:border-orange-500"
                  >
                    <option value="MiniMax-M3">MiniMax-M3 (CoT)</option>
                    <option value="claude-haiku-5-5">Claude Haiku 5.5</option>
                    <option value="claude-sonnet-5-5">Claude Sonnet 5.5</option>
                    <option value="Qwen3.8-27B">Qwen3.8-27B (vLLM)</option>
                  </select>

                  <button
                    onClick={handleRunLivePrediction}
                    disabled={liveLoading}
                    className="px-4 py-1.5 rounded-lg text-xs font-bold bg-gradient-to-r from-orange-500 to-rose-600 hover:from-orange-600 hover:to-rose-700 text-white shadow-md shadow-orange-500/20 disabled:opacity-50 transition-all flex items-center gap-1.5 whitespace-nowrap"
                  >
                    {liveLoading ? (
                      <>
                        <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        <span>Inferring...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>Run Live Call</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Live Inference Output Display */}
              {liveResult && (
                <div className="bg-slate-900/90 border border-orange-500/40 rounded-2xl p-4 shadow-xl space-y-2 animate-fadeIn">
                  <div className="flex items-center justify-between text-xs border-b border-slate-800 pb-2">
                    <span className="font-bold text-orange-400">Live Result ({liveResult.model_name})</span>
                    <span className="text-slate-400 font-mono">Latency: {liveResult.latency_sec}s</span>
                  </div>
                  {liveResult.success ? (
                    <div className="space-y-2 text-xs">
                      <div className="flex items-center gap-3">
                        <span className="text-slate-400">Predicted:</span>
                        <span className="font-bold text-white px-2 py-0.5 rounded bg-slate-800">
                          {liveResult.predicted_survival}
                        </span>
                        <span className="text-slate-400">Vulnerability:</span>
                        <span className="font-bold text-amber-300">{liveResult.predicted_vulnerability}</span>
                        <span className="text-slate-400">Mechanism:</span>
                        <span className="font-bold text-orange-300">{liveResult.predicted_mechanism}</span>
                      </div>
                      {liveResult.thinking_trace && (
                        <div className="p-2.5 rounded-xl bg-slate-950 font-mono text-[11px] text-slate-300 max-h-36 overflow-y-auto border border-slate-800">
                          {liveResult.thinking_trace}
                        </div>
                      )}
                      {liveResult.analysis_text && (
                        <p className="text-slate-300 text-xs italic bg-slate-950/40 p-2 rounded-lg">
                          "{liveResult.analysis_text}"
                        </p>
                      )}
                    </div>
                  ) : (
                    <div className="text-xs text-rose-400">Error: {liveResult.error}</div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Bottom Full-Width Section: Expert Fire Forensic Defect Analysis (错在哪) */}
          <section className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-96 h-96 bg-orange-500/5 rounded-full blur-3xl pointer-events-none" />

            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4 mb-4">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-orange-500/20 text-orange-400 flex items-center justify-center border border-orange-500/30">
                  <AlertTriangle className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white tracking-wide">
                    Expert Fire Forensic Defect Analysis: Why the Models Failed (错在哪)
                  </h3>
                  <span className="text-xs text-slate-400">
                    Physical Causality Audit & Discrepancy Breakdown for Case #{currentCase.id}
                  </span>
                </div>
              </div>

              <span className="px-3 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                {currentCase.expert_diagnosis.title}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-6 text-xs leading-relaxed">
              <div className="md:col-span-8 space-y-3">
                <div className="text-slate-300 text-sm leading-relaxed bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
                  {currentCase.expert_diagnosis.diagnosis}
                </div>
              </div>

              <div className="md:col-span-4 bg-slate-950/80 p-4 rounded-xl border border-slate-800/80 flex flex-col justify-between space-y-3">
                <div>
                  <span className="text-[11px] font-bold text-purple-400 uppercase tracking-wider block mb-1">
                    CVPR 2026 Paper Implication:
                  </span>
                  <p className="text-slate-300 text-xs leading-relaxed">
                    {currentCase.expert_diagnosis.recommendation}
                  </p>
                </div>
                <div className="pt-2 border-t border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
                  <span>Targeted RLVR Reward</span>
                  <span className="font-mono text-emerald-400 font-semibold">R_physics + R_locus</span>
                </div>
              </div>
            </div>
          </section>
        </main>
      ) : (
        <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
          No benchmark cases found matching current filter.
        </div>
      )}

      {/* Data Source & Methodology Modal */}
      {showDataSourceModal && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full p-6 shadow-2xl space-y-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2 text-white font-bold text-base">
                <BookOpen className="w-5 h-5 text-cyan-400" />
                <span>WildfireReason-DINS Benchmark Methodology</span>
              </div>
              <button
                onClick={() => setShowDataSourceModal(false)}
                className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
              <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800">
                <span className="font-bold text-cyan-300 block mb-1 text-sm">
                  1. Ground-Truth Data Source: CAL FIRE DINS
                </span>
                <p>
                  All structural parcel labels originate from the official State of California Damage Inspection Specialist (DINS)
                  forensic database. Trained fire marshals physically survey post-incident structures to document verified damage severity,
                  roof class, vent mesh sizes, decking combustibility, and the precise initial ignition locus (<code>WHEREFIRESTARTEDONSTRUCTURE</code>).
                </p>
              </div>

              <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800">
                <span className="font-bold text-orange-300 block mb-1 text-sm">
                  2. Sub-Meter Aerial Orthoimagery Pairing
                </span>
                <p>
                  Each structural parcel coordinate is paired with Zoom 18 (0.3m ground resolution) aerial imagery tiles from ESRI World Imagery,
                  capturing the actual building geometry, 0-5ft non-combustible perimeter, tree canopy overhangs, and adjacent driveway fuel breaks.
                </p>
              </div>

              <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800">
                <span className="font-bold text-purple-300 block mb-1 text-sm">
                  3. Standardized Multi-Model Evaluation Protocol
                </span>
                <p>
                  Models are evaluated on an identical deductive prompt instructing Chain-of-Thought (CoT) reasoning over physical wildfire dynamics,
                  weather forcing (Red Flag Warning, 25-35 mph gusts, DFM &lt; 6%), and structural hardening specs. Outputs are parsed into
                  binary survival predictions, primary vulnerability, and failure mechanism.
                </p>
              </div>
            </div>

            <div className="pt-2 border-t border-slate-800 flex justify-end">
              <button
                onClick={() => setShowDataSourceModal(false)}
                className="px-4 py-1.5 rounded-lg text-xs font-bold bg-slate-800 hover:bg-slate-700 text-white transition-all"
              >
                Close Documentation
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
